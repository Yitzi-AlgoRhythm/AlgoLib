using AlgoLib.Algorithms.Abstractions;
using AlgoLib.Data.Result_Objects;
using MathNet.Numerics.LinearAlgebra;
using ParameterDataLib;
using ThpUtils.Naming;

namespace AlgoLib.Algorithms.Implementations
{
    public class KalmanFilter : IKalmanFilter
    {
        private const double MinVariance = 1e-9;
        private const ulong StabilizationMs = 60_000;

        private enum Stage
        {
            None,
            Predicted,
            Innovated,
            Updated
        }

        // ---- Fixed by the constructor ----
        private readonly Dictionary<string, int> _parameterIndices;
        private readonly int _stateVariableCount;
        private readonly Matrix<double> _Fk;

        // ---- Baseline, set by Init ----
        private bool _isInitialized;
        private Matrix<double> _Qc = null!;
        private double _b;

        // ---- Filter state, reset by Init ----
        private Matrix<double> _Rk = null!;
        private Matrix<double> _Pk = null!;
        private Vector<double> _Xk = null!;
        private Vector<double> _Z_scores = null!;
        private ulong _lastTimestamp;
        private bool _hasTimestamp;
        private ulong _startTimestamp;
        private bool _hasStartTimestamp;

        // ---- Current step (always assigned by Predict before use) ----
        private Stage _stage;
        private ulong _stepTimestamp;
        private double _deltaT;
        private int[] _indicesInBatch = [];
        private Matrix<double> _Hk = null!, _R_prev_k = null!, _Qk = null!, _Sk = null!, _Kk = null!;
        private Vector<double> _Yk = null!, _Zk = null!;


        private const int LogInterval = 1000;

        private readonly string[] _parameterNames;

        // ---- Z-score statistics, reset by Init ----
        private int _zScoreCallCount;
        private int[] _zStatCounts = [];
        private double[] _zStatMeans = [];
        private double[] _zStatM2 = [];

        public KalmanFilter(NamingData namingData)
        {
            _parameterNames = namingData.ParameterNames;
            _stateVariableCount = _parameterNames.Length;

            _parameterIndices = [];
            for (int i = 0; i < _parameterNames.Length; i++)
            {
                _parameterIndices[_parameterNames[i]] = i;
            }

            _Fk = Matrix<double>.Build.DenseIdentity(_stateVariableCount);
        }

        /// <summary>
        /// Sets the baseline values, resets the filter as if it never ran,
        /// and restarts the stabilization period.
        /// </summary>
        /// <param name="qc">Noise intensities Qc: variance per second of each independent noise source.</param>
        /// <param name="r0">Initial measurement noise covariance (n x n).</param>
        /// <param name="forgettingFactor">Sage-Husa forgetting factor per second. Closer to 1 = longer memory.</param>
        public void Init(Vector<double> qc, Matrix<double> r0, double forgettingFactor)
        {
            _Qc = Matrix<double>.Build.DiagonalOfDiagonalVector(qc);
            _b = forgettingFactor;

            _Rk = r0.Clone();
            _Pk = r0.Clone();
            _Xk = Vector<double>.Build.Dense(_stateVariableCount, double.NaN);
            _Z_scores = Vector<double>.Build.Dense(_stateVariableCount);

            _lastTimestamp = 0;
            _hasTimestamp = false;
            _startTimestamp = 0;
            _hasStartTimestamp = false;

            _stage = Stage.None;

            _zScoreCallCount = 0;
            _zStatCounts = new int[_stateVariableCount];
            _zStatMeans = new double[_stateVariableCount];
            _zStatM2 = new double[_stateVariableCount];

            _isInitialized = true;
        }

        /// <summary>
        /// Starts a new step: registers the measurements, then predicts over dt.
        /// If not all parameters have been measured yet, the step is skipped
        /// and the following steps do nothing.
        /// </summary>
        public void Predict(ParameterData[] measurements)
        {
            EnsureInitialized();
            _stage = Stage.None;

            int measurementsCount = measurements.Length;
            if (measurementsCount == 0)
                return;

            List<int> indicesInBatch = [];
            List<double> valuesInBatch = [];
            ulong newestTimestamp = measurements[0].TimestampMS;

            foreach (ParameterData paramData in measurements)
            {
                if (_parameterIndices.TryGetValue(paramData.Name, out int parameterIndex))
                {
                    indicesInBatch.Add(parameterIndex);
                    valuesInBatch.Add(paramData.Value);

                    // If this parameter is being measured for the first time, add the measurement to X0
                    if (double.IsNaN(_Xk[parameterIndex]))
                    {
                        _Xk[parameterIndex] = paramData.Value;
                    }
                }
            }

            if (!_hasStartTimestamp)
            {
                _startTimestamp = newestTimestamp;
                _hasStartTimestamp = true;
            }
            _stepTimestamp = newestTimestamp;

            if (!AllParametersMeasured())
                return;

            _indicesInBatch = indicesInBatch.ToArray();
            _deltaT = ComputeDeltaT(newestTimestamp);

            AdjustQ(_deltaT);
            PredictStep();
            BuildMeasurementModel(indicesInBatch.ToArray(), valuesInBatch.ToArray());

            _stage = Stage.Predicted;
        }

        public void ComputeInnovation()
        {
            EnsureInitialized();
            if (_stage != Stage.Predicted)
                return;

            _Yk = _Zk - _Hk * _Xk;
            _Sk = Symmetrize(_Hk * _Pk * _Hk.Transpose() + ExtractMeasuredR(_Rk));

            _stage = Stage.Innovated;
        }

        public void Update()
        {
            EnsureInitialized();
            if (_stage != Stage.Innovated)
                return;

            // S is symmetric, so K^T = S^-1 (H P)  =>  K = (S.Solve(H P))^T
            _Kk = _Sk.Solve(_Hk * _Pk).Transpose();

            _Xk = _Xk + _Kk * _Yk;

            Matrix<double> identity = Matrix<double>.Build.DenseIdentity(_stateVariableCount);
            Matrix<double> iMinusKH = identity - _Kk * _Hk;
            Matrix<double> rMeasured = ExtractMeasuredR(_Rk);

            _Pk = Symmetrize(iMinusKH * _Pk * iMinusKH.Transpose() + _Kk * rMeasured * _Kk.Transpose());

            _stage = Stage.Updated;
        }

        /// <summary>
        /// Full-length z-score vector (only measured indices refreshed this step).
        /// Returns null if no valid step, or during the first minute after Init.
        /// </summary>
        public Vector<double>? ComputeZScores()
        {
            EnsureInitialized();
            if (_stage != Stage.Updated)
                return null;

            Vector<double> newZScores = Vector<double>.Build.Dense(_Yk.Count, i =>
            {
                double s = _Sk[i, i];
                return s > 0 ? _Yk[i] / Math.Sqrt(s) : 0.0;
            });

            for (int i = 0; i < _indicesInBatch.Length; i++)
            {
                _Z_scores[_indicesInBatch[i]] = newZScores[i];
            }

            if (!IsStabilized())
                return null;

            // Only the z-scores refreshed this step are accumulated, so stale values are not double counted
            for (int i = 0; i < _indicesInBatch.Length; i++)
            {
                AccumulateZScore(_indicesInBatch[i], newZScores[i]);
            }

            _zScoreCallCount++;
            if (_zScoreCallCount % LogInterval == 0)
            {
                LogZScoreStats();
            }

            if (_Z_scores.Any(double.IsNaN))
            {
                throw new NotFiniteNumberException();
            }

            return _Z_scores.Clone();
        }

        /// <summary>
        /// r_ii = (1 - d) * r_ii + d * (Y_i^2 - (H P- H^T)_ii),  d = 1 - b^dt. R is floored to stay positive.
        /// </summary>
        public void SageHusa()
        {
            EnsureInitialized();
            if (_stage != Stage.Updated || _deltaT <= 0.0)
                return;

            double d = 1.0 - Math.Pow(_b, _deltaT);

            // H P- H^T = S - R (both from before this update)
            Matrix<double> hphT = _Sk - ExtractMeasuredR(_R_prev_k);

            for (int measuredParamIndex = 0; measuredParamIndex < _indicesInBatch.Length; measuredParamIndex++)
            {
                int stateVariableIndex = _indicesInBatch[measuredParamIndex];
                double estimate = _Yk[measuredParamIndex] * _Yk[measuredParamIndex] - hphT[measuredParamIndex, measuredParamIndex];
                double updated = (1.0 - d) * _R_prev_k[stateVariableIndex, stateVariableIndex] + d * estimate;
                _Rk[stateVariableIndex, stateVariableIndex] = Math.Max(updated, MinVariance);
            }
        }

        /// <summary>
        /// Returns null if no valid step, or during the first minute after Init.
        /// </summary>
        public KalmanFilterResult? GetStepResult()
        {
            EnsureInitialized();
            if (_stage != Stage.Updated || !IsStabilized())
                return null;

            return new KalmanFilterResult()
            {
                // Copied so later steps don't mutate a result a consumer is still reading
                ParameterZScores = _Z_scores.Clone(),
                IndicesMeasured = _indicesInBatch,
                ParameterValues = _Xk.ToArray(),
                DeltaT = _deltaT,
            };
        }

        private void EnsureInitialized()
        {
            if (!_isInitialized)
                throw new InvalidOperationException($"{nameof(Init)} must be called first.");
        }

        private bool IsStabilized()
        {
            return _stepTimestamp >= _startTimestamp
                && _stepTimestamp - _startTimestamp >= StabilizationMs;
        }

        private double ComputeDeltaT(ulong newestTimestamp)
        {
            if (!_hasTimestamp)
            {
                _lastTimestamp = newestTimestamp;
                _hasTimestamp = true;
                return 0.0;
            }

            if (newestTimestamp <= _lastTimestamp)
                return 0.0;

            double dt = (newestTimestamp - _lastTimestamp) / 1000.0;
            _lastTimestamp = newestTimestamp;
            return dt;
        }

        private void AdjustQ(double dt)
        {
            _Qk = dt > 0.0
                ? _Qc * dt
                : Matrix<double>.Build.Dense(_stateVariableCount, _stateVariableCount);
        }

        private void PredictStep()
        {
            _Xk = _Fk * _Xk;
            _Pk = Symmetrize(_Pk + _Qk);
        }

        private void BuildMeasurementModel(int[] indices, double[] values)
        {
            _Hk = Matrix<double>.Build.Dense(indices.Length, _stateVariableCount);

            for (int row = 0; row < indices.Length; row++)
            {
                _Hk[row, indices[row]] = 1.0;
            }

            _Zk = Vector<double>.Build.DenseOfArray(values);
            _R_prev_k = _Rk.Clone();
        }

        private Matrix<double> ExtractMeasuredR(Matrix<double> fullR)
        {
            return _Hk * fullR * _Hk.Transpose();
        }

        private static Matrix<double> Symmetrize(Matrix<double> a)
        {
            return (a + a.Transpose()) * 0.5;
        }

        private bool AllParametersMeasured()
        {
            return !_Xk.Any(double.IsNaN);
        }

        // Welford's online algorithm: numerically stable running mean and variance
        private void AccumulateZScore(int stateVariableIndex, double z)
        {
            _zStatCounts[stateVariableIndex]++;
            double delta = z - _zStatMeans[stateVariableIndex];
            _zStatMeans[stateVariableIndex] += delta / _zStatCounts[stateVariableIndex];
            _zStatM2[stateVariableIndex] += delta * (z - _zStatMeans[stateVariableIndex]);
        }

        private void LogZScoreStats()
        {
            Console.WriteLine($"[KalmanFilter] Z-score statistics after {_zScoreCallCount} computations:");

            for (int i = 0; i < _stateVariableCount; i++)
            {
                int count = _zStatCounts[i];
                double mean = count > 0 ? _zStatMeans[i] : double.NaN;
                double stdDev = count > 1 ? Math.Sqrt(_zStatM2[i] / (count - 1)) : double.NaN; // sample standard deviation

                Console.WriteLine($"  {_parameterNames[i]}: mean = {mean:F4}, stdDev = {stdDev:F4}, n = {count}");
            }
        }
    }
}