using AlgoLib.Algorithms.Abstractions;
using AlgoLib.Data.Result_Objects;
using MathNet.Numerics.LinearAlgebra;
using ThpUtils.Naming;

namespace AlgoLib.Algorithms.Implementations
{
    public class Mahalanobis : IMahalanobis
    {
        private readonly string[] _parameterNames;
        private readonly int _parameterCount;

        // ---- Baseline, set by Init ----
        private bool _isInitialized;
        private double _stepSize;
        private Matrix<double> _correlationMatrixInversed = null!;
        private double _distanceSquaredThreshold;

        // ---- Current step only (null until ComputeD2 is called) ----
        private KalmanFilterResult? _kalmanResult;
        private double _distanceSquared;
        private Dictionary<string, double> _outliers = [];

        public Mahalanobis(NamingData namingData)
        {
            _parameterNames = namingData.ParameterNames;
            _parameterCount = _parameterNames.Length;
        }

        /// <summary>Sets the baseline values and discards the current step's state.</summary>
        /// <param name="stepSize">
        /// How far a z-score is moved towards its mean in each iteration of finding outlier parameters.
        /// </param>
        /// <param name="covarianceMatrix">Covariance of the parameters' Z-scores.</param>
        /// <param name="distanceSquaredThreshold">A set of Z-scores is anomalous if MD^2 crosses this threshold.</param>
        public void Init(double stepSize, Matrix<double> covarianceMatrix, double distanceSquaredThreshold)
        {
            _stepSize = stepSize;
            _correlationMatrixInversed = CovarianceToCorrelationInversed(covarianceMatrix);
            _distanceSquaredThreshold = distanceSquaredThreshold;

            _kalmanResult = null;
            _distanceSquared = 0.0;
            _outliers = [];

            _isInitialized = true;
        }

        public double ComputeD2(KalmanFilterResult kalmanResult)
        {
            EnsureInitialized();

            _kalmanResult = kalmanResult;
            _outliers = [];
            _distanceSquared = GetDistanceSquared(kalmanResult.ParameterZScores);

            if (_distanceSquared < 0.0)
                LogNegativeDistanceSquared(kalmanResult.ParameterZScores);

            return _distanceSquared;
        }

        public bool IsAnomaly()
        {
            EnsureComputed();
            return _distanceSquared > _distanceSquaredThreshold;
        }

        public void FindOutlierParameters()
        {
            KalmanFilterResult kalmanResult = EnsureComputed();

            if (!IsAnomaly())
                return;

            Vector<double> xTilde = kalmanResult.ParameterZScores.Clone();
            HashSet<int> flagged = [];
            double currentDistanceSquared = _distanceSquared;

            double flaggedMax;
            double unflaggedMax;

            while (currentDistanceSquared > _distanceSquaredThreshold && flagged.Count < _parameterCount)
            {
                Vector<double> phi = GetShapleyContributions(xTilde);

                int nextFlaggedIndex = -1;
                double maxContribution = double.NegativeInfinity;
                for (int i = 0; i < _parameterCount; i++)
                {
                    if (!flagged.Contains(i) && phi[i] > maxContribution)
                    {
                        maxContribution = phi[i];
                        nextFlaggedIndex = i;
                    }
                }

                flagged.Add(nextFlaggedIndex);

                do
                {
                    phi = GetShapleyContributions(xTilde);

                    flaggedMax = double.NegativeInfinity;
                    unflaggedMax = double.NegativeInfinity;
                    for (int i = 0; i < _parameterCount; i++)
                    {
                        if (flagged.Contains(i))
                        {
                            flaggedMax = Math.Max(flaggedMax, phi[i]);
                        }
                        else
                        {
                            unflaggedMax = Math.Max(unflaggedMax, phi[i]);
                        }
                    }

                    foreach (int flaggedIndex in flagged)
                        xTilde[flaggedIndex] -= xTilde[flaggedIndex] * _stepSize;
                }
                while (flaggedMax <= unflaggedMax);

                currentDistanceSquared = GetDistanceSquared(xTilde);
            }

            _outliers = flagged.ToDictionary(i => _parameterNames[i], i => kalmanResult.ParameterValues[i]);
        }

        public MahalanobisResult GetResult()
        {
            KalmanFilterResult kalmanResult = EnsureComputed();

            return new MahalanobisResult()
            {
                DistanceSquared = _distanceSquared,
                OutlierParameters = _outliers,
                DeltaT = kalmanResult.DeltaT
            };
        }

        private void EnsureInitialized()
        {
            if (!_isInitialized)
                throw new InvalidOperationException($"{nameof(Init)} must be called first.");
        }

        private KalmanFilterResult EnsureComputed()
        {
            EnsureInitialized();

            return _kalmanResult
                ?? throw new InvalidOperationException($"{nameof(ComputeD2)} must be called first.");
        }

        private double GetDistanceSquared(Vector<double> zScores)
        {
            return zScores.DotProduct(_correlationMatrixInversed * zScores);
        }

        /// <summary>
        /// φ = (x − μ) ∘ Ω(x − μ), with μ = 0 because the inputs are z-scores.
        /// Each entry is that parameter's contribution to the Mahalanobis Distance.
        /// </summary>
        private Vector<double> GetShapleyContributions(Vector<double> zScores)
        {
            return zScores.PointwiseMultiply(_correlationMatrixInversed * zScores);
        }

        private static Matrix<double> CovarianceToCorrelationInversed(Matrix<double> covariance)
        {
            Vector<double> standardDeviations = covariance.Diagonal().Map(Math.Sqrt);
            return Matrix<double>.Build.Dense(covariance.RowCount, covariance.ColumnCount,
                (i, j) => covariance[i, j] / (standardDeviations[i] * standardDeviations[j])).Inverse();
        }


        private void LogNegativeDistanceSquared(Vector<double> zScores)
        {
            Matrix<double> inverseTranspose = _correlationMatrixInversed.Transpose();
            Matrix<double> symmetrized = (_correlationMatrixInversed + inverseTranspose) / 2.0;

            double minEigenvalue = symmetrized.Evd().EigenValues
                .Select((System.Numerics.Complex eigenvalue) => eigenvalue.Real)
                .Min();

            double asymmetry = (_correlationMatrixInversed - inverseTranspose).InfinityNorm();

            Console.WriteLine("=== Negative D2 detected ===");
            Console.WriteLine($"D2 = {_distanceSquared:R}");
            Console.WriteLine($"Min eigenvalue of inverse correlation matrix = {minEigenvalue:R}");
            Console.WriteLine($"Asymmetry of inverse correlation matrix (infinity norm) = {asymmetry:R}");
            Console.WriteLine($"Matrix dimension = {_correlationMatrixInversed.RowCount}, z-score count = {zScores.Count}, parameter name count = {_parameterNames.Length}");
            Console.WriteLine("Z-scores:");

            for (int i = 0; i < zScores.Count; i++)
            {
                string name = i < _parameterNames.Length ? _parameterNames[i] : "<no name>";
                Console.WriteLine($"  [{i}] {name} = {zScores[i]:R}");
            }

            Console.WriteLine("=== End ===");
        }
    }
}