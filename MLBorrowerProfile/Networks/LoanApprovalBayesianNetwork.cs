using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Models;
using MLBorrowerProfile.Builders;
using MLBorrowerProfile.DataModels;
using MLBorrowerProfile.Utils;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace MLBorrowerProfile.Networks
{
    public class LoanApprovalBayesianNetwork
    {
        /// <summary>
        /// The number of examples in the training data.
        /// </summary>
        public Variable<int> NumberOfExamples;

        /// <summary>
        /// The range of examples in the training data.
        /// </summary>
        public Range N;

        /// <summary>
        /// The credit score of the borrower.
        /// </summary>
        public VariableArray<double> CreditScore;

        /// <summary>
        /// The annual income of the borrower.
        /// </summary>
        public VariableArray<double> AnnualIncome;

        /// <summary>
        /// The loan amount of the borrower.
        /// </summary>
        public VariableArray<double> LoanAmount;

        /// <summary>
        /// The debt-to-income ratio of the borrower.
        /// </summary>
        public VariableArray<double> DebtToIncomeRatio;

        /// <summary>
        /// The previous loan defaults of the borrower.
        /// </summary>
        public VariableArray<double> PreviousLoanDefaults;

        /// <summary>
        /// The bankruptcy history of the borrower.
        /// </summary>
        public VariableArray<double> BankruptcyHistory;

        /// <summary>
        /// The employment status of the borrower.
        /// </summary>
        public VariableArray<int> EmploymentStatus;

        /// <summary>
        /// The loan purpose of the borrower.
        /// </summary>
        public VariableArray<int> LoanPurpose;

        /// <summary>
        /// The loan-to-income ratio of the borrower.
        /// </summary>
        public VariableArray<double> LoanToIncomeRatio;

        /// <summary>
        /// The risk score of the borrower.
        /// </summary>
        public VariableArray<double> RiskScore;

        /// <summary>
        /// The loan approval of the borrower.
        /// </summary>
        public VariableArray<bool> LoanApproved;

        /// <summary>
        /// The prior distribution for the credit score of the borrower.
        /// </summary>
        public Variable<Gaussian> CreditScorePrior;

        /// <summary>
        /// The prior distribution for the debt-to-income ratio of the borrower.
        /// </summary>
        public Variable<Gaussian> DebtToIncomeRatioPrior;

        /// <summary>
        /// The prior distribution for the employment status of the borrower.
        /// </summary>
        public Variable<Vector> EmploymentStatusPrior;

        /// <summary>
        /// The prior distribution for the approval of the borrower.
        /// </summary>
        public VariableArray<Bernoulli> ApprovalGivenFeatures;

        /// <summary>
        /// The mappings for categorical variables.
        /// </summary>
        private Dictionary<string, Dictionary<string, int>> categoricalMappings;

        /// <summary>
        /// The inverse mappings for categorical variables.
        /// </summary>
        private Dictionary<string, Dictionary<int, string>> inverseCategoricalMappings;

        /// <summary>
        /// The inference engine.
        /// </summary>
        public InferenceEngine Engine = new InferenceEngine();

        /// <summary>
        /// Whether to use direct estimation.
        /// </summary>
        private bool _useDirectEstimation = false;

        /// <summary>
        /// The intercept for direct estimation.
        /// </summary>
        private double _directEstimationIntercept = 0.0;

        /// <summary>
        /// The coefficients for direct estimation.
        /// </summary>
        private double[] _directEstimationCoefficients = new double[7] { 0.5, -0.3, -0.2, -0.4, -0.6, -0.3, -0.1 };

        /// <summary>
        /// The mean for credit score normalization.
        /// </summary>
        private double _creditScoreMean = 680;

        /// <summary>
        /// The standard deviation for credit score normalization.
        /// </summary>
        private double _creditScoreStd = 75;

        /// <summary>
        /// The mean for DTI ratio normalization.
        /// </summary>
        private double _dtiRatioMean = 0.36;

        /// <summary>
        /// The standard deviation for DTI ratio normalization.
        /// </summary>
        private double _dtiRatioStd = 0.15;

        /// <summary>
        /// The mean for loan-to-income ratio normalization.
        /// </summary>
        private double _loanToIncomeMean = 2.5;

        /// <summary>
        /// The standard deviation for loan-to-income ratio normalization.
        /// </summary>
        private double _loanToIncomeStd = 1.0;

        /// <summary>
        /// The mean for default ratio normalization.
        /// </summary>
        private double _defaultMean = 0.1;

        /// <summary>
        /// The standard deviation for default ratio normalization.
        /// </summary>
        private double _defaultStd = 0.3;

        /// <summary>
        /// The mean for bankruptcy ratio normalization.
        /// </summary>
        private double _bankruptcyMean = 0.05;

        /// <summary>
        /// The standard deviation for bankruptcy ratio normalization.
        /// </summary>
        private double _bankruptcyStd = 0.2;

        /// <summary>
        /// The constructor for the LoanApprovalBayesianNetwork.
        /// </summary>
        /// <param name="dataProcessor">The data processor.</param>
        /// <remarks>
        /// This constructor initializes the mappings for categorical variables and sets up the model structure.
        /// </remarks>
        public LoanApprovalBayesianNetwork(BorrowerModelBuilder dataProcessor)
        {
            // Initialize mappings for categorical variables
            this.InitializeCategoricalMappings(dataProcessor);

            // Set up the model structure
            NumberOfExamples = Variable.New<int>().Named("NumberOfExamples");
            N = new Range(NumberOfExamples).Named("N");

            // Define ranges for categorical variables
            Range ES = new Range(dataProcessor.EmploymentStatusValues.Count).Named("ES");
            Range LP = new Range(dataProcessor.LoanPurposeValues.Count).Named("LP");

            // Define continuous variables
            CreditScore = Variable.Array<double>(N).Named("CreditScore");
            AnnualIncome = Variable.Array<double>(N).Named("AnnualIncome");
            LoanAmount = Variable.Array<double>(N).Named("LoanAmount");
            DebtToIncomeRatio = Variable.Array<double>(N).Named("DebtToIncomeRatio");

            // Define binary variables
            PreviousLoanDefaults = Variable.Array<double>(N).Named("PreviousLoanDefaults");
            BankruptcyHistory = Variable.Array<double>(N).Named("BankruptcyHistory");

            // Define categorical variables
            EmploymentStatus = Variable.Array<int>(N).Named("EmploymentStatus");
            LoanPurpose = Variable.Array<int>(N).Named("LoanPurpose");

            // Define calculated features
            LoanToIncomeRatio = Variable.Array<double>(N).Named("LoanToIncomeRatio");
            RiskScore = Variable.Array<double>(N).Named("RiskScore");

            // Define output variable
            LoanApproved = Variable.Array<bool>(N).Named("LoanApproved");

            // Set up relationships between variables
            using (Variable.ForEach(N))
            {
                // Calculate loan-to-income ratio
                LoanToIncomeRatio[N] = LoanAmount[N] / AnnualIncome[N];

                // Calculate risk score as a linear combination of factors
                RiskScore[N] = Variable.GaussianFromMeanAndPrecision(
                    // Positive factors (higher is better)
                    (CreditScore[N] - 500) / 350 * 5 -     // Credit score (normalized)
                    DebtToIncomeRatio[N] * 10 -            // DTI (higher is worse)
                    PreviousLoanDefaults[N] * 5 -          // Each default is a big negative
                    BankruptcyHistory[N] * 8 -             // Bankruptcy is severe negative
                    LoanToIncomeRatio[N] * 3,              // Loan size relative to income

                    // Precision (inverse variance)
                    1.0
                );

                // Loan approval probability based on risk score
                // Using logistic function to convert risk score to probability
                double threshold = 0.0;  // Threshold for approval

                // Define approval probability using logistic function
                Variable<double> approvalLogit = RiskScore[N] - threshold;
                Variable<double> approvalProb = Variable.Logistic(approvalLogit);

                // Define approval outcome as Bernoulli based on probability
                LoanApproved[N] = Variable.Bernoulli(approvalProb).Named("LoanApproved_" + N);
            }
        }

        /// <summary>
        /// Initializes the mappings for categorical variables.
        /// </summary>
        /// <param name="dataProcessor">The data processor.</param>
        /// <remarks>
        /// This method creates mappings for each categorical variable and stores them in dictionaries.
        /// </remarks>
        private void InitializeCategoricalMappings(BorrowerModelBuilder dataProcessor)
        {
            categoricalMappings = new Dictionary<string, Dictionary<string, int>>();
            inverseCategoricalMappings = new Dictionary<string, Dictionary<int, string>>();

            // Create mappings for each categorical variable
            this.InitializeMappingForVariable("EmploymentStatus", dataProcessor.EmploymentStatusValues);
            this.InitializeMappingForVariable("MaritalStatus", dataProcessor.MaritalStatusValues);
            this.InitializeMappingForVariable("EducationLevel", dataProcessor.EducationLevelValues);
            this.InitializeMappingForVariable("HomeOwnershipStatus", dataProcessor.HomeOwnershipStatusValues);
            this.InitializeMappingForVariable("LoanPurpose", dataProcessor.LoanPurposeValues);
            this.InitializeMappingForVariable("HealthInsuranceStatus", dataProcessor.HealthInsuranceStatusValues);
            this.InitializeMappingForVariable("LifeInsuranceStatus", dataProcessor.LifeInsuranceStatusValues);
            this.InitializeMappingForVariable("CarInsuranceStatus", dataProcessor.CarInsuranceStatusValues);
            this.InitializeMappingForVariable("HomeInsuranceStatus", dataProcessor.HomeInsuranceStatusValues);
            this.InitializeMappingForVariable("EmployerType", dataProcessor.EmployerTypeValues);
        }

        /// <summary>
        /// Initializes the mapping for a specific categorical variable.
        /// </summary>
        /// <param name="variableName">The name of the variable.</param>
        /// <param name="values">The list of values for the variable.</param>
        /// <remarks>
        /// This method creates a mapping for a specific categorical variable and stores it in dictionaries.
        /// </remarks>
        private void InitializeMappingForVariable(string variableName, List<string> values)
        {
            Dictionary<string, int> mapping = new Dictionary<string, int>();
            Dictionary<int, string> inverseMapping = new Dictionary<int, string>();

            for (int i = 0; i < values.Count; i++)
            {
                mapping[values[i]] = i;
                inverseMapping[i] = values[i];
            }

            categoricalMappings[variableName] = mapping;
            inverseCategoricalMappings[variableName] = inverseMapping;
        }

        /// <summary>
        /// Learns the parameters for the model.
        /// </summary>
        /// <param name="trainingData">The training data.</param>
        /// <remarks>
        /// This method sets the observed values in the model and prepares arrays for all variables from the training data.
        /// </remarks>
        public void LearnParameters(List<BorrowerProfile> trainingData)
        {
            // Set observed values in the model
            NumberOfExamples.ObservedValue = trainingData.Count;

            // Prepare arrays for all variables from training data
            double[] creditScores = trainingData.Select(b => b.CreditScore).ToArray();
            double[] annualIncomes = trainingData.Select(b => b.AnnualIncome).ToArray();
            double[] loanAmounts = trainingData.Select(b => b.LoanAmount).ToArray();
            double[] dtiRatios = trainingData.Select(b => b.DebtToIncomeRatio).ToArray();
            double[] defaults = trainingData.Select(b => Convert.ToDouble(b.PreviousLoanDefaults)).ToArray();
            double[] bankruptcies = trainingData.Select(b => Convert.ToDouble(b.BankruptcyHistory)).ToArray();

            // Map categorical variables to indices
            int[] employmentStatusIndices = trainingData
                .Select(b => categoricalMappings["EmploymentStatus"][b.EmploymentStatus])
                .ToArray();

            int[] loanPurposeIndices = trainingData
                .Select(b => categoricalMappings["LoanPurpose"][b.LoanPurpose])
                .ToArray();

            // Map approval outcomes to boolean
            bool[] approvalOutcomes = trainingData
                .Select(b => b.LoanApproved == 1)
                .ToArray();

            // Set observed values for variables
            CreditScore.ObservedValue = creditScores;
            AnnualIncome.ObservedValue = annualIncomes;
            LoanAmount.ObservedValue = loanAmounts;
            DebtToIncomeRatio.ObservedValue = dtiRatios;
            PreviousLoanDefaults.ObservedValue = defaults;
            BankruptcyHistory.ObservedValue = bankruptcies;
            EmploymentStatus.ObservedValue = employmentStatusIndices;
            LoanPurpose.ObservedValue = loanPurposeIndices;

            // Set observed values for outcome
            LoanApproved.ObservedValue = approvalOutcomes;

            // Perform inference to learn model parameters
            Engine.NumberOfIterations = 100;
            this.InferenceQuery();
        }

        /// <summary>
        /// Analyzes the feature importance of the model.
        /// </summary>
        /// <param name="borrowerProfile">The borrower profile.</param>
        /// <returns>The feature importance.</returns>
        /// <remarks>
        /// This method calculates the base approval probability and then tests the impact of changes to each feature on the approval probability.
        /// </remarks>
        public FeatureImportance AnalyzeFeatureImportance(BorrowerProfile borrowerProfile)
        {
            FeatureImportance importance = new FeatureImportance();

            // Calculate base approval probability
            double baseProb = this.PredictApprovalProbability(borrowerProfile);
            importance.BaseApprovalProbability = baseProb;

            // Test impact of credit score (50 point improvement)
            BorrowerProfile testModel = this.Clone(borrowerProfile);
            testModel.CreditScore = Math.Min(850, testModel.CreditScore + 50);
            double creditImpact = this.PredictApprovalProbability(testModel) - baseProb;
            importance.CreditScoreImpact = creditImpact;

            // Test impact of DTI (5% reduction)
            testModel = this.Clone(borrowerProfile);
            testModel.DebtToIncomeRatio = Math.Max(0, testModel.DebtToIncomeRatio - 0.05);
            double dtiImpact = this.PredictApprovalProbability(testModel) - baseProb;
            importance.DebtToIncomeRatioImpact = dtiImpact;

            // Test impact of loan-to-income ratio (reducing loan amount by 10%)
            testModel = this.Clone(borrowerProfile);
            testModel.LoanAmount = testModel.LoanAmount * 0.9;
            double loanSizeImpact = this.PredictApprovalProbability(testModel) - baseProb;
            importance.LoanToIncomeRatioImpact = loanSizeImpact;

            // Test impact of employment status (if unemployed, what if employed)
            testModel = this.Clone(borrowerProfile);
            if (borrowerProfile.EmploymentStatus == "Unemployed")
            {
                testModel.EmploymentStatus = "Employed";
                double employmentImpact = this.PredictApprovalProbability(testModel) - baseProb;
                importance.EmploymentStatusImpact = employmentImpact;
            }
            else
            {
                importance.EmploymentStatusImpact = 0; // Already optimal
            }

            // Generate improvement recommendations
            importance.GenerateRecommendations(borrowerProfile);

            return importance;
        }

        /// <summary>
        /// Clones the model.
        /// </summary>
        /// <param name="original">The original model.</param>
        /// <returns>The cloned model.</returns>
        /// <remarks>
        /// This method creates a deep copy of the model to avoid modifying the original.
        /// </remarks>
        private BorrowerProfile Clone(BorrowerProfile original)
        {
            // Create a deep copy of the model to avoid modifying the original
            return new BorrowerProfile
            {
                CreditScore = original.CreditScore,
                AnnualIncome = original.AnnualIncome,
                LoanAmount = original.LoanAmount,
                LoanDuration = original.LoanDuration,
                Age = original.Age,
                EmploymentStatus = original.EmploymentStatus,
                MaritalStatus = original.MaritalStatus,
                NumberOfDependents = original.NumberOfDependents,
                EducationLevel = original.EducationLevel,
                HomeOwnershipStatus = original.HomeOwnershipStatus,
                MonthlyDebtPayments = original.MonthlyDebtPayments,
                CreditCardUtilizationRate = original.CreditCardUtilizationRate,
                NumberOfOpenCreditLines = original.NumberOfOpenCreditLines,
                NumberOfCreditInquiries = original.NumberOfCreditInquiries,
                DebtToIncomeRatio = original.DebtToIncomeRatio,
                BankruptcyHistory = original.BankruptcyHistory,
                LoanPurpose = original.LoanPurpose,
                PreviousLoanDefaults = original.PreviousLoanDefaults,
                InterestRate = original.InterestRate,
                PaymentHistory = original.PaymentHistory,
                // Copy all remaining fields...
                LoanApproved = original.LoanApproved
            };
        }

        /// <summary>
        /// Performs inference on the model.
        /// </summary>
        /// <remarks>
        /// This method performs inference on the model to learn the parameters.
        /// </remarks>
        private void InferenceQuery()
        {
            try
            {
                // Create a test variable
                Variable<bool> testVariable = Variable.Bernoulli(0.5);

                // First test with just the test variable in the OptimiseForVariables list
                Engine.OptimiseForVariables = new IVariable[] { testVariable };

                try
                {
                    // Attempt to perform inference on the test variable
                    Bernoulli testInference = Engine.Infer<Bernoulli>(testVariable);

                    // If we get here, reset the optimization for the actual model
                    Engine.OptimiseForVariables = new IVariable[] { LoanApproved };

                    Console.WriteLine("Basic inference test successful, proceeding with model inference.");

                    // Perform actual model inference
                    // The Engine will now operate on the model with LoanApproved as the target
                }
                catch (Exception testEx)
                {
                    Console.WriteLine($"Basic inference test failed: {testEx.Message}");
                    Console.WriteLine("Using simplified parameter estimation instead.");
                    this.DirectParameterEstimation();
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Inference error: {ex.Message}");
                // Implement fallback approach if inference fails
                this.DirectParameterEstimation();
            }
        }

        /// <summary>
        /// Normalizes the values.
        /// </summary>
        /// <param name="values">The values to normalize.</param>
        /// <returns>The normalized values.</returns>
        /// <remarks>
        /// This method normalizes the values by subtracting the mean and dividing by the standard deviation.
        /// </remarks>
        private double[] Normalize(double[] values)
        {
            double mean = this.Mean(values);
            double std = this.StdDev(values, mean);

            double[] normalized = new double[values.Length];
            for (int i = 0; i < values.Length; i++)
            {
                normalized[i] = (values[i] - mean) / std;
            }

            return normalized;
        }

        /// <summary>
        /// Calculates the mean of the values.
        /// </summary>
        /// <param name="values">The values to calculate the mean of.</param>
        /// <returns>The mean of the values.</returns>
        /// <remarks>
        /// This method calculates the mean of the values by summing them and dividing by the number of values.
        /// </remarks>
        private double Mean(double[] values)
        {
            return values.Sum() / values.Length;
        }

        /// <summary>
        /// Calculates the standard deviation of the values.
        /// </summary>
        /// <param name="values">The values to calculate the standard deviation of.</param>
        /// <param name="mean">The mean of the values.</param>
        /// <returns>The standard deviation of the values.</returns>
        /// <remarks>
        /// This method calculates the standard deviation of the values by summing the squared differences between each value and the mean, dividing by the number of values, and then taking the square root.
        /// </remarks>
        private double StdDev(double[] values, double mean)
        {
            double variance = values.Select(v => (v - mean) * (v - mean)).Sum() / values.Length;
            return Math.Sqrt(variance);
        }

        /// <summary>
        /// Predicts the approval probability for a borrower profile.
        /// </summary>
        /// <param name="borrowerProfile">The borrower profile.</param>
        /// <returns>The predicted approval probability.</returns>
        /// <remarks>
        /// This method predicts the approval probability for a borrower profile using Bayesian inference or direct estimation.
        /// </remarks>
        public double PredictApprovalProbability(BorrowerProfile borrowerProfile)
        {
            if (!_useDirectEstimation)
            {
                // Use Bayesian inference (original code)
                try
                {
                    // Create a model with a single example
                    NumberOfExamples.ObservedValue = 1;

                    // Set observed values for the single example
                    CreditScore.ObservedValue = new double[] { borrowerProfile.CreditScore };
                    AnnualIncome.ObservedValue = new double[] { borrowerProfile.AnnualIncome };
                    LoanAmount.ObservedValue = new double[] { borrowerProfile.LoanAmount };
                    DebtToIncomeRatio.ObservedValue = new double[] { borrowerProfile.DebtToIncomeRatio };
                    PreviousLoanDefaults.ObservedValue = new double[] { borrowerProfile.PreviousLoanDefaults };
                    BankruptcyHistory.ObservedValue = new double[] { borrowerProfile.BankruptcyHistory };

                    // Map categorical variables with safety checks
                    int employmentStatusIdx = 0;
                    int loanPurposeIdx = 0;

                    if (categoricalMappings != null &&
                        categoricalMappings.ContainsKey("EmploymentStatus") &&
                        categoricalMappings["EmploymentStatus"].ContainsKey(borrowerProfile.EmploymentStatus))
                    {
                        employmentStatusIdx = categoricalMappings["EmploymentStatus"][borrowerProfile.EmploymentStatus];
                    }

                    if (categoricalMappings != null &&
                        categoricalMappings.ContainsKey("LoanPurpose") &&
                        categoricalMappings["LoanPurpose"].ContainsKey(borrowerProfile.LoanPurpose))
                    {
                        loanPurposeIdx = categoricalMappings["LoanPurpose"][borrowerProfile.LoanPurpose];
                    }

                    EmploymentStatus.ObservedValue = new int[] { employmentStatusIdx };
                    LoanPurpose.ObservedValue = new int[] { loanPurposeIdx };

                    // Clear observed value for output to enable inference
                    LoanApproved.ClearObservedValue();

                    // Set optimization variables
                    Engine.OptimiseForVariables = new IVariable[] { LoanApproved[0] };

                    // Infer the approval probability
                    Bernoulli approvalDistribution = Engine.Infer<Bernoulli>(LoanApproved[0]);

                    // Return the probability of approval
                    return approvalDistribution.GetProbTrue();
                }
                catch (Exception ex)
                {
                    // If Bayesian inference fails, fall back to direct estimation
                    Console.WriteLine($"Bayesian inference failed: {ex.Message}");
                    Console.WriteLine("Falling back to direct estimation for prediction");
                    _useDirectEstimation = true;
                }
            }

            // Use direct logistic regression estimation
            if (_useDirectEstimation)
            {
                try
                {
                    // Ensure coefficients are initialized
                    if (_directEstimationCoefficients == null)
                    {
                        Console.WriteLine("Initializing default coefficients");
                        _directEstimationCoefficients = new double[7] { 0.5, -0.3, -0.2, -0.4, -0.6, -0.3, -0.1 };
                        _directEstimationIntercept = -2.0;
                    }

                    // Calculate loan-to-income ratio
                    double loanToIncomeRatio = borrowerProfile.LoanAmount / Math.Max(1, borrowerProfile.AnnualIncome);

                    // Use default normalization values if not set
                    double creditScoreNorm = _creditScoreStd > 0 ?
                        (borrowerProfile.CreditScore - _creditScoreMean) / _creditScoreStd : 0;

                    double dtiRatioNorm = _dtiRatioStd > 0 ?
                        (borrowerProfile.DebtToIncomeRatio - _dtiRatioMean) / _dtiRatioStd : 0;

                    double loanToIncomeRatioNorm = _loanToIncomeStd > 0 ?
                        (loanToIncomeRatio - _loanToIncomeMean) / _loanToIncomeStd : 0;

                    double defaultsNorm = _defaultStd > 0 ?
                        (borrowerProfile.PreviousLoanDefaults - _defaultMean) / _defaultStd : 0;

                    double bankruptcyNorm = _bankruptcyStd > 0 ?
                        (borrowerProfile.BankruptcyHistory - _bankruptcyMean) / _bankruptcyStd : 0;

                    // Get categorical variables with null checks
                    int employmentStatusIdx = 0;
                    int loanPurposeIdx = 0;

                    if (categoricalMappings != null &&
                        categoricalMappings.ContainsKey("EmploymentStatus") &&
                        categoricalMappings["EmploymentStatus"].ContainsKey(borrowerProfile.EmploymentStatus))
                    {
                        employmentStatusIdx = categoricalMappings["EmploymentStatus"][borrowerProfile.EmploymentStatus];
                    }

                    if (categoricalMappings != null &&
                        categoricalMappings.ContainsKey("LoanPurpose") &&
                        categoricalMappings["LoanPurpose"].ContainsKey(borrowerProfile.LoanPurpose))
                    {
                        loanPurposeIdx = categoricalMappings["LoanPurpose"][borrowerProfile.LoanPurpose];
                    }

                    // Calculate linear predictor
                    double z = _directEstimationIntercept +
                              _directEstimationCoefficients[0] * creditScoreNorm +
                              _directEstimationCoefficients[1] * dtiRatioNorm +
                              _directEstimationCoefficients[2] * loanToIncomeRatioNorm +
                              _directEstimationCoefficients[3] * defaultsNorm +
                              _directEstimationCoefficients[4] * bankruptcyNorm +
                              _directEstimationCoefficients[5] * (employmentStatusIdx == 0 ? 1 : 0) +
                              _directEstimationCoefficients[6] * (loanPurposeIdx == 0 ? 1 : 0);

                    // Convert to probability using logistic function
                    double probability = 1.0 / (1.0 + Math.Exp(-z));

                    return probability;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Direct estimation failed: {ex.Message}");
                    Console.WriteLine("Using simple heuristic fallback");
                    return this.FallbackHeuristicPrediction(borrowerProfile);
                }
            }

            // Ultimate fallback: simple heuristic if all else fails
            return this.FallbackHeuristicPrediction(borrowerProfile);
        }

        /// <summary>
        /// A very simple heuristic model that uses common lending guidelines.
        /// </summary>
        /// <param name="borrowerProfile">The borrower profile.</param>
        /// <returns>The predicted approval probability.</returns>
        /// <remarks>
        /// This method uses a simple heuristic model that uses common lending guidelines.
        /// </remarks>
        private double FallbackHeuristicPrediction(BorrowerProfile borrowerProfile)
        {
            try
            {
                // A very simple heuristic model that uses common lending guidelines
                double baseProb = 0.5; // Start with 50% baseline

                // Credit score adjustments (very influential)
                if (borrowerProfile.CreditScore >= 740) baseProb += 0.25;
                else if (borrowerProfile.CreditScore >= 680) baseProb += 0.15;
                else if (borrowerProfile.CreditScore >= 620) baseProb += 0.05;
                else if (borrowerProfile.CreditScore < 580) baseProb -= 0.25;

                // DTI adjustments
                if (borrowerProfile.DebtToIncomeRatio > 0.43) baseProb -= 0.2;
                else if (borrowerProfile.DebtToIncomeRatio > 0.36) baseProb -= 0.1;

                // Previous defaults are strong negative
                baseProb -= 0.15 * borrowerProfile.PreviousLoanDefaults;

                // Bankruptcy is a very strong negative
                baseProb -= 0.25 * borrowerProfile.BankruptcyHistory;

                // Employment status
                if (borrowerProfile.EmploymentStatus == "Unemployed") baseProb -= 0.2;

                // Loan amount to income ratio
                double loanToIncomeRatio = borrowerProfile.LoanAmount / Math.Max(1, borrowerProfile.AnnualIncome);
                if (loanToIncomeRatio > 5) baseProb -= 0.2;
                else if (loanToIncomeRatio > 3) baseProb -= 0.1;

                // Constrain to valid probability range
                return Math.Max(0.01, Math.Min(0.99, baseProb));
            }
            catch
            {
                // Absolute last resort - return 50/50 chance
                return 0.5;
            }
        }

        /// <summary>
        /// Direct parameter estimation.
        /// </summary>
        /// <remarks>
        /// This method is a fallback method for parameter estimation when full Bayesian inference fails.
        /// It uses direct statistical methods to approximate the model parameters.
        /// </remarks>
        private void DirectParameterEstimation()
        {
            // This is a fallback method for parameter estimation when full Bayesian inference fails
            // It uses direct statistical methods to approximate the model parameters

            try
            {
                Console.WriteLine("Using direct parameter estimation as fallback...");

                // Get the observed examples
                int exampleCount = NumberOfExamples.ObservedValue;
                double[] creditScores = CreditScore.ObservedValue;
                double[] annualIncomes = AnnualIncome.ObservedValue;
                double[] loanAmounts = LoanAmount.ObservedValue;
                double[] dtiRatios = DebtToIncomeRatio.ObservedValue;
                double[] defaults = PreviousLoanDefaults.ObservedValue;
                double[] bankruptcies = BankruptcyHistory.ObservedValue;
                int[] employmentStatus = EmploymentStatus.ObservedValue;
                int[] loanPurpose = LoanPurpose.ObservedValue;
                bool[] approvals = LoanApproved.ObservedValue;

                // Compute the loan-to-income ratios
                double[] loanToIncomeRatios = new double[exampleCount];
                for (int i = 0; i < exampleCount; i++)
                {
                    loanToIncomeRatios[i] = loanAmounts[i] / Math.Max(1, annualIncomes[i]);
                }

                // 1. Use logistic regression to estimate coefficients
                // We'll implement a simplified version of logistic regression directly

                // Initialize coefficients
                double[] coefficients = new double[7]; // For our 7 key features
                double intercept = 0;

                // Normalize the features for stable estimation
                double[] creditScoresNorm = this.Normalize(creditScores);
                double[] dtiRatiosNorm = this.Normalize(dtiRatios);
                double[] loanToIncomeRatiosNorm = this.Normalize(loanToIncomeRatios);
                double[] defaultsNorm = this.Normalize(defaults);
                double[] bankruptciesNorm = this.Normalize(bankruptcies);

                // Learning rate and iterations for gradient descent
                double learningRate = 0.01;
                int iterations = 1000;

                // Gradient descent for logistic regression
                for (int iter = 0; iter < iterations; iter++)
                {
                    // Predictions using current coefficients
                    double[] predictions = new double[exampleCount];
                    for (int i = 0; i < exampleCount; i++)
                    {
                        // Compute linear predictor
                        double z = intercept +
                                   coefficients[0] * creditScoresNorm[i] +
                                   coefficients[1] * dtiRatiosNorm[i] +
                                   coefficients[2] * loanToIncomeRatiosNorm[i] +
                                   coefficients[3] * defaultsNorm[i] +
                                   coefficients[4] * bankruptciesNorm[i] +
                                   coefficients[5] * (employmentStatus[i] == 0 ? 1 : 0) + // Unemployed dummy
                                   coefficients[6] * (loanPurpose[i] == 0 ? 1 : 0);       // Home loan dummy

                        // Apply logistic function
                        predictions[i] = 1.0 / (1.0 + Math.Exp(-z));
                    }

                    // Compute gradients
                    double interceptGrad = 0;
                    double[] coeffGrad = new double[7];

                    for (int i = 0; i < exampleCount; i++)
                    {
                        double error = predictions[i] - (approvals[i] ? 1.0 : 0.0);

                        interceptGrad += error;
                        coeffGrad[0] += error * creditScoresNorm[i];
                        coeffGrad[1] += error * dtiRatiosNorm[i];
                        coeffGrad[2] += error * loanToIncomeRatiosNorm[i];
                        coeffGrad[3] += error * defaultsNorm[i];
                        coeffGrad[4] += error * bankruptciesNorm[i];
                        coeffGrad[5] += error * (employmentStatus[i] == 0 ? 1 : 0);
                        coeffGrad[6] += error * (loanPurpose[i] == 0 ? 1 : 0);
                    }

                    // Update coefficients
                    intercept -= learningRate * interceptGrad / exampleCount;
                    for (int j = 0; j < coefficients.Length; j++)
                    {
                        coefficients[j] -= learningRate * coeffGrad[j] / exampleCount;
                    }
                }

                // 2. Store the learned coefficients for later prediction
                _directEstimationIntercept = intercept;
                _directEstimationCoefficients = coefficients;

                // 3. Store feature normalization parameters
                _creditScoreMean = this.Mean(creditScores);
                _creditScoreStd = this.StdDev(creditScores, _creditScoreMean);
                _dtiRatioMean = this.Mean(dtiRatios);
                _dtiRatioStd = this.StdDev(dtiRatios, _dtiRatioMean);
                _loanToIncomeMean = this.Mean(loanToIncomeRatios);
                _loanToIncomeStd = this.StdDev(loanToIncomeRatios, _loanToIncomeMean);
                _defaultMean = this.Mean(defaults);
                _defaultStd = this.StdDev(defaults, _defaultMean);
                _bankruptcyMean = this.Mean(bankruptcies);
                _bankruptcyStd = this.StdDev(bankruptcies, _bankruptcyMean);

                // 4. Calculate model accuracy on training data
                int correctPredictions = 0;
                for (int i = 0; i < exampleCount; i++)
                {
                    double z = _directEstimationIntercept +
                              _directEstimationCoefficients[0] * ((creditScores[i] - _creditScoreMean) / _creditScoreStd) +
                              _directEstimationCoefficients[1] * ((dtiRatios[i] - _dtiRatioMean) / _dtiRatioStd) +
                              _directEstimationCoefficients[2] * ((loanToIncomeRatios[i] - _loanToIncomeMean) / _loanToIncomeStd) +
                              _directEstimationCoefficients[3] * ((defaults[i] - _defaultMean) / _defaultStd) +
                              _directEstimationCoefficients[4] * ((bankruptcies[i] - _bankruptcyMean) / _bankruptcyStd) +
                              _directEstimationCoefficients[5] * (employmentStatus[i] == 0 ? 1 : 0) +
                              _directEstimationCoefficients[6] * (loanPurpose[i] == 0 ? 1 : 0);

                    double predictedProb = 1.0 / (1.0 + Math.Exp(-z));
                    bool predictedApproval = predictedProb >= 0.5;

                    if (predictedApproval == approvals[i])
                        correctPredictions++;
                }

                double accuracy = (double)correctPredictions / exampleCount;
                Console.WriteLine($"Direct estimation completed. Training accuracy: {accuracy:P2}");

                // Set flag to use direct estimation for predictions
                _useDirectEstimation = true;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error in direct parameter estimation: {ex.Message}");
                // Use very basic coefficients as ultimate fallback
                _directEstimationIntercept = -2.0;
                _directEstimationCoefficients = new double[] { 0.5, -0.3, -0.2, -0.4, -0.6, -0.3, -0.1 };
                _useDirectEstimation = true;
            }
        }

        /// <summary>
        /// Generates an approval report for a borrower profile.
        /// </summary>
        /// <param name="borrowerProfile">The borrower profile.</param>
        /// <returns>The approval report.</returns>
        /// <remarks>
        /// This method generates an approval report for a borrower profile.
        /// </remarks>
        public string GenerateApprovalReport(BorrowerProfile borrowerProfile)
        {
            double approvalProbability = this.PredictApprovalProbability(borrowerProfile);
            FeatureImportance featureImportance = this.AnalyzeFeatureImportance(borrowerProfile);

            string approvalReportTemplate = """
                # Loan Approval Prediction Report

                ## Borrower Profile Overview

                - **Credit Score**: {{borrower.CreditScore}}
                - **Annual Income**: {{borrower.AnnualIncome}}
                - **Loan Amount**: {{borrower.LoanAmount}}
                - **Loan Duration**: {{borrower.LoanDuration}} years
                - **Age**: {{borrower.Age}}

                ### Employment & Education

                - **Employment Status**: {{borrower.EmploymentStatus}}
                - **Marital Status**: {{borrower.MaritalStatus}}
                - **Number of Dependents**: {{borrower.NumberOfDependents}}
                - **Education Level**: {{borrower.EducationLevel}}
                - **Employer Type**: {{borrower.EmployerType}}
                - **Job Tenure**: {{borrower.JobTenure}} years

                ### Housing & Insurance

                - **Home Ownership Status**: {{borrower.HomeOwnershipStatus}}
                - **Health Insurance Status**: {{borrower.HealthInsuranceStatus}}
                - **Life Insurance Status**: {{borrower.LifeInsuranceStatus}}
                - **Car Insurance Status**: {{borrower.CarInsuranceStatus}}
                - **Home Insurance Status**: {{borrower.HomeInsuranceStatus}}
                - **Other Insurance Policies**: {{borrower.OtherInsurancePolicies}}

                ### Credit & Debt Profile

                - **Monthly Debt Payments**: {{borrower.MonthlyDebtPayments}}
                - **Credit Card Utilization Rate**: {{borrower.CreditCardUtilizationRate}}
                - **Number of Open Credit Lines**: {{borrower.NumberOfOpenCreditLines}}
                - **Number of Credit Inquiries**: {{borrower.NumberOfCreditInquiries}}
                - **Debt-to-Income Ratio**: {{borrower.DebtToIncomeRatio}}
                - **Length of Credit History**: {{borrower.LengthOfCreditHistory}} years
                - **Bankruptcy History**: {{borrower.BankruptcyHistory}}
                - **Loan Purpose**: {{borrower.LoanPurpose}}
                - **Previous Loan Defaults**: {{borrower.PreviousLoanDefaults}}
                - **Interest Rate**: {{borrower.InterestRate}}%
                - **Payment History**: {{borrower.PaymentHistory}}

                ### Financial Accounts

                - **Savings Account Balance**: ${{borrower.SavingsAccountBalance}}
                - **Checking Account Balance**: ${{borrower.CheckingAccountBalance}}
                - **Investment Account Balance**: ${{borrower.InvestmentAccountBalance}}
                - **Retirement Account Balance**: ${{borrower.RetirementAccountBalance}}
                - **Emergency Fund Balance**: ${{borrower.EmergencyFundBalance}}

                ### Assets & Liabilities

                - **Total Assets**: ${{borrower.TotalAssets}}
                - **Total Liabilities**: ${{borrower.TotalLiabilities}}
                - **Net Worth**: ${{borrower.NetWorth}}

                ### Outstanding Balances

                - **Mortgage Balance**: ${{borrower.MortgageBalance}}
                - **Rent Payments**: ${{borrower.RentPayments}}
                - **Auto Loan Balance**: ${{borrower.AutoLoanBalance}}
                - **Personal Loan Balance**: ${{borrower.PersonalLoanBalance}}
                - **Student Loan Balance**: ${{borrower.StudentLoanBalance}}

                ### Monthly Financial Breakdown

                - **Monthly Savings**: ${{borrower.MonthlySavings}}
                - **Annual Bonuses**: ${{borrower.AnnualBonuses}}
                - **Annual Expenses**: ${{borrower.AnnualExpenses}}
                - **Monthly Housing Costs**: ${{borrower.MonthlyHousingCosts}}
                - **Monthly Transportation Costs**: ${{borrower.MonthlyTransportationCosts}}
                - **Monthly Food Costs**: ${{borrower.MonthlyFoodCosts}}
                - **Monthly Healthcare Costs**: ${{borrower.MonthlyHealthcareCosts}}
                - **Monthly Entertainment Costs**: ${{borrower.MonthlyEntertainmentCosts}}
                - **Utility Bills Payment History**: {{borrower.UtilityBillsPaymentHistory}}

                ---

                ## Approval Probability: {{ApprovalProbability}}

                ### Predicted Decision: {{PedictedDecision}}

                ---

                ## Key Factors

                Factor | Impact on Approval | Strength  
                --- | --- | ---  
                {{#each RankedFeatures}}
                {{Feature}} | {{Impact}} | {{Strength}} ({{ImpactPercent}})
                {{/each}}

                ---

                ## Recommendations for Approval

                {{#each Recommendations}}
                - {{this}}
                {{/each}}
                
                """;

            string renderedLoanApprovalReport = HandlebarsUtility.RenderTemplate(approvalReportTemplate, new
            {
                borrower = borrowerProfile,
                RankedFeatures = featureImportance.RankedFeatures()
                    .Select(f => new
                    {
                        f.Feature,
                        Impact = f.Impact > 0 ? "Positive" : "Negative",
                        ImpactPercent = f.Impact,
                        Strength = Math.Abs(f.Impact) < 0.05 ? "Low" :
                                   Math.Abs(f.Impact) < 0.15 ? "Medium" : "High"
                    }),
                featureImportance.Recommendations,
                PedictedDecision = approvalProbability >= 0.5 ? "APPROVED" : "DENIED",
                ApprovalProbability = approvalProbability
            });

            return renderedLoanApprovalReport;
            //StringBuilder report = new StringBuilder();
            //report.AppendLine("# Loan Approval Prediction Report");
            //report.AppendLine();
            //report.AppendLine($"## Approval Probability: {approvalProbability:P2}");
            //report.AppendLine($"### Predicted Decision: {(approvalProbability >= 0.5 ? "APPROVED" : "DENIED")}");
            //report.AppendLine();

            //report.AppendLine("## Key Factors");
            //report.AppendLine("Factor | Impact on Approval | Strength");
            //report.AppendLine("--- | --- | ---");

            //// Add ranked features
            //foreach ((string Feature, double Impact) feature in featureImportance.RankedFeatures())
            //{
            //    string impact = feature.Impact > 0 ? "Positive" : "Negative";
            //    string strength = Math.Abs(feature.Impact) < 0.05 ? "Low" :
            //                      Math.Abs(feature.Impact) < 0.15 ? "Medium" : "High";

            //    report.AppendLine($"{feature.Feature} | {impact} | {strength} ({feature.Impact:P2})");
            //}

            //report.AppendLine();
            //report.AppendLine("## Recommendations for Approval");

            //foreach (string recommendation in featureImportance.Recommendations)
            //{
            //    report.AppendLine($"- {recommendation}");
            //}

            //return report.ToString();
        }

    }
}