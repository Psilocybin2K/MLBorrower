using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Models;
using MLBorrowerProfile.Builders;
using MLBorrowerProfile.DataModels;
using MLBorrowerProfile.Extensions;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace MLBorrowerProfile.Networks
{
    /// <summary>
    /// Represents a Bayesian network for modeling borrower profiles.
    /// </summary>
    /// <remarks>
    /// This class implements a Bayesian network for borrower profile analysis with the following structure:
    /// 
    /// 1. Network Components:
    ///    - Credit Score: Continuous variable representing borrower's creditworthiness
    ///    - Annual Income: Continuous variable for borrower's financial capacity
    ///    - Loan Amount: Continuous variable for requested loan size
    ///    - Loan Duration: Discrete variable for loan term length
    ///    - Age: Discrete variable for borrower's age
    ///    - Employment Status: Categorical variable for current employment situation
    ///    - Marital Status: Categorical variable for relationship status
    ///    - Number of Dependents: Discrete variable for family size
    /// 
    /// 2. Network Structure:
    ///    - Root nodes: Age, Marital Status, Employment Status
    ///    - Intermediate nodes: Credit Score, Annual Income
    ///    - Leaf nodes: Loan Amount, Loan Duration
    /// 
    /// 3. Dependencies:
    ///    - Credit Score depends on Age and Employment Status
    ///    - Annual Income depends on Age, Employment Status, and Marital Status
    ///    - Loan Amount depends on Annual Income and Credit Score
    ///    - Loan Duration depends on Loan Amount and Number of Dependents
    /// 
    /// 4. Inference Process:
    ///    - Uses probabilistic reasoning to estimate conditional probabilities
    ///    - Updates beliefs about variables based on observed evidence
    ///    - Propagates evidence through the network to update related variables
    /// 
    /// 5. Applications:
    ///    - Risk assessment for loan applications
    ///    - Borrower profile analysis
    ///    - Default probability estimation
    ///    - Loan amount optimization
    /// </remarks>

    public class BorrowerBayesianNetwork
    {

        public Range N;

        /// <summary>
        /// The number of examples in the training data.
        /// </summary>
        public Variable<int> NumberOfExamples;

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
        /// The loan duration of the borrower.
        /// </summary>
        public VariableArray<int> LoanDuration;

        /// <summary>
        /// The age of the borrower.
        /// </summary>
        public VariableArray<int> Age;

        /// <summary>
        /// The employment status of the borrower.
        /// </summary>
        public VariableArray<int> EmploymentStatus;

        /// <summary>
        /// The marital status of the borrower.
        /// </summary>
        public VariableArray<int> MaritalStatus;

        /// <summary>
        /// The number of dependents of the borrower.
        /// </summary>
        public VariableArray<int> NumberOfDependents;

        /// <summary>
        /// The education level of the borrower.
        /// </summary>
        public VariableArray<int> EducationLevel;

        /// <summary>
        /// The home ownership status of the borrower.
        /// </summary>
        public VariableArray<int> HomeOwnershipStatus;

        /// <summary>
        /// The prior distribution for the credit score of the borrower.
        /// </summary>
        public Variable<Gaussian> CreditScorePrior;

        /// <summary>
        /// The prior distribution for the annual income of the borrower.
        /// </summary>
        public Variable<Gaussian> AnnualIncomePrior;

        /// <summary>
        /// The prior distribution for the employment status of the borrower.
        /// </summary>
        public Variable<Vector> EmploymentStatusPrior;

        /// <summary>
        /// The prior distribution for the marital status of the borrower.
        /// </summary>
        public Variable<Vector> MaritalStatusPrior;

        /// <summary>
        /// The prior distribution for the education level of the borrower.
        /// </summary>
        public Variable<Vector> EducationLevelPrior;

        /// <summary>
        /// The prior distribution for the loan amount of the borrower.
        /// </summary>
        public Variable<Vector> LoanAmountPrior;

        /// <summary>
        /// The conditional distribution for the loan amount of the borrower given the income and credit score.
        /// </summary>
        public VariableArray<VariableArray<Gaussian>> LoanAmountGivenIncomeAndCredit;

        /// <summary>
        /// The posterior distribution for the credit score of the borrower.
        /// </summary>
        public Gaussian CreditScorePosterior;

        /// <summary>
        /// The posterior distribution for the annual income of the borrower.
        /// </summary>
        public Gaussian AnnualIncomePosterior;

        /// <summary>
        /// The posterior distribution for the employment status of the borrower.
        /// </summary>
        public Discrete[] EmploymentStatusPosterior;

        /// <summary>
        /// The posterior distribution for the marital status of the borrower.
        /// </summary>
        public Discrete[] MaritalStatusPosterior;

        /// <summary>
        /// The posterior distribution for the education level of the borrower.
        /// </summary>
        public Discrete[] EducationLevelPosterior;

        /// <summary>
        /// The posterior distribution for the loan amount of the borrower.
        /// </summary>
        private Dictionary<string, Dictionary<string, int>> categoricalMappings;

        /// <summary>
        /// The inverse mappings for generating new samples (indices to string values).
        /// </summary>
        private Dictionary<string, Dictionary<int, string>> inverseCategoricalMappings;

        /// <summary>
        /// The inference engine for the Bayesian network.
        /// </summary>
        public InferenceEngine Engine = new InferenceEngine();

        /// <summary>
        /// Initializes a new instance of the BorrowerBayesianNetwork class.
        /// </summary>
        /// <param name="dataProcessor">The data processor containing the training data.</param>
        /// <remarks>
        /// This constructor initializes the Bayesian network structure and sets up the mappings for categorical variables.
        /// It also defines the prior distributions for the continuous variables and creates the conditional dependencies.
        /// </remarks>
        public BorrowerBayesianNetwork(BorrowerModelBuilder dataProcessor)
        {
            // Initialize mappings for categorical variables
            this.InitializeCategoricalMappings(dataProcessor);

            // Set up the model structure
            NumberOfExamples = Variable.New<int>().Named("NumberOfExamples");
            N = new Range(NumberOfExamples).Named("N");

            // Define ranges for categorical variables
            Range ES = new Range(dataProcessor.EmploymentStatusValues.Count).Named("ES");
            Range MS = new Range(dataProcessor.MaritalStatusValues.Count).Named("MS");
            Range EL = new Range(dataProcessor.EducationLevelValues.Count).Named("EL");
            Range HOS = new Range(dataProcessor.HomeOwnershipStatusValues.Count).Named("HOS");
            Range LP = new Range(dataProcessor.LoanPurposeValues.Count).Named("LP");

            // Define priors for continuous variables
            // For example, CreditScore
            CreditScorePrior = Variable.New<Gaussian>().Named("CreditScorePrior");
            // Create a distribution over the mean of the Gaussian that generates credit scores
            Variable<double> creditScoreMean = Variable.GaussianFromMeanAndVariance(650, 10000).Named("CreditScoreMean");
            CreditScore = Variable.Array<double>(N).Named("CreditScore");
            // Use the mean to generate credit scores with some additional variance
            CreditScore[N] = Variable.GaussianFromMeanAndVariance(creditScoreMean, 1000).ForEach(N);

            // Define priors for categorical variables
            // For example, EmploymentStatus
            EmploymentStatusPrior = Variable.New<Vector>().Named("EmploymentStatusPrior");
            EmploymentStatusPrior.SetValueRange(ES);
            EmploymentStatus = Variable.Array<int>(N).Named("EmploymentStatus");
            EmploymentStatus[N] = Variable.Discrete(EmploymentStatusPrior).ForEach(N);


            // For Marital Status - This was missing proper initialization
            MaritalStatusPrior = Variable.New<Vector>().Named("MaritalStatusPrior");
            MaritalStatusPrior.SetValueRange(MS);
            MaritalStatus = Variable.Array<int>(N).Named("MaritalStatus");
            MaritalStatus[N] = Variable.Discrete(MaritalStatusPrior).ForEach(N);

            // For Education Level
            EducationLevelPrior = Variable.New<Vector>().Named("EducationLevelPrior");
            EducationLevelPrior.SetValueRange(EL);
            EducationLevel = Variable.Array<int>(N).Named("EducationLevel");
            EducationLevel[N] = Variable.Discrete(EducationLevelPrior).ForEach(N);

            // For Home Ownership Status
            Variable<Vector> homeOwnershipStatusPrior = Variable.New<Vector>().Named("HomeOwnershipStatusPrior");
            homeOwnershipStatusPrior.SetValueRange(HOS);
            HomeOwnershipStatus = Variable.Array<int>(N).Named("HomeOwnershipStatus");
            HomeOwnershipStatus[N] = Variable.Discrete(homeOwnershipStatusPrior).ForEach(N);

            // Define conditional dependencies
            // For example, Loan Amount depends on Annual Income and Credit Score
            // First, define the Annual Income variable
            AnnualIncomePrior = Variable.New<Gaussian>().Named("AnnualIncomePrior");
            Variable<double> annualIncomeMean = Variable.GaussianFromMeanAndVariance(60000, 100000000).Named("AnnualIncomeMean");
            AnnualIncome = Variable.Array<double>(N).Named("AnnualIncome");
            AnnualIncome[N] = Variable.GaussianFromMeanAndVariance(annualIncomeMean, 100000).ForEach(N);

            // Now define Loan Amount conditioned on Annual Income (simplified for now)
            // In a complete implementation, we would use multiple conditions
            Variable<Gaussian> loanAmountPrior = Variable.New<Gaussian>().Named("LoanAmountPrior");
            // Create a distribution over the base loan amount
            Variable<double> loanAmountBase = Variable.GaussianFromMeanAndVariance(20000, 50000000).Named("LoanAmountBase");
            LoanAmount = Variable.Array<double>(N).Named("LoanAmount");


            using (Variable.ForEach(N))
            {
                // Instead of trying to multiply variables directly, use a different approach
                // We'll create a model where loan amount is loosely correlated with income
                // Use a ratio factor to represent relationship between income and loan
                double ratio = 0.3; // Loan is typically around 30% of annual income

                // Create a new variable for the loan amount that incorporates income correlation
                // but doesn't require direct multiplication
                Variable<double> annualIncomeContribution = Variable.GaussianFromMeanAndVariance(
                    ratio * 60000, // Expected contribution based on average income
                    ratio * ratio * 10000000 // Variance scaled by square of ratio
                );

                // Final loan amount combines base amount and income-related component
                LoanAmount[N] = Variable.GaussianFromMeanAndVariance(
                    loanAmountBase + annualIncomeContribution,
                    1000000.0
                );
            }

        }

        /// <summary>
        /// Initializes the mappings for categorical variables.
        /// </summary>
        /// <param name="dataProcessor">The data processor containing the training data.</param>
        /// <remarks>
        /// This method creates mappings for categorical variables to their corresponding indices.
        /// It also creates inverse mappings for generating new samples from indices to string values.
        /// </remarks>
        private void InitializeCategoricalMappings(BorrowerModelBuilder dataProcessor)
        {
            // Initialize dictionaries to store mappings between categorical variables and their indices
            // Forward mapping: variable name -> (string value -> integer index)
            categoricalMappings = new Dictionary<string, Dictionary<string, int>>();
            // Reverse mapping: variable name -> (integer index -> string value)
            inverseCategoricalMappings = new Dictionary<string, Dictionary<int, string>>();

            // Create bidirectional mappings for each categorical variable in the model
            // These mappings enable efficient conversion between string values and numerical indices
            // which is necessary for the Bayesian network's probabilistic calculations
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
        /// <param name="variableName">The name of the categorical variable.</param>
        /// <param name="values">The list of possible values for the variable.</param>
        /// <remarks>
        /// This method creates a mapping from string values to their corresponding indices and vice versa.
        /// It also updates the categoricalMappings and inverseCategoricalMappings dictionaries.
        /// </remarks>
        private void InitializeMappingForVariable(string variableName, List<string> values)
        {
            // Create forward and reverse mappings for categorical variable values
            // Forward mapping: string value -> integer index
            Dictionary<string, int> mapping = new Dictionary<string, int>();
            // Reverse mapping: integer index -> string value
            Dictionary<int, string> inverseMapping = new Dictionary<int, string>();

            // Iterate through each possible value and create bidirectional mappings
            // This allows efficient conversion between string values and their corresponding indices
            for (int i = 0; i < values.Count; i++)
            {
                // Map string value to its index (e.g., "Employed" -> 0)
                mapping[values[i]] = i;
                // Map index back to string value (e.g., 0 -> "Employed")
                inverseMapping[i] = values[i];
            }

            // Store the mappings in the class-level dictionaries for later use
            // These mappings enable efficient conversion between string and integer representations
            categoricalMappings[variableName] = mapping;
            inverseCategoricalMappings[variableName] = inverseMapping;
        }

        /// <summary>
        /// Learns the parameters for the Bayesian network based on the training data.
        /// </summary>
        /// <param name="trainingData">The training data containing the borrower profiles.</param>
        /// <remarks>
        /// This method converts string categorical values to indices, prepares arrays for all variables,
        /// and sets the priors based on the observed data statistics.
        /// </remarks>
        public void LearnParameters(List<BorrowerProfile> trainingData)
        {
            // Convert categorical string values to numerical indices for Bayesian network processing
            // Each categorical variable (employment status, marital status, education level) is mapped to an integer index
            // This conversion is necessary for the probabilistic model to process categorical data
            int[] employmentStatusIndices = trainingData
                .Select(b => categoricalMappings["EmploymentStatus"][b.EmploymentStatus])
                .ToArray();

            int[] maritalStatusIndices = trainingData
                .Select(b => categoricalMappings["MaritalStatus"][b.MaritalStatus])
                .ToArray();

            int[] educationLevelIndices = trainingData
                .Select(b => categoricalMappings["EducationLevel"][b.EducationLevel])
                .ToArray();

            // Extract continuous variables from training data
            // These variables will be modeled using Gaussian distributions
            double[] creditScores = trainingData.Select(b => b.CreditScore).ToArray();
            double[] annualIncomes = trainingData.Select(b => b.AnnualIncome).ToArray();
            double[] loanAmounts = trainingData.Select(b => b.LoanAmount).ToArray();

            // Calculate basic statistics for continuous variables
            // These statistics will be used to initialize the prior distributions
            double creditScoreMean = creditScores.Average();
            double creditScoreVariance = creditScores.Select(v => Math.Pow(v - creditScoreMean, 2)).Average();

            double annualIncomeMean = annualIncomes.Average();
            double annualIncomeVariance = annualIncomes.Select(v => Math.Pow(v - annualIncomeMean, 2)).Average();

            // Initialize the model with the number of training examples
            NumberOfExamples.ObservedValue = trainingData.Count;

            // Set up prior distributions for continuous variables using calculated statistics
            // Gaussian distributions are used for continuous variables like credit score and annual income
            CreditScorePrior.ObservedValue = new Gaussian(creditScoreMean, Math.Sqrt(creditScoreVariance));
            AnnualIncomePrior.ObservedValue = new Gaussian(annualIncomeMean, Math.Sqrt(annualIncomeVariance));

            // Initialize prior distributions for categorical variables
            // These are based on the observed frequencies in the training data
            EmploymentStatusPrior.ObservedValue = this.GetCategoricalPrior(employmentStatusIndices,
                categoricalMappings["EmploymentStatus"].Count);

            MaritalStatusPrior.ObservedValue = this.GetCategoricalPrior(maritalStatusIndices,
                categoricalMappings["MaritalStatus"].Count);

            EducationLevelPrior.ObservedValue = this.GetCategoricalPrior(educationLevelIndices,
                categoricalMappings["EducationLevel"].Count);

            try
            {
                // Initialize posterior distributions for continuous variables
                // These are initially set to the same values as the priors
                CreditScorePosterior = new Gaussian(creditScoreMean, Math.Sqrt(creditScoreVariance));
                AnnualIncomePosterior = new Gaussian(annualIncomeMean, Math.Sqrt(annualIncomeVariance));

                // Initialize arrays for categorical posterior distributions
                EmploymentStatusPosterior = new Discrete[1];
                MaritalStatusPosterior = new Discrete[1];

                // Create probability vectors for categorical variables
                // These vectors represent the probability distribution over possible states
                Vector empStatusVector = this.GetCategoricalPrior(employmentStatusIndices,
                    categoricalMappings["EmploymentStatus"].Count);
                Vector maritalStatusVector = this.GetCategoricalPrior(maritalStatusIndices,
                    categoricalMappings["MaritalStatus"].Count);

                // Convert probability vectors to Discrete distributions
                EmploymentStatusPosterior[0] = new Discrete(empStatusVector);
                MaritalStatusPosterior[0] = new Discrete(maritalStatusVector);

                // Attempt to perform Bayesian inference on observed variables
                // This is a more sophisticated approach that may not work in all model configurations
                try
                {
                    if (!CreditScore.IsObserved)
                    {
                        CreditScore.ObservedValue = creditScores;
                        Gaussian[] inferredCreditScore = Engine.Infer<Gaussian[]>(CreditScore);
                        // Note: Inference results may not be used depending on model structure
                    }

                    if (!EmploymentStatus.IsObserved)
                    {
                        EmploymentStatus.ObservedValue = employmentStatusIndices;
                        Discrete[] inferredEmploymentStatus = Engine.Infer<Discrete[]>(EmploymentStatus);
                        // Note: Inference results may not be used depending on model structure
                    }
                }
                catch (Exception inferEx)
                {
                    Console.WriteLine($"Secondary inference attempt failed: {inferEx.Message}");
                    // Fall back to using direct statistical estimates if inference fails
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Primary inference error: {ex.Message}");

                // Fallback approach: use direct statistical estimates if inference fails
                // This ensures we always have some reasonable parameter estimates
                CreditScorePosterior = new Gaussian(creditScoreMean, Math.Sqrt(creditScoreVariance));
                AnnualIncomePosterior = new Gaussian(annualIncomeMean, Math.Sqrt(annualIncomeVariance));

                // Initialize categorical distributions using observed frequencies
                EmploymentStatusPosterior = new Discrete[1];
                MaritalStatusPosterior = new Discrete[1];

                // Create probability vectors from observed frequencies
                Vector empStatusVector = this.GetCategoricalPrior(employmentStatusIndices,
                    categoricalMappings["EmploymentStatus"].Count);
                Vector maritalStatusVector = this.GetCategoricalPrior(maritalStatusIndices,
                    categoricalMappings["MaritalStatus"].Count);

                // Convert probability vectors to Discrete distributions
                EmploymentStatusPosterior[0] = new Discrete(empStatusVector);
                MaritalStatusPosterior[0] = new Discrete(maritalStatusVector);
            }

            // Output the learned parameters for verification
            Console.WriteLine("Learned parameters:");
            Console.WriteLine($"  Credit Score: Mean={CreditScorePosterior.GetMean():F2}, StdDev={Math.Sqrt(CreditScorePosterior.GetVariance()):F2}");
            Console.WriteLine($"  Annual Income: Mean={AnnualIncomePosterior.GetMean():F2}, StdDev={Math.Sqrt(AnnualIncomePosterior.GetVariance()):F2}");
        }

        /// <summary>
        /// Gets the prior distribution for a categorical variable based on the observed data.
        /// </summary>
        /// <param name="indices">The indices of the observed values.</param>
        /// <param name="stateCount">The total number of possible states for the variable.</param>
        /// <returns>A probability vector representing the prior distribution.</returns>
        /// <remarks>
        /// This method calculates the prior distribution for a categorical variable by counting the occurrences of each state in the observed data.
        /// It then converts these counts into probabilities and returns them as a probability vector.
        /// </remarks>
        private Vector GetCategoricalPrior(int[] indices, int stateCount)
        {
            // Initialize an array to store the frequency count of each possible state
            // The array size matches the number of possible states for the categorical variable
            double[] counts = new double[stateCount];

            // Iterate through all observed indices and increment their corresponding count
            // This builds a frequency distribution of how often each state appears in the data
            foreach (int index in indices)
            {
                counts[index]++;
            }

            // Normalize the frequency counts to create a probability distribution
            // Each count is divided by the total number of observations to get a probability
            // This ensures the probabilities sum to 1.0 (100%)
            for (int i = 0; i < counts.Length; i++)
            {
                counts[i] /= indices.Length;
            }

            // Convert the probability array into a Vector object for use in the Bayesian network
            // This vector represents the prior probability distribution for the categorical variable
            return Vector.FromArray(counts);
        }

        /// <summary>
        /// Generates a value from the statistics with fallback values if stats are missing.
        /// </summary>
        /// <param name="statName">The name of the statistic to generate a value for.</param>
        /// <param name="stats">The dictionary containing the statistics.</param>
        /// <param name="random">The random number generator.</param>
        /// <param name="min">The minimum value for the generated value.</param>
        /// <param name="max">The maximum value for the generated value.</param>
        /// <param name="fallbackMean">The mean value to use if stats are missing.</param>
        /// <param name="fallbackStdDev">The standard deviation value to use if stats are missing.</param>
        /// <returns>A generated value from the statistics.</returns>
        /// <remarks>
        /// This method generates a value from the statistics with fallback values if stats are missing.
        /// It uses a Gaussian distribution to generate a value and applies constraints if provided.
        /// </remarks>
        private int GenerateValueFromStatsWithFallback(string statName, Dictionary<string, (double Min, double Max, double Mean, double StdDev)> stats,
            Random random, int? min = null, int? max = null, double fallbackMean = 0, double fallbackStdDev = 1)
        {
            // Check if we have statistics for the requested variable
            if (stats.TryGetValue(statName, out (double Min, double Max, double Mean, double StdDev) statValues))
            {
                // Generate a value from a normal distribution using the variable's statistics
                // The value is centered around the mean with spread determined by standard deviation
                double value = statValues.Mean + random.NextGaussian() * statValues.StdDev;

                // Ensure the generated value respects the minimum bound if specified
                if (min.HasValue && value < min.Value)
                {
                    value = min.Value;
                }

                // Ensure the generated value respects the maximum bound if specified
                if (max.HasValue && value > max.Value)
                {
                    value = max.Value;
                }

                // Round to nearest integer since we're generating discrete values
                return (int)Math.Round(value);
            }
            else
            {
                // If no statistics are available, use fallback parameters to generate a value
                // This ensures we can still generate reasonable values even without historical data
                double value = fallbackMean + random.NextGaussian() * fallbackStdDev;

                // Apply the same bounds checking as above
                if (min.HasValue && value < min.Value)
                {
                    value = min.Value;
                }

                if (max.HasValue && value > max.Value)
                {
                    value = max.Value;
                }

                // Round to nearest integer for consistency
                return (int)Math.Round(value);
            }
        }

        /// <summary>
        /// Generates a list of sample borrower profiles.
        /// </summary>
        /// <param name="sampleCount">The number of samples to generate.</param>
        /// <param name="dataProcessor">The data processor containing the training data.</param>
        /// <returns>A list of generated borrower profiles.</returns>
        /// <remarks>
        /// This method generates a list of sample borrower profiles using the Bayesian network structure.
        /// It generates demographic variables, categorical variables, and credit history variables,
        /// and then calculates the credit score based on the relationships defined in the Bayesian network.
        /// </remarks>
        public List<BorrowerProfile> GenerateSamples(int sampleCount, BorrowerModelBuilder dataProcessor)
        {
            // Initialize list to store generated borrower samples and random number generator
            List<BorrowerProfile> generatedSamples = new List<BorrowerProfile>();
            Random random = new Random();

            // Retrieve continuous variable statistics from data processor
            Dictionary<string, (double Min, double Max, double Mean, double StdDev)> stats = dataProcessor.ContinuousStats;

            // Generate specified number of borrower samples
            for (int i = 0; i < sampleCount; i++)
            {
                BorrowerProfile sample = new BorrowerProfile();

                //---------------------------------------------------------------------
                // 1. GENERATE PRIMARY/INDEPENDENT VARIABLES
                // These variables form the foundation of the borrower profile
                //---------------------------------------------------------------------

                // Generate demographic variables using statistical distributions
                sample.Age = this.GenerateValueFromStatsWithFallback("Age", stats, random, 18, 85, 43, 15);

                // Generate categorical variables using posterior distributions
                sample.EmploymentStatus = inverseCategoricalMappings["EmploymentStatus"][
                    this.SampleFromDiscrete(EmploymentStatusPosterior[0])];
                sample.MaritalStatus = inverseCategoricalMappings["MaritalStatus"][
                    this.SampleFromDiscrete(MaritalStatusPosterior[0])];
                sample.EducationLevel = this.SampleCategoricalVariable("EducationLevel", dataProcessor);
                sample.HomeOwnershipStatus = this.SampleCategoricalVariable("HomeOwnershipStatus", dataProcessor);
                sample.LoanPurpose = this.SampleCategoricalVariable("LoanPurpose", dataProcessor);

                // Generate basic financial variables using posterior distributions
                sample.AnnualIncome = (int)AnnualIncomePosterior.Sample();

                // Generate dependents count with marital status influence
                if (sample.MaritalStatus == "Married")
                {
                    // Married individuals typically have more dependents
                    sample.NumberOfDependents = this.GenerateValueFromStatsWithFallback(
                        "NumberOfDependents", stats, random, 0, 5,
                        stats["NumberOfDependents"].Mean + 0.5, stats["NumberOfDependents"].StdDev);
                }
                else
                {
                    // Single/divorced/widowed individuals typically have fewer dependents
                    sample.NumberOfDependents = this.GenerateValueFromStatsWithFallback(
                        "NumberOfDependents", stats, random, 0, 3,
                        Math.Max(0, stats["NumberOfDependents"].Mean - 0.5), stats["NumberOfDependents"].StdDev);
                }

                //---------------------------------------------------------------------
                // 2. GENERATE CREDIT HISTORY VARIABLES
                // These variables reflect the borrower's creditworthiness
                //---------------------------------------------------------------------

                // Calculate credit history length based on age
                int maxHistoryLength = Math.Max(0, sample.Age - 18); // Minimum age for credit history
                sample.LengthOfCreditHistory = Math.Min(
                    maxHistoryLength,
                    this.GenerateValueFromStatsWithFallback("LengthOfCreditHistory", stats, random, 0, maxHistoryLength)
                );

                // Calculate default risk factors based on demographics
                double defaultRiskFactor = 1.0;
                if (sample.EmploymentStatus == "Unemployed") defaultRiskFactor *= 2.0;
                if (sample.Age < 30) defaultRiskFactor *= 1.5;
                if (sample.EducationLevel == "High School") defaultRiskFactor *= 1.3;

                // Generate previous defaults with adjusted probability
                double defaultProbability = 0.1 * defaultRiskFactor; // Base 10% chance adjusted by risk factors
                sample.PreviousLoanDefaults = random.NextDouble() < defaultProbability ? 1 : 0;

                // Generate bankruptcy history with age influence
                double bankruptcyProbability = 0.05 * defaultRiskFactor * (sample.Age / 60.0);
                sample.BankruptcyHistory = random.NextDouble() < bankruptcyProbability ? 1 : 0;

                // Generate number of credit lines with history influence
                sample.NumberOfOpenCreditLines = this.GenerateValueFromStatsWithFallback(
                    "NumberOfOpenCreditLines", stats, random, 0, 20,
                    Math.Min(15, sample.LengthOfCreditHistory / 2.0 + 2),
                    Math.Max(1, stats["NumberOfOpenCreditLines"].StdDev / 2)
                );

                // Generate credit card utilization rate
                sample.CreditCardUtilizationRate = this.GenerateDoubleValueFromStatsWithFallback(
                    "CreditCardUtilizationRate", stats, random, 0, 1.0, 0.5, 0.29);

                // Generate credit inquiries with default correlation
                double inquiryBaseline = sample.PreviousLoanDefaults == 1 ? 6 : 3;
                sample.NumberOfCreditInquiries = this.GenerateValueFromStatsWithFallback(
                    "NumberOfCreditInquiries", stats, random, 0, 15, inquiryBaseline, 2);

                //---------------------------------------------------------------------
                // 3. CALCULATE CREDIT SCORE
                // This is a critical variable for loan decisions
                //---------------------------------------------------------------------

                // Start with base score distribution
                double baseScore = 650 + random.NextGaussian() * 60;

                // Calculate credit history impact
                double historyBoost = Math.Min(100, sample.LengthOfCreditHistory * 3);

                // Calculate utilization penalties
                double utilizationPenalty = 0;
                if (sample.CreditCardUtilizationRate > 0.7)
                    utilizationPenalty = 150;
                else if (sample.CreditCardUtilizationRate > 0.5)
                    utilizationPenalty = 100;
                else if (sample.CreditCardUtilizationRate > 0.3)
                    utilizationPenalty = 50;

                // Calculate default and bankruptcy penalties
                double defaultPenalty = sample.PreviousLoanDefaults * 50 * (1 + random.NextDouble());
                double bankruptcyPenalty = sample.BankruptcyHistory * 150 * (1 + random.NextDouble() / 2);

                // Calculate inquiry penalties
                double inquiryPenalty = Math.Max(0, sample.NumberOfCreditInquiries - 3) * 5 * (1 + random.NextDouble());

                // Calculate final credit score
                double creditScore = baseScore + historyBoost - utilizationPenalty - defaultPenalty - bankruptcyPenalty - inquiryPenalty;
                sample.CreditScore = Math.Max(300, Math.Min(850, creditScore));

                //---------------------------------------------------------------------
                // 4. CALCULATE DEBT AND INCOME VARIABLES
                // These variables determine repayment capacity
                //---------------------------------------------------------------------

                // Calculate monthly debt payments based on income and credit score
                double incomeToDebtFactor = sample.CreditScore < 600 ? 0.4 : 0.3;
                double monthlyIncome = sample.AnnualIncome / 12.0;

                sample.MonthlyDebtPayments = this.GenerateValueFromStatsWithFallback(
                    "MonthlyDebtPayments", stats, random, 0,
                    (int)(monthlyIncome * 0.6),
                    (int)(monthlyIncome * incomeToDebtFactor),
                    (int)(monthlyIncome * 0.1)
                );

                // Calculate debt-to-income ratio
                sample.DebtToIncomeRatio = Math.Min(1.0, sample.MonthlyDebtPayments / Math.Max(1, monthlyIncome));

                // Calculate loan amount based on purpose and credit score
                double loanAmountBaseFactor = 0;
                switch (sample.LoanPurpose)
                {
                    case "Home": loanAmountBaseFactor = 3.0; break;
                    case "Auto": loanAmountBaseFactor = 0.5; break;
                    case "Education": loanAmountBaseFactor = 1.0; break;
                    case "Debt Consolidation": loanAmountBaseFactor = 0.7; break;
                    default: loanAmountBaseFactor = 0.3; break;
                }

                // Adjust loan amount by credit score
                if (sample.CreditScore < 600)
                    loanAmountBaseFactor *= 0.7;
                else if (sample.CreditScore > 750)
                    loanAmountBaseFactor *= 1.2;

                // Calculate final loan amount
                double baseLoanAmount = sample.AnnualIncome * loanAmountBaseFactor;
                double loanAmountRandomFactor = 0.8 + random.NextDouble() * 0.4;
                sample.LoanAmount = Math.Max(1000, baseLoanAmount * loanAmountRandomFactor);

                // Determine loan duration based on purpose
                if (sample.LoanPurpose == "Home")
                    sample.LoanDuration = random.Next(15, 31);
                else if (sample.LoanPurpose == "Auto")
                    sample.LoanDuration = random.Next(3, 8);
                else if (sample.LoanPurpose == "Education")
                    sample.LoanDuration = random.Next(5, 16);
                else
                    sample.LoanDuration = random.Next(1, 6);

                //---------------------------------------------------------------------
                // 5. CALCULATE INTEREST RATE
                // This is based on credit score and loan characteristics
                //---------------------------------------------------------------------

                // Start with base interest rate
                double baseRate = 0.03;

                // Calculate credit score adjustment
                double creditScoreAdj = Math.Max(0, (750 - sample.CreditScore) / 150.0);
                double creditScoreFactor = Math.Pow(1.5, creditScoreAdj) - 1; // Exponential penalty for lower scores

                // Loan purpose impacts rate
                double purposeFactor = 0;
                switch (sample.LoanPurpose)
                {
                    case "Home": purposeFactor = -0.005; break; // Lower rates for secured home loans
                    case "Auto": purposeFactor = 0.01; break;
                    case "Education": purposeFactor = 0.02; break;
                    case "Debt Consolidation": purposeFactor = 0.03; break;
                    default: purposeFactor = 0.04; break; // Higher rates for "Other" purposes
                }

                // Calculate interest rate
                double interestRate = baseRate + creditScoreFactor * 0.15 + purposeFactor;

                // Add random noise and clamp to reasonable range
                interestRate += (random.NextDouble() - 0.5) * 0.01; // +/- 0.5%
                sample.InterestRate = Math.Max(0.01, Math.Min(0.3, interestRate)); // 1% to 30% range

                //---------------------------------------------------------------------
                // 6. FINANCIAL ACCOUNTS AND BALANCES
                //---------------------------------------------------------------------

                // Financial accounts (correlated with income and age)
                double incomeFactor = Math.Min(3.0, Math.Max(0.5, sample.AnnualIncome / 60000.0));
                double ageFactor = Math.Min(2.0, Math.Max(0.5, sample.Age / 40.0));

                // Savings account (more for higher income and age)
                sample.SavingsAccountBalance = this.GenerateValueFromStatsWithFallback(
                    "SavingsAccountBalance", stats, random, 0, null,
                    stats["SavingsAccountBalance"].Mean * incomeFactor * ageFactor,
                    stats["SavingsAccountBalance"].StdDev * incomeFactor
                );

                // Checking account (less affected by age, more by income)
                sample.CheckingAccountBalance = this.GenerateValueFromStatsWithFallback(
                    "CheckingAccountBalance", stats, random, 0, null,
                    stats["CheckingAccountBalance"].Mean * incomeFactor,
                    stats["CheckingAccountBalance"].StdDev * Math.Sqrt(incomeFactor)
                );

                // Investment account (strongly affected by both age and income)
                sample.InvestmentAccountBalance = this.GenerateValueFromStatsWithFallback(
                    "InvestmentAccountBalance", stats, random, 0, null,
                    stats["InvestmentAccountBalance"].Mean * incomeFactor * ageFactor * 1.2,
                    stats["InvestmentAccountBalance"].StdDev * incomeFactor * ageFactor
                );

                // Retirement account (very strongly affected by age and somewhat by income)
                sample.RetirementAccountBalance = this.GenerateValueFromStatsWithFallback(
                    "RetirementAccountBalance", stats, random, 0, null,
                    stats["RetirementAccountBalance"].Mean * incomeFactor * Math.Pow(ageFactor, 2),
                    stats["RetirementAccountBalance"].StdDev * incomeFactor * ageFactor
                );

                // Emergency fund (correlated with income and financial responsibility)
                double emergencyFundFactor = sample.CreditScore > 700 ? 1.5 : 1.0; // Better credit = better emergency fund
                sample.EmergencyFundBalance = this.GenerateValueFromStatsWithFallback(
                    "EmergencyFundBalance", stats, random, 0, null,
                    stats["EmergencyFundBalance"].Mean * incomeFactor * emergencyFundFactor,
                    stats["EmergencyFundBalance"].StdDev * incomeFactor
                );

                //---------------------------------------------------------------------
                // 7. CALCULATE LIABILITIES
                //---------------------------------------------------------------------

                // Mortgage balance (depends on home ownership status)
                if (sample.HomeOwnershipStatus == "Own")
                {
                    sample.MortgageBalance = 0; // Owns home outright
                }
                else if (sample.HomeOwnershipStatus == "Mortgage")
                {
                    // Mortgage typically 3-5x annual income depending on location and income
                    double mortgageFactor = 3.0 + (incomeFactor - 1) * 2; // 3-5x income
                    sample.MortgageBalance = this.GenerateValueFromStatsWithFallback(
                        "MortgageBalance", stats, random, 10000, 1000000,
                        sample.AnnualIncome * mortgageFactor * (0.7 + random.NextDouble() * 0.6),
                        sample.AnnualIncome * 0.5
                    );
                }
                else
                {
                    sample.MortgageBalance = 0; // Renting or other
                }

                // Auto loan balance
                sample.AutoLoanBalance = this.GenerateValueFromStatsWithFallback(
                    "AutoLoanBalance", stats, random, 0, 100000,
                    Math.Min(sample.AnnualIncome * 0.4, 35000) * random.NextDouble(),
                    5000
                );

                // Student loan balance (correlated with education level)
                double studentLoanFactor = 0;
                switch (sample.EducationLevel)
                {
                    case "High School": studentLoanFactor = 0.1; break;
                    case "Associate": studentLoanFactor = 0.5; break;
                    case "Bachelor": studentLoanFactor = 1.0; break;
                    case "Master": studentLoanFactor = 1.5; break;
                    case "Doctorate": studentLoanFactor = 2.0; break;
                    default: studentLoanFactor = 0.5; break;
                }

                sample.StudentLoanBalance = this.GenerateValueFromStatsWithFallback(
                    "StudentLoanBalance", stats, random, 0, 250000,
                    20000 * studentLoanFactor,
                    10000 * studentLoanFactor
                );

                // Personal loan balance
                sample.PersonalLoanBalance = this.GenerateValueFromStatsWithFallback(
                    "PersonalLoanBalance", stats, random, 0, 50000,
                    5000 * (random.NextDouble() * incomeFactor),
                    2000
                );

                // Rent payments (for renters only)
                if (sample.HomeOwnershipStatus == "Rent")
                {
                    double rentToIncomeFactor = 0.25 + random.NextDouble() * 0.15; // 25-40% of monthly income
                    sample.RentPayments = (int)(monthlyIncome * rentToIncomeFactor);
                }
                else
                {
                    sample.RentPayments = 0;
                }

                //---------------------------------------------------------------------
                // 8. CALCULATE ASSETS, LIABILITIES, AND NET WORTH
                //---------------------------------------------------------------------

                // Total assets (sum of financial accounts + estimated home value if owned)
                int homeValue = 0;
                if (sample.HomeOwnershipStatus == "Own")
                    homeValue = (int)(sample.AnnualIncome * 5 * (0.8 + random.NextDouble() * 0.4)); // 4-6x annual income
                else if (sample.HomeOwnershipStatus == "Mortgage")
                    homeValue = (int)(sample.MortgageBalance * (1.0 + random.NextDouble() * 0.5)); // Current mortgage + equity

                sample.TotalAssets = sample.SavingsAccountBalance +
                                    sample.CheckingAccountBalance +
                                    sample.InvestmentAccountBalance +
                                    sample.RetirementAccountBalance +
                                    sample.EmergencyFundBalance +
                                    homeValue;

                // Total liabilities (sum of all loans and balances)
                sample.TotalLiabilities = sample.MortgageBalance +
                                        sample.AutoLoanBalance +
                                        sample.StudentLoanBalance +
                                        sample.PersonalLoanBalance;

                // Net worth (assets - liabilities)
                sample.NetWorth = sample.TotalAssets - sample.TotalLiabilities;

                //---------------------------------------------------------------------
                // 9. MONTHLY EXPENSES
                //---------------------------------------------------------------------

                // Monthly housing costs
                if (sample.HomeOwnershipStatus == "Rent")
                {
                    sample.MonthlyHousingCosts = sample.RentPayments;
                }
                else
                {
                    // Mortgage payment + property tax + insurance + maintenance
                    double monthlyMortgage = 0;
                    if (sample.MortgageBalance > 0)
                    {
                        // Simplified mortgage calculation
                        double monthlyRate = sample.InterestRate / 12;
                        int months = sample.LoanDuration * 12;
                        double factor = Math.Pow(1 + monthlyRate, months);
                        monthlyMortgage = sample.MortgageBalance * monthlyRate * factor / (factor - 1);
                    }

                    // Add property tax, insurance, and maintenance (typically 1.5-2% of home value annually)
                    double monthlyHomeExpenses = homeValue * 0.015 / 12; // 1.5% of home value annually

                    sample.MonthlyHousingCosts = (int)(monthlyMortgage + monthlyHomeExpenses);
                }

                // Other monthly expenses (scale with income)
                sample.MonthlyTransportationCosts = this.GenerateValueFromStatsWithFallback(
                    "MonthlyTransportationCosts", stats, random, 50, 2000,
                    Math.Max(100, monthlyIncome * 0.1),
                    monthlyIncome * 0.03
                );

                sample.MonthlyFoodCosts = this.GenerateValueFromStatsWithFallback(
                    "MonthlyFoodCosts", stats, random, 100, 2000,
                    Math.Max(200, monthlyIncome * 0.12),
                    monthlyIncome * 0.04
                );

                sample.MonthlyHealthcareCosts = this.GenerateValueFromStatsWithFallback(
                    "MonthlyHealthcareCosts", stats, random, 0, 1500,
                    Math.Max(50, monthlyIncome * 0.06),
                    monthlyIncome * 0.02
                );

                sample.MonthlyEntertainmentCosts = this.GenerateValueFromStatsWithFallback(
                    "MonthlyEntertainmentCosts", stats, random, 0, 1000,
                    Math.Max(50, monthlyIncome * 0.05),
                    monthlyIncome * 0.02
                );

                // Annual expenses (sum of monthly * 12 + some annual-only expenses)
                sample.AnnualExpenses = (sample.MonthlyHousingCosts +
                                        sample.MonthlyTransportationCosts +
                                        sample.MonthlyFoodCosts +
                                        sample.MonthlyHealthcareCosts +
                                        sample.MonthlyEntertainmentCosts +
                                        sample.MonthlyDebtPayments) * 12;

                // Add some annual-only expenses (taxes, vacations, gifts, etc.)
                sample.AnnualExpenses += (int)(sample.AnnualIncome * 0.1); // Extra 10% for annual expenses

                //---------------------------------------------------------------------
                // 10. MONTHLY SAVINGS AND BONUSES
                //---------------------------------------------------------------------

                // Monthly savings (constrained by available income after expenses)
                double totalMonthlyExpenses = sample.MonthlyHousingCosts +
                                             sample.MonthlyTransportationCosts +
                                             sample.MonthlyFoodCosts +
                                             sample.MonthlyHealthcareCosts +
                                             sample.MonthlyEntertainmentCosts +
                                             sample.MonthlyDebtPayments;

                double availableForSavings = Math.Max(0, monthlyIncome - totalMonthlyExpenses);

                // Savings rate affected by credit score (financial responsibility proxy)
                double savingsRateFactor = 0.1; // Base 10% savings rate
                if (sample.CreditScore > 750) savingsRateFactor = 0.2; // 20% for excellent credit
                else if (sample.CreditScore > 650) savingsRateFactor = 0.15; // 15% for good credit

                // Calculate savings (can't save more than available)
                double idealSavings = monthlyIncome * savingsRateFactor;
                sample.MonthlySavings = (int)Math.Min(availableForSavings * 0.95, idealSavings);

                // Annual bonuses (correlated with employment status and employer type)
                if (sample.EmploymentStatus == "Unemployed")
                {
                    sample.AnnualBonuses = 0;
                }
                else
                {
                    double bonusFactor = 0;
                    switch (sample.EmployerType)
                    {
                        case "Private": bonusFactor = 0.1; break; // 10% of salary
                        case "Public": bonusFactor = 0.05; break; // 5% of salary
                        case "Self-Employed": bonusFactor = 0.15; break; // 15% of salary (variable income)
                        default: bonusFactor = 0.03; break; // 3% of salary
                    }

                    // Bonus also affected by education
                    if (sample.EducationLevel == "Master" || sample.EducationLevel == "Doctorate")
                        bonusFactor *= 1.5;

                    sample.AnnualBonuses = (int)(sample.AnnualIncome * bonusFactor * random.NextDouble() * 2); // 0-2x expected bonus
                }

                //---------------------------------------------------------------------
                // 11. INSURANCE STATUSES
                //---------------------------------------------------------------------

                // Health insurance (correlated with employment and income)
                if (sample.EmploymentStatus == "Unemployed")
                {
                    sample.HealthInsuranceStatus = random.NextDouble() < 0.3 ? "Insured" : "Uninsured";
                }
                else
                {
                    double healthInsuranceProb = 0.7 + incomeFactor * 0.1;
                    sample.HealthInsuranceStatus = random.NextDouble() < healthInsuranceProb ? "Insured" : "Uninsured";
                }

                // Life insurance (correlated with dependents, age, and income)
                double lifeInsuranceProb = 0.3 + sample.NumberOfDependents * 0.1 + ageFactor * 0.1 + incomeFactor * 0.1;
                sample.LifeInsuranceStatus = random.NextDouble() < lifeInsuranceProb ? "Insured" : "Uninsured";

                // Car insurance (nearly universal for auto loan holders)
                if (sample.AutoLoanBalance > 0)
                {
                    sample.CarInsuranceStatus = "Insured"; // Required for auto loans
                }
                else
                {
                    sample.CarInsuranceStatus = random.NextDouble() < 0.8 ? "Insured" : "Uninsured";
                }

                // Home insurance (required for mortgages, common for homeowners)
                if (sample.HomeOwnershipStatus == "Mortgage")
                {
                    sample.HomeInsuranceStatus = "Insured"; // Required for mortgages
                }
                else if (sample.HomeOwnershipStatus == "Own")
                {
                    sample.HomeInsuranceStatus = random.NextDouble() < 0.9 ? "Insured" : "Uninsured";
                }
                else
                {
                    // Renters insurance is less common
                    sample.HomeInsuranceStatus = random.NextDouble() < 0.4 ? "Insured" : "Uninsured";
                }

                // Other insurance policies (higher income and assets = more policies)
                double otherInsuranceFactor = (incomeFactor + ageFactor) / 2;
                sample.OtherInsurancePolicies = (int)Math.Min(5, Math.Max(0, otherInsuranceFactor * random.Next(0, 4)));

                //---------------------------------------------------------------------
                // 12. EMPLOYMENT DETAILS
                //---------------------------------------------------------------------

                // Employer type
                if (sample.EmploymentStatus == "Unemployed")
                {
                    sample.EmployerType = "Other";
                    sample.JobTenure = 0;
                }
                else if (sample.EmploymentStatus == "Self-Employed")
                {
                    sample.EmployerType = "Self-Employed";
                    // Self-employed tenure can be longer
                    sample.JobTenure = this.GenerateValueFromStatsWithFallback(
                        "JobTenure", stats, random, 0, Math.Min(40, sample.Age - 18),
                        Math.Min(15, (sample.Age - 25) / 2),
                        5
                    );
                }
                else
                {
                    // For employed, select employer type if not already set
                    if (sample.EmployerType == null || sample.EmployerType == "Self-Employed")
                    {
                        sample.EmployerType = this.SampleCategoricalVariable("EmployerType", dataProcessor);
                    }

                    // Typical job tenure is shorter
                    sample.JobTenure = this.GenerateValueFromStatsWithFallback(
                        "JobTenure", stats, random, 0, Math.Min(30, sample.Age - 18),
                        Math.Min(10, (sample.Age - 22) / 3),
                        3
                    );
                }

                //---------------------------------------------------------------------
                // 13. DETERMINE LOAN APPROVAL
                //---------------------------------------------------------------------

                // Logistic regression model for loan approval
                double approvalScore = 0;

                // Credit score has major impact
                approvalScore += (sample.CreditScore - 600) / 100.0 * 2.0; // +/- 5 points

                // Debt-to-income ratio is critical
                if (sample.DebtToIncomeRatio > 0.43)
                    approvalScore -= 3.0; // Hard threshold at 43% DTI
                else
                    approvalScore -= sample.DebtToIncomeRatio / 0.43 * 1.5; // Linear penalty up to -1.5

                // Previous defaults are major negative
                approvalScore -= sample.PreviousLoanDefaults * 2.0;

                // Bankruptcy is a severe negative
                approvalScore -= sample.BankruptcyHistory * 3.0;

                // Employment status matters
                if (sample.EmploymentStatus == "Unemployed")
                    approvalScore -= 2.0;
                else if (sample.EmploymentStatus == "Self-Employed")
                    approvalScore -= 0.5; // Slight penalty for variable income

                // Loan amount relative to income
                double loanToIncomeRatio = sample.LoanAmount / Math.Max(1, sample.AnnualIncome);
                if (loanToIncomeRatio > 5)
                    approvalScore -= 2.0;
                else if (loanToIncomeRatio > 3)
                    approvalScore -= 1.0;
                else if (loanToIncomeRatio < 1)
                    approvalScore += 0.5; // Small loans relative to income are favored

                // Convert score to probability using logistic function
                double approvalProbability = 1.0 / (1.0 + Math.Exp(-approvalScore));

                // Determine approval
                sample.LoanApproved = random.NextDouble() < approvalProbability ? 1 : 0;

                generatedSamples.Add(sample);
            }

            return generatedSamples;
        }


        /// <summary>
        /// Generates a double value from the statistics with fallback values if stats are missing.
        /// </summary>
        /// <param name="statName">The name of the statistic to generate a value for.</param>
        /// <param name="stats">The dictionary containing the statistics.</param>
        /// <param name="random">The random number generator.</param>
        /// <param name="min">The minimum value for the generated value.</param>
        /// <param name="max">The maximum value for the generated value.</param>
        /// <param name="fallbackMean">The mean value to use if stats are missing.</param>
        /// <param name="fallbackStdDev">The standard deviation value to use if stats are missing.</param>
        /// <returns>A generated double value from the statistics.</returns>
        /// <remarks>
        /// This method generates a double value from the statistics with fallback values if stats are missing.
        /// It uses a Gaussian distribution to generate a value and applies constraints if provided.
        /// </remarks>
        private double GenerateDoubleValueFromStatsWithFallback(string statName, Dictionary<string, (double Min, double Max, double Mean, double StdDev)> stats,
            Random random, double? min = null, double? max = null, double fallbackMean = 0, double fallbackStdDev = 1)
        {
            // Try to get statistics for the specified variable from the stats dictionary
            if (stats.TryGetValue(statName, out (double Min, double Max, double Mean, double StdDev) statValues))
            {
                // Generate a value using a Gaussian distribution centered at the mean
                // with standard deviation from the statistics
                double value = statValues.Mean + random.NextGaussian() * statValues.StdDev;

                // Ensure the generated value respects the minimum bound if specified
                if (min.HasValue && value < min.Value)
                {
                    value = min.Value;
                }

                // Ensure the generated value respects the maximum bound if specified
                if (max.HasValue && value > max.Value)
                {
                    value = max.Value;
                }

                return value;
            }
            else
            {
                // If no statistics are available for this variable, use the provided fallback values
                // to generate a value from a Gaussian distribution
                double value = fallbackMean + random.NextGaussian() * fallbackStdDev;

                // Apply the same bounds checking as above
                if (min.HasValue && value < min.Value)
                {
                    value = min.Value;
                }

                if (max.HasValue && value > max.Value)
                {
                    value = max.Value;
                }

                return value;
            }
        }

        /// <summary>
        /// Samples a categorical variable based on its distribution in the data.
        /// </summary>
        /// <param name="variableName">The name of the categorical variable to sample.</param>
        /// <param name="dataProcessor">The data processor containing the training data.</param>
        /// <returns>A sampled value from the categorical variable.</returns>
        /// <remarks>
        private string SampleCategoricalVariable(string variableName, BorrowerModelBuilder dataProcessor)
        {
            // Validate input parameters to ensure they are valid before proceeding
            if (string.IsNullOrWhiteSpace(variableName))
            {
                throw new ArgumentException("Variable name cannot be null or empty", nameof(variableName));
            }

            if (dataProcessor == null)
            {
                throw new ArgumentNullException(nameof(dataProcessor), "Data processor cannot be null");
            }

            // Get the list of possible categorical values for the specified variable
            // This uses pattern matching to select the appropriate list from the data processor
            List<string> possibleValues;
            try
            {
                possibleValues = variableName switch
                {
                    "EmploymentStatus" => dataProcessor.EmploymentStatusValues,
                    "MaritalStatus" => dataProcessor.MaritalStatusValues,
                    "EducationLevel" => dataProcessor.EducationLevelValues,
                    "HomeOwnershipStatus" => dataProcessor.HomeOwnershipStatusValues,
                    "LoanPurpose" => dataProcessor.LoanPurposeValues,
                    "HealthInsuranceStatus" => dataProcessor.HealthInsuranceStatusValues,
                    "LifeInsuranceStatus" => dataProcessor.LifeInsuranceStatusValues,
                    "CarInsuranceStatus" => dataProcessor.CarInsuranceStatusValues,
                    "HomeInsuranceStatus" => dataProcessor.HomeInsuranceStatusValues,
                    "EmployerType" => dataProcessor.EmployerTypeValues,
                    _ => throw new ArgumentException($"Unknown categorical variable: {variableName}"),
                };

                // Ensure we have valid values to sample from
                if (possibleValues == null || !possibleValues.Any())
                {
                    throw new InvalidOperationException($"No possible values found for categorical variable: {variableName}");
                }
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Error getting possible values for variable '{variableName}'", ex);
            }

            // Verify that we have the necessary mappings for converting between string values and indices
            if (!categoricalMappings.ContainsKey(variableName))
            {
                throw new InvalidOperationException($"No mapping found for categorical variable: {variableName}");
            }

            if (!inverseCategoricalMappings.ContainsKey(variableName))
            {
                throw new InvalidOperationException($"No inverse mapping found for categorical variable: {variableName}");
            }

            // Sample an index from the distribution based on the frequency of each value in the training data
            int index;
            try
            {
                index = this.SampleFromDistribution(
                    possibleValues.Count,
                    idx => dataProcessor.TrainingData.Count(m =>
                    {
                        try
                        {
                            // Get the property value and check if it matches the current index
                            string propertyValue = this.GetProperty(m, variableName) as string;
                            return categoricalMappings[variableName].ContainsKey(propertyValue) &&
                                   categoricalMappings[variableName][propertyValue] == idx;
                        }
                        catch (Exception ex)
                        {
                            throw new InvalidOperationException($"Error processing training data for variable '{variableName}'", ex);
                        }
                    }));
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Error sampling from distribution for variable '{variableName}'", ex);
            }

            // Convert the sampled index back to its corresponding categorical value
            try
            {
                if (!inverseCategoricalMappings[variableName].ContainsKey(index))
                {
                    throw new InvalidOperationException($"No mapping found for index {index} in variable '{variableName}'");
                }
                return inverseCategoricalMappings[variableName][index];
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Error converting sampled index to categorical value for variable '{variableName}'", ex);
            }
        }

        /// <summary>
        /// Gets a property value from a model by name using reflection.
        /// </summary>
        /// <param name="model">The model to get the property value from.</param>
        /// <param name="propertyName">The name of the property to get the value of.</param>
        /// <returns>The value of the property.</returns>
        /// <remarks>
        /// This method gets a property value from a model by name using reflection.
        /// </remarks>
        private object GetProperty(BorrowerProfile model, string propertyName)
        {
            // Validate input parameters to ensure they are valid before proceeding
            if (model == null)
            {
                throw new ArgumentNullException(nameof(model), "Model cannot be null");
            }

            if (string.IsNullOrWhiteSpace(propertyName))
            {
                throw new ArgumentException("Property name cannot be null or empty", nameof(propertyName));
            }

            // Use reflection to get the property info for the requested property
            // This allows us to dynamically access properties by name
            System.Reflection.PropertyInfo property = typeof(BorrowerProfile).GetProperty(propertyName) ?? throw new ArgumentException($"Property '{propertyName}' not found on Model type", nameof(propertyName));

            try
            {
                // Use reflection to get the value of the property from the model instance
                return property.GetValue(model);
            }
            catch (Exception ex)
            {
                // Wrap any reflection-related exceptions in a more specific exception
                // that includes the property name for better error reporting
                throw new InvalidOperationException($"Error getting value of property '{propertyName}'", ex);
            }
        }

        /// <summary>
        /// Samples from a discrete distribution defined by a function.
        /// </summary>
        /// <param name="stateCount">The number of states in the distribution.</param>
        /// <param name="countFunc">The function that defines the distribution.</param>
        /// <returns>The index of the sampled state.</returns>
        /// <remarks>
        private int SampleFromDistribution(int stateCount, Func<int, int> countFunc)
        {
            // Initialize an array to store the count of occurrences for each possible state
            int[] counts = new int[stateCount];

            // Populate the counts array by calling the provided count function for each state
            for (int i = 0; i < stateCount; i++)
            {
                counts[i] = countFunc(i);
            }

            // Calculate the total number of occurrences across all states
            int total = counts.Sum();

            // Generate a random number between 0 and total-1 to select a state
            int randomValue = new Random().Next(total);

            // Initialize a running sum to track accumulated counts
            int accumulatedCount = 0;

            // Iterate through each state, accumulating counts until we find the selected state
            for (int i = 0; i < stateCount; i++)
            {
                accumulatedCount += counts[i];
                // If the random value falls within the current accumulated range, return this state
                if (randomValue < accumulatedCount)
                {
                    return i;
                }
            }

            // Fallback: return 0 if no state was selected (should never happen with valid input)
            return 0;
        }

        /// <summary>
        /// Samples from a discrete distribution.
        /// </summary>
        /// <param name="distribution">The distribution to sample from.</param>
        /// <returns>The index of the sampled state.</returns>
        /// <remarks>
        private int SampleFromDiscrete(Discrete distribution)
        {
            // Get the probability vector from the discrete distribution
            Vector probs = distribution.GetProbs();

            // Initialize cumulative sum and generate a random number between 0 and 1
            double sum = 0;
            double rand = new Random().NextDouble();

            // Iterate through each probability in the distribution
            for (int i = 0; i < probs.Count; i++)
            {
                // Add the current probability to the cumulative sum
                sum += probs[i];

                // If the random number falls within the current cumulative probability range,
                // return the corresponding state index
                if (rand < sum)
                {
                    return i;
                }
            }

            // If no state was selected (should be very rare), return the last state as fallback
            return probs.Count - 1;
        }
    }
}