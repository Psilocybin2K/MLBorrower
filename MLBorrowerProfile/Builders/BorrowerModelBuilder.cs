using Microsoft.ML.Probabilistic.Math;
using MLBorrowerProfile.DataModels;
using MLBorrowerProfile.Utils;

namespace MLBorrowerProfile.Builders
{
    /// <summary>
    /// Builder class for creating borrower models from CSV data.
    /// </summary>
    /// <remarks>
    /// The BorrowerModelBuilder class implements a comprehensive data processing pipeline for borrower data:
    /// 
    /// 1. Data Loading and Initialization:
    ///    - Loads borrower data from CSV files
    ///    - Initializes storage for categorical and continuous variables
    ///    - Sets up statistical tracking for numerical features
    /// 
    /// 2. Categorical Data Processing:
    ///    - Collects unique values for categorical variables (employment status, marital status, etc.)
    ///    - Maintains mappings between string values and numerical indices
    ///    - Handles missing or invalid categorical values
    /// 
    /// 3. Continuous Variable Analysis:
    ///    - Calculates statistical measures (mean, standard deviation, min, max)
    ///    - Tracks distributions for numerical features
    ///    - Handles outliers and data normalization
    /// 
    /// 4. Model Building:
    ///    - Constructs Bayesian network structure
    ///    - Sets up prior distributions for variables
    ///    - Establishes conditional dependencies between variables
    /// 
    /// 5. Data Validation and Quality Control:
    ///    - Implements data validation rules
    ///    - Performs consistency checks
    ///    - Handles missing data imputation
    /// 
    /// Key Methods:
    /// - LoadData(string filePath): Loads and parses CSV data
    /// - ProcessCategoricalData(): Processes and maps categorical variables
    /// - CalculateStatistics(): Computes statistical measures for continuous variables
    /// - BuildModel(): Constructs the Bayesian network model
    /// - ValidateData(): Performs data quality checks
    public class BorrowerModelBuilder
    {
        /// <summary>
        /// Lists to store unique categorical values for various borrower attributes.
        /// </summary>
        private List<string> employmentStatusValues = new List<string>();
        /// <summary>
        /// List to store unique marital status values.
        /// </summary>
        private List<string> maritalStatusValues = new List<string>();
        /// <summary>
        /// List to store unique education level values.
        /// </summary>
        private List<string> educationLevelValues = new List<string>();
        /// <summary>
        /// List to store unique home ownership status values.
        /// </summary>
        private List<string> homeOwnershipStatusValues = new List<string>();
        /// <summary>
        /// List to store unique loan purpose values.
        /// </summary>
        private List<string> loanPurposeValues = new List<string>();

        /// <summary>
        /// List to store unique health insurance status values.
        /// </summary>
        private List<string> healthInsuranceStatusValues = new List<string>();

        /// <summary>
        /// List to store unique life insurance status values.
        /// </summary>
        private List<string> lifeInsuranceStatusValues = new List<string>();

        /// <summary>
        /// List to store unique car insurance status values.
        /// </summary>
        private List<string> carInsuranceStatusValues = new List<string>();

        /// <summary>
        /// List to store unique home insurance status values.
        /// </summary>
        private List<string> homeInsuranceStatusValues = new List<string>();

        /// <summary>
        /// List to store unique employer type values.
        /// </summary>
        private List<string> employerTypeValues = new List<string>();

        /// <summary>
        /// Dictionary to store continuous variable statistics.
        /// </summary>
        private Dictionary<string, (double Min, double Max, double Mean, double StdDev)> continuousStats =
            new Dictionary<string, (double, double, double, double)>();

        /// <summary>
        /// List to store processed borrower models.
        /// </summary>
        private List<BorrowerProfile> trainingData = new List<BorrowerProfile>();


        /// <summary>
        /// Loads and processes the borrower data from a CSV file.
        /// </summary>
        /// <param name="csvFilePath">The path to the CSV file containing borrower data.</param>
        /// <remarks>
        /// The method reads the CSV file, parses the header row to get column names and their positions,
        /// and processes each data row (skipping the header row) to create a new borrower model instance.
        /// </remarks>
        public void LoadAndProcessData(string csvFilePath)
        {
            // Read the CSV file and split into lines for processing
            string[] lines = File.ReadAllLines(csvFilePath);

            // Parse the header row to get column names and their positions
            string[] headers = lines[0].Split(',');

            // Process each data row (skipping the header row)
            for (int i = 1; i < lines.Length; i++)
            {
                // Split the current line into individual values
                string[] values = lines[i].Split(',');

                // Create a new borrower model instance and populate it with parsed data
                // Note: Using Array.IndexOf to dynamically map CSV columns to model properties
                BorrowerProfile borrower = new BorrowerProfile
                {
                    // Core financial metrics
                    CreditScore = int.Parse(values[Array.IndexOf(headers, "CreditScore")]),
                    AnnualIncome = int.Parse(values[Array.IndexOf(headers, "AnnualIncome")]),
                    LoanAmount = int.Parse(values[Array.IndexOf(headers, "LoanAmount")]),
                    LoanDuration = int.Parse(values[Array.IndexOf(headers, "LoanDuration")]),

                    // Personal information
                    Age = int.Parse(values[Array.IndexOf(headers, "Age")]),
                    EmploymentStatus = values[Array.IndexOf(headers, "EmploymentStatus")],
                    MaritalStatus = values[Array.IndexOf(headers, "MaritalStatus")],
                    NumberOfDependents = int.Parse(values[Array.IndexOf(headers, "NumberOfDependents")]),
                    EducationLevel = values[Array.IndexOf(headers, "EducationLevel")],

                    // Housing and property details
                    HomeOwnershipStatus = values[Array.IndexOf(headers, "HomeOwnershipStatus")],
                    MonthlyDebtPayments = int.Parse(values[Array.IndexOf(headers, "MonthlyDebtPayments")]),
                    MortgageBalance = int.Parse(values[Array.IndexOf(headers, "MortgageBalance")]),
                    RentPayments = int.Parse(values[Array.IndexOf(headers, "RentPayments")]),

                    // Credit and debt information
                    CreditCardUtilizationRate = double.Parse(values[Array.IndexOf(headers, "CreditCardUtilizationRate")]),
                    NumberOfOpenCreditLines = int.Parse(values[Array.IndexOf(headers, "NumberOfOpenCreditLines")]),
                    NumberOfCreditInquiries = int.Parse(values[Array.IndexOf(headers, "NumberOfCreditInquiries")]),
                    DebtToIncomeRatio = double.Parse(values[Array.IndexOf(headers, "DebtToIncomeRatio")]),
                    BankruptcyHistory = int.Parse(values[Array.IndexOf(headers, "BankruptcyHistory")]),
                    PreviousLoanDefaults = int.Parse(values[Array.IndexOf(headers, "PreviousLoanDefaults")]),

                    // Loan details
                    LoanPurpose = values[Array.IndexOf(headers, "LoanPurpose")],
                    InterestRate = double.Parse(values[Array.IndexOf(headers, "InterestRate")]),
                    PaymentHistory = int.Parse(values[Array.IndexOf(headers, "PaymentHistory")]),

                    // Asset and account balances
                    SavingsAccountBalance = int.Parse(values[Array.IndexOf(headers, "SavingsAccountBalance")]),
                    CheckingAccountBalance = int.Parse(values[Array.IndexOf(headers, "CheckingAccountBalance")]),
                    InvestmentAccountBalance = int.Parse(values[Array.IndexOf(headers, "InvestmentAccountBalance")]),
                    RetirementAccountBalance = int.Parse(values[Array.IndexOf(headers, "RetirementAccountBalance")]),
                    EmergencyFundBalance = int.Parse(values[Array.IndexOf(headers, "EmergencyFundBalance")]),
                    TotalAssets = int.Parse(values[Array.IndexOf(headers, "TotalAssets")]),
                    TotalLiabilities = int.Parse(values[Array.IndexOf(headers, "TotalLiabilities")]),
                    NetWorth = int.Parse(values[Array.IndexOf(headers, "NetWorth")]),

                    // Credit history
                    LengthOfCreditHistory = int.Parse(values[Array.IndexOf(headers, "LengthOfCreditHistory")]),

                    // Loan balances
                    AutoLoanBalance = int.Parse(values[Array.IndexOf(headers, "AutoLoanBalance")]),
                    PersonalLoanBalance = int.Parse(values[Array.IndexOf(headers, "PersonalLoanBalance")]),
                    StudentLoanBalance = int.Parse(values[Array.IndexOf(headers, "StudentLoanBalance")]),

                    // Payment history
                    UtilityBillsPaymentHistory = double.Parse(values[Array.IndexOf(headers, "UtilityBillsPaymentHistory")]),

                    // Insurance information
                    HealthInsuranceStatus = values[Array.IndexOf(headers, "HealthInsuranceStatus")],
                    LifeInsuranceStatus = values[Array.IndexOf(headers, "LifeInsuranceStatus")],
                    CarInsuranceStatus = values[Array.IndexOf(headers, "CarInsuranceStatus")],
                    HomeInsuranceStatus = values[Array.IndexOf(headers, "HomeInsuranceStatus")],
                    OtherInsurancePolicies = int.Parse(values[Array.IndexOf(headers, "OtherInsurancePolicies")]),

                    // Employment details
                    EmployerType = values[Array.IndexOf(headers, "EmployerType")],
                    JobTenure = int.Parse(values[Array.IndexOf(headers, "JobTenure")]),

                    // Income and expenses
                    MonthlySavings = int.Parse(values[Array.IndexOf(headers, "MonthlySavings")]),
                    AnnualBonuses = int.Parse(values[Array.IndexOf(headers, "AnnualBonuses")]),
                    AnnualExpenses = int.Parse(values[Array.IndexOf(headers, "AnnualExpenses")]),
                    MonthlyHousingCosts = int.Parse(values[Array.IndexOf(headers, "MonthlyHousingCosts")]),
                    MonthlyTransportationCosts = int.Parse(values[Array.IndexOf(headers, "MonthlyTransportationCosts")]),
                    MonthlyFoodCosts = int.Parse(values[Array.IndexOf(headers, "MonthlyFoodCosts")]),
                    MonthlyHealthcareCosts = int.Parse(values[Array.IndexOf(headers, "MonthlyHealthcareCosts")]),
                    MonthlyEntertainmentCosts = int.Parse(values[Array.IndexOf(headers, "MonthlyEntertainmentCosts")]),

                    // Loan outcome
                    LoanApproved = int.Parse(values[Array.IndexOf(headers, "LoanApproved")])
                };

                // Add the processed borrower to the training dataset
                trainingData.Add(borrower);
            }

            // Process the loaded data:
            // 1. Extract unique values for categorical variables
            this.CollectCategoricalValues();

            // 2. Calculate statistical measures for continuous variables
            this.CalculateContinuousStatistics();

            // Output processing summary
            Console.WriteLine($"Loaded {trainingData.Count} borrower records");
            this.PrintCategoricalValuesSummary();
            this.PrintContinuousStatisticsSummary();
        }

        /// <summary>
        /// Collects unique categorical values from the training data for each field.
        /// </summary>
        /// <remarks>
        /// This method extracts unique values from the training data for each categorical field,
        /// which will be used to create discrete probability distributions and encode categorical variables in the Bayesian network.
        /// </remarks>
        private void CollectCategoricalValues()
        {
            // Extract unique categorical values from training data for each field
            // These values will be used to create discrete probability distributions
            // and for encoding categorical variables in the Bayesian network

            // Employment status categories (e.g., Full-time, Part-time, Self-employed)
            employmentStatusValues = trainingData.Select(b => b.EmploymentStatus).Distinct().ToList();

            // Marital status categories (e.g., Single, Married, Divorced)
            maritalStatusValues = trainingData.Select(b => b.MaritalStatus).Distinct().ToList();

            // Education level categories (e.g., High School, Bachelor's, Master's)
            educationLevelValues = trainingData.Select(b => b.EducationLevel).Distinct().ToList();

            // Home ownership status categories (e.g., Own, Rent, Mortgage)
            homeOwnershipStatusValues = trainingData.Select(b => b.HomeOwnershipStatus).Distinct().ToList();

            // Loan purpose categories (e.g., Home Purchase, Debt Consolidation, Education)
            loanPurposeValues = trainingData.Select(b => b.LoanPurpose).Distinct().ToList();

            // Insurance status categories for different types of insurance
            // Each status typically has values like "Yes", "No", or specific coverage levels
            healthInsuranceStatusValues = trainingData.Select(b => b.HealthInsuranceStatus).Distinct().ToList();
            lifeInsuranceStatusValues = trainingData.Select(b => b.LifeInsuranceStatus).Distinct().ToList();
            carInsuranceStatusValues = trainingData.Select(b => b.CarInsuranceStatus).Distinct().ToList();
            homeInsuranceStatusValues = trainingData.Select(b => b.HomeInsuranceStatus).Distinct().ToList();

            // Employer type categories (e.g., Private, Public, Non-profit)
            employerTypeValues = trainingData.Select(b => b.EmployerType).Distinct().ToList();
        }

        /// <summary>
        /// Calculates statistical measures for continuous variables in the training data.
        /// </summary>
        /// <remarks>
        /// This method computes basic statistical measures (min, max, mean, standard deviation)
        /// for each continuous variable in the training data. These statistics are used to normalize
        /// data and inform the Bayesian network's probability distributions.
        /// </remarks>
        private void CalculateContinuousStatistics()
        {
            // Calculate statistical measures (min, max, mean, standard deviation) for each continuous variable
            // These statistics will be used to normalize data and inform the Bayesian network's probability distributions

            // Core financial metrics
            this.CalculateStats("CreditScore", b => b.CreditScore);                    // FICO or similar credit score
            this.CalculateStats("AnnualIncome", b => b.AnnualIncome);                  // Gross annual income
            this.CalculateStats("LoanAmount", b => b.LoanAmount);                      // Requested loan amount
            this.CalculateStats("LoanDuration", b => b.LoanDuration);                  // Loan term in months
            this.CalculateStats("Age", b => b.Age);                                    // Borrower's age
            this.CalculateStats("NumberOfDependents", b => b.NumberOfDependents);      // Number of dependents

            // Debt and credit metrics
            this.CalculateStats("MonthlyDebtPayments", b => b.MonthlyDebtPayments);    // Total monthly debt obligations
            this.CalculateStats("CreditCardUtilizationRate", b => b.CreditCardUtilizationRate);  // Credit card usage ratio
            this.CalculateStats("NumberOfOpenCreditLines", b => b.NumberOfOpenCreditLines);      // Active credit accounts
            this.CalculateStats("NumberOfCreditInquiries", b => b.NumberOfCreditInquiries);      // Recent credit applications
            this.CalculateStats("DebtToIncomeRatio", b => b.DebtToIncomeRatio);        // DTI ratio (monthly debt/income)
            this.CalculateStats("BankruptcyHistory", b => b.BankruptcyHistory);        // Years since bankruptcy
            this.CalculateStats("PreviousLoanDefaults", b => b.PreviousLoanDefaults);  // Number of past defaults
            this.CalculateStats("InterestRate", b => b.InterestRate);                  // Current loan interest rate
            this.CalculateStats("PaymentHistory", b => b.PaymentHistory);              // Payment performance score
            this.CalculateStats("LengthOfCreditHistory", b => b.LengthOfCreditHistory); // Years of credit history

            // Asset and account balances
            this.CalculateStats("SavingsAccountBalance", b => b.SavingsAccountBalance);          // Liquid savings
            this.CalculateStats("CheckingAccountBalance", b => b.CheckingAccountBalance);        // Checking account balance
            this.CalculateStats("InvestmentAccountBalance", b => b.InvestmentAccountBalance);    // Investment portfolio value
            this.CalculateStats("RetirementAccountBalance", b => b.RetirementAccountBalance);    // Retirement savings
            this.CalculateStats("EmergencyFundBalance", b => b.EmergencyFundBalance);            // Emergency savings
            this.CalculateStats("TotalAssets", b => b.TotalAssets);                              // Total asset value
            this.CalculateStats("TotalLiabilities", b => b.TotalLiabilities);                    // Total debt obligations
            this.CalculateStats("NetWorth", b => b.NetWorth);                                    // Assets minus liabilities

            // Specific loan balances
            this.CalculateStats("MortgageBalance", b => b.MortgageBalance);              // Outstanding mortgage amount
            this.CalculateStats("RentPayments", b => b.RentPayments);                    // Monthly rent if applicable
            this.CalculateStats("AutoLoanBalance", b => b.AutoLoanBalance);              // Outstanding auto loan amount
            this.CalculateStats("PersonalLoanBalance", b => b.PersonalLoanBalance);      // Outstanding personal loan amount
            this.CalculateStats("StudentLoanBalance", b => b.StudentLoanBalance);        // Outstanding student loan amount

            // Employment and income details
            this.CalculateStats("JobTenure", b => b.JobTenure);                          // Years at current job
            this.CalculateStats("MonthlySavings", b => b.MonthlySavings);                // Regular monthly savings
            this.CalculateStats("AnnualBonuses", b => b.AnnualBonuses);                  // Annual bonus income
            this.CalculateStats("AnnualExpenses", b => b.AnnualExpenses);                // Total annual expenses

            // Monthly living expenses
            this.CalculateStats("MonthlyHousingCosts", b => b.MonthlyHousingCosts);      // Housing-related expenses
            this.CalculateStats("MonthlyTransportationCosts", b => b.MonthlyTransportationCosts);  // Transportation expenses
            this.CalculateStats("MonthlyFoodCosts", b => b.MonthlyFoodCosts);            // Food and grocery expenses
            this.CalculateStats("MonthlyHealthcareCosts", b => b.MonthlyHealthcareCosts);  // Healthcare expenses
            this.CalculateStats("MonthlyEntertainmentCosts", b => b.MonthlyEntertainmentCosts);  // Entertainment expenses

            // Insurance metrics
            this.CalculateStats("OtherInsurancePolicies", b => b.OtherInsurancePolicies);  // Number of additional insurance policies
        }

        /// <summary>
        /// Calculates basic descriptive statistics for a continuous variable in the training data.
        /// </summary>
        /// <typeparam name="T">The type of the variable to analyze.</typeparam>
        /// <param name="variableName">The name of the variable to analyze.</param>
        /// <param name="selector">A function that selects the value of the variable from a borrower model.</param>
        /// <remarks>
        /// This method converts all values to doubles for statistical analysis,
        /// calculates basic descriptive statistics (min, max, mean, standard deviation),
        /// and stores the results in the continuousStats dictionary.
        /// </remarks>
        private void CalculateStats<T>(string variableName, Func<BorrowerProfile, T> selector) where T : IConvertible
        {
            // Convert all values to doubles for statistical analysis
            List<double> values = trainingData.Select(b => Convert.ToDouble(selector(b))).ToList();

            // Calculate basic descriptive statistics
            double min = values.Min();        // Minimum value in the dataset
            double max = values.Max();        // Maximum value in the dataset
            double mean = values.Average();   // Arithmetic mean (average) of all values

            // Calculate variance and standard deviation for dispersion measures
            double variance = values.Select(v => Math.Pow(v - mean, 2)).Sum() / values.Count;  // Average squared deviation from mean
            double stdDev = Math.Sqrt(variance);  // Square root of variance, measures spread of data

            // Store the calculated statistics in the continuousStats dictionary
            continuousStats[variableName] = (min, max, mean, stdDev);
        }

        /// <summary>
        /// Prints a summary of categorical values from the training data.
        /// </summary>
        /// <remarks>
        /// This method formats and displays a summary of categorical values from the training data,
        /// using a template string with Handlebars-style placeholders for dynamic content.
        /// </remarks>
        private void PrintCategoricalValuesSummary()
        {
            // Define a template string using C# 11 raw string literals for better readability
            // The template uses Handlebars-style placeholders ({{variableName}}) for dynamic content
            string template = """
                Categorical Variables Summary:
                Employment Status: {{employmentStatusValues}}
                Marital Status: {{maritalStatusValues}}
                Education Level: {{educationLevelValues}}
                Home Ownership Status: {{homeOwnershipStatusValues}}
                Loan Purpose: {{loanPurposeValues}}
                Health Insurance Status: {{healthInsuranceStatusValues}}
                Life Insurance Status: {{lifeInsuranceStatusValues}}
                Car Insurance Status: {{carInsuranceStatusValues}}
                Home Insurance Status: {{homeInsuranceStatusValues}}
                Employer Type: {{employerTypeValues}}
                """;

            // Create an anonymous object containing all categorical values
            // Each property joins the corresponding list of values into a comma-separated string
            var data = new
            {
                employmentStatusValues = string.Join(", ", employmentStatusValues),
                maritalStatusValues = string.Join(", ", maritalStatusValues),
                educationLevelValues = string.Join(", ", educationLevelValues),
                homeOwnershipStatusValues = string.Join(", ", homeOwnershipStatusValues),
                loanPurposeValues = string.Join(", ", loanPurposeValues),
                healthInsuranceStatusValues = string.Join(", ", healthInsuranceStatusValues),
                lifeInsuranceStatusValues = string.Join(", ", lifeInsuranceStatusValues),
                carInsuranceStatusValues = string.Join(", ", carInsuranceStatusValues),
                homeInsuranceStatusValues = string.Join(", ", homeInsuranceStatusValues),
                employerTypeValues = string.Join(", ", employerTypeValues)
            };

            // Render the template with the data and write to console
            // HandlebarsUtility.RenderTemplate replaces placeholders with actual values
            Console.WriteLine(HandlebarsUtility.RenderTemplate(template, data));
        }

        /// <summary>
        /// Prints a summary of continuous variable statistics from the training data.
        /// </summary>
        /// <remarks>
        /// This method formats and displays a summary of continuous variable statistics from the training data,
        /// using a template string with Handlebars-style placeholders for dynamic content.
        /// </remarks>
        private void PrintContinuousStatisticsSummary()
        {
            // Print a header for the continuous variables statistics section
            Console.WriteLine("\nContinuous Variables Statistics:");

            // Iterate through each continuous variable's statistics stored in the dictionary
            // Each entry contains min, max, mean, and standard deviation values
            foreach (KeyValuePair<string, (double Min, double Max, double Mean, double StdDev)> stat in continuousStats)
            {
                // Format and display the statistics for each variable
                // Using F2 format specifier to show 2 decimal places for better readability
                Console.WriteLine($"{stat.Key}: Min={stat.Value.Min:F2}, Max={stat.Value.Max:F2}, " +
                                 $"Mean={stat.Value.Mean:F2}, StdDev={stat.Value.StdDev:F2}");
            }
        }

        // Properties to access the processed data
        /// <summary>
        /// Gets the list of processed borrower models used for training.
        /// </summary>
        public List<BorrowerProfile> TrainingData => trainingData;

        /// <summary>
        /// Gets the statistics (min, max, mean, standard deviation) for continuous variables in the training data.
        /// </summary>
        public Dictionary<string, (double Min, double Max, double Mean, double StdDev)> ContinuousStats => continuousStats;

        /// <summary>
        /// Gets the list of unique employment status values found in the training data.
        /// </summary>
        public List<string> EmploymentStatusValues => employmentStatusValues;

        /// <summary>
        /// Gets the list of unique marital status values found in the training data.
        /// </summary>
        public List<string> MaritalStatusValues => maritalStatusValues;

        /// <summary>
        /// Gets the list of unique education level values found in the training data.
        /// </summary>
        public List<string> EducationLevelValues => educationLevelValues;

        /// <summary>
        /// Gets the list of unique home ownership status values found in the training data.
        /// </summary>
        public List<string> HomeOwnershipStatusValues => homeOwnershipStatusValues;

        /// <summary>
        /// Gets the list of unique loan purpose values found in the training data.
        /// </summary>
        public List<string> LoanPurposeValues => loanPurposeValues;

        /// <summary>
        /// Gets the list of unique health insurance status values found in the training data.
        /// </summary>
        public List<string> HealthInsuranceStatusValues => healthInsuranceStatusValues;

        /// <summary>
        /// Gets the list of unique life insurance status values found in the training data.
        /// </summary>
        public List<string> LifeInsuranceStatusValues => lifeInsuranceStatusValues;

        /// <summary>
        /// Gets the list of unique car insurance status values found in the training data.
        /// </summary>
        public List<string> CarInsuranceStatusValues => carInsuranceStatusValues;

        /// <summary>
        /// Gets the list of unique home insurance status values found in the training data.
        /// </summary>
        public List<string> HomeInsuranceStatusValues => homeInsuranceStatusValues;

        /// <summary>
        /// Gets the list of unique employer type values found in the training data.
        /// </summary>
        public List<string> EmployerTypeValues => employerTypeValues;
    }
}