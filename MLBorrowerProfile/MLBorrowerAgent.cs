using System.ComponentModel;
using System.Text;
using Microsoft.ML.Probabilistic.Distributions.Kernels;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Agents;
using Microsoft.SemanticKernel.Connectors.AzureOpenAI;
using Microsoft.SemanticKernel.Connectors.OpenAI;
using Microsoft.SemanticKernel.PromptTemplates.Liquid;
using Microsoft.SemanticKernel.Prompty;
using MLBorrowerProfile.Builders;
using MLBorrowerProfile.DataModels;
using MLBorrowerProfile.Extensions;
using MLBorrowerProfile.Networks;
using MLBorrowerProfile.Utils;

namespace MLBorrowerProfile
{
    /// <summary>
    /// Represents a borrower agent that uses a Bayesian network to predict loan approval probabilities.
    /// </summary>
    /// <remarks>
    /// This class contains two main plugins:
    /// 1. LoanApprovalPredictorPlugin: Predicts loan approval probabilities based on borrower profile.
    /// 2. ProfileImprovementAnalyzerPlugin: Analyzes borrower profile and suggests improvements to increase approval odds.
    /// </remarks>
    public class MLBorrowerAgent
    {

        /// <summary>
        /// Represents a plugin for predicting loan approval probabilities using a Bayesian network.
        /// </summary>
        /// <remarks>
        /// This class contains a method for predicting loan approval probabilities based on a borrower profile.
        /// It uses a loan approval Bayesian network to calculate the probability of loan approval.
        /// </remarks>
        public class LoanApprovalPredictorPlugin
        {
            /// <summary>
            /// The loan approval Bayesian network.
            /// </summary>
            private LoanApprovalBayesianNetwork _approvalNetwork;

            /// <summary>
            /// The borrower model builder.
            /// </summary>
            private BorrowerModelBuilder _modelBuilder;

            /// <summary>
            /// Initializes a new instance of the LoanApprovalPredictorPlugin class.
            /// </summary>
            /// <param name="approvalNetwork">The loan approval Bayesian network.</param>
            /// <param name="modelBuilder">The borrower model builder.</param>
            /// <remarks>
            /// This constructor initializes the plugin with the provided loan approval Bayesian network and borrower model builder.
            /// </remarks>
            public LoanApprovalPredictorPlugin(LoanApprovalBayesianNetwork approvalNetwork,
                                             BorrowerModelBuilder modelBuilder)
            {
                _approvalNetwork = approvalNetwork;
                _modelBuilder = modelBuilder;
            }

            /// <summary>
            /// Predicts whether a loan application would be approved and provides detailed analysis.
            /// </summary>
            /// <param name="creditScore">The credit score of the borrower.</param>
            /// <param name="annualIncome">The annual income of the borrower.</param>
            /// <param name="loanAmount">The amount of the loan requested.</param>
            /// <param name="loanDuration">The duration of the loan in years.</param>
            /// <param name="employmentStatus">The employment status of the borrower.</param>
            /// <param name="maritalStatus">The marital status of the borrower.</param>
            /// <param name="numberOfDependents">The number of dependents of the borrower.</param>
            /// <param name="educationLevel">The education level of the borrower.</param>
            /// <param name="homeOwnershipStatus">The home ownership status of the borrower.</param>
            /// <param name="debtToIncomeRatio">The debt-to-income ratio of the borrower.</param>
            /// <param name="monthlyDebtPayments">The monthly debt payments of the borrower.</param>
            /// <param name="creditCardUtilizationRate">The credit card utilization rate of the borrower.</param>
            /// <param name="numberOfOpenCreditLines">The number of open credit lines of the borrower.</param>
            /// <param name="numberOfCreditInquiries">The number of credit inquiries of the borrower.</param>
            /// <param name="previousLoanDefaults">The number of previous loan defaults of the borrower.</param>
            /// <param name="bankruptcyHistory">The number of bankruptcy history of the borrower.</param>
            /// <param name="loanPurpose">The purpose of the loan.</param>
            /// <param name="interestRate">The interest rate of the loan.</param>
            /// <param name="paymentHistory">The payment history of the borrower.</param>
            /// <param name="savingsAccountBalance">The balance of the borrower's savings account.</param>
            /// <param name="checkingAccountBalance">The balance of the borrower's checking account.</param>
            /// <param name="investmentAccountBalance">The balance of the borrower's investment account.</param>
            /// <param name="retirementAccountBalance">The balance of the borrower's retirement account.</param>
            /// <param name="emergencyFundBalance">The balance of the borrower's emergency fund.</param>
            /// <param name="totalAssets">The total assets of the borrower.</param>
            /// <param name="totalLiabilities">The total liabilities of the borrower.</param>
            /// <param name="lengthOfCreditHistory">The length of the borrower's credit history.</param>
            /// <param name="mortgageBalance">The balance of the borrower's mortgage.</param>
            /// <param name="rentPayments">The amount of the borrower's rent payments.</param>
            /// <param name="autoLoanBalance">The balance of the borrower's auto loan.</param>
            /// <param name="personalLoanBalance">The balance of the borrower's personal loan.</param>
            /// <param name="studentLoanBalance">The balance of the borrower's student loan.</param>
            /// <param name="utilityBillsPaymentHistory">The payment history of the borrower's utility bills.</param>
            /// <param name="healthInsuranceStatus">The status of the borrower's health insurance.</param>
            /// <param name="lifeInsuranceStatus">The status of the borrower's life insurance.</param>
            /// <param name="carInsuranceStatus">The status of the borrower's car insurance.</param>
            /// <param name="homeInsuranceStatus">The status of the borrower's home insurance.</param>
            /// <param name="otherInsurancePolicies">The number of other insurance policies the borrower has.</param>
            /// <param name="employerType">The type of employer of the borrower.</param>
            /// <param name="jobTenure">The tenure of the borrower's job.</param>
            /// <param name="monthlySavings">The monthly savings of the borrower.</param>
            /// <param name="annualBonuses">The annual bonuses of the borrower.</param>
            /// <param name="age">The age of the borrower.</param>
            /// <returns>A detailed analysis of the loan approval probability and recommendations for improvement.</returns>
            /// <remarks>
            /// This method uses the loan approval Bayesian network to predict the probability of loan approval based on the provided borrower profile.
            /// It then generates a detailed report with the results and recommendations for improving the borrower's chances of approval.
            /// </remarks>
            [KernelFunction("PredictLoanApproval")]
            [Description("Predicts whether a loan application would be approved and provides detailed analysis.")]
            public async Task<string> PredictLoanApproval(
                double creditScore,
                double annualIncome,
                double loanAmount,
                int loanDuration = 30,
                string employmentStatus = "Employed",
                string maritalStatus = "Single",
                int numberOfDependents = 0,
                string educationLevel = "Bachelor",
                string homeOwnershipStatus = "Mortgage",
                double debtToIncomeRatio = 0.36,
                int monthlyDebtPayments = 0,
                double creditCardUtilizationRate = 0.3,
                int numberOfOpenCreditLines = 3,
                int numberOfCreditInquiries = 1,
                int previousLoanDefaults = 0,
                int bankruptcyHistory = 0,
                string loanPurpose = "Home",
                double interestRate = 0.04,
                int paymentHistory = 95,
                int savingsAccountBalance = 10000,
                int checkingAccountBalance = 5000,
                int investmentAccountBalance = 20000,
                int retirementAccountBalance = 50000,
                int emergencyFundBalance = 15000,
                int totalAssets = 0,
                int totalLiabilities = 0,
                int lengthOfCreditHistory = 8,
                int mortgageBalance = 0,
                int rentPayments = 0,
                int autoLoanBalance = 0,
                int personalLoanBalance = 0,
                int studentLoanBalance = 0,
                double utilityBillsPaymentHistory = 0.95,
                string healthInsuranceStatus = "Insured",
                string lifeInsuranceStatus = "Insured",
                string carInsuranceStatus = "Insured",
                string homeInsuranceStatus = "Insured",
                int otherInsurancePolicies = 0,
                string employerType = "Private",
                int jobTenure = 5,
                int monthlySavings = 1000,
                int annualBonuses = 5000,
                int age = 35)
            {
                // Calculate derived values
                if (monthlyDebtPayments == 0)
                {
                    monthlyDebtPayments = (int)(annualIncome / 12 * debtToIncomeRatio);
                }

                if (totalAssets == 0)
                {
                    totalAssets = savingsAccountBalance +
                                 checkingAccountBalance +
                                 investmentAccountBalance +
                                 retirementAccountBalance +
                                 emergencyFundBalance;

                    // Add home value if owned
                    if (homeOwnershipStatus == "Own" || homeOwnershipStatus == "Mortgage")
                    {
                        totalAssets += (int)(annualIncome * 4); // Rough estimate of home value
                    }
                }

                if (totalLiabilities == 0)
                {
                    totalLiabilities = mortgageBalance +
                                      autoLoanBalance +
                                      personalLoanBalance +
                                      studentLoanBalance;
                }

                int netWorth = totalAssets - totalLiabilities;

                // Calculate monthly expenses from annual income and debt-to-income ratio
                int monthlyIncome = (int)(annualIncome / 12);
                int annualExpenses = (int)(annualIncome * 0.7); // Estimate

                int monthlyHousingCosts = rentPayments;
                if (homeOwnershipStatus == "Mortgage" && mortgageBalance > 0)
                {
                    // Estimate monthly mortgage payment if not provided
                    monthlyHousingCosts = (int)(mortgageBalance * interestRate / 12 *
                                         Math.Pow(1 + interestRate / 12, loanDuration * 12) /
                                         (Math.Pow(1 + interestRate / 12, loanDuration * 12) - 1));
                }

                int monthlyTransportationCosts = (int)(monthlyIncome * 0.1); // Estimate
                int monthlyFoodCosts = (int)(monthlyIncome * 0.15); // Estimate
                int monthlyHealthcareCosts = (int)(monthlyIncome * 0.05); // Estimate
                int monthlyEntertainmentCosts = (int)(monthlyIncome * 0.1); // Estimate

                // Create a complete borrower profile from the inputs
                BorrowerProfile profile = new BorrowerProfile
                {
                    CreditScore = creditScore,
                    AnnualIncome = annualIncome,
                    LoanAmount = loanAmount,
                    LoanDuration = loanDuration,
                    Age = age,
                    EmploymentStatus = employmentStatus,
                    MaritalStatus = maritalStatus,
                    NumberOfDependents = numberOfDependents,
                    EducationLevel = educationLevel,
                    HomeOwnershipStatus = homeOwnershipStatus,
                    MonthlyDebtPayments = monthlyDebtPayments,
                    CreditCardUtilizationRate = creditCardUtilizationRate,
                    NumberOfOpenCreditLines = numberOfOpenCreditLines,
                    NumberOfCreditInquiries = numberOfCreditInquiries,
                    DebtToIncomeRatio = debtToIncomeRatio,
                    BankruptcyHistory = bankruptcyHistory,
                    LoanPurpose = loanPurpose,
                    PreviousLoanDefaults = previousLoanDefaults,
                    InterestRate = interestRate,
                    PaymentHistory = paymentHistory,
                    SavingsAccountBalance = savingsAccountBalance,
                    CheckingAccountBalance = checkingAccountBalance,
                    InvestmentAccountBalance = investmentAccountBalance,
                    RetirementAccountBalance = retirementAccountBalance,
                    EmergencyFundBalance = emergencyFundBalance,
                    TotalAssets = totalAssets,
                    TotalLiabilities = totalLiabilities,
                    NetWorth = netWorth,
                    LengthOfCreditHistory = lengthOfCreditHistory,
                    MortgageBalance = mortgageBalance,
                    RentPayments = rentPayments,
                    AutoLoanBalance = autoLoanBalance,
                    PersonalLoanBalance = personalLoanBalance,
                    StudentLoanBalance = studentLoanBalance,
                    UtilityBillsPaymentHistory = utilityBillsPaymentHistory,
                    HealthInsuranceStatus = healthInsuranceStatus,
                    LifeInsuranceStatus = lifeInsuranceStatus,
                    CarInsuranceStatus = carInsuranceStatus,
                    HomeInsuranceStatus = homeInsuranceStatus,
                    OtherInsurancePolicies = otherInsurancePolicies,
                    EmployerType = employerType,
                    JobTenure = jobTenure,
                    MonthlySavings = monthlySavings,
                    AnnualBonuses = annualBonuses,
                    AnnualExpenses = annualExpenses,
                    MonthlyHousingCosts = monthlyHousingCosts,
                    MonthlyTransportationCosts = monthlyTransportationCosts,
                    MonthlyFoodCosts = monthlyFoodCosts,
                    MonthlyHealthcareCosts = monthlyHealthcareCosts,
                    MonthlyEntertainmentCosts = monthlyEntertainmentCosts
                };

                // Calculate approval probability
                double approvalProbability = _approvalNetwork.PredictApprovalProbability(profile);

                // Generate detailed report
                string report = _approvalNetwork.GenerateApprovalReport(profile);

                return report;
            }

            /// <summary>
            /// Analyzes a borrower profile and suggests improvements to increase approval odds.
            /// </summary>
            /// <param name="borrowerProfile">The borrower profile to analyze.</param>
            /// <returns>A detailed analysis of the borrower profile and suggestions for improvement.</returns>
            /// <remarks>
            /// This method uses the loan approval Bayesian network to analyze the borrower profile and suggest improvements to increase approval odds.
            /// It generates a detailed report with the results and suggestions for improving the borrower's chances of approval.
            /// </remarks>
            [KernelFunction("AnalyzeProfileImprovements")]
            [Description("Analyzes a borrower profile and suggests improvements to increase approval odds.")]
            public async Task<string> AnalyzeProfileImprovements(
                BorrowerProfile borrowerProfile)
            {
                // Generate feature importance analysis
                FeatureImportance importance = _approvalNetwork.AnalyzeFeatureImportance(borrowerProfile);

                // Format results as a string
                StringBuilder result = new StringBuilder();
                result.AppendLine("# Profile Improvement Analysis");
                result.AppendLine();
                result.AppendLine($"Current approval probability: {importance.BaseApprovalProbability:P2}");
                result.AppendLine();
                result.AppendLine("## Suggested Improvements");

                foreach (string recommendation in importance.Recommendations)
                {
                    result.AppendLine($"- {recommendation}");
                }

                return result.ToString();
            }

            //     /// <summary>
            //     /// Compares two borrower profiles and explains differences in approval probability.
            //     /// </summary>
            //     /// <param name="profile1">The first borrower profile to compare.</param>
            //     /// <param name="profile2">The second borrower profile to compare.</param>
            //     /// <returns>A detailed comparison of the two profiles and their approval probabilities.</returns>
            //     /// <remarks>
            //     /// This method compares two borrower profiles and explains differences in approval probability.
            //     /// It calculates the approval probability for each profile and compares the results.
            //     /// It then generates a detailed comparison of the two profiles and their approval probabilities.
            //     /// </remarks>
            //     [KernelFunction("CompareProfiles")]
            //     [Description("Compares two borrower profiles and explains differences in approval probability.")]
            //     public async Task<string> CompareProfiles(
            //         Model profile1,
            //         Model profile2)
            //     {
            //         double prob1 = _approvalNetwork.PredictApprovalProbability(profile1);
            //         double prob2 = _approvalNetwork.PredictApprovalProbability(profile2);

            //         StringBuilder comparison = new StringBuilder();
            //         comparison.AppendLine("# Borrower Profile Comparison");
            //         comparison.AppendLine();
            //         comparison.AppendLine($"Profile 1 Approval Probability: {prob1:P2}");
            //         comparison.AppendLine($"Profile 2 Approval Probability: {prob2:P2}");
            //         comparison.AppendLine($"Difference: {Math.Abs(prob1 - prob2):P2} in favor of {(prob1 > prob2 ? "Profile 1" : "Profile 2")}");
            //         comparison.AppendLine();

            //         // Compare key metrics
            //         comparison.AppendLine("## Key Differences");
            //         comparison.AppendLine("Factor | Profile 1 | Profile 2 | Impact");
            //         comparison.AppendLine("--- | --- | --- | ---");

            //         // Credit Score
            //         int creditScoreDiff = (int)Math.Round(profile1.CreditScore - profile2.CreditScore);
            //         comparison.AppendLine($"Credit Score | {profile1.CreditScore:F0} | {profile2.CreditScore:F0} | {creditScoreDiff:+#;-#;0}");

            //         // DTI
            //         double dtiDiff = profile1.DebtToIncomeRatio - profile2.DebtToIncomeRatio;
            //         comparison.AppendLine($"Debt-to-Income | {profile1.DebtToIncomeRatio:P2} | {profile2.DebtToIncomeRatio:P2} | {dtiDiff:+0.00%;-0.00%;0.00%}");

            //         // Loan-to-Income
            //         double lti1 = profile1.LoanAmount / profile1.AnnualIncome;
            //         double lti2 = profile2.LoanAmount / profile2.AnnualIncome;
            //         double ltiDiff = lti1 - lti2;
            //         comparison.AppendLine($"Loan-to-Income | {lti1:F2}x | {lti2:F2}x | {ltiDiff:+0.00;-0.00;0.00}x");

            //         // Add more key metrics as needed

            //         return comparison.ToString();
            //     }
        }

        /// <summary>
        /// Predicts the approval probability of a borrower profile.
        /// </summary>
        /// <param name="borrowerProfile">The borrower profile to predict the approval probability for.</param>
        /// <returns>The predicted approval probability of the borrower profile.</returns>
        /// <remarks>
        /// This method predicts the approval probability of a borrower profile using the loan approval Bayesian network.
        /// It uses the network to calculate the probability of loan approval based on the provided borrower profile.
        /// </remarks>
        public class PredictorPlugin
        {
            /// <summary>
            /// The borrower model builder.
            /// </summary>
            private BorrowerModelBuilder _modelBuilder;

            /// <summary>
            /// The borrower Bayesian network.
            /// </summary>
            private BorrowerBayesianNetwork _bayesianNetwork;

            /// <summary>
            /// Initializes a new instance of the PredictorPlugin class.
            /// </summary>
            /// <param name="modelBuilder">The borrower model builder.</param>
            /// <param name="bayesianNetwork">The borrower Bayesian network.</param>
            /// <remarks>
            /// This constructor initializes the plugin with the provided borrower model builder and Bayesian network.
            /// </remarks>
            public PredictorPlugin(BorrowerModelBuilder modelBuilder,
                BorrowerBayesianNetwork bayesianNetwork)
            {
                _modelBuilder = modelBuilder;
                _bayesianNetwork = bayesianNetwork;
            }

            /// <summary>
            /// Generates a random borrower profile (with influences).
            /// </summary>
            /// <param name="maxResults">The maximum number of results to generate.</param>
            /// <param name="maxAnalysisComparers">The maximum number of analysis comparers.</param>
            /// <param name="maxSearchRecords">The maximum number of search records.</param>
            /// <param name="targetCreditScore">The target credit score.</param>
            /// <param name="targetAnnualIncome">The target annual income.</param>
            /// <param name="targetLoanAmount">The target loan amount.</param>
            /// <param name="creditScoreWeight">The credit score weight.</param>
            /// <param name="annualIncomeWeight">The annual income weight.</param>
            /// <param name="loanAmountWeight">The loan amount weight.</param>
            /// <param name="minThreshold">The minimum threshold.</param>
            /// <returns>A random borrower profile (with influences).</returns>
            /// <remarks>
            /// This method generates a random borrower profile (with influences) using the borrower Bayesian network.
            /// It generates a random profile based on the provided parameters and the Bayesian network.
            /// </remarks>
            [KernelFunction("GenerateRandomProfile")]
            [Description("Generate a random borrower profile (with influences).")]
            public async Task<string> GenerateRandomProfile(
                int maxResults = 5,
                int maxAnalysisComparers = 5,
                int maxSearchRecords = 50,
                double? targetCreditScore = null,
                double? targetAnnualIncome = null,
                double? targetLoanAmount = null,
                double creditScoreWeight = 1.0,
                double annualIncomeWeight = 1.0,
                double loanAmountWeight = 1.0,
                double minThreshold = 0.5)
            {
                List<BorrowerProfile> samples = _bayesianNetwork.GenerateSamples(maxSearchRecords, _modelBuilder);

                List<(BorrowerProfile Profile, double SimilarityScore)> similarProfiles = samples.FindSimilarProfiles(
                    targetCreditScore: targetCreditScore ?? new Random().Next((int)samples.Min(s => s.CreditScore), (int)samples.Max(selector: s => s.CreditScore)),
                    targetAnnualIncome: targetAnnualIncome ?? new Random().Next((int)samples.Min(s => s.AnnualIncome), (int)samples.Max(selector: s => s.AnnualIncome)),
                    targetLoanAmount: new Random().Next((int)samples.Min(s => s.LoanAmount), (int)samples.Max(selector: s => s.LoanAmount)),
                    creditScoreWeight: 2.0,  // We care more about credit score
                    annualIncomeWeight: 1.0, // Standard weight for income
                    loanAmountWeight: 1.0,   // Standard weight for loan amount
                    topCount: 5,             // Return top 5 matches
                    minThreshold: 0.7        // Only include profiles with at least 70% similarity
                );

                string similarProfilesTemplate = """
                    ## Borrower Profiles

                    {{#each samples}}

                    ### Borrower {{@index}}

                    - **Credit Score**: {{CreditScore}}
                    - **Annual Income**: {{AnnualIncome}}
                    - **Loan Amount**: {{LoanAmount}}
                    - **Loan Duration**: {{LoanDuration}} years
                    - **Age**: {{Age}}

                    #### Employment & Family

                    - **Employment Status**: {{EmploymentStatus}}
                    - **Marital Status**: {{MaritalStatus}}
                    - **Number of Dependents**: {{NumberOfDependents}}
                    - **Education Level**: {{EducationLevel}}
                    - **Employer Type**: {{EmployerType}}
                    - **Job Tenure**: {{JobTenure}} years

                    #### Housing & Insurance

                    - **Home Ownership Status**: {{HomeOwnershipStatus}}
                    - **Health Insurance**: {{HealthInsuranceStatus}}
                    - **Life Insurance**: {{LifeInsuranceStatus}}
                    - **Car Insurance**: {{CarInsuranceStatus}}
                    - **Home Insurance**: {{HomeInsuranceStatus}}
                    - **Other Insurance Policies**: {{OtherInsurancePolicies}}

                    #### Credit & Debt

                    - **Monthly Debt Payments**: {{MonthlyDebtPayments}}
                    - **Credit Card Utilization Rate**: {{CreditCardUtilizationRate}}
                    - **Number of Open Credit Lines**: {{NumberOfOpenCreditLines}}
                    - **Number of Credit Inquiries**: {{NumberOfCreditInquiries}}
                    - **Debt-to-Income Ratio**: {{DebtToIncomeRatio}}
                    - **Length of Credit History**: {{LengthOfCreditHistory}} years
                    - **Bankruptcy History**: {{BankruptcyHistory}}
                    - **Previous Loan Defaults**: {{PreviousLoanDefaults}}
                    - **Loan Purpose**: {{LoanPurpose}}
                    - **Interest Rate**: {{InterestRate}}%
                    - **Payment History Score**: {{PaymentHistory}}

                    #### Financials

                    - **Savings Account Balance**: ${{SavingsAccountBalance}}
                    - **Checking Account Balance**: ${{CheckingAccountBalance}}
                    - **Investment Account Balance**: ${{InvestmentAccountBalance}}
                    - **Retirement Account Balance**: ${{RetirementAccountBalance}}
                    - **Emergency Fund Balance**: ${{EmergencyFundBalance}}
                    - **Total Assets**: ${{TotalAssets}}
                    - **Total Liabilities**: ${{TotalLiabilities}}
                    - **Net Worth**: ${{NetWorth}}

                    #### Loans & Expenses

                    - **Mortgage Balance**: ${{MortgageBalance}}
                    - **Rent Payments**: ${{RentPayments}}
                    - **Auto Loan Balance**: ${{AutoLoanBalance}}
                    - **Personal Loan Balance**: ${{PersonalLoanBalance}}
                    - **Student Loan Balance**: ${{StudentLoanBalance}}

                    #### Monthly Budget

                    - **Monthly Savings**: ${{MonthlySavings}}
                    - **Annual Bonuses**: ${{AnnualBonuses}}
                    - **Annual Expenses**: ${{AnnualExpenses}}
                    - **Monthly Housing Costs**: ${{MonthlyHousingCosts}}
                    - **Monthly Transportation Costs**: ${{MonthlyTransportationCosts}}
                    - **Monthly Food Costs**: ${{MonthlyFoodCosts}}
                    - **Monthly Healthcare Costs**: ${{MonthlyHealthcareCosts}}
                    - **Monthly Entertainment Costs**: ${{MonthlyEntertainmentCosts}}
                    - **Utility Bills Payment History**: {{UtilityBillsPaymentHistory}}

                    - **Loan Approved**: {{LoanApproved}}
                    
                    -----------------

                    {{/each}}
                    
                    Select the most closely aligned borrower profile.
                    Respond with only a JSON object containing each borrower field.

                        class Model
                        {
                            double CreditScore;
                            double AnnualIncome;
                            double LoanAmount;
                            int LoanDuration;
                            int Age;
                            string EmploymentStatus;
                            string MaritalStatus;
                            int NumberOfDependents;
                            string EducationLevel;
                            string HomeOwnershipStatus;
                            int MonthlyDebtPayments;
                            double CreditCardUtilizationRate;
                            int NumberOfOpenCreditLines;
                            int NumberOfCreditInquiries;
                            double DebtToIncomeRatio;
                            int BankruptcyHistory;
                            string LoanPurpose;
                            int PreviousLoanDefaults;
                            double InterestRate;
                            int PaymentHistory;
                            int SavingsAccountBalance;
                            int CheckingAccountBalance;
                            int InvestmentAccountBalance;
                            int RetirementAccountBalance;
                            int EmergencyFundBalance;
                            int TotalAssets;
                            int TotalLiabilities;
                            int NetWorth;
                            int LengthOfCreditHistory;
                            int MortgageBalance;
                            int RentPayments;
                            int AutoLoanBalance;
                            int PersonalLoanBalance;
                            int StudentLoanBalance;
                            double UtilityBillsPaymentHistory;
                            string HealthInsuranceStatus;
                            string LifeInsuranceStatus;
                            string CarInsuranceStatus;
                            string HomeInsuranceStatus;
                            int OtherInsurancePolicies;
                            string EmployerType;
                            int JobTenure;
                            int MonthlySavings;
                            int AnnualBonuses;
                            int AnnualExpenses;
                            int MonthlyHousingCosts;
                            int MonthlyTransportationCosts;
                            int MonthlyFoodCosts;
                            int MonthlyHealthcareCosts;
                            int MonthlyEntertainmentCosts;
                            int LoanApproved;
                        }
                    """;

                string response = HandlebarsUtility.RenderTemplate(similarProfilesTemplate, new
                {
                    samples = similarProfiles.Select(p => p.Profile).Take(maxAnalysisComparers)
                });

                return response;
            }
        }

        /// <summary>
        /// The kernel.
        /// </summary>
        private Kernel _kernel;

        /// <summary>
        /// The borrower model builder.
        /// </summary>
        private BorrowerModelBuilder _modelBuilder;

        /// <summary>
        /// The borrower Bayesian network.
        /// </summary>
        private BorrowerBayesianNetwork _bayesianNetwork;

        /// <summary>
        /// The agent template.
        /// </summary>
        private PromptTemplateConfig _agentTemplate;

        /// <summary>
        /// The template factory.
        /// </summary>
        private LiquidPromptTemplateFactory _templateFactory;

        /// <summary>
        /// The agent thread.
        /// </summary>
        private ChatHistoryAgentThread _agentThread;

        /// <summary>
        /// The agent.
        /// </summary>
        private ChatCompletionAgent _agent;

        /// <summary>
        /// Initializes a new instance of the MLBorrowerAgent class.
        /// </summary>
        /// <param name="kernel">The kernel.</param>
        /// <param name="modelBuilder">The borrower model builder.</param>
        /// <param name="bayesianNetwork">The borrower Bayesian network.</param>
        /// <param name="approvalNetwork">The loan approval Bayesian network.</param>
        /// <remarks>
        /// This constructor initializes the agent with the provided kernel, borrower model builder, Bayesian network, and loan approval Bayesian network.
        /// It adds the loan approval predictor and borrower profile predictor plugins to the kernel.
        /// It also initializes the agent template and template factory.
        /// </remarks>
        public MLBorrowerAgent(Kernel kernel,
            BorrowerModelBuilder modelBuilder,
            BorrowerBayesianNetwork bayesianNetwork,
            LoanApprovalBayesianNetwork approvalNetwork
            )
        {
            _kernel = kernel.Clone();
            _modelBuilder = modelBuilder;
            _bayesianNetwork = bayesianNetwork;

            _kernel.Plugins.AddFromObject(new LoanApprovalPredictorPlugin(approvalNetwork, modelBuilder));
            _kernel.Plugins.AddFromObject(new PredictorPlugin(_modelBuilder, _bayesianNetwork));

            _agentTemplate = KernelFunctionPrompty.ToPromptTemplateConfig(File.ReadAllText(@".\Templates\Prompts\MLBorrowerProfileAgent.prompty"));

            _templateFactory = new LiquidPromptTemplateFactory()
            {
                AllowDangerouslySetContent = true
            };

            // Chat History Thread - Maintain conversation history
            _agentThread = new ChatHistoryAgentThread();

            _agent = new ChatCompletionAgent(_agentTemplate, _templateFactory)
            {
                Name = "BuyerProfileAnalysisAgent",
                Kernel = _kernel
            };
        }

        /// <summary>
        /// Invokes the agent asynchronously.
        /// </summary>
        /// <param name="instructions">The instructions to invoke the agent with.</param>
        /// <returns>The response from the agent.</returns>
        /// <remarks>
        /// This method invokes the agent asynchronously with the provided instructions.
        /// It uses the Azure OpenAI API to execute the agent's prompt.
        /// </remarks>
        public async Task<string> InvokeAsync(string instructions)
        {

            AzureOpenAIPromptExecutionSettings executionSettings = new AzureOpenAIPromptExecutionSettings()
            {
                Temperature = 0.2,
                ToolCallBehavior = ToolCallBehavior.AutoInvokeKernelFunctions
            };

            // Buyer Profile Invoke Options
            AgentInvokeOptions agentInvocationOptions = new AgentInvokeOptions()
            {
                Kernel = _kernel,
                KernelArguments = new KernelArguments(executionSettings)
                {
                    { "instructions", instructions },
                }
            };

            IAsyncEnumerable<AgentResponseItem<ChatMessageContent>> res = _agent.InvokeAsync(_agentThread, agentInvocationOptions);

            StringBuilder sbBuyerAnalysis = new StringBuilder();

            await foreach (AgentResponseItem<ChatMessageContent> msgBuyerAnalysis in res)
            {
                sbBuyerAnalysis.Append(msgBuyerAnalysis.Message.Content);
            }

            string text = sbBuyerAnalysis.ToString();

            return text;
        }

    }
}