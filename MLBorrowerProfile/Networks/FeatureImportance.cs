namespace MLBorrowerProfile.Networks
{
    using MLBorrowerProfile.DataModels;

    /// <summary>
    /// Represents the importance of features in loan approval.
    /// </summary>
    /// <remarks>
    /// This class provides a comprehensive analysis of the impact of various features on loan approval.
    /// It includes the base approval probability, as well as the impact of each feature on the approval probability.
    /// </remarks>
    public class FeatureImportance
    {
        /// <summary>
        /// The base approval probability for the borrower profile.
        /// </summary>
        public double BaseApprovalProbability { get; set; }

        /// <summary>
        /// The impact of the credit score on the approval probability.
        /// </summary>
        public double CreditScoreImpact { get; set; }

        /// <summary>
        /// The impact of the debt-to-income ratio on the approval probability.
        /// </summary>
        public double DebtToIncomeRatioImpact { get; set; }

        /// <summary>
        /// The impact of the loan-to-income ratio on the approval probability.
        /// </summary>
        public double LoanToIncomeRatioImpact { get; set; }

        /// <summary>
        /// The impact of the default history on the approval probability.
        /// </summary>
        public double DefaultHistoryImpact { get; set; }

        /// <summary>
        /// The impact of the bankruptcy history on the approval probability.
        /// </summary>
        public double BankruptcyHistoryImpact { get; set; }

        /// <summary>
        /// The impact of the employment status on the approval probability.
        /// </summary>
        public double EmploymentStatusImpact { get; set; }

        /// <summary>
        /// The impact of the loan purpose on the approval probability.
        /// </summary>
        public double LoanPurposeImpact { get; set; }

        /// <summary>
        /// The list of recommendations for the borrower profile.
        /// </summary>
        public List<string> Recommendations { get; private set; } = new List<string>();

        /// <summary>
        /// The list of features ranked by their impact on the approval probability.
        /// </summary>
        /// <returns>A list of tuples containing feature names and their corresponding impact scores.</returns>
        /// <remarks>
        /// This method creates a list of tuples containing feature names and their corresponding impact scores.
        /// Each tuple represents a key factor in loan approval decision making.
        /// </remarks>
        public List<(string Feature, double Impact)> RankedFeatures()
        {
            // Create a list of tuples containing feature names and their corresponding impact scores
            // Each tuple represents a key factor in loan approval decision making
            List<(string, double)> features = new List<(string, double)>
            {
                // Credit score is typically the most influential factor in loan decisions
                ("Credit Score", this.CreditScoreImpact),
                // DTI ratio measures borrower's ability to manage monthly payments
                ("Debt-to-Income Ratio", this.DebtToIncomeRatioImpact),
                // Loan-to-income ratio indicates loan affordability
                ("Loan-to-Income Ratio", this.LoanToIncomeRatioImpact),
                // Previous defaults indicate credit risk
                ("Default History", this.DefaultHistoryImpact),
                // Bankruptcy history is a significant negative factor
                ("Bankruptcy History", this.BankruptcyHistoryImpact),
                // Employment status reflects income stability
                ("Employment Status", this.EmploymentStatusImpact),
                // Loan purpose can affect risk assessment
                ("Loan Purpose", this.LoanPurposeImpact)
            };

            // Sort features by absolute impact value in descending order
            // This ensures most influential factors appear first in the ranking
            return features.OrderByDescending(f => Math.Abs(f.Item2)).ToList();
        }

        /// <summary>
        /// Generates personalized recommendations for the borrower profile based on their feature importance.
        /// </summary>
        /// <param name="profile">The borrower profile to generate recommendations for.</param>
        /// <remarks>
        /// This method generates personalized recommendations for the borrower profile based on their feature importance.
        /// It prioritizes recommendations based on the impact of each feature on the approval probability.
        /// </remarks>
        public void GenerateRecommendations(BorrowerProfile profile)
        {
            // Clear any existing recommendations to start fresh
            this.Recommendations.Clear();

            // Get features ranked by their absolute impact on approval probability
            // Higher impact features will be prioritized in recommendations
            List<(string Feature, double Impact)> rankedFeatures = this.RankedFeatures();

            // Generate personalized recommendations based on feature importance
            // Only consider features with meaningful impact (>= 3% effect on approval probability)
            foreach ((string Feature, double Impact) feature in rankedFeatures)
            {
                // Skip features with minimal impact on approval probability
                if (Math.Abs(feature.Impact) < 0.03) continue;

                // Generate specific recommendations based on the feature type
                switch (feature.Feature)
                {
                    case "Credit Score":
                        // Credit score below 680 is considered subprime and needs improvement
                        if (profile.CreditScore < 680)
                            this.Recommendations.Add($"Improve credit score by at least 50 points (current: {profile.CreditScore:F0}) to increase approval odds by {Math.Abs(feature.Impact):P2}");
                        break;

                    case "Debt-to-Income Ratio":
                        // DTI above 36% is considered high risk by most lenders
                        if (profile.DebtToIncomeRatio > 0.36)
                            this.Recommendations.Add($"Reduce debt-to-income ratio from {profile.DebtToIncomeRatio:P2} to below 36% to increase approval odds by {Math.Abs(feature.Impact):P2}");
                        break;

                    case "Loan-to-Income Ratio":
                        // Calculate and check loan-to-income ratio
                        double annualIncome = profile.AnnualIncome;
                        double loanAmount = profile.LoanAmount;
                        double ratio = loanAmount / annualIncome;

                        // Ratio above 3.0x is considered high risk
                        if (ratio > 3.0)
                        {
                            double targetLoan = annualIncome * 3.0;
                            this.Recommendations.Add($"Consider reducing loan amount from ${loanAmount:N0} to ${targetLoan:N0} to keep loan-to-income ratio below 3.0x");
                        }
                        break;

                    case "Employment Status":
                        // Employment stability is a key factor in loan approval
                        if (profile.EmploymentStatus == "Unemployed")
                            this.Recommendations.Add("Securing employment would significantly improve approval odds");
                        else if (profile.JobTenure < 2)
                            this.Recommendations.Add("Maintaining current employment for at least 2 years would improve approval odds");
                        break;
                }
            }

            // Provide fallback recommendations if no specific improvements are needed
            if (this.Recommendations.Count == 0)
            {
                // For strong profiles, provide positive feedback
                if (this.BaseApprovalProbability >= 0.5)
                    this.Recommendations.Add("Your profile has a positive approval outlook. No major changes needed.");
                // For weaker profiles, suggest general improvements
                else
                    this.Recommendations.Add("Consider applying for a smaller loan amount or providing a larger down payment.");
            }
        }
    }
}