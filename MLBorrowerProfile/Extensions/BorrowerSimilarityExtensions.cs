using HandlebarsDotNet;
using MLBorrowerProfile.DataModels;
using MLBorrowerProfile.Utils;

namespace MLBorrowerProfile.Extensions
{
    /// <summary>
    /// Extension methods for finding similar borrower profiles
    /// </summary>
    /// <remarks>
    /// This class contains extension methods for finding similar borrower profiles based on specified criteria.
    /// It provides methods to find the most similar profiles and format them into a readable string.
    /// </remarks>
    public static class BorrowerSimilarityExtensions
    {
        /// <summary>
        /// Finds the most similar borrower profiles based on specified criteria
        /// </summary>
        /// <param name="samples">The list of borrower profiles to search within</param>
        /// <param name="targetCreditScore">The target credit score to match</param>
        /// <param name="targetAnnualIncome">The target annual income to match</param>
        /// <param name="targetLoanAmount">The target loan amount to match</param>
        /// <param name="creditScoreWeight">The weight to assign to credit score similarity (higher = more important)</param>
        /// <param name="annualIncomeWeight">The weight to assign to annual income similarity (higher = more important)</param>
        /// <param name="loanAmountWeight">The weight to assign to loan amount similarity (higher = more important)</param>
        /// <param name="topCount">The number of similar profiles to return</param>
        /// <param name="minThreshold">The minimum similarity threshold (0.0 to 1.0) to include a profile</param>
        /// <returns>A list of the most similar borrower profiles along with their similarity scores</returns>
        /// <remarks>
        /// This method finds the most similar borrower profiles based on specified criteria:
        /// - Credit score
        /// - Annual income
        /// - Loan amount
        /// 
        /// It uses a weighted combination of normalized distances for each attribute to calculate a final similarity score.
        /// The scores are then filtered and sorted to return the top N most similar profiles.
        /// </remarks>        
        public static List<(BorrowerProfile Profile, double SimilarityScore)> FindSimilarProfiles(
            this List<BorrowerProfile> samples,
            double targetCreditScore,
            double targetAnnualIncome,
            double targetLoanAmount,
            double creditScoreWeight = 1.0,
            double annualIncomeWeight = 1.0,
            double loanAmountWeight = 1.0,
            int topCount = 5,
            double minThreshold = 0.5)
        {
            // Input validation to ensure data integrity and prevent runtime errors
            if (samples == null || !samples.Any())
                throw new ArgumentException("Sample list cannot be null or empty", nameof(samples));

            if (creditScoreWeight < 0 || annualIncomeWeight < 0 || loanAmountWeight < 0)
                throw new ArgumentException("Weights cannot be negative");

            if (topCount <= 0)
                throw new ArgumentException("Top count must be greater than zero", nameof(topCount));

            if (minThreshold < 0 || minThreshold > 1)
                throw new ArgumentException("Minimum threshold must be between 0 and 1", nameof(minThreshold));

            // Normalize weights to ensure they sum to 1.0 for proper weighted average calculation
            double totalWeight = creditScoreWeight + annualIncomeWeight + loanAmountWeight;

            if (totalWeight <= 0)
                throw new ArgumentException("At least one weight must be greater than zero");

            creditScoreWeight /= totalWeight;
            annualIncomeWeight /= totalWeight;
            loanAmountWeight /= totalWeight;

            // Calculate min/max ranges for each attribute to enable normalization
            // This allows comparison of values on a consistent 0-1 scale
            double minCreditScore = samples.Min(s => s.CreditScore);
            double maxCreditScore = samples.Max(s => s.CreditScore);
            double creditScoreRange = maxCreditScore - minCreditScore;

            double minAnnualIncome = samples.Min(s => s.AnnualIncome);
            double maxAnnualIncome = samples.Max(s => s.AnnualIncome);
            double annualIncomeRange = maxAnnualIncome - minAnnualIncome;

            double minLoanAmount = samples.Min(s => s.LoanAmount);
            double maxLoanAmount = samples.Max(s => s.LoanAmount);
            double loanAmountRange = maxLoanAmount - minLoanAmount;

            // Calculate similarity scores for each borrower profile
            // This uses a weighted combination of normalized distances for each attribute
            List<(BorrowerProfile Profile, double SimilarityScore)> scoredSamples = samples.Select(sample =>
            {
                // Calculate normalized distances (0 to 1) for each attribute
                // A distance of 0 means exact match, 1 means maximum difference
                double creditScoreDistance = creditScoreRange <= 0 ? 0 :
                    Math.Abs(sample.CreditScore - targetCreditScore) / creditScoreRange;

                double annualIncomeDistance = annualIncomeRange <= 0 ? 0 :
                    Math.Abs(sample.AnnualIncome - targetAnnualIncome) / annualIncomeRange;

                double loanAmountDistance = loanAmountRange <= 0 ? 0 :
                    Math.Abs(sample.LoanAmount - targetLoanAmount) / loanAmountRange;

                // Convert distances to similarities (1 - distance)
                // A similarity of 1 means exact match, 0 means maximum difference
                double creditScoreSimilarity = 1 - creditScoreDistance;
                double annualIncomeSimilarity = 1 - annualIncomeDistance;
                double loanAmountSimilarity = 1 - loanAmountDistance;

                // Calculate final similarity score using weighted average
                // Higher weights mean the attribute has more influence on the final score
                double overallSimilarity =
                    (creditScoreSimilarity * creditScoreWeight) +
                    (annualIncomeSimilarity * annualIncomeWeight) +
                    (loanAmountSimilarity * loanAmountWeight);

                return (Profile: sample, SimilarityScore: overallSimilarity);
            })
            // Filter out profiles below the minimum similarity threshold
            .Where(item => item.SimilarityScore >= minThreshold)
            // Sort by similarity score in descending order (most similar first)
            .OrderByDescending(item => item.SimilarityScore)
            // Take only the top N most similar profiles
            .Take(topCount)
            .ToList();

            return scoredSamples;
        }

        /// <summary>
        /// Displays a formatted string representation of the similar profiles
        /// </summary>
        /// <param name="similarProfiles">The list of similar profiles with scores</param>
        /// <returns>A formatted string with profile information</returns>
        /// <remarks>
        /// This method formats the similar profiles into a readable string with columns for:
        /// - Rank (right-aligned, 4 chars)
        /// - Similarity score (percentage with 2 decimal places)
        /// - Credit score (right-aligned, 11 chars, no decimals)
        /// - Annual income (right-aligned, 12 chars, currency format)
        /// - Loan amount (right-aligned, 10 chars, currency format)
        /// - Loan approval status (converted from binary to text)
        /// </remarks>
        public static string FormatSimilarProfiles(this List<(BorrowerProfile Profile, double SimilarityScore)> similarProfiles)
        {
            // Check if there are any profiles to display
            if (similarProfiles == null || !similarProfiles.Any())
                return "No similar profiles found.";

            // Define the Handlebars template
            string template = """
                Similar Borrower Profiles:
                -------------------------------------------------------------------------
                Rank | Similarity | Credit Score | Annual Income | Loan Amount | Status
                -------------------------------------------------------------------------
                {{#each profiles}}
                {{rank,4}} | {{score:P2}}    | {{creditScore,11:F0}} | {{annualIncome,12:C0}} | {{loanAmount,10:C0}} | {{status}}
                {{/each}}
                -------------------------------------------------------------------------
                """;

            // Transform the data into a format suitable for the template
            var templateData = new
            {
                profiles = similarProfiles.Select((item, index) => new
                {
                    rank = index + 1,
                    score = item.SimilarityScore,
                    creditScore = item.Profile.CreditScore,
                    annualIncome = item.Profile.AnnualIncome,
                    loanAmount = item.Profile.LoanAmount,
                    status = item.Profile.LoanApproved == 1 ? "Approved" : "Rejected"
                })
            };

            // Render the template with the data
            return HandlebarsUtility.RenderTemplate(template, templateData);
        }
    }
}