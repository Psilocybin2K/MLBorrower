# BorrowerSimilarityExtensions

The `BorrowerSimilarityExtensions` class provides extension methods for finding and displaying similar borrower profiles.

## Properties

| Property                   | Type   | Description                                         |
| -------------------------- | ------ | --------------------------------------------------- |

## Methods

### FindSimilarProfiles

**Summary**: Finds the most similar borrower profiles based on specified criteria

**Parameters**:

- `samples`: The list of borrower profiles to search within
- `targetCreditScore`: The target credit score to match
- `targetAnnualIncome`: The target annual income to match
- `targetLoanAmount`: The target loan amount to match
- `creditScoreWeight`: The weight to assign to credit score similarity (higher = more important)
- `annualIncomeWeight`: The weight to assign to annual income similarity (higher = more important)
- `loanAmountWeight`: The weight to assign to loan amount similarity (higher = more important)
- `topCount`: The number of similar profiles to return
- `minThreshold`: The minimum similarity threshold (0.0 to 1.0) to include a profile

**Returns**: A list of the most similar borrower profiles along with their similarity scores

This method finds the most similar borrower profiles based on credit score, annual income, and loan amount. It uses a weighted combination of normalized distances for each attribute to calculate a final similarity score. The scores are then filtered and sorted to return the top N most similar profiles.

### FormatSimilarProfiles

**Summary**: Displays a formatted string representation of the similar profiles

**Parameters**:

- `similarProfiles`: The list of similar profiles with scores

**Returns**: A formatted string with profile information

This method formats the similar profiles into a readable string with columns for rank, similarity score (percentage), credit score, annual income, loan amount, and loan approval status. The output is formatted in a tabular structure with appropriate alignment and formatting for each column.
