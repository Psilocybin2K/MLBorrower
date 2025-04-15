# FeatureImportance

The `FeatureImportance` class represents the importance of features in loan approval and provides analysis of their impact.

## Properties

| Property                | Type         | Description                                                         |
| ----------------------- | ------------ | ------------------------------------------------------------------- |
| BaseApprovalProbability | double       | The base approval probability for the borrower profile.             |
| CreditScoreImpact       | double       | The impact of the credit score on the approval probability.         |
| DebtToIncomeRatioImpact | double       | The impact of the debt-to-income ratio on the approval probability. |
| LoanToIncomeRatioImpact | double       | The impact of the loan-to-income ratio on the approval probability. |
| DefaultHistoryImpact    | double       | The impact of the default history on the approval probability.      |
| BankruptcyHistoryImpact | double       | The impact of the bankruptcy history on the approval probability.   |
| EmploymentStatusImpact  | double       | The impact of the employment status on the approval probability.    |
| LoanPurposeImpact       | double       | The impact of the loan purpose on the approval probability.         |
| Recommendations         | List<string> | The list of recommendations for the borrower profile.               |

## Methods

### RankedFeatures

**Summary**: The list of features ranked by their impact on the approval probability.

**Parameters**: None

**Returns**: A list of tuples containing feature names and their corresponding impact scores.

This method creates a list of tuples containing feature names and their corresponding impact scores. Each tuple represents a key factor in loan approval decision making. The features are sorted by absolute impact value in descending order, ensuring the most influential factors appear first in the ranking.

### GenerateRecommendations

**Summary**: Generates personalized recommendations for the borrower profile based on their feature importance.

**Parameters**:

- `profile`: The borrower profile to generate recommendations for.

**Returns**: void

This method generates personalized recommendations for the borrower profile based on their feature importance. It prioritizes recommendations based on the impact of each feature on the approval probability. The method analyzes features like credit score, debt-to-income ratio, loan-to-income ratio, and employment status to provide specific actionable recommendations for improving loan approval odds.
