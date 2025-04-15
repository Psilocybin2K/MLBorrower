# ML Borrower Agent Dataset Data Card

## Dataset Overview

**Name**: Borrower Profile Dataset  
**Description**: A comprehensive dataset of borrower profiles with associated loan approval outcomes  
**Size**: Approximately 500,000 records  
**Format**: CSV (Comma-Separated Values)  
**Purpose**: Training and evaluation of loan approval prediction models  

## Dataset Features

The dataset contains 51 features spanning various aspects of borrower profiles:

### Financial Attributes
| Feature Name | Data Type | Description | Example Value |
|--------------|-----------|-------------|--------------|
| CreditScore | Numeric | Borrower's credit score | 402, 735 |
| AnnualIncome | Numeric | Annual income in USD | 63295, 55936 |
| LoanAmount | Numeric | Requested loan amount in USD | 18830, 23729 |
| MonthlyDebtPayments | Numeric | Monthly debt payments in USD | 675, 508 |
| DebtToIncomeRatio | Numeric | Ratio of debt payments to income | 0.555, 0.091 |
| CreditCardUtilizationRate | Numeric | Percentage of available credit used | 0.144, 0.690 |
| SavingsAccountBalance | Numeric | Balance in savings account in USD | 7989, 5158 |
| CheckingAccountBalance | Numeric | Balance in checking account in USD | 3144, 2818 |
| InvestmentAccountBalance | Numeric | Balance in investment accounts in USD | 6407, 16030 |
| RetirementAccountBalance | Numeric | Balance in retirement accounts in USD | 12216, 50733 |
| EmergencyFundBalance | Numeric | Emergency fund balance in USD | 3279, 4353 |
| TotalAssets | Numeric | Sum of all assets in USD | 93791, 52891 |
| TotalLiabilities | Numeric | Sum of all liabilities in USD | 34258, 60647 |
| NetWorth | Numeric | Difference between assets and liabilities in USD | -5232, 89318 |
| InterestRate | Numeric | Interest rate for the loan | 0.161, 0.013 |
| MortgageBalance | Numeric | Remaining mortgage balance in USD | 44585, 117295 |
| RentPayments | Numeric | Monthly rent payments in USD | 1088, 1344 |
| AutoLoanBalance | Numeric | Remaining auto loan balance in USD | 17815, 11269 |
| PersonalLoanBalance | Numeric | Remaining personal loan balance in USD | 2752, 1819 |
| StudentLoanBalance | Numeric | Remaining student loan balance in USD | 21989, 20927 |
| MonthlySavings | Numeric | Monthly savings amount in USD | 378, 575 |
| AnnualBonuses | Numeric | Annual bonuses in USD | 3741, 4115 |
| AnnualExpenses | Numeric | Annual expenses in USD | 40058, 16745 |
| MonthlyHousingCosts | Numeric | Monthly housing costs in USD | 977, 695 |
| MonthlyTransportationCosts | Numeric | Monthly transportation costs in USD | 412, 206 |
| MonthlyFoodCosts | Numeric | Monthly food costs in USD | 399, 898 |
| MonthlyHealthcareCosts | Numeric | Monthly healthcare costs in USD | 136, 252 |
| MonthlyEntertainmentCosts | Numeric | Monthly entertainment costs in USD | 124, 131 |

### Demographic Attributes
| Feature Name | Data Type | Description | Example Value |
|--------------|-----------|-------------|--------------|
| Age | Numeric | Age of the borrower in years | 29, 42 |
| EmploymentStatus | Categorical | Current employment status | Self-Employed |
| MaritalStatus | Categorical | Current marital status | Widowed, Divorced |
| NumberOfDependents | Numeric | Number of financial dependents | 2, 3 |
| EducationLevel | Categorical | Highest level of education completed | Doctorate, Master |
| HomeOwnershipStatus | Categorical | Current home ownership status | Other, Own |
| EmployerType | Categorical | Type of employer | Self-Employed, Private |
| JobTenure | Numeric | Time at current job in months | 24, 10 |

### Credit History Attributes
| Feature Name | Data Type | Description | Example Value |
|--------------|-----------|-------------|--------------|
| NumberOfOpenCreditLines | Numeric | Number of open credit accounts | 11, 9 |
| NumberOfCreditInquiries | Numeric | Recent credit inquiries | 2, 8 |
| BankruptcyHistory | Numeric | Number of past bankruptcies | 0, 0 |
| PreviousLoanDefaults | Numeric | Number of previous loan defaults | 0, 0 |
| PaymentHistory | Numeric | Payment history score | 12, 18 |
| LengthOfCreditHistory | Numeric | Length of credit history in months | 3, 7 |
| UtilityBillsPaymentHistory | Numeric | Utility bills payment history score | 0.257, 0.661 |

### Insurance Information
| Feature Name | Data Type | Description | Example Value |
|--------------|-----------|-------------|--------------|
| HealthInsuranceStatus | Categorical | Current health insurance status | Insured, Uninsured |
| LifeInsuranceStatus | Categorical | Current life insurance status | Insured, Uninsured |
| CarInsuranceStatus | Categorical | Current car insurance status | Insured, Uninsured |
| HomeInsuranceStatus | Categorical | Current home insurance status | Insured, Uninsured |
| OtherInsurancePolicies | Numeric | Number of other insurance policies | 4, 4 |

### Loan Information
| Feature Name | Data Type | Description | Example Value |
|--------------|-----------|-------------|--------------|
| LoanDuration | Numeric | Duration of loan in years | 13, 1 |
| LoanPurpose | Categorical | Purpose of the loan | Education, Debt Consolidation |
| LoanApproved | Binary | Whether the loan was approved (target variable) | 0, 0 |

## Data Distribution and Statistics

Based on the sample rows provided, the dataset appears to contain a diverse range of borrower profiles with varying:
- Credit scores (from poor to excellent)
- Income levels
- Loan amounts and purposes
- Employment and marital statuses
- Asset and liability portfolios
- Spending and saving behaviors

A full statistical analysis would provide more detailed insights into:
- Distribution of continuous variables (mean, median, standard deviation, etc.)
- Frequency of categorical variables
- Correlation between features
- Class balance of the target variable (loan approval rate)

## Data Collection Methodology

This dataset represents approximately 500,000 historical loan applications with comprehensive borrower information. Each record contains detailed financial, demographic, and credit history information along with the final loan approval decision.

## Intended Use

This dataset is intended for:
1. Training machine learning models to predict loan approval probability
2. Analyzing factors that influence loan approval decisions
3. Developing borrower profile improvement recommendations
4. Simulating the effects of profile changes on approval odds
5. Understanding the relationship between various borrower attributes and loan outcomes

## Limitations and Considerations

When working with this dataset, consider the following:
- The data may reflect historical biases in lending practices
- Some features may be correlated, potentially leading to multicollinearity in models
- The dataset may not represent all demographic groups equally
- Regional lending differences may not be captured
- Temporal changes in lending standards may not be reflected
- Privacy considerations are important when handling sensitive financial data

## Ethical Considerations

Users of this dataset should be aware of:
- Potential for reinforcing existing biases in lending
- Need for fairness and transparency in model development
- Importance of explainable AI in financial decision-making
- Privacy concerns related to sensitive personal financial information
- Regulatory compliance requirements in the financial industry

## Preprocessing Requirements

Typical preprocessing steps for this dataset include:
- Handling missing values
- Normalizing continuous features
- Encoding categorical variables
- Feature scaling
- Addressing class imbalance (if present)
- Feature selection or dimensionality reduction

## Benchmarks

Model performance on this dataset should be evaluated using appropriate metrics such as:
- Accuracy
- Precision and recall
- F1-score
- Area Under the ROC Curve (AUC-ROC)
- Calibration of probability estimates
- Fairness metrics across different demographic groups

## Maintenance and Updates

Information about how frequently the dataset is updated, who maintains it, and the process for reporting issues or requesting changes should be documented here.
