# BorrowerModelBuilder

The `BorrowerModelBuilder` class implements a comprehensive data processing pipeline for borrower data.

## Properties

| Property                    | Type                                                                     | Description                                                                        |
| --------------------------- | ------------------------------------------------------------------------ | ---------------------------------------------------------------------------------- |
| TrainingData                | List<BorrowerProfile>                                                    | Gets the list of processed borrower models used for training.                      |
| ContinuousStats             | Dictionary<string, (double Min, double Max, double Mean, double StdDev)> | Gets the statistics for continuous variables in the training data.                 |
| EmploymentStatusValues      | List<string>                                                             | Gets the list of unique employment status values found in the training data.       |
| MaritalStatusValues         | List<string>                                                             | Gets the list of unique marital status values found in the training data.          |
| EducationLevelValues        | List<string>                                                             | Gets the list of unique education level values found in the training data.         |
| HomeOwnershipStatusValues   | List<string>                                                             | Gets the list of unique home ownership status values found in the training data.   |
| LoanPurposeValues           | List<string>                                                             | Gets the list of unique loan purpose values found in the training data.            |
| HealthInsuranceStatusValues | List<string>                                                             | Gets the list of unique health insurance status values found in the training data. |
| LifeInsuranceStatusValues   | List<string>                                                             | Gets the list of unique life insurance status values found in the training data.   |
| CarInsuranceStatusValues    | List<string>                                                             | Gets the list of unique car insurance status values found in the training data.    |
| HomeInsuranceStatusValues   | List<string>                                                             | Gets the list of unique home insurance status values found in the training data.   |
| EmployerTypeValues          | List<string>                                                             | Gets the list of unique employer type values found in the training data.           |

## Methods

### LoadAndProcessData

**Summary**: Loads and processes the borrower data from a CSV file.

**Parameters**:

- `csvFilePath`: The path to the CSV file containing borrower data.

**Returns**: void

### CollectCategoricalValues

**Summary**: Collects unique categorical values from the training data for each field.

**Parameters**: None

**Returns**: void

### CalculateContinuousStatistics

**Summary**: Calculates statistical measures for continuous variables in the training data.

**Parameters**: None

**Returns**: void

### CalculateStats

**Summary**: Calculates basic descriptive statistics for a continuous variable in the training data.

**Parameters**:

- `variableName`: The name of the variable to analyze.
- `selector`: A function that selects the value of the variable from a borrower model.

**Returns**: void

### PrintCategoricalValuesSummary

**Summary**: Prints a summary of categorical values from the training data.

**Parameters**: None

**Returns**: void

### PrintContinuousStatisticsSummary

**Summary**: Prints a summary of continuous variable statistics from the training data.

**Parameters**: None

**Returns**: void
