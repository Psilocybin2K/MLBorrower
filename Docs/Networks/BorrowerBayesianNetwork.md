# Borrower Bayesian Network

## Properties

| Property                   | Type                | Description                                         |
| -------------------------- | ------------------- | --------------------------------------------------- |
| N                          | Range               | The range of examples in the training data.         |
| NumberOfExamples           | Variable<int>       | The number of examples in the training data.        |
| CreditScore                | VariableArray<double> | The credit score of the borrower.                |
| CreditScorePrior           | Variable<Gaussian>  | The prior distribution for the credit score.        |
| AnnualIncome               | VariableArray<double> | The annual income of the borrower.               |
| AnnualIncomePrior          | Variable<Gaussian>  | The prior distribution for the annual income.       |
| EmploymentStatus           | VariableArray<int>  | The employment status of the borrower.             |
| EmploymentStatusPrior      | Variable<Vector>    | The prior distribution for the employment status.   |
| MaritalStatus              | VariableArray<int>  | The marital status of the borrower.                |
| MaritalStatusPrior         | Variable<Vector>    | The prior distribution for the marital status.      |
| EducationLevel             | VariableArray<int>  | The education level of the borrower.               |
| EducationLevelPrior        | Variable<Vector>    | The prior distribution for the education level.     |
| HomeOwnershipStatus        | VariableArray<int>  | The home ownership status of the borrower.         |
| NumberOfDependents         | VariableArray<int>  | The number of dependents of the borrower.          |
| AnnualIncomePosterior      | Gaussian            | The posterior distribution for the annual income.   |
| EmploymentStatusPosterior  | Discrete[]          | The posterior distribution for the employment status. |
| MaritalStatusPosterior     | Discrete[]          | The posterior distribution for the marital status.  |
| EducationLevelPosterior    | Discrete[]          | The posterior distribution for the education level. |
| Engine                     | InferenceEngine     | The inference engine for the Bayesian network.      |

## Methods

### BorrowerBayesianNetwork

**Summary**: Initializes a new instance of the BorrowerBayesianNetwork class.

**Parameters**:

- `dataProcessor`: The data processor containing the training data.

**Remarks**: This constructor initializes the Bayesian network structure and sets up the mappings for categorical variables. It also defines the prior distributions for the continuous variables and creates the conditional dependencies.

### InitializeCategoricalMappings

**Summary**: Initializes the mappings for categorical variables.

**Parameters**:

- `dataProcessor`: The data processor containing the categorical variable values.

**Remarks**: This method initializes the mappings for categorical variables, which is necessary for the Bayesian network's probabilistic calculations.

### InitializeMappingForVariable

**Summary**: Initializes the mapping for a specific categorical variable.

**Parameters**:

- `variableName`: The name of the categorical variable.
- `values`: The list of possible values for the variable.

**Remarks**: This method creates a mapping from string values to their corresponding indices and vice versa. It also updates the categoricalMappings and inverseCategoricalMappings dictionaries.

### GenerateSamples

**Summary**: Generates a list of sample borrower profiles.

**Parameters**:

- `sampleCount`: The number of samples to generate.
- `dataProcessor`: The data processor containing the training data.

**Returns**: A list of generated borrower profiles.

**Remarks**: This method generates a list of sample borrower profiles using the Bayesian network structure. It generates demographic variables, categorical variables, and credit history variables, and then calculates the credit score based on the relationships defined in the Bayesian network.

### GenerateValueFromStatsWithFallback

**Summary**: Generates a value from the statistics with fallback values if stats are missing.

**Parameters**:

- `statName`: The name of the statistic to generate a value for.
- `stats`: The dictionary containing the statistics.
- `random`: The random number generator.
- `min`: The minimum value for the generated value.
- `max`: The maximum value for the generated value.
- `fallbackMean`: The mean value to use if stats are missing.
- `fallbackStdDev`: The standard deviation value to use if stats are missing.

**Returns**: A generated value from the statistics.

**Remarks**: This method generates a value from the statistics with fallback values if stats are missing. It uses a Gaussian distribution to generate a value and applies constraints if provided.

### GenerateDoubleValueFromStatsWithFallback

**Summary**: Generates a double value from the statistics with fallback values if stats are missing.

**Parameters**:

- `statName`: The name of the statistic to generate a value for.
- `stats`: The dictionary containing the statistics.
- `random`: The random number generator.
- `min`: The minimum value for the generated value.
- `max`: The maximum value for the generated value.
- `fallbackMean`: The mean value to use if stats are missing.
- `fallbackStdDev`: The standard deviation value to use if stats are missing.

**Returns**: A generated double value from the statistics.

**Remarks**: This method generates a double value from the statistics with fallback values if stats are missing. It uses a Gaussian distribution to generate a value and applies constraints if provided.

### SampleCategoricalVariable

**Summary**: Samples a categorical variable based on its distribution in the data.

**Parameters**:

- `variableName`: The name of the categorical variable to sample.
- `dataProcessor`: The data processor containing the training data.

**Returns**: A sampled value from the categorical variable.

### GetProperty

**Summary**: Gets a property value from a model by name using reflection.

**Parameters**:

- `model`: The model to get the property value from.
- `propertyName`: The name of the property to get the value of.

**Returns**: The value of the property.

**Remarks**: This method gets a property value from a model by name using reflection.
