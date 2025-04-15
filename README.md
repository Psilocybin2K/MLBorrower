# ML Borrower Agent

## Table of Contents

- [Overview](#overview)
- [Technical Documentation](#technical-documentation)
  - [Class Structure](#class-structure)
  - [Constructor](#constructor)
  - [Methods](#methods)
- [Agent Plugins](#agent-plugins)
  - [LoanApprovalPredictorPlugin](#loanapprovalpredictorplugin)
  - [ProfileImprovementAnalyzerPlugin](#profileimprovementanalyzerplugin)
- [User Documentation](#user-documentation)
  - [Getting Started](#getting-started)
  - [Using the Agent](#using-the-agent)
  - [Understanding Results](#understanding-results)
  - [Best Practices](#best-practices)

## Overview

The MLBorrowerAgent is an intelligent agent that leverages Bayesian networks to predict loan approval probabilities and provide actionable recommendations to improve borrower profiles. This system combines machine learning with natural language processing to deliver personalized financial advice and loan approval predictions.

Refer to the [docs](./Docs/README.md)

## Technical Documentation

### Class Structure

```csharp
public class MLBorrowerAgent
```

The `MLBorrowerAgent` class represents a borrower agent that uses a Bayesian network to predict loan approval probabilities. The agent incorporates multiple plugins to analyze borrower profiles and predict loan outcomes.

#### Dependencies

The agent relies on the following key components:

- `Kernel`: Core processing component for the agent
- `BorrowerModelBuilder`: Builds borrower models from input data
- `BorrowerBayesianNetwork`: Bayesian network for borrower profile analysis
- `LoanApprovalBayesianNetwork`: Bayesian network for loan approval prediction
- `LiquidPromptTemplateFactory`: Factory for creating prompt templates
- `ChatHistoryAgentThread`: Maintains conversation history
- `ChatCompletionAgent`: Handles chat completion functionality


## Kaggle Dataset

Ensure that you download the following CSV dataset, and add to your project as `loans.csv` in the root folder of your project

[View Data](https://www.kaggle.com/datasets/deboleenamukherjee/financial-risk-data-large?select=financial_risk_analysis_large.csv)

### Constructor

```csharp
public MLBorrowerAgent(
    Kernel kernel,
    BorrowerModelBuilder modelBuilder,
    BorrowerBayesianNetwork bayesianNetwork,
    LoanApprovalBayesianNetwork approvalNetwork
)
```

#### Parameters

- `kernel`: The kernel instance for agent operations
- `modelBuilder`: The borrower model builder for creating borrower profiles
- `bayesianNetwork`: The borrower Bayesian network for profile analysis
- `approvalNetwork`: The loan approval Bayesian network for prediction

#### Implementation Details

The constructor:

1. Clones the provided kernel to avoid side effects
2. Sets the model builder and Bayesian network
3. Adds the LoanApprovalPredictorPlugin and PredictorPlugin to the kernel
4. Initializes the agent template from a file
5. Creates a template factory
6. Sets up a chat history thread for maintaining conversation context
7. Initializes the chat completion agent with the name "BuyerProfileAnalysisAgent"

### Methods

```csharp
public async Task<string> InvokeAsync(string instructions)
```

Invokes the agent asynchronously with the provided instructions.

#### Parameters

- `instructions`: The instructions to pass to the agent

#### Returns

- `Task<string>`: A string containing the agent's response

#### Implementation Details

The method:

1. Sets up Azure OpenAI execution settings with temperature 0.2 and automatic kernel function invocation
2. Creates agent invocation options with the kernel and provided instructions
3. Invokes the agent asynchronously
4. Collects the response items and builds a string
5. Returns the final text response

## Agent Plugins

The MLBorrowerAgent includes two specialized plugins that enable its core functionality:

### LoanApprovalPredictorPlugin

```csharp
public class LoanApprovalPredictorPlugin
```

This plugin predicts loan approval probabilities using a Bayesian network.

#### Functionality

- Analyzes borrower profile data
- Calculates probability of loan approval
- Provides numerical probability scores
- Identifies key factors influencing approval

#### Implementation Details

The plugin uses a loan approval Bayesian network to calculate conditional probabilities based on various borrower attributes such as:

- Credit score
- Debt-to-income ratio
- Loan amount relative to income
- Employment history
- Previous loan history

### ProfileImprovementAnalyzerPlugin

While not fully shown in the code snippet, this plugin is referenced in the documentation.

```csharp
public class PredictorPlugin
```

This appears to be an implementation of the profile improvement analyzer functionality.

#### Functionality

- Analyzes borrower profiles for weaknesses
- Suggests specific improvements to increase approval chances
- Prioritizes recommendations based on impact
- Provides actionable advice

#### Implementation Details

The plugin uses the borrower model builder and Bayesian network to:

- Identify profile attributes with negative impact
- Simulate profile changes to measure impact on approval probability
- Generate prioritized recommendations
- Provide quantitative improvement estimates

## User Documentation

### Getting Started

#### System Requirements

- .NET runtime environment
- Access to Azure OpenAI API
- Configuration for Bayesian network models
- Proper template files in the specified locations

#### Installation

1. Include the MLBorrowerAgent package in your project
2. Configure the required dependencies
3. Ensure template files are properly placed at `.\Templates\Prompts\MLBorrowerProfileAgent.prompty`

#### Configuration

The agent requires proper configuration of:

- Azure OpenAI API credentials
- Bayesian network model parameters
- Template files for agent prompts

### Using the Agent

#### Initializing the Agent

```csharp
// Example initialization
var kernel = new Kernel(/* configuration */);
var modelBuilder = new BorrowerModelBuilder(/* configuration */);
var borrowerNetwork = new BorrowerBayesianNetwork(/* configuration */);
var approvalNetwork = new LoanApprovalBayesianNetwork(/* configuration */);

var borrowerAgent = new MLBorrowerAgent(
    kernel,
    modelBuilder,
    borrowerNetwork,
    approvalNetwork
);
```

#### Invoking the Agent

```csharp
// Example invocation
string instructions = "Analyze this borrower profile and predict loan approval: Credit score 720, income $75,000, debt-to-income ratio 28%, requesting $250,000 loan";
string response = await borrowerAgent.InvokeAsync(instructions);
Console.WriteLine(response);
```

### Understanding Results

The agent provides responses that typically include:

1. **Loan Approval Probability**: A numerical estimation of approval likelihood
2. **Key Factors**: Identification of profile strengths and weaknesses
3. **Recommendations**: Specific actions to improve approval chances
4. **Explanations**: Natural language explanations of the underlying factors

Example response:

```
Based on the provided profile, the estimated loan approval probability is 78%.

Key strengths:
- Good credit score (720)
- Reasonable debt-to-income ratio (28%)

Areas for improvement:
- The loan amount is somewhat high relative to income
- Consider increasing down payment to reduce loan-to-value ratio
- Maintaining credit score above 740 would significantly increase approval odds

If you increase your down payment by $25,000 and improve your credit score to 740+, your approval probability would increase to approximately 92%.
```

### Best Practices

#### For Optimal Results

1. Provide complete borrower information
2. Include specific financial metrics when available
3. Be explicit about the loan type and purpose
4. Use clear, unambiguous instructions
5. Follow up on recommendations for iterative improvement

#### Limitations

- Predictions are probabilistic and not guarantees
- Results depend on the quality of input data
- The agent does not replace formal loan pre-approval
- Recommendations should be evaluated in context of individual circumstances
- Bayesian networks reflect historical patterns which may not apply to all lenders
