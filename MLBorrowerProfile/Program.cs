using System.Diagnostics;
using Microsoft.SemanticKernel;
using MLBorrowerProfile.Builders;
using MLBorrowerProfile.Networks;

namespace MLBorrowerProfile
{

    internal class Program
    {
        private static async Task Main(string[] args)
        {
            Kernel _kernel = Kernel.CreateBuilder()
                .AddAzureOpenAIChatCompletion("gpt-4o-mini",
                    Environment.GetEnvironmentVariable("AOAI_ENDPOINT"),
                    Environment.GetEnvironmentVariable("AOAI_API_KEY"))
                .Build();

            Console.WriteLine("Borrower Profile Generator");

            // Initialize the data processor
            BorrowerModelBuilder modelBuilder = new BorrowerModelBuilder();

            // Load and process the CSV data
            modelBuilder.LoadAndProcessData("loans.csv");

            Console.WriteLine("Data preparation completed successfully.");

            // Create and train the Bayesian network model
            BorrowerBayesianNetwork bayesianNetwork = new BorrowerBayesianNetwork(modelBuilder);
            bayesianNetwork.LearnParameters(modelBuilder.TrainingData);

            // Initialize the loan approval prediction network
            LoanApprovalBayesianNetwork approvalNetwork = new LoanApprovalBayesianNetwork(modelBuilder);
            approvalNetwork.LearnParameters(modelBuilder.TrainingData);

            //Console.WriteLine("Model training completed successfully.");

            //// Generate new samples
            //int sampleCount = 10;
            //List<Model> samples = bayesianNetwork.GenerateSamples(sampleCount, modelBuilder);


            //string borrowerProfilesTemplate = """
            //        ## Generated {{sampleCount}} new borrower profiles:

            //        {{#each samples}}
            //        - **Credit Score**: {{this.creditScore}},  
            //          **Annual Income**: {{this.annualIncome}},  
            //          **Loan Amount**: {{this.loanAmount}},  
            //          **Employment Status**: {{this.employmentStatus}},  
            //          **Marital Status**: {{this.maritalStatus}}

            //        {{/each}}
            //        """;

            //Console.WriteLine(HandlebarsUtility.RenderTemplate(borrowerProfilesTemplate, new
            //{
            //    samples,
            //    sampleCount
            //}));

            //Console.WriteLine("\nPress any key to exit...");

            MLBorrowerAgent agent = new MLBorrowerAgent(_kernel, modelBuilder, bayesianNetwork, approvalNetwork);

            //string res1 = await agent.InvokeAsync("""
            //    Predict the liklihood of loan approval:
            //        - Credit Score = 600
            //        - Loan Amount = 120K
            //        - Annual Income = 340K
            //    """);

            string res = await agent.InvokeAsync("""
                Generate a single borrower profile with a similar profile:
                    - Ltv <= 40%
                    - Fico >= 600
                """);

            Console.WriteLine($"""
                
                ---

                {res}

                ---

                """);

            Debugger.Break();
        }
    }
}