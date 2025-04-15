namespace MLBorrowerProfile.Extensions
{
    /// <summary>
    /// Extension methods for the Random class
    /// </summary>
    /// <remarks>
    /// This class contains extension methods for the Random class.
    /// </remarks>
    public static class RandomExtensions
    {
        /// <summary>
        /// Generates a random number from a Gaussian distribution
        /// </summary>
        /// <param name="random">The Random instance</param>
        /// <returns>A random number from a Gaussian distribution</returns>
        /// <remarks>
        /// This method generates a random number from a Gaussian distribution using the Box-Muller transform.
        /// </remarks>
        public static double NextGaussian(this Random random)
        {
            // Generate two independent uniform random variables between 0 and 1
            // Subtracting from 1.0 ensures we avoid taking log of 0
            double u1 = 1.0 - random.NextDouble();
            double u2 = 1.0 - random.NextDouble();

            // Apply Box-Muller transform to convert uniform random variables to standard normal distribution
            // The transform uses polar coordinates to generate two independent standard normal variables
            // We only return one of them (the other would be the cosine term)
            return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        }
    }
}