# RandomExtensions

The `RandomExtensions` class provides extension methods for the Random class to enhance its functionality.

## Properties

| Property                   | Type   | Description                                         |
| -------------------------- | ------ | --------------------------------------------------- |

## Methods

### NextGaussian

**Summary**: Generates a random number from a Gaussian distribution

**Parameters**:

- `random`: The Random instance

**Returns**: A random number from a Gaussian distribution with mean 0 and standard deviation 1

This method generates a random number from a Gaussian distribution using the Box-Muller transform. It takes two independent uniform random variables and converts them to a standard normal distribution. The implementation ensures numerical stability by avoiding taking the logarithm of zero.
