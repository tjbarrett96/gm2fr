import numpy as np

# A Gaussian-mixture probability density function.
class GaussianMixture:

  # Constructor. ===============================================================

  def __init__(self, weights, means, widths):
  
      # Ensure there is one weight, mean, and width for each Gaussian.
      assert len(weights) == len(means) and len(means) == len(widths)
      
      # Store the Gaussian parameters as NumPy arrays.
      self.weights = np.array(weights, dtype = np.float64)
      self.means = np.array(means, dtype = np.float64)
      self.widths = np.array(widths, dtype = np.float64)
      
      # Ensure the weights are normalized.
      self.weights *= 1 / np.sum(self.weights)
      
      # The number of Gaussians in the mixture.
      self.gaussians = len(weights)
    
  # Random number generator. ===================================================
    
  def draw(self, choices = 1, shifts = 0):
  
    # Ensure the number of choices is an integer.
    choices = int(choices)
  
    # For each choice, pick a Gaussian from the mixture.
    indices = np.random.choice(self.gaussians, size = choices, p = self.weights)
    
    # For each choice, draw from the corresponding Gaussian.
    return np.random.normal(self.means.take(indices) + shifts, self.widths.take(indices))
    
  # Saving/loading. ============================================================
  
  def save(self, filename):
    np.savez(filename, weights = self.weights, means = self.means, widths = self.widths)
    
  @classmethod
  def load(cls, filename):
    data = np.load(filename)
    return cls(data["weights"], data["means"], data["widths"])
    
  # Evaluating the PDF. ========================================================
  
  def value(self, x):
  
    result = 0
    for mean, width, weight in zip(self.means, self.widths, self.weights):
      result += weight / (np.sqrt(2*np.pi) * width) * np.exp(-(x - mean)**2 / (2 * width**2))
    return result
    
  # Plotting the PDF. ==========================================================
  
  def plot(self, plt, x, label = None):
    return plt.plot(x, self.value(x), label = label)
  
