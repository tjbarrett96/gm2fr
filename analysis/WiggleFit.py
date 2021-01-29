import numpy as np
import scipy.optimize as opt
import gm2fr.utilities as util

# ==============================================================================

# Two-parameter wiggle fit function, only modeling exponential decay.
def two(t, N, tau):
  return N * np.exp(-t / tau)
  
# ==============================================================================

# Five-parameter wiggle fit function, adding precession wiggle on top of decay.
def five(t, N, tau, A, f, phi):
  return two(t, N, tau) \
         * (1 + A * np.cos(2 * np.pi * f * t + phi))
  
# ==============================================================================

# Nine-parameter wiggle fit function, adding CBO wiggle on top of the above.
def nine(t, N, tau, A, f, phi, tau_cbo, A_cbo, f_cbo, phi_cbo):
  return five(t, N, tau, A, f, phi) \
         * (1 + np.exp(-t / tau_cbo) * A_cbo * np.cos(2 * np.pi * f_cbo * t + phi_cbo))
  
# ==============================================================================

# Mapping function names (i.e. strings) to the actual functions.
modelFunctions = {
  "two": two,
  "five": five,
  "nine": nine
}

# ==============================================================================

# Parameter names and units, for printing.
modelLabels = {}

modelLabels["two"] = [
  ("normalization", ""),
  ("lifetime", "us")
]

modelLabels["five"] = modelLabels["two"] + [
  ("asymmetry", ""),
  ("frequency", "1/us"),
  ("phase", "rad")
]

modelLabels["nine"] = modelLabels["five"] + [
  ("CBO lifetime", "us"),
  ("CBO asymmetry", ""),
  ("CBO frequency", "1/us"),
  ("CBO phase", "rad")
]

# ==============================================================================

# Initial guesses for each function's parameters.
modelSeeds = {}

modelSeeds["two"] = [
  1,    # normalization
  64.4  # lifetime (us)
]

modelSeeds["five"] = modelSeeds["two"] + [
  0.5,   # wiggle asymmetry
  0.229, # wiggle frequency (MHz)
  np.pi  # wiggle phase (rad)
]

modelSeeds["nine"] = modelSeeds["five"] + [
  150,   # CBO lifetime (us)
  0.005, # CBO asymmetry
  0.370, # CBO frequency (MHz)
  np.pi  # CBO phase (rad)
]

# ==============================================================================

# Upper and lower bounds for each function's parameters.
modelBounds = {}

modelBounds["two"] = (
    [0,             64.1    ],
    [np.inf,        64.9    ]
) # [normalization, lifetime]

modelBounds["five"] = (
  modelBounds["two"][0] + [-0.3,      0.2288,    0      ],
  modelBounds["two"][1] + [ 1.0,      0.2292,    2*np.pi]
) #          (precession) [asymmetry, frequency, phase  ]

modelBounds["nine"] = (
  modelBounds["five"][0] + [100,      0   ,      0.350,     0      ],
  modelBounds["five"][1] + [400,      0.01,      0.430,     2*np.pi]
) #                  (CBO) [lifetime, asymmetry, frequency, phase  ]

# ==============================================================================

class WiggleFit:
 
  # ============================================================================
 
  def __init__(
    self,
    time,           # time bin centers
    signal   ,      # positron counts per time bin
    model = "five", # wiggle fit model ("two" / "five" / "nine")
    start = 30,     # fit start time (us)
    end = 650,      # fit end time (us)
    n = None        # the n-value, if known, helps inform CBO frequency seed
  ):
  
    # Signal data.
    self.time = time
    self.signal = signal
    self.error = np.sqrt(signal)
    
    # Ensure the errors are all non-zero.
    self.error[self.error == 0] = 1
    
    # Select the sequence of fits to perform, based on the supplied model.
    self.models = ["two", "five", "nine"]
    if model in self.models:
      self.model = model
      self.models = self.models[:(self.models.index(self.model) + 1)]
    else:
      raise ValueError(f"Wiggle fit model '{model}' not recognized.")
      
    # Fit options.
    self.start = start
    self.end = end
    self.n = n
    self.function = None
    
    # Fit portion of the data, with start/end mask applied.
    mask = (self.time >= self.start) & (self.time <= self.end)
    self.fitTime = self.time[mask]
    self.fitSignal = self.signal[mask]
    self.fitError = self.error[mask]
    
    # Fit results.
    self.pOpt = None
    self.pCov = None
    self.pErr = None
    self.fitResult = None
    self.chi2dof = None
    
  # ============================================================================

  def fit(self):
    
    # Iterate through the models, updating fit seeds each time.
    for model in self.models:
      
      # Get the function, parameter seeds, and parameter bounds for this model.
      self.function = modelFunctions[model]
      seeds = modelSeeds[model]
      bounds = modelBounds[model]
      
      # Update the seeds with the previous fit's results.
      if self.pOpt is not None:
        seeds[:len(self.pOpt)] = self.pOpt
      
      # Set the CBO frequency seed based on the expected n-value.
      if model == "nine":
        seeds[7] = (1 - np.sqrt(1 - self.n)) * util.magicFrequency * 1E-3
      
      # Perform the fit.
      self.pOpt, self.pCov = opt.curve_fit(
        self.function,
        self.fitTime,
        self.fitSignal,
        sigma = self.fitError,
        p0 = seeds,
        bounds = bounds
      )
      
      # Get each parameter's uncertainty from the covariance matrix diagonal.
      self.pErr = np.sqrt(np.diag(self.pCov))
      
      # Calculate the best fit and residuals.
      self.fitResult = self.function(self.fitTime, *self.pOpt)
      fitResiduals = self.fitSignal - self.fitResult
      
      # Calculate the chi-squared, and reduced chi-squared.
      chi2 = np.sum((fitResiduals / self.fitError)**2)
      self.chi2dof = chi2 / (len(self.fitTime) - len(self.pOpt))
      
      # Status update.
      print(f"\nCompleted {model}-parameter wiggle fit.")
      print(f"{'chi2/dof':>16} = {self.chi2dof:.4f}")
      
      # Print parameter values.
      for i in range(len(modelLabels[model])):
      
        # Don't reveal omega_a.
        if modelLabels[model][i][0] == "frequency":
          continue
        
        print((
          f"{modelLabels[model][i][0]:>16} = "
          f"{self.pOpt[i]:.4f} +/- {self.pErr[i]:.4f} "
          f"{modelLabels[model][i][1]}"
        ))

# ==============================================================================
