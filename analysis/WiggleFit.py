import numpy as np
import scipy.optimize as opt
import gm2fr.utilities as util

import matplotlib.pyplot as plt
import gm2fr.style as style
style.setStyle()

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

# Parameter symbols, for saving results.
# TODO: move to utilities, like transform labels
# TODO: maybe make small class for holding each label's data
modelLabels = [
  {"printing": "normalization", "output": "N", "units": ""},
  {"printing": "lifetime", "output": "tau", "units": "us"},
  {"printing": "asymmetry", "output": "A", "units": ""},
  {"printing": "frequency", "output": "f_a", "units": "1/us"},
  {"printing": "phase", "output": "phi_a", "units": "rad"},
  {"printing": "CBO lifetime", "output": "tau_cbo", "units": "us"},
  {"printing": "CBO asymmetry", "output": "A_cbo", "units": ""},
  {"printing": "CBO frequency", "output": "f_cbo", "units": "1/us"},
  {"printing": "CBO phase", "output": "phi_cbo", "units": "rad"}
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
    [0,             00.0    ],
    [np.inf,        64.9    ]
) # [normalization, lifetime]

modelBounds["five"] = (
  modelBounds["two"][0] + [-0.3,      0.2288,    0      ],
  modelBounds["two"][1] + [ 1.0,      0.2292,    2*np.pi]
) #          (precession) [asymmetry, frequency, phase  ]

modelBounds["nine"] = (
  modelBounds["five"][0] + [0,        0   ,      0.350,     0      ],
  modelBounds["five"][1] + [500,      0.01,      0.430,     2*np.pi]
) #                  (CBO) [lifetime, asymmetry, frequency, phase  ]

# ==============================================================================

class WiggleFit:

  # ============================================================================

  def __init__(
    self,
    time,           # time bin centers
    signal,         # positron counts per time bin
    model = "five", # wiggle fit model ("two" / "five" / "nine")
    start = 30,     # fit start time (us)
    end = 650,      # fit end time (us)
    n = None        # the n-value, if known, helps inform CBO frequency seed
  ):

    # Finely-binned signal data for the fast rotation signal.
    self.fineTime = time
    self.fineSignal = signal
    self.fineError = np.sqrt(np.abs(self.fineSignal))

    # Coarsely-binned signal data for the wiggle fit.
    groups = len(self.fineTime) // 149
    self.time = self.fineTime[:(groups * 149)].reshape((groups, 149)).mean(axis = 1)
    self.signal = self.fineSignal[:(groups * 149)].reshape((groups, 149)).sum(axis = 1)
    self.error = np.sqrt(np.abs(self.signal))

    # Ensure the errors are all non-zero.
    self.fineError[self.fineError == 0] = 1
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
    self.fineResult = None

    self.chi2 = None
    self.ndf = None
    self.chi2ndf = None
    self.pval = None

    # Initialize the structured array of results, with column headers.
    # TODO: add uncertainties, chi2/dof, and n-value
    self.results = []

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
        seeds[7] = (1 - np.sqrt(1 - self.n)) * util.magic["f"] * 1E-3

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

      # Calculate the chi-squared, reduced chi-squared, and p-value.
      self.chi2 = np.sum((fitResiduals / self.fitError)**2)
      self.ndf = len(self.fitTime) - len(self.pOpt)
      self.chi2ndf = self.chi2 / self.ndf
      self.pval = util.pval(self.chi2, self.ndf)

      # Status update.
      print(f"\nCompleted {model}-parameter wiggle fit.")
      print(f"{'chi2/ndf':>16} = {self.chi2ndf:.4f}")
      print(f"{'p-value':>16} = {self.pval:.4f}")

      # Print and save parameter values.
      for i in range(len(self.pOpt)):

        # Don't reveal omega_a.
        if modelLabels[i]["printing"] == "frequency":
          continue

        print((
          f"{modelLabels[i]['printing']:>16} = "
          f"{self.pOpt[i]:.4f} +/- {self.pErr[i]:.4f} "
          f"{modelLabels[i]['units']}"
        ))

    # Evaluate the finely-binned fit result.
    self.fineResult = self.function(self.fineTime, *self.pOpt) / 149

    # Copy the parameters into the results array.
    for i in range(len(self.pOpt)):
      if modelLabels[i]["printing"] == "frequency":
        continue
      self.results.append((f"wg_{modelLabels[i]['output']}", self.pOpt[i]))
    self.results.append(("wg_chi2", self.chi2))
    self.results.append(("wg_ndf", self.ndf))
    self.results.append(("wg_chi2ndf", self.chi2ndf))
    self.results.append(("wg_pval", self.pval))

# ==============================================================================

  # Plot the wiggle fit.
  def plot(self, output, endTimes):

    if output is not None:

      # Plot the signal.
      plt.plot(self.time, self.signal, label = "Signal")
      plt.plot(self.fitTime, self.fitResult, 'r', label = "Fit")

      # Label the axes.
      style.xlabel(r"Time ($\mu$s)")
      style.ylabel("Intensity")
      plt.legend()

      # Save the figure over a range of time axis limits (in us).
      for end in endTimes:

        # Set the time limits.
        plt.xlim(4, end)

        # Update the intensity limits.
        view = self.signal[(self.time >= 4) & (self.time <= end)]
        plt.ylim(np.min(view), np.max(view))

        # Save the figure.
        plt.savefig(f"{output}/signal/WiggleFit_{end}us.pdf")

      # Clear the figure.
      plt.clf()

# ==============================================================================

  # Plot the raw positron signal.
  def plotFine(self, output, endTimes):

    if output is not None:

      # Plot the signal.
      plt.plot(self.fineTime, self.fineSignal)

      # Label the axes.
      style.xlabel(r"Time ($\mu$s)")
      style.ylabel("Intensity")

      # Plot the early-time contamination.
      plt.xlim(0, 5)
      plt.savefig(f"{output}/signal/EarlyTime.pdf")

      # Save the figure over a range of time axis limits (in us).
      for end in endTimes:

        # Set the time limits.
        plt.xlim(4, end)

        # Update the intensity limits.
        view = self.fineSignal[(self.fineTime >= 4) & (self.fineTime <= end)]
        plt.ylim(np.min(view), np.max(view))

        # Save the figure.
        plt.savefig(f"{output}/signal/RawSignal_{end}us.pdf")

      # Clear the figure.
      plt.clf()

# ==============================================================================

  # Plot an FFT of the raw positron signal.
  def plotFFT(self, output):

    # Calculate the FFT magnitude.
    f, fft = util.fft(self.fineTime, self.fineSignal)
    mag = np.abs(fft)

    # Plot the FFT magnitude.
    plt.plot(f, mag)

    # Show the frequency region used for nominal fast rotation analysis.
    plt.axvspan(util.min["f"], util.max["f"], alpha = 0.2, fc = "k", ec = None, label = "Cyclotron Region")

    # Axis limits.
    plt.xlim(0, 8000)
    plt.ylim(0, np.max(mag[(f > 1000)]) * 1.05)

    style.xlabel("Frequency (kHz)")
    style.ylabel("Arbitrary Units")
    plt.legend()

    plt.savefig(f"{output}/fft_raw.pdf")

    # Save with a log scale.
    plt.yscale("log")
    plt.ylim(np.min(mag[(f < 8000)]), None)
    plt.savefig(f"{output}/fft_raw_log.pdf")

    # Clear the figure.
    plt.clf()
