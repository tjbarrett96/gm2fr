import numpy as np
import scipy.optimize as opt
import gm2fr.utilities as util

from gm2fr.analysis.WiggleModels import *
import matplotlib.pyplot as plt
import gm2fr.style as style
style.setStyle()

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
    n = 0.108       # the n-value, if known, helps inform CBO frequency seed
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

    self.models = []
    if model == "two":
      self.models.append(TwoParameter())
    if model == "five":
      self.models.append(TwoParameter())
      self.models.append(FiveParameter())
    if model == "nine":
      self.models.append(TwoParameter())
      self.models.append(FiveParameter())
      self.models.append(NineParameter(n))
    if len(self.models) == 0:
      raise ValueError(f"Wiggle fit model '{model}' not recognized.")
    self.model = None

    # Fit options.
    self.start = start
    self.end = end
    self.n = n

    # Fit portion of the data, with start/end mask applied.
    mask = (self.time >= self.start) & (self.time <= self.end)
    self.fitTime = self.time[mask]
    self.fitSignal = self.signal[mask]
    self.fitError = self.error[mask]

    # Fit results.
    self.fitResult = None
    self.fineResult = None

    # Initialize the structured array of results, with column headers.
    # self.results = []

  # ============================================================================

  def results(self):
    return self.model.results(prefix = "wg")

  def fit(self):

    # Iterate through the models, updating fit seeds each time.
    for model in self.models:

      # Update the new seeds with the latest fit's results.
      if self.model is not None:
        model.seeds[:len(self.model.pOpt)] = self.model.pOpt

      model.fit(self.fitTime, self.fitSignal, self.fitError)

      # Calculate the best fit and residuals.
      self.fitResult = model.eval(self.fitTime)

      # Status update.
      model.print()

      self.model = model

    # Evaluate the finely-binned fit result.
    self.fineResult = self.model.eval(self.fineTime) / 149

    # self.model.plot(wrap = 102.5)
    # plt.yscale("log")
    # plt.show()

    # self.results = self.model.results(prefix = "wg")

# ==============================================================================

  # Plot the wiggle fit.
  def plot(self, output):

    if output is not None:

      # Time used to wrap the plot around.
      modTime = 102.5

      # Calculate the indices (exclusive) at which a wrap occurs.
      breaks = np.nonzero(np.diff(self.fitTime % modTime) < 0)[0] + 1

      # Split the signal data into a list of chunks, one for each wrap.
      times = np.split(self.fitTime, breaks)
      signals = np.split(self.fitSignal, breaks)
      errors = np.split(self.fitError, breaks)

      # Plot each wrapped chunk of the data and fit.
      for i, (time, signal, error) in enumerate(zip(times, signals, errors)):
        style.errorbar(time % modTime, signal, error, c = "C0", zorder = 0)
        plt.plot(time % modTime, self.model.eval(time), c = "C1")

      # Set a logarithmic vertical scale.
      plt.yscale("log")

      # Axis labels.
      style.ylabel("Counts / 149 ns")
      style.xlabel("Time mod 102.5 us")

      # Annotate the fit quality.
      style.databox(
        (r"\chi^2/\mathrm{ndf}", self.model.chi2ndf, None, None),
        ("p", self.model.pval, None, None)
      )

      # Save and clear.
      plt.savefig(output)
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
