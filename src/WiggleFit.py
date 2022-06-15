import numpy as np
import gm2fr.src.constants as const

from gm2fr.src.WiggleModels import *
import matplotlib.pyplot as plt
import gm2fr.src.style as style
style.set_style()

# ==============================================================================

class WiggleFit:

  # ============================================================================

  def __init__(
    self,
    signal,
    model = "five", # wiggle fit model ("two" / "five" / "nine")
    start = 30,     # fit start time (us)
    end = 650,      # fit end time (us)
    n = 0.108       # the n-value, if known, helps inform CBO frequency seed
  ):

    # Finely-binned signal data for the fast rotation signal.
    self.fine_signal = signal.copy()

    # Fit options.
    self.start = start
    self.end = end
    self.n = n

    self.group = int((const.info["T"].magic * 1E-3) // self.fine_signal.width)
    self.coarse_signal = self.fine_signal.copy().rebin(self.group, discard = True).mask((self.start, self.end))

    # Ensure the errors are all non-zero.
    self.fine_signal.errors[self.fine_signal.errors == 0] = 1
    self.coarse_signal.errors[self.coarse_signal.errors == 0] = 1

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

    # Fit results.
    self.fit_result = None
    self.fine_result = None

  # ============================================================================

  def results(self):
    return self.model.results(prefix = "wg")

  def fit(self):

    # Iterate through the models, updating fit seeds each time.
    for model in self.models:

      # Update the new seeds with the latest fit's results.
      if self.model is not None:
        model.seeds[:len(self.model.p_opt)] = self.model.p_opt

      model.fit(self.coarse_signal)

      # Calculate the best fit and residuals.
      self.fit_result = model.eval(self.coarse_signal.centers)

      # Status update.
      model.print()

      self.model = model

    # Evaluate the finely-binned fit result.
    self.fine_result = self.model.eval(self.fine_signal) / self.group

# ==============================================================================

  # Plot the wiggle fit.
  def plot(self, output):

    if output is not None:

      # Time used to wrap the plot around.
      modTime = 102.5

      # Calculate the indices (exclusive) at which a wrap occurs.
      breaks = np.nonzero(np.diff(self.coarse_signal.centers % modTime) < 0)[0] + 1

      # Split the signal data into a list of chunks, one for each wrap.
      times = np.split(self.coarse_signal.centers, breaks)
      signals = np.split(self.coarse_signal.heights, breaks)
      errors = np.split(self.coarse_signal.errors, breaks)

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
        style.Entry(self.model.chi2_ndf, r"\chi^2/\mathrm{ndf}"),
        style.Entry(self.model.p_value, "p")
      )

      # Save and clear.
      plt.savefig(output)
      plt.clf()

# ==============================================================================

  # Plot the raw positron signal.
  def plot_fine(self, output, endTimes):

    if output is not None:

      # Plot the signal.
      plt.plot(self.fine_signal.centers, self.fine_signal.heights)

      self.fine_signal.plot(errors = False, skip = 10)

      # Label the axes.
      style.xlabel(r"Time ($\mu$s)")
      style.ylabel("Intensity")

      # Plot the early-time contamination.
      plt.xlim(0, 5)
      plt.ylim(0, None)

      pdf = style.make_pdf(f"{output}/RawSignal.pdf")
      pdf.savefig()

      # Save the figure over a range of time axis limits (in us).
      for end in endTimes:

        # Set the time limits.
        plt.xlim(4, end)

        # Update the intensity limits.
        view = self.fine_signal.heights[(self.fine_signal.centers >= 4) & (self.fine_signal.centers <= end)]
        plt.ylim(0, np.max(view))

        # Save the figure.
        pdf.savefig()

      # Clear the figure.
      plt.clf()
      pdf.close()
