import numpy as np
import matplotlib.pyplot as plt

import gm2fr.io as io
import gm2fr.constants as const
import gm2fr.calculations as calc
import gm2fr.style as style
from gm2fr.analysis.Transform import Transform
from gm2fr.Histogram1D import Histogram1D
from gm2fr.analysis.Optimizer import Optimizer
from gm2fr.analysis.Results import Results
from gm2fr.analysis.WiggleFit import WiggleFit
from gm2fr.analysis.BackgroundFit import BackgroundFit
from gm2fr.analysis.Iterator import Iterator
from gm2fr.analysis.Corrector import Corrector

import ROOT as root

# Filesystem management.
import os
import gm2fr.analysis
import time
import inspect
import itertools

# ==================================================================================================

# Organizes the high-level logic of the fast rotation analysis.
class Analyzer:

  # ================================================================================================

  # Constructor.
  def __init__(
    self,
    filename, # Filename containing data to analyze.
    signal_label = None, # Label for signal data inside each file.
    pileup_label = None, # Label for pileup data inside file.
    output_label = None, # Output directory name, within gm2fr/analysis/results.
    truth_filename = None, # Truth .npz file from gm2fr simulation.
    fr_method = None,
    n = 0.108,
    time_units = 1E-6
  ):

    self.filename = filename
    self.signal_label = signal_label
    self.pileup_label = pileup_label
    self.output_label = output_label
    self.truth_filename = truth_filename
    self.n = n
    self.time_units = time_units

    self.output_path = f"{io.results_path}/{output_label}" if output_label is not None else None
    if self.output_path is not None:
      io.make_if_absent(self.output_path)

    self.raw_signal = None
    self.wiggle_fit = None
    self.fr_signal = None
    self.load_fr_signal(fr_method)

    self.results = Results()
    self.transform = None

  # ================================================================================================

  def load_fr_signal(self, method):

    self.raw_signal = Histogram1D.load(self.filename, self.signal_label)
    if self.pileup_label is not None:
      try:
        self.raw_signal.heights -= Histogram1D.load(self.filename, self.pileup_label).heights
      except:
        print(f"\nWarning: could not load pileup histogram; continuing without pileup correction.")

    # Convert time units to microseconds.
    self.raw_signal.map(lambda t: t * self.time_units / 1E-6)

    # Create the fast rotation signal using the ratio method.
    if method == "ratio":

      numerator = Histogram1D.load(self.filename, f"{self.signal_label}_Num")
      denominator = Histogram1D.load(self.filename, f"{self.signal_label}_Den")
      if self.pileup_label is not None:
        try:
          numerator.heights -= Histogram1D.load(self.filename, f"{self.pileup_label}_Num").heights
          denominator.heights -= Histogram1D.load(self.filename, f"{self.pileup_label}_Den").heights
        except:
          print(f"\nWarning: could not load pileup histogram; continuing without pileup correction.")
      self.fr_signal = numerator.divide(denominator, zero = 1)

      # Convert time units to microseconds.
      self.fr_signal.map(lambda t: t * self.time_units / 1E-6)

    # Create the fast rotation signal using a wiggle fit.
    elif method in ("two", "five", "nine"):

      self.wiggle_fit = WiggleFit(self.raw_signal, model = method, n = self.n)
      self.wiggle_fit.fit()
      self.fr_signal = self.raw_signal.divide(self.wiggle_fit.fine_result)

    # The signal is already a fast rotation signal, and requires no further processing.
    elif method is None:
      self.fr_signal = self.raw_signal

    else:
      raise ValueError(f"Fast rotation signal production method '{method}' not recognized.")

  # ================================================================================================

  def analyze(
    self,
    start = 4, # Start time (us) for the cosine transform.
    end = 250, # End time (in us) for the cosine transform.
    t0 = None, # t0 time (in us) for the cosine transform.
    iterate = False,
    bg_model = "sinc", # Background fit model: "constant" / "parabola" / "sinc" / "error".
    df = 2, # Frequency interval (in kHz) for the cosine transform.
    coarse_t0_width = 20, # full range (in ns) for the initial coarse t0 scan range.
    coarse_t0_steps = 5, # Number of steps for the initial coarse t0 scan range.
    fine_t0_width = 1, # full range (in ns) for the subsequent fine t0 scan ranges.
    fine_t0_steps = 10, # Number of steps for the subsequent fine t0 scan ranges.
    plot_level = 1, # Plotting option. 0 = nothing, 1 = main plots, 2 = more plots (slower).
    save_output = True,
    rebin = 1 # Integer rebinning group size for the fast rotation signal.
  ):

    begin_time = time.time()

    fr_signal_masked = self.fr_signal.copy().mask((start, end))

    self.transform = Transform(fr_signal_masked, df)

    coarse_t0_optimizer = None
    fine_t0_optimizer = None
    optimize_t0 = (t0 is None)

    if optimize_t0:

      coarse_t0_optimizer = Optimizer(self.transform, bg_model, coarse_t0_width * 1E-3, coarse_t0_steps)
      t0 = coarse_t0_optimizer.optimize()

      fine_t0_optimizer = Optimizer(self.transform, bg_model, fine_t0_width * 1E-3, fine_t0_steps, seed = t0)
      t0 = fine_t0_optimizer.optimize()

      if save_output and plot_level > 0:
        fine_t0_optimizer.plot_chi2(f"{self.output_path}/BackgroundChi2.pdf")

    self.transform.set_t0(t0, fine_t0_optimizer.err_t0 if fine_t0_optimizer is not None else 0)
    corr_transform = self.transform.optCosine.copy()

    self.bg_fit = None
    if bg_model is not None:
      self.bg_fit = BackgroundFit(self.transform.optCosine, t0, start, bg_model).fit()
      corr_transform = self.transform.optCosine.subtract(self.bg_fit.result)

      iterator = None
      if iterate:
        iterator = Iterator(self.transform, self.bg_fit)
        corr_transform = iterator.iterate(optimize_t0)
        if save_output and plot_level > 0:
          iterator.plot(f"{self.output_path}/Iterations.pdf")

    corrector = None
    if self.truth_filename is not None:
      corrector = Corrector(self.transform, corr_transform, self.truth_filename if self.truth_filename != "same" else self.filename)
      corrector.correct()
      if save_output and plot_level > 0:
        corrector.plot(self.output_path)

    # Compile results.
    results = Results({"start": start, "end": end, "df": df, "t0": t0, "err_t0": fine_t0_optimizer.err_t0 if fine_t0_optimizer is not None else 0})
    histograms = dict()

    output_variables = ["f", "x", "dp_p0", "tau", "gamma", "c_e"]
    for unit in output_variables:
      histograms[unit] = corr_transform.copy().map(const.info[unit].from_frequency)
      mean, mean_err = histograms[unit].mean(error = True)
      std, std_err = histograms[unit].std(error = True)
      results.merge(Results(
        {unit: mean, f"err_{unit}": mean_err, f"sig_{unit}": std, f"err_sig_{unit}": std_err}
      ))

    if self.bg_fit is not None:
      results.merge(self.bg_fit.results())
    if self.wiggle_fit is not None:
      results.merge(self.wiggle_fit.results())

    self.results.append(results)

    # Save the results to disk.
    if save_output and self.output_path is not None:

      # Plot the fast rotation signal, and wiggle fit (if present).
      if plot_level >= 1:

        self.fr_signal.save(f"{self.output_path}/signal.npz")
        self.fr_signal.save(f"{self.output_path}/signal.root", "signal")

        pdf = style.make_pdf(f"{self.output_path}/FastRotation.pdf")
        endTimes = [5, 100, 300]
        for endTime in endTimes:
          self.fr_signal.plot(errors = False, start = 4, end = endTime, skip = int(np.clip(endTime - 4, 1, 10)))
          if endTime - 4 > 10:
            plt.xlim(0, None)
          style.label_and_save(r"Time ($\mu$s)", "Arbitrary Units", pdf)
        pdf.close()

        if self.wiggle_fit is not None:
          self.wiggle_fit.plot(f"{self.output_path}/WiggleFit.pdf")
          self.wiggle_fit.plot_fine(self.output_path, endTimes)
          calc.plot_fft(self.raw_signal.centers, self.raw_signal.heights, f"{self.output_path}/RawSignalFFT.pdf")

      plotbegin_time = time.time()

      # Determine which units to plot a distribution for.
      axesToPlot = []
      if plot_level > 0:
        axesToPlot = ["f", "x", "dp_p0"]
        if plot_level > 1:
          axesToPlot = output_variables.copy()
          axesToPlot.remove("c_e")

      pdf = style.make_pdf(f"{self.output_path}/AllDistributions.pdf")
      rootFile = root.TFile(f"{self.output_path}/transform.root", "RECREATE")

      # Compile the results list of (name, value) pairs from each object.

      for unit in axesToPlot:
        # Plot the truth-level distribution for comparison, if present.
        # if truth is not None:
        #   ref_predicted.plot(label = "Predicted")
        histograms[unit].to_root(f"transform_{unit}", f";{const.info[unit].formatLabel()};").Write()
        histograms[unit].plot(label = None if self.truth_filename is None else "Result")
        plt.axvline(const.info[unit].magic, ls = ":", c = "k", label = "Magic")
        style.draw_horizontal()
        style.databox(
          style.Entry(mean, rf"\langle {const.info[unit].symbol} \rangle", mean_err, const.info[unit].units),
          style.Entry(std, rf"\sigma_{{{const.info[unit].symbol}}}", std_err, const.info[unit].units)
        )
        plt.xlim(const.info[unit].min, const.info[unit].max)
        style.label_and_save(const.info[unit].formatLabel(), "Arbitrary Units", pdf)

      pdf.close()
      rootFile.Close()

      # Include the differences from the reference distribution, if provided.
      # if truth_results is not None:
      #   diff_results = truth_results.copy()
      #   columnsToDrop = [x for x in diff_results.table.columns if "err_" in x]
      #   diff_results.table.drop(columns = columnsToDrop, inplace = True)
      #   # Set the ref_results column data to the difference from the results.
      #   for (name, data) in diff_results.table.iteritems():
      #     diff_results.table[name] = results.table[name] - diff_results.table[name]
      #   # Change the column names with "diff" prefix.
      #   diff_results.table.columns = [f"diff_{name}" for name in diff_results.table.columns]
      #   results.merge(diff_results)

      self.results.save(self.output_path)

      if plot_level > 0:

        if corrector is not None:
          corrector.truth_frequency.plot(errors = False, label = "Truth")
        self.transform.optCosine.plot(label = "Cosine Transform")
        self.transform.optSine.plot(label = "Sine Transform")
        self.transform.magnitude.plot(label = "Fourier Magnitude")
        style.draw_horizontal()
        plt.axvline(const.info["f"].magic, ls = ":", c = "k", label = "Magic")
        style.label_and_save("Frequency (kHz)", "Arbitrary Units", f"{self.output_path}/magnitude.pdf")

        calc.plot_fft(fr_signal_masked.centers, fr_signal_masked.heights, f"{self.output_path}/FastRotationFFT.pdf")

        # Plot the final background fit.
        if self.bg_fit is not None:
          self.bg_fit.plot(f"{self.output_path}/BackgroundFit.pdf")

        # if self.newBGFit is not None:
        #   self.newBGFit.plot(f"{self.output_path}/TemplateFit.pdf")

          # corr_transform.bgFit.save(f"{self.output_path}/background.npz")

        corr_transform.save(f"{self.output_path}/transform.npz")
        # corr_transform.save(f"{self.output_path}/transform.root", "transform")

        print(f"\nFinished plotting and saving results in {time.time() - plotbegin_time:.2f} seconds.")

    print(f"\nCompleted {self.output_label} in {time.time() - begin_time:.2f} seconds.")

  # ================================================================================================

  def scan_parameters(self, **parameters):

    # Ensure parameters are valid, and values are iterable objects.
    analyzer_args = inspect.getargspec(self.analyze).args
    if not all(parameter in analyzer_args for parameter in parameters):
      print("Scan parameter(s) unrecognized.")
      return
    if not all(io.is_iterable(value) for value in parameters.values()):
      print("Scan parameter values are not iterable.")
      return

    for parameter_set in itertools.product(*parameters.values()):
      self.analyze(
        **{parameter: parameter_set[i] for i, parameter in enumerate(parameters)},
        save_output = False
      )

    if self.output_path is not None:
      self.results.save(self.output_path)
