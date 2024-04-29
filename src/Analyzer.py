import numpy as np
import matplotlib.pyplot as plt

import gm2fr.src.io as io
import gm2fr.src.constants as const
import gm2fr.src.calculations as calc
import gm2fr.src.style as style
from gm2fr.src.Transform import Transform
from gm2fr.src.Histogram1D import Histogram1D
from gm2fr.src.Optimizer import Optimizer
from gm2fr.src.Results import Results
from gm2fr.src.WiggleFit import WiggleFit
from gm2fr.src.BackgroundFit import BackgroundFit
from gm2fr.src.Iterator import Iterator
from gm2fr.src.Corrector import Corrector

import ROOT as root

# import warnings
# warnings.filterwarnings("error")

# Filesystem management.
import os
import gm2fr.src
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
    filename = None, # Filename containing the fast rotation histogram to analyze.
    signal_label = None, # Label for signal histogram inside file.
    pileup_label = None, # Label for pileup histogram inside file.
    output_label = None, # Output directory name, to be created within gm2fr/results.
    output_prefix = "", # Prefix prepended to default output filenames.
    ref_filename = None, # Truth .npz file from gm2fr simulation.
    ref_t0 = None, # t0 time for reference signal.
    n = 0.108, # Expected n-value for CBO frequency fit seed.
    time_units = 1E-6 # Units of FR signal times relative to seconds.
  ):

    self.filename = filename
    self.signal_label = signal_label
    self.pileup_label = pileup_label
    self.output_label = output_label
    self.output_prefix = output_prefix
    self.ref_filename = ref_filename
    self.ref_t0 = ref_t0
    self.n = n
    self.time_units = time_units

    self.output_path = f"{io.results_path}/{output_label}" if output_label is not None else None
    if self.output_path is not None:
      io.make_if_absent(self.output_path)

    self.raw_signal = None
    self.wiggle_fit = None
    self.fr_signal = None
    self.fr_method = None
    self.fr_rebin = 1

    self.transform = None
    self.coarse_t0_optimizer = None
    self.fine_t0_optimizer = None
    self.bg_fit = None
    self.bg_iterator = None
    self.corrector = None

    self.converted_transforms = dict()
    self.converted_ref_distributions = dict()
    self.converted_corrected_transforms = dict()

    self.results = Results()
    self.transforms = []
    self.bg_fits = []

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
    t0_seed = None, # initial guess for t0 time
    err_t0 = 0, # error to assign to t0 when fixed t0 is supplied
    iterate = False, # perform empirical iteration procedure for the background fit
    fr_method = None, # wiggle fit option or ratio method: "two" / "five" / "nine" / "ratio"
    bg_model = "sinc", # Background fit model: "constant" / "parabola" / "sinc" / "error".
    df = 2, # Frequency interval (in kHz) for the cosine transform.
    harmonic = 1, # Which harmonic of the Fourier transform to evaluate.
    bg_space = None, # Space on either side of f_magic before background boundaries. Default is +/- 43 kHz (from collimator limits).
    bg_width = 33, # Extra interval (in kHz) beyond background boundaries for background fit data.
    coarse_t0_width = 20, # full range (in ns) for the initial coarse t0 scan range.
    coarse_t0_steps = 5, # Number of steps for the initial coarse t0 scan range.
    fine_t0_width = 1, # full range (in ns) for the subsequent fine t0 scan ranges.
    fine_t0_steps = 10, # Number of steps for the subsequent fine t0 scan ranges.
    plot_level = 1, # Plotting option. 0 = nothing, 1 = main plots, 2 = more plots (slower).
    save_output = True,
    rebin = 1, # Integer rebinning group size for the fast rotation signal.
    tweak = False
  ):

    begin_time = time.time()

    if self.fr_signal is None or fr_method != self.fr_method or rebin != self.fr_rebin:
      rebin = int(rebin)
      self.fr_method = fr_method
      self.fr_rebin = rebin
      self.load_fr_signal(fr_method)
      self.fr_signal.rebin(rebin, discard = True)

    if bg_space is None:
      bg_space = int(round(const.info["f"].max - const.info["f"].magic))

    # Compute the Fourier transform of the fast rotation signal, masked between the requested times.
    self.transform = Transform(self.fr_signal, start, end, df, bg_space + bg_width, harmonic)

    # Determine whether or not to optimize t0. If t0 value supplied, then use it; otherwise, optimize.
    optimize_t0 = (t0 is None)

    if optimize_t0:

      self.coarse_t0_optimizer = Optimizer(self.transform, bg_model, coarse_t0_width * 1E-3, coarse_t0_steps, seed = t0_seed, bg_space = bg_space)
      t0 = self.coarse_t0_optimizer.optimize()

      self.fine_t0_optimizer = Optimizer(self.transform, bg_model, fine_t0_width * 1E-3, fine_t0_steps, seed = t0, bg_space = bg_space)
      t0 = self.fine_t0_optimizer.optimize()
      err_t0 = self.fine_t0_optimizer.err_t0

    # Set the t0 value for the Fourier transform.
    self.transform.set_t0(t0, err_t0)

    # Copy the optimal cosine transform before the background correction.
    distcorr_transform = self.transform.opt_cosine.copy()

    # TODO: if truth supplied, subtract distortion term HERE, before background fit. then fit, then divide A(f).
    # this should enable a better background fit!
    # If reference data is supplied, calculate and apply corrections.
    if self.ref_filename is not None:
      self.corrector = Corrector(self.transform, distcorr_transform, self.ref_filename if self.ref_filename != "same" else self.filename, self.ref_t0, tweak)
      self.corrector.correct(peak = False, distortion = False, background = False)
      distcorr_transform = distcorr_transform.subtract(self.corrector.distortion.interpolate(distcorr_transform.centers))
      # self.transform.opt_cosine = self.transform.opt_cosine.subtract(self.corrector.distortion.interpolate(corr_transform.centers))

    # Copy the optimal cosine transform before the background correction.
    corr_transform = self.transform.opt_cosine.copy()

    # Perform the background fit, and subtract it from the optimal cosine transform.
    if bg_model is not None:

      # nominal background fit
      self.bg_fit = BackgroundFit(self.transform, bg_model, bg_space = bg_space)
      self.bg_fit.fit()
      corr_transform = self.transform.opt_cosine.subtract(self.bg_fit.result)

      # background fit on distortion-subtracted transform
      if self.ref_filename is not None:
        dist_bg_fit = BackgroundFit(self.transform, bg_model, bg_space = bg_space)
        dist_bg_fit.cosine_histogram = distcorr_transform
        dist_bg_fit.update_data()
        dist_bg_fit.fit()
        distcorr_transform = distcorr_transform.subtract(dist_bg_fit.result)
      # corr_transform = corr_transform.subtract(self.bg_fit.result)
      # If enabled, perform empirical iteration of the background fit.
      # if iterate:
      #   self.bg_iterator = Iterator(self.transform, self.bg_fit)
      #   corr_transform = self.bg_iterator.iterate(optimize_t0)
      #   self.bg_fit = self.bg_iterator.fits[-1]

    # If reference data is supplied, calculate and apply corrections.
    if self.ref_filename is not None:
      self.corrector = Corrector(self.transform, distcorr_transform, self.ref_filename if self.ref_filename != "same" else self.filename, self.ref_t0, tweak, plain_cosine = corr_transform)
      self.corrector.correct(distortion = False, background = (bg_model is None))

    n_data = None
    if self.wiggle_fit is not None and len(self.wiggle_fit.model.p_opt) >= 8:
      n_data = 1 - (1 - 1E3 * self.wiggle_fit.model.p_opt[7] / const.info["f"].magic)**2

    # Compile results.
    results = Results({
      "start": start,
      "end": end,
      "df": df,
      "harmonic": harmonic,
      "t0": t0,
      "err_t0": err_t0,
      "bg_model": bg_model,
      "bg_width": bg_width,
      "bg_space": bg_space,
      "fr_method": self.fr_method,
      "dt": np.mean(self.fr_signal.width),
      "n_used": self.n,
      "n_data": n_data
    })

    # Convert final transform to other units.
    output_variables = ["f", "x", "dp_p0", "T", "tau", "gamma"]
    masked_transform = corr_transform.copy().mask((const.info["f"].min, const.info["f"].max))
    for unit in output_variables:
      self.converted_transforms[unit] = masked_transform.copy().map(const.info[unit].from_frequency)

    # Convert reference and corrected distributions, if they exist.
    if self.corrector is not None:
      masked_ref_distribution = self.corrector.ref_frequency.copy().mask((const.info["f"].min, const.info["f"].max))
      masked_corr_transform = self.corrector.corrected_transform.copy().mask((const.info["f"].min, const.info["f"].max))
      for unit in output_variables:
        self.converted_ref_distributions[unit] = masked_ref_distribution.copy().map(const.info[unit].from_frequency)
        self.converted_corrected_transforms[unit] = masked_corr_transform.copy().map(const.info[unit].from_frequency)

    def add_all_transform_results(transform_dict, prefix = None):
      prefix = "" if prefix is None else f"{prefix}_"
      for unit, transform in transform_dict.items():
        mean, mean_err = transform.mean(error = True)
        std, std_err = transform.std(error = True)
        results.merge(
          Results({
            f"{prefix}{unit}": mean,
            f"{prefix}err_{unit}": mean_err,
            f"{prefix}sig_{unit}": std,
            f"{prefix}err_sig_{unit}": std_err
          })
        )
        if unit == "x":
          avg_x2, err_avg_x2 = transform.moment(2, central = False, error = True)
          c_e = 2*self.n*(1-self.n)*(const.info["beta"].magic/const.info["r"].magic)**2*avg_x2*1E9
          results.merge(
            Results({
              f"{prefix}c_e": c_e,
              f"{prefix}err_c_e": (c_e / avg_x2) * err_avg_x2
            })
          )

    add_all_transform_results(self.converted_transforms)
    add_all_transform_results(self.converted_ref_distributions, prefix = "ref")
    add_all_transform_results(self.converted_corrected_transforms, prefix = "corr")

    if self.bg_fit is not None:
      results.merge(self.bg_fit.results())
    if self.wiggle_fit is not None:
      results.merge(self.wiggle_fit.results())

    self.results.append(results)
    self.transforms.append(self.converted_transforms["f"])
    self.bg_fits.append(self.bg_fit)

    # Save the results to disk.
    if save_output:
      self.save()
      self.plot(plot_level)

    print(f"\nCompleted {self.output_label} in {time.time() - begin_time:.2f} seconds.")

  # ================================================================================================

  def save(self):

    if self.output_path is not None:

      self.fr_signal.save(f"{self.output_path}/{self.output_prefix}signal.npz")
      self.fr_signal.save(f"{self.output_path}/{self.output_prefix}signal.root", "signal")

      root_file = root.TFile(f"{self.output_path}/{self.output_prefix}transform.root", "RECREATE")
      numpy_dict = dict()

      for unit, transform in self.converted_transforms.items():
        transform.to_root(f"transform_{unit}", f";{const.info[unit].format_label()};").Write()
        numpy_dict.update(transform.collect(f"transform_{unit}"))

      for unit, transform in self.converted_ref_distributions.items():
        transform.to_root(f"ref_transform_{unit}", f";{const.info[unit].format_label()};").Write()
        numpy_dict.update(transform.collect(f"ref_transform_{unit}"))

      for unit, transform in self.converted_corrected_transforms.items():
        transform.to_root(f"corr_transform_{unit}", f";{const.info[unit].format_label()};").Write()
        numpy_dict.update(transform.collect(f"corr_transform_{unit}"))

      self.transform.opt_cosine.to_root("opt_cosine", ";Frequency (kHz)").Write()
      self.bg_fit.result.to_root("bg_fit", ";Frequency (kHz)").Write()

      np.savez(f"{self.output_path}/{self.output_prefix}transform.npz", **numpy_dict)
      root_file.Close()

      self.results.save(self.output_path, filename = f"{self.output_prefix}results")

  # ================================================================================================

  def plot(self, level = 1):

    if level > 0 and (self.output_path is not None):

      begin_time = time.time()

      pdf = style.make_pdf(f"{self.output_path}/{self.output_prefix}AllDistributions.pdf")

      if level == 1:
        units_to_plot = ["f", "x", "dp_p0"]
      else:
        units_to_plot = self.converted_transforms.keys()

      for unit, transform in self.converted_transforms.items():
        if unit not in units_to_plot:
          continue
        transform.plot()
        plt.axvline(const.info[unit].magic, ls = ":", c = "k", label = "Magic")
        style.draw_horizontal()
        style.databox(
          style.Entry(self.results.get(unit, -1), rf"\langle {const.info[unit].symbol} \rangle", self.results.get(f"err_{unit}", -1), const.info[unit].units),
          style.Entry(self.results.get(f"sig_{unit}", -1), rf"\sigma_{{{const.info[unit].symbol}}}", self.results.get(f"err_sig_{unit}", -1), const.info[unit].units)
        )
        style.set_physical_limits(unit)
        style.label_and_save(const.info[unit].format_label(), "Arbitrary Units", pdf)

      pdf.close()

      # print(f"Finished plotting distributions in {time.time() - begin_time:.2f} seconds.")
      # begin_time = time.time()

      # ~1.3 seconds
      pdf = style.make_pdf(f"{self.output_path}/{self.output_prefix}FastRotation.pdf")
      endTimes = [5, 100, 300]
      for endTime in endTimes:
        self.fr_signal.plot(errors = False, start = 4, end = endTime, skip = int(np.clip(endTime - 4, 1, 10)))
        if endTime - 4 > 10:
          plt.xlim(0, None)
        style.label_and_save(r"Time ($\mu$s)", "Arbitrary Units", pdf)
      pdf.close()

      # print(f"Finished plotting FR signal in {time.time() - begin_time:.2f} seconds.")
      # begin_time = time.time()

      if self.wiggle_fit is not None:
        #slowest: ~2 seconds
        self.wiggle_fit.plot(f"{self.output_path}/{self.output_prefix}WiggleFit.pdf")
        #slow: ~1.5 seconds
        self.wiggle_fit.plot_fine(self.output_path, endTimes)
        #fast: ~0.5 seconds
        calc.plot_fft(self.raw_signal.centers, self.raw_signal.heights, f"{self.output_path}/{self.output_prefix}RawSignalFFT.pdf")

      # print(f"Finished plotting wiggle fit in {time.time() - begin_time:.2f} seconds.")
      # begin_time = time.time()

      # <1 second
      if self.fine_t0_optimizer is not None:
        pdf = style.make_pdf(f"{self.output_path}/{self.output_prefix}BackgroundChi2.pdf")
        self.coarse_t0_optimizer.plot_chi2(pdf)
        self.fine_t0_optimizer.plot_chi2(pdf)
        pdf.close()

      # print(f"Finished plotting optimizers in {time.time() - begin_time:.2f} seconds.")
      # begin_time = time.time()

      if self.bg_iterator is not None:
        self.bg_iterator.plot(f"{self.output_path}/{self.output_prefix}Iterations.pdf")

      if self.corrector is not None:
        self.corrector.plot(self.output_path)

      # ~0.5 seconds
      style.draw_horizontal()
      if self.corrector is not None:
        self.corrector.ref_frequency.plot(label = "Truth", color = "k", ls = "--", scale = self.transform.opt_cosine)
      self.transform.opt_cosine.plot(label = "Cosine Transform")
      self.transform.opt_sine.plot(label = "Sine Transform")
      self.transform.magnitude.plot(label = "Fourier Magnitude")
      plt.axvline(const.info["f"].magic, ls = ":", c = "k", label = "Magic")
      style.label_and_save("Frequency (kHz)", "Arbitrary Units", f"{self.output_path}/{self.output_prefix}FourierMagnitude.pdf")

      # print(f"Finished plotting FFT magnitude in {time.time() - begin_time:.2f} seconds.")

      calc.plot_fft(self.transform.signal.centers, self.transform.signal.heights, f"{self.output_path}/{self.output_prefix}FastRotationFFT.pdf")

      # Plot the final background fit.
      if self.bg_fit is not None:
        self.bg_fit.plot(f"{self.output_path}/{self.output_prefix}BackgroundFit.pdf")

      if level > 1 and self.coarse_t0_optimizer is not None:
        self.coarse_t0_optimizer.plot_fits(f"{self.output_path}/{self.output_prefix}CoarseOptimizerFits.pdf")

  # ================================================================================================

  def scan_parameters(self, **parameters):

    # Ensure parameters are valid, and values are iterable objects.
    analyzer_args = inspect.getfullargspec(self.analyze).args
    if not all(parameter in analyzer_args for parameter in parameters):
      print("Scan parameter(s) unrecognized.")
      return
    if not all(io.is_iterable(value) for value in parameters.values()):
      print("Scan parameter values are not iterable.")
      return

    # Calculate how many steps there will be, as the product of all parameter list lengths.
    num_iterations = 1
    for parameter_list in parameters.values():
      num_iterations *= len(parameter_list)

    # Form an iterator that produces all combinations of input parameter values.
    parameter_sets = itertools.product(*parameters.values())

    for i, parameter_set in enumerate(parameter_sets):
      print(f"\nWorking on step {i + 1} of {num_iterations}.")
      
      kwargs = {parameter: parameter_set[i] for i, parameter in enumerate(parameters)}
      if "bg_space" in kwargs and "bg_width" in kwargs:
        if kwargs["bg_space"] + kwargs["bg_width"] > 250:
          continue
      
      self.analyze(
        **kwargs,
        save_output = False,
        # if the analyzer already knows a t0 seed, re-use it (it won't change with input parameters)
        t0_seed = self.coarse_t0_optimizer.seed if self.coarse_t0_optimizer is not None else None
      )

    if self.output_path is not None:

      self.results.save(self.output_path, filename = f"{self.output_prefix}results")

      numpy_dict = dict()
      for i, transform in enumerate(self.transforms):
        numpy_dict.update(transform.collect(f"index_{i}"))
      np.savez(f"{self.output_path}/{self.output_prefix}transforms.npz", **numpy_dict)

      bg_pdf = style.make_pdf(f"{self.output_path}/{self.output_prefix}BackgroundFits.pdf")
      for bg_fit in self.bg_fits:
        if bg_fit is not None:
          bg_fit.plot(bg_pdf)
      bg_pdf.close()
