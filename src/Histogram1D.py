import numpy as np
import array
import scipy.interpolate as interp
import scipy.stats as stats
import scipy.integrate

import matplotlib.pyplot as plt
import gm2fr.src.style as style
style.set_style()
import gm2fr.src.io as io
import gm2fr.src.calculations as calc

import ROOT as root
import root_numpy as rnp

# ==================================================================================================

class Histogram1D:

  # ================================================================================================

  def __init__(self, bins, range = None, heights = None, cov = None):

    # Binning parameters for internal usage of np.histogram.
    self.bins, self.range = None, None

    # Initialize the histogram bin variables.
    self.edges, self.centers, self.width, self.length = None, None, None, None
    self.heights, self.errors, self.cov = None, None, None

    self.edges = Histogram1D.parse_binning(bins, range)
    self.update_bins()

    # Determine the shape for the bin height and error arrays.
    shape = (self.length,)

    self.pdf = None
    self.spline = None

    # Initialize the bin heights: from an argument if supplied, or else zeros.
    if heights is not None:
      if heights.shape == shape:
        self.heights = heights.copy()
      else:
        raise ValueError(f"Supplied histogram heights do not match bin shape.")
    else:
      self.heights = np.zeros(shape)

    # Initialize the covariance matrix: from an argument if supplied, else zeros.
    if cov is not None:
      if cov.shape == (self.length, self.length) or cov.shape == shape:
        self.cov = cov.copy()
      else:
        raise ValueError(f"Covariance matrix does not match the size of the histogram.")
    else:
      self.cov = np.zeros(shape)
    self.update_errors()

  # ================================================================================================

  @staticmethod
  def parse_binning(bins, range):
    if io.is_number(bins) and io.is_numeric_pair(range):
      return np.linspace(range[0], range[1], bins + 1)
    elif io.is_array(bins, 1) and range is None:
      return bins.copy()
    else:
      raise ValueError(f"Histogram bins '{bins}' and range '{range}' not understood.")

  # ================================================================================================

  # Compute bin centers, widths, and binning parameters for np.histogram.
  def update_bins(self):

    # Compute bin centers and widths.
    self.centers = (self.edges[1:] + self.edges[:-1]) / 2
    self.width = self.edges[1:] - self.edges[:-1]
    self.length = len(self.centers)

    # If all bins are the same width, update the bin and range format accordingly.
    # TODO: np.isclose is too small to catch bin widths from simulation
    if np.all(np.abs(self.width - self.width[0]) < 0.01 * self.width[0]):
      self.width = self.width[0]
      self.bins = len(self.centers)
      self.range = [self.edges[0], self.edges[-1]]
    else:
      self.bins = self.edges
      self.range = None

    return self

  # ================================================================================================

  # Add new entries to the histogram.
  def fill(self, values):
    change, _ = np.histogram(values, bins = self.bins, range = self.range)
    self.heights += change
    self.cov += change
    self.update_errors()
    return self

  # ================================================================================================

  # Clear the histogram contents.
  def clear(self):
    self.heights.fill(0)
    self.errors.fill(0)
    self.cov.fill(0)
    return self

  # ================================================================================================

  def __neg__(self):
    result = self.copy()
    result.heights = -result.heights
    return result

  # ================================================================================================

  def draw(self, choices = 1, update = False):

    # Create a probability distribution out of this histogram, if it hasn't already been made.
    if self.pdf is None or update:
      # Interpolate the histogram with 10x smaller bin widths, for smoothness.
      interpolated = self.interpolate(self.width / 10)
      self.pdf = stats.rv_histogram((interpolated.heights, interpolated.edges))

    # Sample the smoothed probability distribution for this histogram.
    return self.pdf.rvs(size = choices)

  # ================================================================================================

  # a(self) + b.
  # 'cov' is the covariance between 'a' and 'b'.
  # 'err' is the uncertainty in 'b', only used if 'b' is a scalar.
  def add(a, b, cov = None, err = None):
    result = a.copy()
    if isinstance(b, Histogram1D) and np.isclose(a.edges, b.edges).all():
      result.heights += b.heights
      result.cov = a.cov + b.cov
      if cov is not None:
        result.cov += cov + cov.T
      if err is not None:
        raise ValueError("Argument 'err' is unused when both operands are Histogram1Ds.")
    elif io.is_number(b):
      result.heights += b
      result.cov = a.cov
      if err is not None:
        result.cov += err**2
      if cov is not None:
        raise NotImplementedError()
    else:
      raise ValueError()
    result.update_errors()
    return result

  # ================================================================================================

  # a(self) - b
  def subtract(a, b, cov = None, err = None):
    return a.add(-b, cov = (-cov if cov is not None else None), err = err)

  # ================================================================================================

  # a(self) * b
  def multiply(a, b, cov = None, err = None):
    result = a.copy()
    if isinstance(b, Histogram1D) and np.isclose(a.edges, b.edges).all():
      result.heights *= b.heights
      result.cov = np.outer(b.heights, b.heights) * a.cov + np.outer(a.heights, a.heights) * b.cov
      if cov is not None:
        temp = np.outer(b.heights, a.heights) * cov
        result.cov += temp + temp.T
      if err is not None:
        raise ValueError("Argument 'err' is unused when both operands are Histogram1Ds.")
    elif io.is_number(b):
      result.heights *= b
      result.cov = b**2 * a.cov
      if err is not None:
        result.cov += np.outer(a.heights, a.heights) * err**2
      if cov is not None:
        raise NotImplementedError()
    elif io.is_array(b):
      result.heights *= b
      if result.cov.ndim == 2:
        result.cov = np.outer(b, b) * a.cov + np.outer(a.heights, a.heights)
      else:
        result.cov = b**2 * a.cov
    else:
      raise ValueError()
    result.update_errors()
    return result

  # ================================================================================================

  # a(self) / b
  def divide(a, b, cov = None, err = None, zero = 0):
    result = a.copy()
    if isinstance(b, Histogram1D) and np.isclose(a.edges, b.edges).all() and a.cov.ndim == b.cov.ndim:
      fixed_heights = np.where(np.isclose(b.heights, 0), zero, b.heights)
      result.heights /= fixed_heights
      if a.cov.ndim == 2:
        a_coeff = 1 / np.outer(fixed_heights, fixed_heights)
        b_coeff = np.outer(a.heights, a.heights) / np.outer(fixed_heights, fixed_heights)**2
      else:
        a_coeff = 1 / fixed_heights**2
        b_coeff = a.heights**2 / fixed_heights**4
      result.cov = a_coeff * a.cov + b_coeff * b.cov
      if cov is not None:
        temp = np.outer(1 / fixed_heights, a.heights / fixed_heights**2) * cov
        result.cov -= temp + temp.T
      if err is not None:
        raise ValueError("Argument 'err' is unused when both operands are Histogram1Ds.")
    elif io.is_number(b):
      result.heights /= b
      result.cov = a.cov / b**2
      if err is not None:
        result.cov += np.outer(a.heights, a.heights) / b**4 * err**2
      if cov is not None:
        raise NotImplementedError()
    elif io.is_array(b):
      result.heights /= b
      if result.cov.ndim == 2:
        result.cov = a.cov / np.outer(b, b)
      else:
        result.cov = a.cov / b**2
    else:
      raise ValueError()
    result.update_errors()
    return result

  # ================================================================================================

  # a(self)^b
  def power(a, b):
    result = a.copy()
    if io.is_number(b):
      result.heights **= b
      result.cov = b**2 * np.outer(a.heights, a.heights)**(b - 1) * a.cov
    else:
      raise NotImplementedError()
    result.update_errors()
    return result

  # ================================================================================================

  def set_heights(self, heights):
    self.heights = heights.copy()
    return self

  def set_cov(self, cov):
    self.cov = cov.copy()
    self.update_errors()
    return self

  # ================================================================================================

  def convolve(self, function):
    result = self.copy().clear()
    # extra = len(result.heights)
    # if io.is_number(result.width):
    #   df = result.width
    #   leftPad = np.arange(result.centers[0] - extra * df, result.centers[0], df)
    #   rightPad = np.arange(result.centers[-1] + df, result.centers[-1] + (extra + 1) * df, df)
    #   paddedCenters = np.concatenate((leftPad, result.centers, rightPad))
    #   paddedHeights = np.concatenate((np.zeros(len(leftPad)), self.heights, np.zeros(len(rightPad))))
    # else:
    paddedCenters, paddedHeights = result.centers, self.heights
    fDifferences = function(np.subtract.outer(self.centers, paddedCenters))
    centralDifferences = function(np.subtract.outer(self.centers, self.centers))
    result.heights = np.einsum("i, ki -> k", paddedHeights, fDifferences)
    result.cov = np.einsum(f"ki, {'lj, ij' if self.cov.ndim == 2 else 'li, i'} -> kl", centralDifferences, centralDifferences, self.cov)
    result.heights *= np.mean(self.width)
    result.cov *= np.mean(self.width)**2
    result.update_errors()
    return result

  # ================================================================================================

  # Transform this histogram's bin edges according to the supplied function.
  def map(self, function):

    # Apply the function to the bin edges.
    new_edges = function(self.edges)

    if np.all(np.diff(new_edges) > 0):
      self.edges = new_edges
    elif np.all(np.diff(new_edges) < 0):
      self.edges = np.flip(new_edges)
      self.heights = np.flip(self.heights)
      if self.cov.ndim == 1:
        self.cov = np.flip(self.cov)
      else:
        self.cov = np.fliplr(np.flipud(self.cov))
    else:
      raise ValueError("Cannot map bin edges using non-monotonic function.")

    self.update_errors()
    self.update_bins()
    return self

  # ================================================================================================

  def moment(self, degree, central = True, error = False):

    if not io.is_integer(degree) or degree < 1:
      raise ValueError(f"Cannot calculate moment of degree '{degree}'.")

    # clip any negative heights to zero
    # masked_heights = np.where(self.heights > 0, self.heights, 0)
    masked_heights = self.heights

    # normalize the heights and covariance to unit area using Simpson's rule
    area = calc.area(self.centers, masked_heights)
    norm_heights = masked_heights / area
    norm_cov = self.cov / area**2

    # determine moment factor (x - <x>)^n
    if degree == 1 or not central:
      moment_factor = self.centers**degree
    else:
      moment_factor = (self.centers - self.mean())**degree

    moment = calc.area(self.centers, moment_factor * norm_heights)
    if not error:
      return moment

    if central:
      central_term = degree * (self.moment(degree - 1, central) if degree > 1 else 0) * self.centers
    else:
      central_term = 0

    # get covariance of the integrand: scale covariance of heights with moment factor
    if self.cov.ndim == 1:
      scaled_cov = norm_cov * (moment_factor - central_term - moment)**2
    else:
      scaled_cov = norm_cov * np.outer(moment_factor - central_term - moment, moment_factor - central_term - moment)

    # integrate (x - <x>)^n * rho(x)
    return calc.area(self.centers, moment_factor * norm_heights, cov = scaled_cov)

  # ================================================================================================

  # Calculate the mean.
  def mean(self, error = False):
    return self.moment(degree = 1, error = error)

  # ================================================================================================

  # Calculate the standard deviation.
  def std(self, error = False):
    result = self.moment(degree = 2, error = error)
    if error:
      var, var_err = result
      return np.sqrt(var), 1/(2*np.sqrt(var)) * var_err
    else:
      var = result
      return np.sqrt(var)

  # ================================================================================================

  def normalize(self):
    scale = calc.area(self.centers, self.heights)
    self.heights /= scale
    self.cov /= scale**2
    self.update_errors()
    return self

  # ================================================================================================

  def copy(self):
    return Histogram1D(self.edges, heights = self.heights, cov = self.cov)

  # ================================================================================================

  # TODO: don't discard original data, just make a new view (so we can re-mask)
  def mask(self, range):

    if io.is_numeric_pair(range):
      minIndex = np.searchsorted(self.centers, range[0], side = "left")
      maxIndex = np.searchsorted(self.centers, range[1], side = "right")

      self.edges = self.edges[minIndex:(maxIndex + 1)]
      self.update_bins()

      self.heights = self.heights[minIndex:maxIndex]
      self.errors = self.errors[minIndex:maxIndex]
      self.cov = self.cov[minIndex:maxIndex]
      if self.cov.ndim == 2:
        self.cov = self.cov[:, minIndex:maxIndex]
    else:
      raise ValueError(f"Data range '{range}' not understood.")

    return self

  # ================================================================================================

  # Split an array into equal-length blocks along an axis, then sum and recombine along that axis.
  @staticmethod
  def split_sum(array, step, axis = 0, discard = False):

    length = array.shape[axis]
    even = (length % step == 0)
    if not even and not discard:
      raise ValueError(f"Cannot rebin {length} bins into {length / step:.2f} blocks.")

    # Note: indices are SPLIT points; including 0 puts an empty array!
    blocks = np.split(array, np.arange(step, length, step), axis)
    if not even and discard:
      blocks = blocks[:-1]
    return np.concatenate([block.sum(axis, keepdims = True) for block in blocks], axis)

  # ================================================================================================

  def update_errors(self):
    if self.cov.ndim == 1:
      self.errors = np.sqrt(self.cov)
    elif self.cov.ndim == 2:
      self.errors = np.sqrt(np.diag(self.cov))
    else:
      raise ValueError(f"Invalid covariance dimensions: {self.cov.ndim}")

  # ================================================================================================

  # Merge integer groups of bins along the x- or y-axes.
  def rebin(self, step, discard = False):

    if io.is_integer(step):
      if step > 1:
        self.heights = Histogram1D.split_sum(self.heights, step, 0, discard)
        self.cov = Histogram1D.split_sum(self.cov, step, 0, discard)
        if self.cov.ndim == 2:
          self.cov = Histogram1D.split_sum(self.cov, step, 1, discard)
        self.update_errors()

        self.edges = self.edges[::step]
        self.update_bins()

    else:
      raise ValueError(f"Cannot rebin histogram in groups of '{step}'.")

    return self

  # ================================================================================================

  def interpolate(self, x, spline = True):

    # Interpolate new bin heights at the desired bin centers.
    if io.is_number(x):
      x = np.arange(self.edges[0], self.edges[-1] + x, x)

    if spline:
      tck = interp.splrep(self.centers, self.heights)
      heights = interp.splev(x, tck)
    else:
      heights = np.interp(x, self.centers, self.heights)

    # Assume the bin edges are halfway between each bin center.
    edges = np.zeros(len(x) + 1)
    edges[1:-1] = (x[:-1] + x[1:]) / 2
    edges[0] = x[0] - (edges[1] - x[0])
    edges[-1] = x[-1] + (x[-1] - edges[-2])

    # Construct a new, interpolated histogram.
    return Histogram1D(edges, heights = heights)

  # ================================================================================================

  # Plot this histogram.
  def plot(self, errors = True, bar = False, start = None, end = None, skip = 1, label = None, scale = 1, ls = "-", **kwargs):

    # If 'scale' is another histogram, set the plotting scale factor to match their largest values.
    if isinstance(scale, Histogram1D):
      scale = np.max(scale.heights) / np.max(self.heights)

    # Select points to plot, skipping by 'skip' and scaling by 'scale'.
    plot_centers = self.centers[::skip]
    plot_heights, plot_errors = self.heights[::skip] * scale, self.errors[::skip] * scale

    # Mask the data points between 'start' and 'end'.
    plot_start = np.min(plot_centers) if start is None else start
    plot_end = np.max(plot_centers) if end is None else end
    mask = (plot_centers >= plot_start) & (plot_centers <= plot_end)
    plot_centers = plot_centers[mask]
    plot_heights, plot_errors = plot_heights[mask], plot_errors[mask]

    # Set the x and y limits.
    plt.xlim(plot_centers[0], plot_centers[-1])
    # if np.min(plot_heights) > 0:
    #   plt.ylim(0, 1.05 * np.max(plot_heights))

    # Make a bar plot if requested, or else errorbar/line plot.
    if bar:
      return plt.bar(
        plot_centers,
        plot_heights,
        yerr = plot_errors if errors else None,
        width = self.width,
        label = label,
        linewidth = 0.5,
        edgecolor = "k",
        **kwargs
      )
    else:
      if errors:
        return style.errorbar(plot_centers, plot_heights, plot_errors, label = label, ls = ls, **kwargs)
      else:
        return plt.plot(plot_centers, plot_heights, label = label, ls = ls, **kwargs)

  # ================================================================================================

  # Save this histogram to disk in NumPy format.
  # TODO: match ROOT format
  def save(self, filename, name = None, labels = ""):
    if filename.endswith(".root") and name is not None:
      file = root.TFile(filename, "RECREATE")
      self.to_root(name, labels).Write()
      file.Close()
    elif filename.endswith(".npz"):
      np.savez(filename, **self.collect())
    else:
      raise ValueError("Histogram file format not recognized.")

  # ================================================================================================

  # Collect this histogram's data into a dictionary with keyword labels.
  def collect(self, path = None):
    prefix = "" if path is None else f"{path}/"
    return {
      f"{prefix}edges": self.edges,
      f"{prefix}heights": self.heights,
      f"{prefix}errors": self.errors,
      f"{prefix}cov": self.cov
    }

  # ================================================================================================

  def to_root(self, name, labels = ""):
    edges = np.sort(self.edges)
    sortedIndices = np.argsort(self.centers)
    heights, errors = self.heights[sortedIndices], self.errors[sortedIndices]
    histogram = root.TH1F(name, labels, self.length, array.array("f", list(edges)))
    rnp.array2hist(heights, histogram, errors = errors)
    histogram.ResetStats()
    return histogram

  # ================================================================================================

  # Load a histogram previously saved to disk in NumPy or ROOT format.
  @staticmethod
  def load(filename, label = None):
    if filename.endswith(".root") and label is not None:
      try:
        rootFile = root.TFile(filename)
        histogram = rootFile.Get(label)
        heights, edges = rnp.hist2array(histogram, return_edges = True)
      except:
        raise FileNotFoundError()
      edges = edges[0]
      cov = np.array([histogram.GetBinError(i + 1)**2 for i in range(histogram.GetNbinsX())])
      rootFile.Close()
    elif filename.endswith(".npz"):
      prefix = "" if label is None else f"{label}/"
      try:
        data = np.load(filename)
      except:
        raise FileNotFoundError()
      edges = data[f'{prefix}edges']
      heights = data[f'{prefix}heights']
      cov = data[f"{prefix}cov"]
    else:
      raise ValueError("Histogram file format not recognized.")
    return Histogram1D(edges, heights = heights, cov = cov)
