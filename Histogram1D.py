import numpy as np
import array

import matplotlib.pyplot as plt
import gm2fr.style as style
style.setStyle()
import gm2fr.utilities as util

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

    self.edges = Histogram1D.parseBinning(bins, range)
    self.updateBins()

    # Determine the shape for the bin height and error arrays.
    shape = (self.length,)

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
    self.updateErrors()

  # ================================================================================================

  @staticmethod
  def parseBinning(bins, range):
    if util.isNumber(bins) and util.isNumericPair(range):
      return np.linspace(range[0], range[1], bins + 1)
    elif util.isArray(bins, 1) and range is None:
      return bins.copy()
    else:
      raise ValueError(f"Histogram bins '{bins}' and range '{range}' not understood.")

  # ================================================================================================

  # Compute bin centers, widths, and binning parameters for np.histogram.
  def updateBins(self):

    # Compute bin centers and widths.
    self.centers = (self.edges[1:] + self.edges[:-1]) / 2
    self.width = self.edges[1:] - self.edges[:-1]
    self.length = len(self.centers)

    # If all bins are the same width, update the bin and range format accordingly.
    if np.all(np.isclose(self.width, self.width[0])):
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
    self.updateErrors()
    return self

  # ================================================================================================

  # Clear the histogram contents.
  def clear(self):
    self.heights.fill(0)
    self.errors.fill(0)
    self.cov.fill(0)
    return self

  # ================================================================================================

  # Override the *= operator to scale the bin entries in-place.
  def __imul__(self, scale):
    if util.isNumber(scale):
      self.heights *= scale
      self.cov *= abs(scale)**2
    elif isinstance(scale, Histogram1D) and (scale.edges == self.edges).all():
      self.heights *= scale.heights
      if self.cov.ndim == 1 and scale.cov.ndim == 1:
        self.cov = scale.heights**2 * self.cov + self.heights**2 * scale.cov
      elif self.cov.ndim == 1 and scale.cov.ndim == 2:
        self.cov = scale.heights**2 * np.diag(self.cov) + np.outer(self.heights, self.heights) * scale.cov
      elif self.cov.ndim == 2 and scale.cov.ndim == 1:
        self.cov = self.heights**2 * np.diag(scale.cov) + np.outer(scale.heights, scale.heights) * self.cov
      else:
        self.cov = np.outer(self.heights, self.heights) * scale.cov + np.outer(scale.heights, scale.heights) * self.cov
    else:
      raise ValueError("Cannot multiply histogram by given scale.")
    self.updateErrors()
    return self

  # ================================================================================================

  # Override the * operator.
  def __mul__(self, other):
    result = self.copy()
    result *= other
    return result

  def __rmul__(other, self):
    return self * other

  # ================================================================================================

  # Divide bin entries in-place, optionally replacing dividends near zero.
  def divide(self, other, zeros = 0):
    if util.isNumber(other):
      self.heights /= other
      self.cov /= abs(other)**2
    elif isinstance(other, Histogram1D) and np.allclose(other.edges, self.edges):
      otherHeights = np.where(np.isclose(other.heights, 0), zeros, other.heights)
      self.heights /= otherHeights
      if self.cov.ndim == 1 and other.cov.ndim == 1:
        self.cov = self.cov / otherHeights**2 + (self.heights / otherHeights**2)**2 * other.cov
      elif self.cov.ndim == 1 and other.cov.ndim == 2:
        self.cov = np.diag(self.cov) / otherHeights**2 + np.outer(self.heights, self.heights) / np.outer(otherHeights, otherHeights)**2 * other.cov
      elif self.cov.ndim == 2 and other.cov.ndim == 1:
        self.cov = (self.heights / otherHeights**2)**2 * np.diag(other.cov) + 1/np.outer(otherHeights, otherHeights) * self.cov
      else:
        self.cov = 1/np.outer(otherHeights, otherHeights) * self.cov + np.outer(self.heights, self.heights) / np.outer(otherHeights, otherHeights)**2 * other.cov
    else:
      raise ValueError("Cannot divide histogram by given scale.")
    self.updateErrors()
    return self

  # Override the /= operator to divide the bin entries in-place.
  def __idiv__(self, other):
    return self.divide(other)

  # Override the / operator.
  def __div__(self, other):
    result = self.copy()
    result /= other
    return result

  def __rdiv__(other, self):
    return self / other

  # ================================================================================================

  def setHeights(self, heights):
    self.heights = heights
    return self

  def setCov(self, cov):
    self.cov = cov
    self.updateErrors()
    return self

  # ================================================================================================

  # Override the += operator to add bin entries in-place, assuming statistical independence.
  def __iadd__(self, other):
    self.heights += other.heights
    if self.cov.ndim == other.cov.ndim:
      self.cov += other.cov
    elif self.cov.ndim == 1 and other.cov.ndim == 2:
      self.cov = np.diag(self.cov) + other.cov
    else:
      self.cov += np.diag(other.cov)
    self.updateErrors()
    return self

  # ================================================================================================

  def __add__(self, other):
    result = self.copy()
    result += other
    return result

  # ================================================================================================

  # Transform this histogram's bin edges according to the supplied function.
  def map(self, function):
    self.edges = function(self.edges)
    self.updateBins()
    return self

  # ================================================================================================

  # Calculate the mean.
  def mean(self, error = False):

    total = np.sum(self.heights)
    if total == 0:
      return np.nan, np.nan if error else np.nan
    avg = np.average(self.centers, weights = self.heights)
    if not error:
      return avg

    temp = self.centers - avg
    if self.cov.ndim == 1:
      err = 1/total * np.sqrt(np.sum(temp**2 * self.cov))
    else:
      err = 1/total * np.sqrt(temp.T @ self.cov @ temp)

    return avg, err

  # ================================================================================================

  # Calculate the standard deviation.
  def std(self, error = False):

    total = np.sum(self.heights)
    if total == 0:
      return np.nan, np.nan if error else np.nan

    avg = self.mean(False)
    std = np.sqrt(np.average((self.centers - avg)**2, weights = self.heights))
    if not error:
      return std

    temp = (self.centers - avg)**2 - std**2
    if self.cov.ndim == 1:
      err = 1/(2*std*total) * np.sqrt(np.sum(temp**2 * self.cov))
    else:
      err = 1/(2*std*total) * np.sqrt(temp.T @ self.cov @ temp)

    return std, err

  # ================================================================================================

  def normalize(self, area = True):
    scale = np.sum(self.heights * (self.width if area else 1))
    self.heights /= scale
    self.cov /= scale**2
    self.updateErrors()
    return self

  # ================================================================================================

  def copy(self):
    return Histogram1D(self.edges, heights = self.heights, cov = self.cov)

  # ================================================================================================

  # TODO: don't discard original data, just make a new view (so we can re-mask)
  def mask(self, range):

    if util.isNumericPair(range):
      minIndex = np.searchsorted(self.centers, range[0], side = "left")
      maxIndex = np.searchsorted(self.centers, range[1], side = "right")

      self.edges = self.edges[minIndex:(maxIndex + 1)]
      self.updateBins()

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
  def splitSum(array, step, axis = 0, discard = False):

    length = array.shape[axis]
    even = (length % step != 0)
    if not even and not discard:
      raise ValueError(f"Cannot rebin {length} bins into {blocks} blocks.")

    blocks = np.split(array, np.arange(0, length, step), axis)
    if not even and discard:
      blocks = blocks[:-1]
    return np.concatenate([block.sum(axis) for block in blocks], axis)

  def updateErrors(self):
    if self.cov.ndim == 1:
      self.errors = np.sqrt(self.cov)
    elif self.cov.ndim == 2:
      self.errors = np.sqrt(np.diag(self.cov))
    else:
      raise ValueError(f"Invalid covariance dimensions: {self.cov.ndim}")

  # ================================================================================================

  # Merge integer groups of bins along the x- or y-axes.
  def rebin(self, step, discard = False):

    if util.isInteger(step) and step > 0:
      self.heights = Histogram.splitSum(self.heights, step, 0, discard)
      self.cov = Histogram.splitSum(self.cov, step, 0, discard)
      if self.cov.ndim == 2:
        self.cov = Histogram.splitSum(self.cov, step, 1, discard)
      self.updateErrors()

      self.edges = self.edges[::step]
      self.updateBins()

    else:
      raise ValueError(f"Cannot rebin histogram in groups of '{step}'.")

    return self

  # ================================================================================================

  def interpolate(self, x):

    # Interpolate new bin heights at the desired bin centers.
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
  def plot(self, errors = True, bar = False, label = None, scale = 1, **kwargs):

    # Set the x and y limits.
    # plt.xlim(self.edges[0], self.edges[-1])
    # if np.min(self.heights) > 0:
    #   plt.ylim(0, None)

    # Make a bar plot if requested, or else errorbar/line plot.
    if bar:
      return plt.bar(
        self.centers,
        self.heights * scale,
        yerr = self.errors * scale if errors else None,
        width = self.width,
        label = label,
        linewidth = 0.5,
        edgecolor = "k",
        **kwargs
      )
    else:
      if errors:
        return style.errorbar(self.centers, self.heights * scale, self.errors * scale, label = label, **kwargs)
      else:
        return plt.plot(self.centers, self.heights * scale, label = label, **kwargs)

  # ================================================================================================

  # Save this histogram to disk in NumPy format.
  def save(self, filename, name = None, labels = ""):
    if filename.endswith(".root") and name is not None:
      file = root.TFile(filename, "RECREATE")
      self.toRoot(name, labels).Write()
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

  def toRoot(self, name, labels = ""):
    histogram = root.TH1F(name, labels, self.length, array.array("f", list(self.edges)))
    rnp.array2hist(self.heights, histogram, errors = self.errors)
    histogram.ResetStats()
    return histogram

  # ================================================================================================

  # Load a histogram previously saved to disk in NumPy or ROOT format.
  @staticmethod
  def load(filename, label = None):
    if filename.endswith(".root") and label is not None:
      rootFile = root.TFile(filename)
      histogram = rootFile.Get(label)
      heights, edges = rnp.hist2array(histogram, return_edges = True)
      edges = edges[0]
      cov = np.array([histogram.GetBinError(i + 1)**2 for i in range(histogram.GetNbinsX())])
      rootFile.Close()
    elif filename.endswith(".npz"):
      prefix = "" if label is None else f"{label}/"
      data = np.load(filename)
      edges = data[f'{prefix}edges']
      heights = data[f'{prefix}heights']
      cov = data[f"{prefix}cov"]
    else:
      raise ValueError("Histogram file format not recognized.")
    return Histogram1D(edges, heights = heights, cov = cov)

  # ================================================================================================

  @staticmethod
  def transform(signal, frequencies, t0, type = "cosine", errors = True, wiggle = True):

    if isinstance(frequencies, gm2fr.Histogram1D.Histogram1D):
      result = frequencies
    else:
      df = frequencies[1] - frequencies[0]
      result = gm2fr.Histogram1D.Histogram1D(np.arange(frequencies[0] - df/2, frequencies[-1] + df, df))

    differences = np.arange(result.length) * result.width
    cov = None

    # Note: to first order, cosine and sine transforms have the same covariance. (Not a typo.)
    if type == "cosine":

      heights = cosineTransform(result.centers, signal.heights, signal.centers, t0, wiggle)
      if errors:
        cov = 0.5 * linalg.toeplitz(cosineTransform(differences, signal.errors**2, signal.centers, t0, False))

    elif type == "sine":

      heights = sineTransform(result.centers, signal.heights, signal.centers, t0, wiggle)
      if errors:
        cov = 0.5 * linalg.toeplitz(cosineTransform(differences, signal.errors**2, signal.centers, t0, False))

    elif type == "magnitude":

      cosine = cosineTransform(result.centers, signal.heights, signal.centers, t0, wiggle)
      sine = sineTransform(result.centers, signal.heights, signal.centers, t0, wiggle)

      tempCos = linalg.toeplitz(cosineTransform(differences, signal.errors**2, signal.centers, t0, False))
      tempSin = linalg.toeplitz(sineTransform(differences, signal.errors**2, signal.centers, t0, False))

      heights = np.sqrt(cosine**2 + sine**2)
      if errors:
        cov = 0.5 / np.outer(heights, heights) * (
          (np.outer(cosine, cosine) + np.outer(sine, sine)) * tempCos \
          + (np.outer(sine, cosine) - np.outer(cosine, sine)) * tempSin
        )

    else:
      raise ValueError(f"Frequency transform type '{type}' not recognized.")

    result.setHeights(heights)
    if errors:
      result.setCov(cov)
    return result
