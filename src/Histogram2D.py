import numpy as np
import array

import matplotlib.pyplot as plt
import gm2fr.src.style as style
style.set_style()
import gm2fr.src.io as io
from gm2fr.src.Histogram1D import Histogram1D

# import ROOT as root
# import root_numpy as rnp
import uproot

class Histogram2D:

  # ================================================================================================

  def __init__(self, x_bins, y_bins, x_range = None, y_range = None, heights = None, errors = None):

    # Binning parameters for internal usage of np.histogram.
    self.bins, self.range = [None, None], [None, None]

    # Initialize the histogram bin variables.
    self.edges, self.x_edges, self.y_edges = [None, None], None, None
    self.centers, self.x_centers, self.y_centers = [None, None], None, None
    self.widths, self.x_width, self.y_width = [None, None], None, None
    self.lengths, self.x_length, self.y_length = [None, None], None, None
    self.heights, self.errors = None, None

    self.edges[0] = Histogram1D.parse_binning(x_bins, x_range)
    self.edges[1] = Histogram1D.parse_binning(y_bins, y_range)
    self.update_bins()

    # Determine the shape for the bin height and error arrays.
    shape = tuple(self.lengths)

    def copy_if_present(array):
      if array is not None:
        if array.shape == shape:
          return array.copy()
        else:
          raise ValueError(f"Histogram data does not match bin shape.")
      else:
        return np.zeros(shape)

    # Initialize the bin heights: from an argument if supplied, or else zeros.
    self.heights = copy_if_present(heights)
    self.errors = copy_if_present(errors)

  # ================================================================================================

  # Compute bin centers, widths, and binning parameters for np.histogram.
  def update_bins(self, axis = None):

    if axis is None:
      axes = [0, 1]
    elif axis in (0, 1):
      axes = [axis]
    else:
      raise ValueError(f"Axis option '{axis}' not understood.")

    for i in axes:

      self.centers[i] = (self.edges[i][1:] + self.edges[i][:-1]) / 2
      self.widths[i] = self.edges[i][1:] - self.edges[i][:-1]
      self.lengths[i] = len(self.centers[i])

      # If all bins are the same width, update the bin and range format accordingly.
      if np.all(np.isclose(self.widths[i], self.widths[i][0])):
        self.widths[i] = self.widths[i][0]
        self.bins[i] = self.lengths[i]
        self.range[i] = [self.edges[i][0], self.edges[i][-1]]
      else:
        self.bins[i] = self.edges[i]
        self.range[i] = None

    if 0 in axes:
      self.x_edges = self.edges[0]
      self.x_centers, self.x_width, self.x_length = self.centers[0], self.widths[0], self.lengths[0]
    if 1 in axes:
      self.y_edges = self.edges[1]
      self.y_centers, self.y_width, self.y_length = self.centers[1], self.widths[1], self.lengths[1]

  # ================================================================================================

  # Add new entries to the histogram.
  def fill(self, xValues, yValues):

    # np.histogram2d returns heights and edges, so use index [0] to just take the heights.
    self.heights += np.histogram2d(xValues, yValues, bins = self.bins, range = self.range)[0]

    # Update the bin errors, assuming Poisson statistics.
    self.errors = np.sqrt(self.heights)
    # self.errors[self.errors == 0] = 1

  # ================================================================================================

  # Clear the histogram contents.
  def clear(self):
    self.heights.fill(0)
    self.errors.fill(0)
    return self

  # ================================================================================================

  # Override the *= operator to scale the bin entries in-place.
  def __imul__(self, scale):
    if io.is_number(scale):
      self.heights *= scale
      self.errors *= abs(scale)
    elif isinstance(scale, Histogram2D):
      self.heights *= scale.heights
      self.errors = np.sqrt((self.heights * scale.errors)**2 + (scale.heights * self.errors)**2)
    else:
      raise ValueError("Cannot multiply histogram by given scale.")
    return self

  # ================================================================================================

  # Override the * operator.
  def __mul__(a, b):
    result = a.copy()
    result *= b
    return result

  # ================================================================================================

  # Override the += operator to add bin entries in-place, assuming statistical independence.
  def __iadd__(self, other):
    self.heights += other.heights
    self.errors = np.sqrt(self.errors**2 + other.errors**2)
    return self

  # ================================================================================================

  # Transform this histogram's bin edges according to the supplied function.
  def map(self, x = None, y = None):
    functions = [x, y]
    for axis, function in enumerate(functions):
      if function is None:
        continue
      new_edges = function(self.edges[axis])
      #if np.all(np.diff(new_edges) > 0):
      #  self.edges[axis] = new_edges
      if np.all(np.diff(new_edges) < 0):
        self.edges[axis] = np.flip(new_edges)
        self.heights = np.flip(self.heights, axis = axis)
        self.errors = np.flip(self.errors, axis = axis)
      else:
        self.edges[axis] = new_edges
      #  raise ValueError("Cannot map bin edges using non-monotonic function.")
      self.update_bins(axis)
    return self

  # ================================================================================================

  # Calculate the mean along each axis.
  def mean(self, axis = 1, empty = np.nan):

    if axis not in (0, 1):
      raise ValueError(f"Histogram axis '{axis}' invalid.")
    otherAxis = (axis + 1) % 2

    means = np.zeros(self.lengths[otherAxis])
    errors = np.zeros(self.lengths[otherAxis])

    for i in range(len(means)):
      weights = self.heights[i, :] if axis == 1 else self.heights[:, i]
      wErrors = self.errors[i, :] if axis == 1 else self.errors[:, i]
      total = np.sum(weights)
      if total == 0:
        means[i], errors[i] = empty, empty
        continue
      means[i] = np.average(self.centers[axis], weights = weights)
      errors[i] = 1/total * np.sqrt(np.sum((self.centers[axis] - means[i])**2 * wErrors**2))

    return Histogram1D(self.edges[otherAxis], heights = means, cov = errors**2)

  # ================================================================================================

  # Calculate the standard deviation along each axis.
  def std(self, axis = 1):

    if axis not in (0, 1):
      raise ValueError(f"Histogram axis '{axis}' invalid.")
    otherAxis = (axis + 1) % 2

    means = self.mean(axis).heights
    std = np.zeros(self.lengths[otherAxis])
    errors = np.zeros(self.lengths[otherAxis])

    for i in range(len(means)):
      weights = self.heights[i, :] if axis == 1 else self.heights[:, i]
      wErrors = self.errors[i, :] if axis == 1 else self.errors[:, i]
      total = np.sum(weights)
      if total == 0:
        std[i], errors[i] = np.nan, np.nan
        continue
      std[i] = np.sqrt(np.average((self.centers[axis] - means[i])**2, weights = weights))
      errors[i] = 1/(2*std[i]*total) * np.sqrt(
        np.sum(((self.centers[axis] - means[i])**2 - std[i]**2) * wErrors**2)
      )

    return Histogram1D(self.edges[otherAxis], heights = std, cov = errors**2)

  # ================================================================================================

  def normalize(self, area = True):
    scale = np.sum(self.heights * (self.x_width * self.y_width if area else 1))
    self.heights /= scale
    self.errors /= scale
    return self

  # ================================================================================================

  def transpose(self):
    self.edges[0], self.edges[1] = self.edges[1], self.edges[0]
    self.heights = self.heights.T
    self.errors = self.errors.T
    self.update_bins()
    return self

  # ================================================================================================

  def copy(self):
    return Histogram2D(self.x_edges, self.y_edges, heights = self.heights, errors = self.errors)

  # ================================================================================================

  # TODO: don't discard original data, just make a new view (so we can re-mask)
  def mask(self, x_range = None, y_range = None):

    ranges = [x_range, y_range]
    for axis, range in enumerate(ranges):
      if range is None:
        continue
      if io.is_numeric_pair(range):
        min = np.searchsorted(self.centers[axis], range[0], side = "left")
        max = np.searchsorted(self.centers[axis], range[1], side = "right")

        self.edges[axis] = self.edges[axis][min:(max + 1)]
        self.update_bins(axis)

        self.heights = self.heights[min:max] if axis == 0 else self.heights[:, min:max]
        self.errors = self.errors[min:max] if axis == 0 else self.errors[:, min:max]
      else:
        raise ValueError(f"Data range '{range}' not understood.")

    return self

  # ================================================================================================

  # Merge integer groups of bins along the x- or y-axes.
  def rebin(self, xStep = None, yStep = None, discard = False):

    steps = [xStep, yStep]
    for axis, step in enumerate(steps):
      if step is None:
        continue
      if io.is_integer(step) and step > 0:
        self.heights = Histogram1D.split_sum(self.heights, step, axis, discard)
        self.errors = np.sqrt(Histogram1D.split_sum(self.errors**2, step, axis, discard))
        self.edges[axis] = self.edges[axis][::step]
        self.update_bins(axis)
      else:
        raise ValueError(f"Histogram rebinning '{step}' invalid.")

    return self

  # ================================================================================================

  # Plot this histogram.
  def plot(self, **kwargs):
    return style.colormesh(
      self.x_edges,
      self.y_edges,
      np.where(self.heights == 0, np.nan, self.heights),
      **kwargs
    )

  # ================================================================================================

  # Save this histogram to disk.
  def save(self, filename, name = None, labels = ""):
    if filename.endswith(".root") and name is not None:
      with uproot.recreate(filename) as root_file:
        # file = root.TFile(filename, "RECREATE")
        root_file[name] = self.to_root(name, labels)#.Write()
        # file.Close()
    elif filename.endswith(".npz"):
      np.savez(filename, **self.collect())
    else:
      raise ValueError(f"Histogram file format invalid. Use '.root' or '.npz'.")

  # Collect this histogram's data into a dictionary with keyword labels.
  def collect(self, path = None):
    prefix = "" if path is None else f"{path}/"
    return {
      f"{prefix}x_edges": self.x_edges,
      f"{prefix}y_edges": self.y_edges,
      f"{prefix}heights": self.heights,
      f"{prefix}errors": self.errors
    }

  # ================================================================================================

  # Load a histogram previously saved to disk.
  @staticmethod
  def load(filename, label = None, transpose = False):
    if filename.endswith(".root") and label is not None:
      with uproot.open(filename) as root_file:
        # rootFile = root.TFile(filename)
        # histogram = rootFile.Get(label)
        # heights, edges = rnp.hist2array(histogram, return_edges = True)
        # x_edges, y_edges = edges[0], edges[1]
        # errors = np.array([
        #   [histogram.GetBinError(i + 1, j + 1) for j in range(histogram.GetNbinsY())]
        #   for i in range(histogram.GetNbinsX())
        # ])
        histogram = root_file[label]
        heights, errors = histogram.values(), histogram.errors()
        x_edges = histogram.axis(0).edges()
        y_edges = histogram.axis(1).edges()
        # rootFile.Close()
    elif filename.endswith(".npz"):
      prefix = "" if label is None else f"{label}/"
      data = np.load(filename)
      x_edges = data[f'{prefix}x_edges']
      y_edges = data[f'{prefix}y_edges']
      heights = data[f'{prefix}heights']
      errors = data[f'{prefix}errors']
      if transpose:
        heights = heights.T
        errors = errors.T
    else:
      raise ValueError(f"Could not load histogram from '{filename}' with label '{label}'.")
    return Histogram2D(x_edges, y_edges, heights = heights, errors = errors)

  # ================================================================================================

  # Convert this Histogram2D to a TH2F.
  def to_root(self, name = "", labels = ""):
    # histogram = root.TH2F(
    #   name, labels,
    #   self.x_length, array.array("f", list(self.x_edges)),
    #   self.y_length, array.array("f", list(self.y_edges))
    # )
    # rnp.array2hist(self.heights, histogram, errors = self.errors)
    # histogram.ResetStats()
    # return histogram
    return (self.heights, self.x_edges, self.y_edges)
