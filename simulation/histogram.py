# import numpy as np
# import array
#
# import matplotlib.pyplot as plt
# import gm2fr.style as style
# style.setStyle()
# import gm2fr.utilities as util
#
# import ROOT as root
# import root_numpy as rnp
#
# # ==================================================================================================
#
# class Histogram1D:
#
#   # ============================================================================
# 
#   def __init__(self, bins, range = None, heights = None, cov = None):
#
#     # Binning parameters for internal usage of np.histogram.
#     self.bins, self.range = None, None
#
#     # Initialize the histogram bin variables.
#     self.edges, self.centers, self.width, self.length = None, None, None, None
#     self.heights, self.errors, self.cov = None, None, None
#
#     self.edges = Histogram1D.parseBinning(bins, range)
#     self.updateBins()
#
#     # Determine the shape for the bin height and error arrays.
#     shape = (self.length,)
#
#     # Initialize the bin heights: from an argument if supplied, or else zeros.
#     if heights is not None:
#       if heights.shape == shape:
#         self.heights = heights.copy()
#       else:
#         raise ValueError(f"Supplied histogram heights do not match bin shape.")
#     else:
#       self.heights = np.zeros(shape)
#
#     # Initialize the covariance matrix: from an argument if supplied, else zeros.
#     if cov is not None:
#       if cov.shape == (self.length, self.length) or cov.shape == shape:
#         self.cov = cov.copy()
#       else:
#         raise ValueError(f"Covariance matrix does not match the size of the histogram.")
#     else:
#       self.cov = np.zeros(shape)
#     self.updateErrors()
#
#   @staticmethod
#   def parseBinning(bins, range):
#     if util.isNumber(bins) and util.isNumericPair(range):
#       return np.linspace(range[0], range[1], bins + 1)
#     elif util.isArray(bins, 1) and range is None:
#       return bins.copy()
#     else:
#       raise ValueError(f"Histogram bins '{bins}' and range '{range}' not understood.")
#
#   # ============================================================================
#
#   # Compute bin centers, widths, and binning parameters for np.histogram.
#   def updateBins(self):
#
#     # Compute bin centers and widths.
#     self.centers = (self.edges[1:] + self.edges[:-1]) / 2
#     self.width = self.edges[1:] - self.edges[:-1]
#     self.length = len(self.centers)
#
#     # If all bins are the same width, update the bin and range format accordingly.
#     if np.all(np.isclose(self.width, self.width[0])):
#       self.width = self.width[0]
#       self.bins = len(self.centers)
#       self.range = [self.edges[0], self.edges[-1]]
#     else:
#       self.bins = self.edges
#       self.range = None
#
#   # ============================================================================
#
#   # Add new entries to the histogram.
#   def fill(self, values):
#     change, _ = np.histogram(values, bins = self.bins, range = self.range)
#     self.heights += change
#     self.cov += change
#     self.updateErrors()
#
#   # ============================================================================
#
#   # Clear the histogram contents.
#   def clear(self):
#     self.heights.fill(0)
#     self.errors.fill(0)
#     self.cov.fill(0)
#     return self
#
#   # ============================================================================
#
#   # Override the *= operator to scale the bin entries in-place.
#   def __imul__(self, scale):
#     if util.isNumber(scale):
#       self.heights *= scale
#       self.cov *= abs(scale)**2
#     elif isinstance(scale, Histogram1D) and (scale.edges == self.edges).all():
#       self.heights *= scale.heights
#       if self.cov.ndim == 1 and scale.cov.ndim == 1:
#         self.cov = scale.heights**2 * self.cov + self.heights**2 * scale.cov
#       elif self.cov.ndim == 1 and scale.cov.ndim == 2:
#         self.cov = scale.heights**2 * np.diag(self.cov) + np.outer(self.heights, self.heights) * scale.cov
#       elif self.cov.ndim == 2 and scale.cov.ndim == 1:
#         self.cov = self.heights**2 * np.diag(scale.cov) + np.outer(scale.heights, scale.heights) * self.cov
#       else:
#         self.cov = np.outer(self.heights, self.heights) * scale.cov + np.outer(scale.heights, scale.heights) * self.cov
#     else:
#       raise ValueError("Cannot multiply histogram by given scale.")
#     self.updateErrors()
#     return self
#
#   # Override the * operator.
#   def __mul__(self, b):
#     result = self.copy()
#     result *= b
#     return result
#
#   # ============================================================================
#
#   def setHeights(self, heights):
#     self.heights = heights
#     return self
#
#   # Override the += operator to add bin entries in-place, assuming statistical independence.
#   def __iadd__(self, other):
#     self.heights += other.heights
#     if self.cov.ndim == other.cov.ndim:
#       self.cov += other.cov
#     elif self.cov.ndim == 1 and other.cov.ndim == 2:
#       self.cov = np.diag(self.cov) + other.cov
#     else:
#       self.cov += np.diag(other.cov)
#     self.updateErrors()
#     return self
#
#   def __add__(self, other):
#     result = self.copy()
#     result += other
#     return result
#
#   # ============================================================================
#
#   # Transform this histogram's bin edges according to the supplied function.
#   def map(self, function):
#     self.edges = function(self.edges)
#     self.updateBins()
#     return self
#
#   # ============================================================================
#
#   # Calculate the mean.
#   def mean(self):
#
#     total = np.sum(self.heights)
#     if total == 0:
#       return np.nan, np.nan
#     avg = np.average(self.centers, weights = self.heights)
#
#     temp = self.centers - avg
#     if self.cov.ndim == 1:
#       err = 1/total * np.sqrt(np.sum(temp**2 * self.cov))
#     else:
#       err = 1/total * np.sqrt(temp.T @ self.cov @ temp)
#
#     return avg, err
#
#   # ============================================================================
#
#   # Calculate the standard deviation.
#   def std(self):
#
#     total = np.sum(self.heights)
#     avg, _ = self.mean()
#     std = np.sqrt(np.average((self.centers - avg)**2, weights = self.heights))
#
#     temp = (self.centers - avg)**2 - std**2
#     if self.cov.ndim == 1:
#       err = 1/(2*std*total) * np.sqrt(np.sum(temp**2 * self.cov))
#     else:
#       err = 1/(2*std*total) * np.sqrt(temp.T @ self.cov @ temp)
#     return std, err
#
#   # ============================================================================
#
#   def normalize(self, area = True):
#     scale = np.sum(self.heights * (self.width if area else 1))
#     self.heights /= scale
#     self.cov /= scale**2
#     self.updateErrors()
#     return self
#
#   # ============================================================================
#
#   def copy(self):
#     return Histogram1D(self.edges, heights = self.heights, cov = self.cov)
#
#   # ============================================================================
#
#   # TODO: don't discard original data, just make a new view (so we can re-mask)
#   def mask(self, range):
#
#     if util.isNumericPair(range):
#       minIndex = np.searchsorted(self.centers, range[0], side = "left")
#       maxIndex = np.searchsorted(self.centers, range[1], side = "right")
#
#       self.edges = self.edges[minIndex:(maxIndex + 1)]
#       self.updateBins()
#
#       self.heights = self.heights[minIndex:maxIndex]
#       self.errors = self.errors[minIndex:maxIndex]
#       self.cov = self.cov[minIndex:maxIndex]
#       if self.cov.ndim == 2:
#         self.cov = self.cov[:, minIndex:maxIndex]
#     else:
#       raise ValueError(f"Data range '{range}' not understood.")
#
#     return self
#
#   # ============================================================================
#
#   # Split an array into equal-length blocks along an axis, then sum and recombine along that axis.
#   @staticmethod
#   def splitSum(array, step, axis = 0, discard = False):
#
#     length = array.shape[axis]
#     even = (length % step != 0)
#     if not even and not discard:
#       raise ValueError(f"Cannot rebin {length} bins into {blocks} blocks.")
#
#     blocks = np.split(array, np.arange(0, length, step), axis)
#     if not even and discard:
#       blocks = blocks[:-1]
#     return np.concatenate([block.sum(axis) for block in blocks], axis)
#
#   def updateErrors(self):
#     if self.cov.ndim == 1:
#       self.errors = np.sqrt(self.cov)
#     elif self.cov.ndim == 2:
#       self.errors = np.sqrt(np.diag(self.cov))
#     else:
#       raise ValueError(f"Invalid covariance dimensions: {self.cov.ndim}")
#
#   # Merge integer groups of bins along the x- or y-axes.
#   def rebin(self, step, discard = False):
#
#     if util.isInteger(step) and step > 0:
#       self.heights = Histogram.splitSum(self.heights, step, 0, discard)
#       self.cov = Histogram.splitSum(self.cov, step, 0, discard)
#       if self.cov.ndim == 2:
#         self.cov = Histogram.splitSum(self.cov, step, 1, discard)
#       self.updateErrors()
#
#       self.edges = self.edges[::step]
#       self.updateBins()
#
#     else:
#       raise ValueError(f"Cannot rebin histogram in groups of '{step}'.")
#
#     return self
#
#   # ============================================================================
#
#   # Plot this histogram.
#   def plot(self, errors = False, bar = False, label = None, **kwargs):
#
#     # Set the x and y limits.
#     plt.xlim(self.edges[0], self.edges[-1])
#     if np.min(self.heights) > 0:
#       plt.ylim(0, None)
#
#     # Make a bar plot if requested, or else errorbar/line plot.
#     if bar:
#       return plt.bar(
#         self.centers,
#         self.heights,
#         yerr = self.errors if errors else None,
#         width = self.width,
#         label = label,
#         linewidth = 0.5,
#         edgecolor = "k",
#         **kwargs
#       )
#     else:
#       if errors:
#         return style.errorbar(self.centers, self.heights, self.errors, label = label, **kwargs)
#       else:
#         return plt.plot(self.centers, self.heights, label = label, **kwargs)
#
#   # ============================================================================
#
#   # Save this histogram to disk in NumPy format.
#   def save(self, filename, name = None, labels = ""):
#     if filename.endswith(".root") and name is not None:
#       file = root.TFile(filename, "RECREATE")
#       self.toRoot(name, labels).Write()
#       file.Close()
#     elif filename.endswith(".npz"):
#       np.savez(filename, **self.collect())
#     else:
#       raise ValueError("Histogram file format not recognized.")
#
#   # Collect this histogram's data into a dictionary with keyword labels.
#   def collect(self, path = None):
#     prefix = "" if path is None else f"{path}/"
#     return {
#       f"{prefix}edges": self.edges,
#       f"{prefix}heights": self.heights,
#       f"{prefix}errors": self.errors,
#       f"{prefix}cov": self.cov
#     }
#
#   def toRoot(self, name, labels = ""):
#     histogram = root.TH1F(name, labels, self.length, array.array("f", list(self.edges)))
#     rnp.array2hist(self.heights, histogram, errors = self.errors)
#     histogram.ResetStats()
#     return histogram
#
#   # ============================================================================
#
#   # Load a histogram previously saved to disk in NumPy or ROOT format.
#   @staticmethod
#   def load(filename, label = None):
#     if filename.endswith(".root") and label is not None:
#       rootFile = root.TFile(filename)
#       histogram = rootFile.Get(label)
#       heights, edges = rnp.hist2array(histogram, return_edges = True)
#       edges = edges[0]
#       cov = np.array([histogram.GetBinError(i + 1)**2 for i in range(histogram.GetNbinsX())])
#       rootFile.Close()
#     elif filename.endswith(".npz"):
#       prefix = "" if label is None else f"{label}/"
#       data = np.load(filename)
#       edges = data[f'{prefix}edges']
#       heights = data[f'{prefix}heights']
#       cov = data[f"{prefix}cov"]
#     else:
#       raise ValueError("Histogram file format not recognized.")
#     return Histogram1D(edges, heights = heights, cov = cov)
#
# # ==================================================================================================
#
# # A simple histogram, in one or two dimensions.
# class Histogram2D:
#
#   # ============================================================================
#
#   def __init__(self, xBins, yBins, xRange = None, yRange = None, heights = None, errors = None):
#
#     # Binning parameters for internal usage of np.histogram.
#     self.bins, self.range = [None, None], [None, None]
#
#     # Initialize the histogram bin variables.
#     self.edges, self.xEdges, self.yEdges = [None, None], None, None
#     self.centers, self.xCenters, self.yCenters = [None, None], None, None
#     self.widths, self.xWidth, self.yWidth = [None, None], None, None
#     self.lengths, self.xLength, self.yLength = [None, None], None, None
#     self.heights, self.errors = None, None
#
#     self.edges[0] = Histogram1D.parseBinning(xBins, xRange)
#     self.edges[1] = Histogram1D.parseBinning(yBins, yRange)
#     self.updateBins()
#
#     # Determine the shape for the bin height and error arrays.
#     shape = tuple(self.lengths)
#
#     def copyIfPresent(array):
#       if array is not None:
#         if array.shape == shape:
#           return array.copy()
#         else:
#           raise ValueError(f"Histogram data does not match bin shape.")
#       else:
#         return np.zeros(shape)
#
#     # Initialize the bin heights: from an argument if supplied, or else zeros.
#     self.heights = copyIfPresent(heights)
#     self.errors = copyIfPresent(errors)
#
#   # ============================================================================
#
#   # Compute bin centers, widths, and binning parameters for np.histogram.
#   def updateBins(self, axis = None):
#
#     if axis is None:
#       axes = [0, 1]
#     elif axis in (0, 1):
#       axes = [axis]
#     else:
#       raise ValueError(f"Axis option '{axis}' not understood.")
#
#     for i in axes:
#
#       self.centers[i] = (self.edges[i][1:] + self.edges[i][:-1]) / 2
#       self.widths[i] = self.edges[i][1:] - self.edges[i][:-1]
#       self.lengths[i] = len(self.centers[i])
#
#       # If all bins are the same width, update the bin and range format accordingly.
#       if np.all(np.isclose(self.widths[i], self.widths[i][0])):
#         self.widths[i] = self.widths[i][0]
#         self.bins[i] = self.lengths[i]
#         self.range[i] = [self.edges[i][0], self.edges[i][-1]]
#       else:
#         self.bins[i] = self.edges[i]
#         self.range[i] = None
#
#     if 0 in axes:
#       self.xEdges = self.edges[0]
#       self.xCenters, self.xWidth, self.xLength = self.centers[0], self.widths[0], self.lengths[0]
#     if 1 in axes:
#       self.yEdges = self.edges[1]
#       self.yCenters, self.yWidth, self.yLength = self.centers[1], self.widths[1], self.lengths[1]
#
#   # ============================================================================
#
#   # Add new entries to the histogram.
#   def fill(self, xValues, yValues):
#
#     # np.histogram2d returns heights and edges, so use index [0] to just take the heights.
#     self.heights += np.histogram2d(xValues, yValues, bins = self.bins, range = self.range)[0]
#
#     # Update the bin errors, assuming Poisson statistics.
#     self.errors = np.sqrt(self.heights)
#     # self.errors[self.errors == 0] = 1
#
#   # ============================================================================
#
#   # Clear the histogram contents.
#   def clear(self):
#     self.heights.fill(0)
#     self.errors.fill(0)
#     return self
#
#   # ============================================================================
#
#   # Override the *= operator to scale the bin entries in-place.
#   def __imul__(self, scale):
#     if util.isNumber(scale):
#       self.heights *= scale
#       self.errors *= abs(scale)
#     elif isinstance(scale, Histogram2D):
#       self.heights *= scale.heights
#       self.errors = np.sqrt((self.heights * scale.errors)**2 + (scale.heights * self.errors)**2)
#     else:
#       raise ValueError("Cannot multiply histogram by given scale.")
#     return self
#
#   # ============================================================================
#
#   # Override the * operator.
#   def __mul__(a, b):
#     result = a.copy()
#     result *= b
#     return result
#
#   # ============================================================================
#
#   # Override the += operator to add bin entries in-place, assuming statistical independence.
#   def __iadd__(self, other):
#     self.heights += other.heights
#     self.errors = np.sqrt(self.errors**2 + other.errors**2)
#     return self
#
#   # ============================================================================
#
#   # Transform this histogram's bin edges according to the supplied function.
#   def map(self, x = None, y = None):
#     functions = [x, y]
#     for axis, function in enumerate(functions):
#       if function is None:
#         continue
#       self.edges[axis] = function(self.edges[axis])
#       self.updateBins(axis)
#     return self
#
#   # ============================================================================
#
#   # Calculate the mean along each axis.
#   def mean(self, axis = 1, empty = np.nan):
#
#     if axis not in (0, 1):
#       raise ValueError(f"Histogram axis '{axis}' invalid.")
#     otherAxis = (axis + 1) % 2
#
#     means = np.zeros(self.lengths[otherAxis])
#     errors = np.zeros(self.lengths[otherAxis])
#
#     for i in range(len(means)):
#       weights = self.heights[i, :] if axis == 1 else self.heights[:, i]
#       wErrors = self.errors[i, :] if axis == 1 else self.errors[:, i]
#       total = np.sum(weights)
#       if total == 0:
#         means[i], errors[i] = empty, empty
#         continue
#       means[i] = np.average(self.centers[axis], weights = weights)
#       errors[i] = 1/total * np.sqrt(np.sum((self.centers[axis] - means[i])**2 * wErrors**2))
#
#     return Histogram1D(self.edges[otherAxis], heights = means, cov = errors**2)
#
#   # ============================================================================
#
#   # Calculate the standard deviation along each axis.
#   def std(self, axis = 1):
#
#     if axis not in (0, 1):
#       raise ValueError(f"Histogram axis '{axis}' invalid.")
#     otherAxis = (axis + 1) % 2
#
#     means = self.mean(axis).heights
#     std = np.zeros(self.lengths[otherAxis])
#     errors = np.zeros(self.lengths[otherAxis])
#
#     for i in range(len(means)):
#       weights = self.heights[i, :] if axis == 1 else self.heights[:, i]
#       wErrors = self.errors[i, :] if axis == 1 else self.errors[:, i]
#       total = np.sum(weights)
#       if total == 0:
#         std[i], errors[i] = np.nan, np.nan
#         continue
#       std[i] = np.sqrt(np.average((self.centers[axis] - means[i])**2, weights = weights))
#       errors[i] = 1/(2*std[i]*total) * np.sqrt(
#         np.sum(((self.centers[axis] - means[i])**2 - std[i]**2) * wErrors**2)
#       )
#
#     return Histogram1D(self.edges[otherAxis], heights = std, cov = errors**2)
#
#   # ============================================================================
#
#   def normalize(self, area = True):
#     scale = np.sum(self.heights * (self.xWidth * self.yWidth if area else 1))
#     self.heights /= scale
#     self.errors /= scale
#     return self
#
#   # ============================================================================
#
#   def transpose(self):
#     self.edges[0], self.edges[1] = self.edges[1], self.edges[0]
#     self.heights = self.heights.T
#     self.errors = self.errors.T
#     self.updateBins()
#     return self
#
#   # ============================================================================
#
#   def copy(self):
#     return Histogram2D(self.xEdges, self.yEdges, heights = self.heights, errors = self.errors)
#
#   # ============================================================================
#
#   # TODO: don't discard original data, just make a new view (so we can re-mask)
#   def mask(self, xRange = None, yRange = None):
#
#     ranges = [xRange, yRange]
#     for axis, range in enumerate(ranges):
#       if range is None:
#         continue
#       if util.isNumericPair(range):
#         min = np.searchsorted(self.centers[axis], range[0], side = "left")
#         max = np.searchsorted(self.centers[axis], range[1], side = "right")
#
#         self.edges[axis] = self.edges[axis][min:(max + 1)]
#         self.updateBins(axis)
#
#         self.heights = self.heights[min:max] if axis == 0 else self.heights[:, min:max]
#         self.errors = self.errors[min:max] if axis == 0 else self.errors[:, min:max]
#       else:
#         raise ValueError(f"Data range '{range}' not understood.")
#
#     return self
#
#   # ============================================================================
#
#   # Merge integer groups of bins along the x- or y-axes.
#   def rebin(self, xStep = None, yStep = None, discard = False):
#
#     steps = [xStep, yStep]
#     for axis, step in enumerate(steps):
#       if step is None:
#         continue
#       if util.isInteger(step) and step > 0:
#         self.heights = Histogram1D.splitSum(self.heights, step, axis, discard)
#         self.errors = np.sqrt(Histogram1D.splitSum(self.errors**2, step, axis, discard))
#         self.edges[axis] = self.edges[axis][::step]
#         self.updateBins(axis)
#       else:
#         raise ValueError(f"Histogram rebinning '{step}' invalid.")
#
#     return self
#
#   # ============================================================================
#
#   # Plot this histogram.
#   def plot(self, **kwargs):
#     return style.colormesh(
#       self.xEdges,
#       self.yEdges,
#       np.where(self.heights == 0, np.nan, self.heights),
#       **kwargs
#     )
#
#   # ============================================================================
#
#   # Save this histogram to disk.
#   def save(self, filename, name = None, labels = ""):
#     if filename.endswith(".root") and name is not None:
#       file = root.TFile(filename, "RECREATE")
#       self.toRoot(name, labels).Write()
#       file.Close()
#     elif filename.endswith(".npz"):
#       np.savez(filename, **self.collect())
#     else:
#       raise ValueError(f"Histogram file format invalid. Use '.root' or '.npz'.")
#
#   # Collect this histogram's data into a dictionary with keyword labels.
#   def collect(self, path = None):
#     prefix = "" if path is None else f"{path}/"
#     return {
#       f"{prefix}xEdges": self.xEdges,
#       f"{prefix}yEdges": self.yEdges,
#       f"{prefix}heights": self.heights,
#       f"{prefix}errors": self.errors
#     }
#
#   # ============================================================================
#
#   # Load a histogram previously saved to disk.
#   @staticmethod
#   def load(filename, label = None):
#     if filename.endswith(".root") and label is not None:
#       rootFile = root.TFile(filename)
#       histogram = rootFile.Get(label)
#       heights, edges = rnp.hist2array(histogram, return_edges = True)
#       xEdges, yEdges = edges[0], edges[1]
#       errors = np.array([
#         [histogram.GetBinError(i + 1, j + 1) for j in range(histogram.GetNbinsY())]
#         for i in range(histogram.GetNbinsX())
#       ])
#       rootFile.Close()
#     elif filename.endswith(".npz"):
#       prefix = "" if label is None else f"{label}/"
#       data = np.load(filename)
#       xEdges = data[f'{prefix}xEdges']
#       yEdges = data[f'{prefix}yEdges']
#       heights = data[f'{prefix}heights'],
#       errors = data[f'{prefix}errors'],
#     else:
#       raise ValueError(f"Could not load histogram from '{filename}' with label '{label}'.")
#     return Histogram2D(xEdges, yEdges, heights = heights, errors = errors)
#
#   # ============================================================================
#
#   # Convert this Histogram2D to a TH2F.
#   def toRoot(self, name, labels = ""):
#     histogram = root.TH2F(
#       name, labels,
#       self.xLength, array.array("f", list(self.xEdges)),
#       self.yLength, array.array("f", list(self.yEdges))
#     )
#     rnp.array2hist(self.heights, histogram, errors = self.errors)
#     histogram.ResetStats()
#     return histogram
#
#
# # A simple histogram, in one or two dimensions.
# # class Histogram:
# #
# #   # ============================================================================
# #
# #   def __init__(self, bins, range = None, heights = None, errors = None, cov = None):
# #
# #     # Binning parameters for internal usage of np.histogram.
# #     self.bins, self.range = None, None
# #
# #     # Initialize the histogram bin variables.
# #     self.xEdges, self.yEdges = None, None
# #     self.xCenters, self.yCenters = None, None
# #     self.xWidth, self.yWidth = None, None
# #     self.xLen, self.yLen = None, None
# #     self.heights, self.errors, self.cov = None, None, None
# #
# #     # Force the 'bins' and 'range' inputs into 2-tuples (for each dimension) for easier parsing.
# #     if not util.isPair(bins):
# #       bins = (bins, None)
# #     if not util.isPair(range) or util.isNumericPair(range):
# #       range = (range, None)
# #
# #     # Parse the binning input for the first dimension.
# #     if util.isNumber(bins[0]) and util.isNumericPair(range[0]):
# #       self.xEdges = np.linspace(range[0][0], range[0][1], bins[0] + 1)
# #     elif util.isArray(bins[0], 1) and range[0] is None:
# #       self.xEdges = bins[0].copy()
# #     else:
# #       raise ValueError(f"Histogram bins '{bins[0]}' and range '{range[0]}' not understood.")
# #
# #     # Parse the binning input for the second dimension.
# #     if util.isNumber(bins[1]) and util.isNumericPair(range[1]):
# #       self.yEdges = np.linspace(range[1][0], range[1][1], bins[1] + 1)
# #     elif util.isArray(bins[1], 1) and range[1] is None:
# #       self.yEdges = bins[1].copy()
# #     elif bins[1] is None and range[1] is None:
# #       pass
# #     else:
# #       raise ValueError(f"Histogram bins '{bins[1]}' and range '{range[1]}' not understood.")
# #
# #     # Use the parsed bin edges to compute bin centers, widths, etc.
# #     self.updateBins()
# #
# #     # Determine the shape for the bin height and error arrays.
# #     if self.yCenters is None:
# #       shape = (len(self.xCenters),)
# #     else:
# #       shape = (len(self.xCenters), len(self.yCenters))
# #
# #     # Initialize the bin heights: from an argument if supplied, or else zeros.
# #     if heights is not None:
# #       if heights.shape == shape:
# #         self.heights = heights.copy()
# #       else:
# #         raise ValueError(f"Supplied histogram heights do not match bin shape.")
# #     else:
# #       self.heights = np.zeros(shape = shape)
# #
# #     # Initialize the bin errors: from an argument if supplied, or else zeros.
# #     if errors is not None:
# #       if errors.shape == shape:
# #         self.errors = errors.copy()
# #       else:
# #         raise ValueError(f"Supplied histogram errors do not match bin shape.")
# #     else:
# #       self.errors = np.zeros(shape = shape)
# #
# #     # Check that supplied errors and covariance do not clash.
# #     if errors is not None and cov is not None:
# #       raise ValueError(f"Cannot supply both 'errors' and 'cov' to histogram.")
# #
# #     # Initialize the covariance matrix for a 1D histogram: from an argument if supplied, else zeros.
# #     if self.heights.ndim == 1:
# #       if cov is not None:
# #         if cov.shape == (self.xLen, self.xLen):
# #           self.cov = cov.copy()
# #           self.errors[:] = np.sqrt(self.cov.diag())
# #         else:
# #           raise ValueError(f"Covariance matrix does not match the size of the histogram.")
# #     elif cov is not None:
# #       raise ValueError(f"Covariance not supported for 2-dimensional histograms.")
# #
# #   # ============================================================================
# #
# #   # Compute bin centers, widths, and binning parameters for np.histogram.
# #   def updateBins(self, axis = "both"):
# #
# #     if axis.lower() in ["x", "both"]:
# #
# #       # Compute x-axis bin centers and widths.
# #       self.xCenters = (self.xEdges[1:] + self.xEdges[:-1]) / 2
# #       self.xWidth = self.xEdges[1:] - self.xEdges[:-1]
# #       self.xLen = len(self.xCenters)
# #
# #       # If all bins are the same width, update the bin and range format accordingly.
# #       if np.all(np.isclose(self.xWidth, self.xWidth[0])):
# #         self.xWidth = self.xWidth[0]
# #         self.bins = len(self.xCenters)
# #         self.range = [self.xEdges[0], self.xEdges[-1]]
# #       else:
# #         self.bins = self.xEdges
# #         self.range = None
# #
# #     # Process the second dimension, if there is one.
# #     if axis.lower() in ["y", "both"] and self.yEdges is not None:
# #
# #       # Compute y-axis bin centers and widths.
# #       self.yCenters = (self.yEdges[1:] + self.yEdges[:-1]) / 2
# #       self.yWidth = self.yEdges[1:] - self.yEdges[:-1]
# #       self.yLen = len(self.yCenters)
# #
# #       # If all bins are the same width, update the bin and range format accordingly.
# #       if np.all(np.isclose(self.yWidth, self.yWidth[0])):
# #         self.yWidth = self.yWidth[0]
# #         self.bins = [self.bins, len(self.yCenters)]
# #         self.range = [self.range, [self.yEdges[0], self.yEdges[-1]]]
# #       else:
# #         self.bins = [self.bins, self.yEdges]
# #         self.range = None
# #
# #   # ============================================================================
# #
# #   # Add new entries to the histogram.
# #   def fill(self, xValues, yValues = None):
# #
# #     # np.histogram(2d) returns heights and edges, so use index [0] to just take the heights.
# #     if yValues is None:
# #       self.heights += np.histogram(xValues, bins = self.bins, range = self.range)[0]
# #     else:
# #       self.heights += np.histogram2d(xValues, yValues, bins = self.bins, range = self.range)[0]
# #
# #     # Update the bin errors, assuming Poisson statistics.
# #     self.errors = np.sqrt(self.heights)
# #     self.errors[self.errors == 0] = 1
# #
# #     # Update the bin covariances.
# #     if self.cov is not None:
# #       np.fill_diagonal(self.cov, self.errors**2)
# #
# #   # ============================================================================
# #
# #   # Clear the histogram contents.
# #   def clear(self):
# #     self.heights.fill(0)
# #     self.errors.fill(0)
# #     if self.cov is not None:
# #       self.cov.fill(0)
# #
# #   # ============================================================================
# #
# #   # Override the *= operator to scale the bin entries in-place.
# #   def __imul__(self, scale):
# #     self.heights *= scale
# #     if self.cov is not None:
# #       self.cov *= abs(scale)**2
# #       self.errors = np.sqrt(np.diag(self.cov))
# #     else:
# #       self.errors *= abs(scale)
# #     return self
# #
# #   # ============================================================================
# #
# #   # Override the += operator to add bin entries in-place, assuming statistical independence.
# #   def __iadd__(self, other):
# #     self.heights += other.heights
# #     if self.cov is not None and other.cov is not None:
# #       self.cov += other.cov
# #       self.errors = np.sqrt(np.diag(self.cov))
# #     else:
# #       self.errors = np.sqrt(self.errors**2 + other.errors**2)
# #     return self
# #
# #   # ============================================================================
# #
# #   # Transform this histogram's bin edges according to the supplied function.
# #   def remap(self, xFunction, yFunction = None):
# #
# #     # Remap the bin edges according to the given function(s).
# #     self.xEdges = xFunction(self.xEdges)
# #     if yFunction is not None and self.yEdges is not None:
# #       self.yEdges = yFunction(self.yEdges)
# #
# #     # Recompute bin centers, widths, etc.
# #     self.updateBins()
# #
# #   # ============================================================================
# #
# #   # Calculate the mean along each axis.
# #   def mean(self):
# #     if self.heights.ndim == 2:
# #       xAvg = np.average(self.xCenters, weights = np.sum(self.heights, axis = 1))
# #       yAvg = np.average(self.yCenters, weights = np.sum(self.heights, axis = 0))
# #       return xAvg, yAvg
# #     else:
# #       return np.average(self.xCenters, weights = self.heights)
# #
# #   # ============================================================================
# #
# #   # Calculate the standard deviation along each axis.
# #   def std(self):
# #     if self.heights.ndim == 2:
# #       xAvg, yAvg = self.mean()
# #       xStd = np.sqrt(np.average((self.xCenters - xAvg)**2, weights = np.sum(self.heights, axis = 1)))
# #       yStd = np.sqrt(np.average((self.yCenters - yAvg)**2, weights = np.sum(self.heights, axis = 0)))
# #       return xStd, yStd
# #     else:
# #       xAvg = self.mean()
# #       return np.sqrt(np.average((self.xCenters - xAvg)**2, weights = self.heights))
# #
# #   # ============================================================================
# #
# #   def normalize(self, area = True):
# #
# #     if area:
# #       if self.heights.ndim == 2:
# #         scale = np.sum(self.heights * self.xWidth * self.yWidth)
# #       else:
# #         scale = np.sum(self.heights * self.xWidth)
# #     else:
# #       scale = np.sum(self.heights)
# #
# #     self.heights /= scale
# #
# #   # ============================================================================
# #
# #   def transpose(self):
# #
# #     if self.heights.ndim != 2:
# #       raise ValueError("Cannot transpose non-2D histogram.")
# #
# #     self.xEdges, self.yEdges = self.yEdges, self.xEdges
# #     self.updateBins()
# #
# #     self.heights = self.heights.T
# #     self.errors = self.errors.T
# #
# #     return self
# #
# #   # ============================================================================
# #
# #   def copy(self):
# #     return Histogram(
# #       (self.xEdges, self.yEdges),
# #       heights = self.heights,
# #       errors = self.errors,
# #       cov = self.cov
# #     )
# #
# #   # ============================================================================
# #
# #   # TODO: don't discard original data, just make a new view (so we can re-mask)
# #   def mask(self, xRange, yRange = None):
# #
# #     if util.isNumericPair(xRange):
# #       xMinIndex = np.searchsorted(self.xCenters, xRange[0], side = "left")
# #       xMaxIndex = np.searchsorted(self.xCenters, xRange[1], side = "right")
# #
# #       self.xEdges = self.xEdges[xMinIndex:(xMaxIndex + 1)]
# #       self.updateBins("x")
# #
# #       self.heights = self.heights[xMinIndex:xMaxIndex]
# #       self.errors = self.errors[xMinIndex:xMaxIndex]
# #       if self.cov is not None:
# #         self.cov = self.cov[xMinIndex:xMaxIndex, :][:, xMinIndex:xMaxIndex]
# #     else:
# #       raise ValueError(f"Data range '{range}' not understood.")
# #
# #     if util.isNumericPair(yRange) and self.heights.ndim == 2:
# #
# #       yMinIndex = np.searchsorted(self.yCenters, yRange[0], side = "left")
# #       yMaxIndex = np.searchsorted(self.yCenters, yRange[1], side = "right")
# #
# #       self.yEdges = self.yEdges[yMinIndex:(yMaxIndex + 1)]
# #       self.updateBins("y")
# #
# #       self.heights = self.heights[:, yMinIndex:yMaxIndex]
# #       self.errors = self.errors[:, yMinIndex:yMaxIndex]
# #
# #     return self
# #
# #   # ============================================================================
# #
# #   # Merge integer groups of bins along the x- or y-axes.
# #   def rebin(self, xGroup = None, yGroup = None, discard = False):
# #
# #     if xGroup is not None:
# #
# #       # TODO: clean up and do cov
# #       if util.isInteger(xGroup) and xGroup > 0:
# #
# #         groups = self.xLen // xGroup
# #         if self.xLen % xGroup == 0:
# #           heights, errors, cov = self.heights, self.errors, self.cov
# #         elif discard:
# #           heights = self.heights[:(groups * xGroup)]
# #           errors = self.errors[:(groups * xGroup)]
# #           cov = self.cov[:(groups * xGroup), :(groups * xGroup)] if self.cov is not None else None
# #         else:
# #           raise ValueError(f"Cannot rebin {self.xLen} bins into groups of {xGroup}.")
# #
# #         # Split along the first axis in steps of xGroup, stack along a new first axis,
# #         # and sum over the old x-axis (now axis 1).
# #         self.heights = np.stack(np.split(heights, groups, axis = 0)).sum(axis = 1)
# #
# #         if self.cov is not None:
# #           self.cov = np.stack(np.split(cov, groups, axis = 0)).sum(axis = 1)
# #           self.cov = np.stack(np.split(self.cov, groups, axis = 1)).sum(axis = 2).T
# #           self.errors = np.sqrt(np.diag(self.cov))
# #         else:
# #           self.errors = np.sqrt(np.stack(np.split(errors**2, groups, axis = 0)).sum(axis = 1))
# #
# #         self.xEdges = self.xEdges[::xGroup]
# #         self.updateBins("x")
# #
# #       else:
# #         raise ValueError(f"Cannot rebin histogram in groups of '{xGroup}'.")
# #
# #     if yGroup is not None and self.heights.ndim == 2:
# #
# #       if util.isInteger(yGroup) and yGroup > 0:
# #
# #         groups = self.yLen // yGroup
# #         if self.yLen % yGroup == 0:
# #           heights, errors = self.heights, self.errors
# #         elif discard:
# #           heights, errors = self.heights[:, :(groups * yGroup)], self.errors[:, :(groups * yGroup)]
# #         else:
# #           raise ValueError(f"Cannot rebin {self.yLen} bins into groups of {yGroup}.")
# #
# #         # Split along the second axis in steps of yGroup, stack along a new first axis, sum over
# #         # the old y-axis (now axis 2), and transpose.
# #         self.heights = np.stack(np.split(heights, groups, axis = 1)).sum(axis = 2).T
# #         self.errors = np.stack(np.split(errors**2, groups, axis = 1)).sum(axis = 2).T
# #
# #         self.yEdges = self.yEdges[::yGroup]
# #         self.updateBins("y")
# #
# #       else:
# #         raise ValueError(f"Cannot rebin histogram in groups of '{yGroup}'.")
# #
# #     return self
# #
# #   # ============================================================================
# #
# #   # Plot this histogram.
# #   def plot(self, errors = False, normalize = False, bar = False, label = None, **kwargs):
# #
# #     # Normalize the data if requested.
# #     if normalize:
# #       binSize = self.xWidth * self.yWidth if self.heights.ndim == 2 else self.xWidth
# #       heights = self.heights / np.sum(self.heights * binSize)
# #     else:
# #       heights = self.heights
# #
# #     # Show a 2D histogram with a colorbar.
# #     if self.heights.ndim == 2:
# #
# #       # Replace empty bins with np.nan, which draws them blank.
# #       return style.colormesh(
# #         self.xEdges,
# #         self.yEdges,
# #         np.where(heights == 0, np.nan, heights),
# #         **kwargs
# #       )
# #
# #     # Show a 1D histogram as a line, bar, or errorbar plot.
# #     else:
# #
# #       # Set the x and y limits.
# #       plt.xlim(self.xEdges[0], self.xEdges[-1])
# #       if np.max(heights) > 0:
# #         plt.ylim(0, np.max(heights) * 1.05)
# #
# #       if bar:
# #
# #         return plt.bar(
# #           self.xCenters,
# #           heights,
# #           yerr = self.errors if errors else None,
# #           width = self.xWidth,
# #           label = label,
# #           linewidth = 0.5,
# #           edgecolor = "k",
# #           **kwargs
# #         )
# #
# #       else:
# #
# #         if errors:
# #           return style.errorbar(
# #             self.xCenters,
# #             heights,
# #             self.errors,
# #             label = label,
# #             **kwargs
# #           )
# #         else:
# #           return plt.plot(self.xCenters, heights, label = label, **kwargs)
# #
# #   # ============================================================================
# #
# #   # Save this histogram to disk in NumPy format.
# #   def save(self, filename):
# #     np.savez(filename, **self.collect())
# #
# #   # Collect this histogram's data into a dictionary with keyword labels.
# #   def collect(self, path = None):
# #
# #     prefix = "" if path is None else f"{path}/"
# #
# #     # For a default one-dimensional histogram.
# #     result = {
# #       f"{prefix}xEdges": self.xEdges,
# #       f"{prefix}xCenters": self.xCenters,
# #       f"{prefix}heights": self.heights,
# #       f"{prefix}errors": self.errors
# #     }
# #
# #     if self.cov is not None:
# #       result[f"{prefix}xCov"] = self.cov
# #
# #     # If there's a second dimension, add the bin data.
# #     if self.heights.ndim == 2:
# #       result[f"{prefix}yEdges"] = self.yEdges
# #       result[f"{prefix}yCenters"] = self.yCenters
# #
# #     return result
# #
# #   # ============================================================================
# #
# #   # Load a histogram previously saved to disk in NumPy format.
# #   @classmethod
# #   def load(cls, filename, label = None):
# #
# #     if filename.endswith(".root"):
# #
# #       return Histogram.fromRoot(filename, label)
# #
# #     elif filename.endswith(".npz"):
# #
# #       prefix = "" if label is None else f"{label}/"
# #       data = np.load(filename)
# #
# #       xEdges = data[f'{prefix}xEdges']
# #       yEdges = data[f'{prefix}yEdges'] if f'{prefix}yEdges' in data.keys() else None
# #       cov = data[f"{prefix}xCov"] if f"{prefix}xCov" in data.keys() else None
# #
# #       # Recreate the histogram object.
# #       return cls(
# #         (xEdges, yEdges),
# #         heights = data[f'{prefix}heights'],
# #         errors = data[f'{prefix}errors'],
# #         cov = cov
# #       )
# #
# #     else:
# #       raise ValueError("Could not load histogram.")
# #
# #   # ============================================================================
# #
# #   # Convert this NumPy-style histogram to a ROOT-style TH1 or TH2.
# #   def toRoot(self, name, labels = ""):
# #
# #     # If there's a second dimension, create a TH2.
# #     if self.heights.ndim == 2:
# #
# #       histogram = root.TH2F(
# #         name, labels,
# #         self.xLen, self.xEdges[0], self.xEdges[-1],
# #         self.yLen, self.yEdges[0], self.yEdges[-1]
# #       )
# #
# #     # If there's no second dimension, create a TH1.
# #     else:
# #
# #       histogram = root.TH1F(
# #         name, labels,
# #         len(self.xCenters), self.xEdges[0], self.xEdges[-1]
# #       )
# #
# #     # Copy the bin contents and update the number of entries.
# #     rnp.array2hist(self.heights, histogram, errors = self.errors)
# #     histogram.ResetStats()
# #
# #     return histogram
# #
# #   # ============================================================================
# #
# #   @classmethod
# #   def fromRoot(cls, filename, label):
# #
# #     # Get the histogram from the file.
# #     rootFile = root.TFile(filename)
# #     histogram = rootFile.Get(label)
# #
# #     # Get the heights and edges using root_numpy.
# #     heights, edges = rnp.hist2array(histogram, return_edges = True)
# #
# #     # Copy the bin errors manually...
# #     errors = np.zeros(shape = heights.shape)
# #     if len(edges) == 1:
# #       for i in range(histogram.GetNbinsX()):
# #         errors[i] = histogram.GetBinError(i + 1)
# #     elif len(edges) == 2:
# #       for i in range(histogram.GetNbinsX()):
# #         for j in range(histogram.GetNbinsY()):
# #           errors[i][j] = histogram.GetBinError(i + 1, j + 1)
# #     else:
# #       raise ValueError("Dimensions of ROOT histogram not supported.")
# #
# #     # Replace any empty errors with one.
# #     errors[errors == 0] = 1
# #
# #     # Close the root file.
# #     rootFile.Close()
# #
# #     # Construct the new Histogram object.
# #     return cls(
# #       bins = tuple(edges) if len(edges) > 1 else edges[0],
# #       heights = heights,
# #       errors = errors
# #     )
