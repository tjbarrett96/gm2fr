import numpy as np

import matplotlib.pyplot as plt
import gm2fr.style as style

import ROOT as root
import root_numpy as rnp

# ==============================================================================

# A simple histogram, in one or two dimensions.
class Histogram:

  # ============================================================================

  # Create the histogram, formatting the bin definitions as tuples (lowerEdge, upperEdge, binWidth).
  def __init__(self, xRange, yRange = None):
    
    # Unpack the bin range tuple as (lowerEdge, upperEdge, binWidth).
    xMin, xMax, self.xWidth = xRange[0], xRange[1], xRange[2]
    
    # Create the lists of bin edges and centers.
    self.xEdges = np.arange(xMin, xMax + self.xWidth, self.xWidth)
    self.xCenters = self.xEdges[1:] - self.xWidth / 2
    
    if yRange is None:
      # Allocate bin heights for a 1D histogram.
      self.heights = np.zeros(shape = len(self.xCenters))
    else:
      # Allocate bins and heights for a 2D histogram.
      yMin, yMax, self.yWidth = yRange[0], yRange[1], yRange[2]
      self.yEdges = np.arange(yMin, yMax + self.yWidth, self.yWidth)
      self.yCenters = self.yEdges[1:] - self.yWidth / 2
      self.heights = np.zeros(shape = (len(self.xCenters), len(self.yCenters)))
      
    # Allocate bin errors.
    self.errors = np.zeros(shape = self.heights.shape)
  
  # ============================================================================
  
  # Add new entries to the histogram.
  def fill(self, xValues, yValues = None):
  
    # 1D filling done manually for speed, because we use this a lot for the fast rotation signal.
    if yValues is None:
    
      # Discard any entries which are out of range.
      xValues = xValues[(xValues >= self.xEdges[0]) & (xValues < self.xEdges[-1])]
      
      if len(xValues) > 0:
      
        # Calculate the bin index for each entry.
        indices = ((xValues - self.xEdges[0]) / self.xWidth).astype(np.int64)
        
        # Count the total for each unique bin index (np.bincount), relative to the minimum index.
        # Increment only the necessary bins, rather than all of them, which gives the speed boost.
        minIndex, maxIndex = np.min(indices), np.max(indices)
        self.heights[minIndex:(maxIndex + 1)] += np.bincount(indices - minIndex)
    
    # 2D filling uses np.histogram2d, which is slower, but simpler and fine for our applications.
    else:
    
      # np.histogram2d returns (heights, xEdges, yEdges), so just take [0], the heights.
      self.heights += np.histogram2d(xValues, yValues, bins = (self.xEdges, self.yEdges))[0]
    
    # Update the bin errors, assuming Poisson statistics.
    self.errors = np.sqrt(self.heights)
    
  # ============================================================================
  
  # Clear the histogram contents.
  def clear(self):
    self.heights.fill(0)
    self.errors.fill(0)
    
  # ============================================================================
  
  # Override the *= operator to scale the bin entries in-place.
  def __imul__(self, scale):
  
    self.heights *= scale
    self.errors *= scale
    
    return self
    
  # ============================================================================
  
  # Calculate the mean along each axis.
  def mean(self):
  
    if self.heights.ndim == 2:
    
      xAvg = np.average(self.xCenters, weights = np.sum(self.heights, axis = 1))
      yAvg = np.average(self.yCenters, weights = np.sum(self.heights, axis = 0))
      return xAvg, yAvg
      
    else:
    
      return np.average(self.xCenters, weights = self.heights)
      
  # ============================================================================
      
  # Calculate the standard deviation along each axis.
  def std(self):
  
    if self.heights.ndim == 2:
    
      xAvg, yAvg = self.mean()
      xStd = np.sqrt(np.average((self.xCenters - xAvg)**2, weights = np.sum(self.heights, axis = 1)))
      yStd = np.sqrt(np.average((self.yCenters - yAvg)**2, weights = np.sum(self.heights, axis = 0)))
      return xStd, yStd
      
    else:
    
      xAvg = self.mean()
      return np.sqrt(np.average((self.xCenters - xAvg)**2, weights = self.heights))
    
  # ============================================================================
  
  # Plot this histogram.
  def plot(self, errors = False, normalize = False, bar = False, label = None, **kwargs):
  
    # Normalize the data if requested.
    if normalize:
      binSize = self.xWidth * self.yWidth if self.heights.ndim == 2 else self.xWidth
      heights = self.heights / np.sum(self.heights * binSize)
    else:
      heights = self.heights
  
    # Show a 2D histogram with a colorbar.
    if self.heights.ndim == 2:
    
      # Replace empty bins with np.nan, which draws them blank.
      return style.imshow(
        self.xEdges,
        self.yEdges,
        np.where(heights == 0, np.nan, heights),
        **kwargs
      )
      
    # Show a 1D histogram as a line, bar, or errorbar plot.
    else:
        
      # Set the x and y limits.
      plt.xlim(self.xEdges[0], self.xEdges[-1])
      if np.max(heights) > 0:
        plt.ylim(0, np.max(heights) * 1.05)
    
      if bar:
      
        return plt.bar(
          self.xCenters,
          heights,
          yerr = self.errors if errors else None,
          width = self.xWidth,
          label = label,
          **kwargs
        )
        
      else:
      
        if errors:
          return style.errorbar(
            self.xCenters,
            heights,
            self.errors,
            label = label,
            **kwargs
          )
        else:
          return plt.plot(self.xCenters, heights, label = label, **kwargs)
      
  # ============================================================================
  
  # Save this histogram to disk in NumPy format.
  def save(self, filename):
  
    if self.heights.ndim == 2:
    
      np.savez(
        filename,
        xRange = (self.xEdges[0], self.xEdges[-1], self.xWidth),
        yRange = (self.yEdges[0], self.yEdges[-1], self.yWidth),
        heights = self.heights,
        errors = self.errors
      )
      
    else:
    
      np.savez(
        filename,
        xRange = (self.xEdges[0], self.xEdges[-1], self.xWidth),
        heights = self.heights,
        errors = self.errors
      )
      
  # ============================================================================
  
  # Load a histogram previously saved to disk in NumPy format.
  @classmethod
  def load(cls, filename):
  
    data = np.load(filename)
    
    if 'yRange' in data.keys():
    
      # Recreate the 2D histogram object.
      obj = cls(data['xRange'], data['yRange'])
      
    else:
    
      # Recreate the 1D histogram object.
      obj = cls(data['xRange'])
      
    # Set the bin heights and errors.
    obj.heights = data['heights']
    obj.errors = data['errors']
    
    return obj
    
  # ============================================================================
  
  # Convert this NumPy-style histogram to a ROOT-style TH1 or TH2.
  def toRoot(self, name, labels = "", xRescale = 1, yRescale = 1):
  
    # If there's a second dimension, create a TH2.
    if self.heights.ndim == 2:
    
      histogram = root.TH2F(
        name,
        labels,
        len(self.xCenters),
        self.xEdges[0] * xRescale,
        self.xEdges[-1] * xRescale,
        len(self.yCenters),
        self.yEdges[0] * yRescale,
        self.yEdges[-1] * yRescale
      )
   
    # If there's no second dimension, create a TH1.
    else:
    
      histogram = root.TH1F(
        name,
        labels,
        len(self.xCenters),
        self.xEdges[0] * xRescale,
        self.xEdges[-1] * xRescale
      )
    
    # Copy the bin contents and update the number of entries.
    rnp.array2hist(self.heights, histogram)
    histogram.ResetStats()
    
    return histogram
