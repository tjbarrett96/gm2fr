import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import gm2fr.utilities as util

# ==============================================================================

# Set the default plotting options.
def setStyle(latex = False):

  # LaTeX options.
  if latex:
    plt.rcParams["text.usetex"] = True
    plt.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
    plt.rcParams["font.family"] = "serif"

  # Font options.
  size = 16 if latex else 14
  plt.rcParams["font.size"] = size * 0.75
  plt.rcParams["axes.labelsize"] = size
  plt.rcParams["axes.titlesize"] = size
  plt.rcParams["xtick.labelsize"] = size
  plt.rcParams["ytick.labelsize"] = size
  plt.rcParams["legend.fontsize"] = size * 0.75

  # Rules for switching to scientific notation in axis tick labels.
  plt.rcParams["axes.formatter.limits"] = (-4, 5)
  plt.rcParams["axes.formatter.use_mathtext"] = True

  # Marker and line options.
  plt.rcParams["lines.markersize"] = 3
  plt.rcParams["lines.linewidth"] = 1

  # Draw grid.
  plt.rcParams["axes.grid"] = True
  plt.rcParams["grid.alpha"] = 0.25

  # Default figure size.
  plt.rcParams["figure.figsize"] = 8, 5
  plt.rcParams["figure.dpi"] = 100

  # Make axis tick marks face inward.
  plt.rcParams["xtick.direction"] = "in"
  plt.rcParams["ytick.direction"] = "in"

  # Draw axis tick marks all around the edges.
  plt.rcParams["xtick.top"] = True
  plt.rcParams["ytick.right"] = True

  # Draw minor axis tick marks in between major labels.
  plt.rcParams["xtick.minor.visible"] = True
  plt.rcParams["ytick.minor.visible"] = True

  # Make all tick marks longer.
  plt.rcParams["xtick.major.size"] = 8
  plt.rcParams["xtick.minor.size"] = 4
  plt.rcParams["ytick.major.size"] = 8
  plt.rcParams["ytick.minor.size"] = 4

  # Dynamically choose the number of histogram bins.
  plt.rcParams["hist.bins"] = "auto"

  # Space between axes and legend, in font units.
  plt.rcParams["legend.borderaxespad"] = 1
  plt.rcParams["legend.handlelength"] = 1
  plt.rcParams["legend.columnspacing"] = 1.4
  plt.rcParams["legend.handletextpad"] = 0.6

  # Default subplot spacing.
  plt.rcParams["figure.subplot.wspace"] = 0.3
  plt.rcParams["figure.subplot.hspace"] = 0.3
  plt.rcParams["figure.subplot.top"] = 0.93
  plt.rcParams["figure.subplot.right"] = 0.93

# ==============================================================================

# Override plt.xlabel with automatic formatting.
def xlabel(label):
  return plt.xlabel(label, ha = "right", x = 1)

# Override plt.ylabel with automatic formatting.
def ylabel(label):
  return plt.ylabel(label, ha = "right", y = 1)

# ==============================================================================

# Override plt.colorbar with automatic formatting.
def colorbar(
  label = None,
  pad = 0.01,
  fraction = 0.10,
  aspect = 18,
  **kwargs
):
  cbar = plt.colorbar(pad = pad, fraction = fraction, aspect = aspect, **kwargs)
  if label is not None:
    cbar.set_label(label, ha = "right", y = 1)
  return cbar

# ==============================================================================

# Override plt.errorbar with automatic formatting.
def errorbar(x, y, yErr, xErr = None, fmt = "o", ms = 3, **kwargs):

  return plt.errorbar(
    x,
    y,
    yErr,
    xErr,
    fmt = fmt,
    ms = ms,
    capsize = 2,
    linewidth = 1,
    **kwargs
  )

# ==============================================================================

# TODO: add a non-latex option, without alignment; check rcParams[latex] to switch
def databox(*args, left = True):

  if plt.rcParams["text.usetex"]:

    string = r"\begin{align*}"
    for arg in args:
      string += f"{arg[0]} &= {arg[1]:.4f}"
      string += fr" \pm {arg[2]:.4f}" if arg[2] is not None else ""
      string += fr" \; \text{{{arg[3]}}}" if arg[3] is not None else ""
      string += r" \\[-0.5ex]"
    string += r"\end{align*}"

  else:

    string = ""
    for arg in args:
      string += f"${arg[0]}$ = {arg[1]:.4f}"
      string += fr" $\pm$ {arg[2]:.4f}" if arg[2] is not None else ""
      string += f" {arg[3]}" if arg[3] is not None else ""
      string += "\n"

  plt.text(
    0.03 if left else 0.97,
    0.96,
    string,
    ha = "left" if left else "right",
    va = "top",
    transform = plt.gca().transAxes
  )

# ==============================================================================

# Override plt.imshow with automatic formatting and colorbar.
def imshow(
  heights,
  x = None,
  y = None,
  label = None,
  cmap = "coolwarm",
  origin = "lower",
  aspect = "auto",
  extent = None,
  **kwargs
):

  if x is not None and y is not None:
    dx, dy = x[1] - x[0], y[1] - y[0]
    extent = (x[0] - dx/2, x[-1] + dx/2, y[0] - dy/2, y[-1] + dy/2)

  result = plt.imshow(
    heights.T,
    extent = extent,
    cmap = cmap,
    origin = origin,
    aspect = aspect,
    **kwargs
  )
  cbar = colorbar(label)
  return result, cbar

# ==============================================================================

# TODO
def collimators(axis = "f"):
  plt.axvspan(plt.xlim()[0], util.min[axis])
  plt.axvspan(util.max[axis], plt.xlim()[-1])

# ==============================================================================

# Keyword arguments for dotted lines.
dotStyle = {"linestyle": ":", "color": "k"}

# Draw a horizontal line at y = 0.
def yZero():
  return plt.axhline(0, **dotStyle)

# Draw a vertical line at x = 0.
def xZero():
  return plt.axvline(0, **dotStyle)

# ==============================================================================

def xStats(avg, std, units = None, color = "k"):

  # Remember the current y-limits.
  yMin, yMax = plt.ylim()

  # Plot the average line.
  line = plt.axvline(avg, linestyle = "--", color = color, linewidth = 1)

  # Plot the spread.
  fill = plt.fill_between(
    [avg - std, avg + std],
    [0, 0],
    [yMax, yMax],
    alpha = 0.1,
    linewidth = 0,
    color = color
  )

  # Restore the original y-limits.
  plt.ylim(yMin, yMax)

  # Return the plot objects and legend label.
  label = f"${avg:.2f} \pm {std:.2f}$" + f" {units}" if units is not None else ""
  return line, fill, label

# ==============================================================================

def yStats(
  x,
  y,
  weights = None,
  units = None,
  width = True,
  xLow = None,
  xHigh = None,
  color = "k"
):

  # Remember the current x-axis viewing range.
  axLeft, axRight = plt.xlim()

  # Set the x-axis limits for the average and standard deviation.
  xLow = axLeft if xLow is None else xLow
  xHigh = axRight if xHigh is None else xHigh
  mask = (x >= xLow) & (x <= xHigh)

  # Calculate the (weighted) average and spread.
  avg = np.average(y[mask], weights = weights)
  std = np.sqrt(np.average((y[mask] - avg)**2, weights = weights))

  # Plot the horizontal average line.
  line, = plt.plot(
    [xLow, xHigh],
    [avg, avg],
    "--",
    linewidth = 1,
    color = color,
    zorder = 0
  )

  # Plot a horizontal shaded region for the spread.
  fill = None
  if width:
    fill = plt.fill_between(
      [xLow, xHigh],
      [avg - std, avg - std],
      [avg + std, avg + std],
      alpha = 0.1,
      color = color,
      linewidth = 0
    )

  # Restore the original axis limits.
  plt.xlim(axLeft, axRight)

  label = f"${avg:.2f} \pm {std:.2f}$" + f" {units}" if units is not None else ""
  return line, fill, label
