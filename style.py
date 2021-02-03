import matplotlib.pyplot as plt
import numpy as np

# ==============================================================================

# Set the default plotting options.
def setStyle():

  # TODO: if LaTeX is turned off, make the font a bit smaller
  # Font options.
  plt.rcParams["font.size"] = 16
  plt.rcParams["font.family"] = "serif"
  plt.rcParams["axes.labelsize"] = 16
  plt.rcParams["axes.titlesize"] = 16
  plt.rcParams["xtick.labelsize"] = 16
  plt.rcParams["ytick.labelsize"] = 16
  plt.rcParams["legend.fontsize"] = 14

  # LaTeX options.
  plt.rcParams["text.usetex"] = True
  plt.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"

  # Marker and line options.
  plt.rcParams["lines.markersize"] = 4
  plt.rcParams["lines.linewidth"] = 2

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

# Keyword arguments for colorbar formatting.
# colorbarStyle = {"pad": 0.01, "fraction": 0.04, "aspect": 18}

# Override plt.colorbar with automatic formatting.
def colorbar(
  cLabel = None,
  pad = 0.01,
  fraction = 0.06,
  aspect = 18,
  **kwargs
):
  cbar = plt.colorbar(pad = pad, fraction = fraction, aspect = aspect, **kwargs)
  if cLabel is not None:
    cbar.set_label(cLabel, ha = "right", y = 1)
  return cbar

# ==============================================================================

# Override plt.errorbar with automatic formatting.
# TODO: what was "obj" for? I don't even remember... is it still necessary?
def errorbar(x, y, err, obj = plt, fmt = "o", ms = 4, **kwargs):
  return obj.errorbar(
    x,
    y,
    err,
    fmt = fmt,
    ms = ms,
    capsize = 2,
    linewidth = 1,
    **kwargs
  )

# ==============================================================================

# TODO: add a non-latex option, without alignment; check rcParams[latex] to switch
def databox(*args):

  if plt.rcParams["text.usetex"]:

    string = r"\begin{align*}"
    for arg in args:
      string += f"{arg[0]} &= {arg[1]:.4f}"
      string += fr" \; \text{{{arg[2]}}}" if arg[2] is not None else ""
      string += r" \\[-0.5ex]"
    string += r"\end{align*}"

  else:

    string = ""
    for arg in args:
      string += f"${arg[0]}$ = {arg[1]:.4f}"
      string += f" {arg[2]}" if arg[2] is not None else ""
      string += "\n"

  plt.text(
    0.03,
    0.96,
    string,
    ha = "left",
    va = "top",
    transform = plt.gca().transAxes
  )

# ==============================================================================

# Keyword arguments for image formatting.
imageStyle = {"cmap": "jet", "origin": "lower", "aspect": "auto"}

# Override plt.imshow with automatic formatting and colorbar.
def imshow(x, y, heights, cLabel = None, **kwargs):
  result = plt.imshow(
    heights.T,
    extent = (x[0], x[-1], y[0], y[-1]),
    **imageStyle,
    **kwargs
  )
  cbar = colorbar(cLabel)
  return result, cbar

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
