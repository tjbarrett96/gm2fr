import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import gm2fr.src.constants as const

# ==================================================================================================

# Set the default plotting options.
def set_style(latex = False):

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
  plt.rcParams["axes.formatter.limits"] = (-2, 5)
  plt.rcParams["axes.formatter.use_mathtext"] = True

  # Marker and line options.
  plt.rcParams["lines.markersize"] = 5
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

# ==================================================================================================

# Override plt.xlabel with automatic formatting.
def xlabel(label):
  return plt.xlabel(label, ha = "right", x = 1)

# Override plt.ylabel with automatic formatting.
def ylabel(label):
  return plt.ylabel(label, ha = "right", y = 1)

# ==================================================================================================

def twinx(right = None):
  plt.twinx()
  plt.grid(False)
  if right is not None:
    plt.subplots_adjust(right = right)

# ==================================================================================================

# Override plt.colorbar with automatic formatting.
def colorbar(label = None, pad = 0.01, fraction = 0.10, aspect = 18, **kwargs):
  cbar = plt.colorbar(pad = pad, fraction = fraction, aspect = aspect, **kwargs)
  if label is not None:
    cbar.set_label(label, ha = "right", y = 1)
  return cbar

# ==================================================================================================

# Override plt.errorbar with automatic formatting.
def errorbar(x, y, yErr, xErr = None, ls = "-", marker = "o", ms = 4, **kwargs):
  return plt.errorbar(x, y, yErr, xErr, fmt = f"{marker}{ls}", ms = ms, capsize = 2, lw = 0.75, elinewidth = 0.5, mew = 0.5, **kwargs)

# ==================================================================================================

# Show a legend on the current plot, containing only unique labels without duplicates.
def make_unique_legend(extend_x = 0, **kwargs):
  # Get the artist handles and text labels for everything in the current plot.
  handles, labels = plt.gca().get_legend_handles_labels()
  # Make a dictionary mapping labels to handles; this ensures each label only appears with one handle.
  labels_to_handles = {label: handle for label, handle in zip(labels, handles)}
  # Make a legend, as long as there are some labels to show.
  if len(labels_to_handles) > 0:
    if extend_x > 0:
      xLow, xHigh = plt.xlim()
      plt.xlim(xLow, xHigh + extend_x * (xHigh - xLow))
    plt.legend(handles = labels_to_handles.values(), labels = labels_to_handles.keys(), **kwargs)

# ==================================================================================================

def set_physical_limits(unit = "f"):
  plt.xlim(const.info[unit].min, const.info[unit].max)

# ==================================================================================================

# Shortcut for labeling axes, adding legend, saving the figure, and clearing the plot.
def label_and_save(xLabel, yLabel, output, **legend_kwargs):
  xlabel(xLabel)
  ylabel(yLabel)
  make_unique_legend(**legend_kwargs)
  if isinstance(output, PdfPages):
    output.savefig()
  else:
    plt.savefig(output)
  plt.clf()

# ==================================================================================================

class Entry:

  def __init__(self, val, sym, err = None, units = None):
    (self.val, self.err) = (val, err)
    (self.sym, self.units) = (sym.symbol, sym.units) if isinstance(sym, const.Quantity) else (sym, units)

  def format(self, align = False, places = 4):
    m = "" if align else "$" # math mode boundary character: "" if already inside math env, else "$"
    amp = "&" if align else "" # alignment character: "&" if inside math align env, else ""
    err = "" if self.err is None else rf" {m}\pm{m} {self.err:.{places}f}"
    units = "" if self.units is None else (rf"\;\text{{{self.units}}}" if align else f" {self.units}")
    return rf"{m}{self.sym}{m} {amp}= {self.val:.{places}f}{err}{units}"

def databox(*entries, left = True):

  if plt.rcParams["text.usetex"]:
    string = r"\begin{align*}" + r"\\[-0.5ex]".join([entry.format(align = True) for entry in entries]) + r"\end{align*}"
  else:
    string = "\n".join([entry.format(align = False) for entry in entries])

  plt.text(
    0.03 if left else 0.97,
    0.96,
    string,
    ha = "left" if left else "right",
    va = "top",
    transform = plt.gca().transAxes
  )

# ==================================================================================================

# Override plt.imshow with automatic formatting and colorbar.
def colormesh(x, y, heights, label = None, cmap = "coolwarm", **kwargs):
  result = plt.pcolormesh(x, y, heights.T, cmap = cmap, **kwargs)
  cbar = colorbar(label)
  return result, cbar

# ==================================================================================================

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

# ==================================================================================================

# Draw a horizontal line.
def draw_horizontal(y = 0, ls = ":", c = "k", **kwargs):
  return plt.axhline(y, linestyle = ls, color = c, **kwargs)

# Draw a vertical line.
def draw_vertical(x = 0, ls = ":", c = "k", **kwargs):
  return plt.axvline(x, linestyle = ls, color = c, **kwargs)

def horizontal_spread(width, y = 0, color = "k", **kwargs):
  return plt.axhspan(y - width/2, y + width/2, color = color, alpha = 0.1, **kwargs)

def vertical_spread(width, x = 0, color = "k", **kwargs):
  return plt.axvspan(x - width/2, x + width/2, color = color, alpha = 0.1, **kwargs)

# ==================================================================================================

def x_stats(avg, std, units = None, color = "k"):

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

# ==================================================================================================

def y_stats(
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

# ==================================================================================================

def make_pdf(path):
  return PdfPages(path)
