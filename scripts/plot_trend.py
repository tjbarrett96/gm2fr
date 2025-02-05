import numpy as np
import matplotlib.pyplot as plt
import gm2fr.src.style as style
style.set_style()
import sys
import gm2fr.src.io as io
import gm2fr.src.constants as const
import argparse
from gm2fr.src.Results import Results

# ==================================================================================================

def plot_trend(x, y, results, label = None, ls = "-", color = None, skip = 1):

  x_data = results[x] if x in results.dtype.names else x
  y_data = results[y]

  if f"err_{y}" in results.dtype.names:
    errors = results[f"err_{y}"]
    # Don't plot error bars if they obscure the variation of the data.
    y_range = np.abs(np.max(y_data) - np.min(y_data))
    errors[errors > 2 * y_range] = 0
  else:
    errors = None

  if y == "t0": # patch the output for t0 being in microseconds
    y_data = y_data * 1E3
    errors = errors * 1E3

  skip = int(skip)

  errorbar = style.errorbar(
    x_data[::skip],
    y_data[::skip],
    errors[::skip] if errors is not None else errors,
    ls = ls,
    label = label,
    color = color
  )

  if f"ref_{y}" in results.dtype.names:
    plt.axhline(results[f"ref_{y}"][0], ls = "--", color = errorbar[0].get_color(), label = "MC Truth")

  return errorbar

# ==================================================================================================

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--results", "-r", required = True, nargs = "+")
  parser.add_argument("-x", required = True)
  parser.add_argument("-y", required = True)
  parser.add_argument("-z", default = None)
  args = parser.parse_args()

  if len(args.results) > 1 and args.z is not None:
    raise ValueError("Cannot make contour plot from multiple results files.")

  for filename in args.results:
    results = Results.load(filename)
    results.plot(args.x, args.y, args.z)

  plt.show()
