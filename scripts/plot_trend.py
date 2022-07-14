import numpy as np
import matplotlib.pyplot as plt
import gm2fr.src.style as style
style.set_style()
import sys
import gm2fr.src.io as io
import gm2fr.src.constants as const

# ==================================================================================================

def plot_trend(x, y, results, label = None, ls = "-", color = None):

  if f"err_{y}" in results.dtype.names:
    errors = results[f"err_{y}"]
  else:
    errors = None

  median_err = np.median(errors)
  errors[errors > 10*median_err] = 0

  x_data = results[x] if x in results.dtype.names else x
  y_data = results[y]

  if "wg_N" in results.dtype.names and len(results) > 1:
    mask = (results["wg_N"] > 5000)
    x_data, y_data, errors = x_data[mask], y_data[mask], errors[mask]

  if y == "t0": # patch the output for t0 being in microseconds
    y_data = y_data * 1E3
    errors = errors * 1E3

  errorbar = style.errorbar(
    x_data,
    y_data,
    errors,
    ls = ls,
    label = label,
    color = color
  )

  if f"ref_{y}" in results.dtype.names:
    plt.axhline(results[f"ref_{y}"][0], ls = "--", color = errorbar[0].get_color(), label = "MC Truth")

  return errorbar

# ==================================================================================================

if __name__ == "__main__":

  if len(sys.argv) < 4:
    print("Usage: python3 plot_trend.py <var_x> <var_y> [dir_name(s)]")
    quit()

  x, y = sys.argv[1], sys.argv[2]
  filenames = sys.argv[3:]

  for filename in filenames:
    results = np.load(filename)
    plot_trend(x, y, results, label = filename)

  style.xlabel(const.info[x].format_label() if x in const.info.keys() else x)
  style.ylabel(const.info[y].format_label() if y in const.info.keys() else y)
  plt.legend()

  plt.show()
