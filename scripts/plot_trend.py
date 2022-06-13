import numpy as np
import matplotlib.pyplot as plt
import gm2fr.style as style
style.set_style()
import sys
import gm2fr.io as io
import gm2fr.constants as const

# ==================================================================================================

def plot_trend(x, y, filename, label = None, ls = "-", color = None):

  # results = np.load(f"{io.gm2fr_path}/analysis/results/{filename}/results.npy")
  results = np.load(filename, allow_pickle = True)

  if f"err_{y}" in results.dtype.names:
    errors = results[f"err_{y}"]
    errors = np.where(errors > 10 * np.median(errors), 0, errors)
  else:
    errors = None

  x_data = results[x] if x in results.dtype.names else x
  if x == "t0": # patch the output for t0 being in microseconds
    x_data *= 1E3

  errorbar = style.errorbar(
    x_data,
    results[y],
    errors,
    ls = ls,
    label = label,
    color = color
  )

  if f"ref_{y}" in results.dtype.names:
    plt.axhline(results[f"ref_{y}"][0], ls = "--", color = errorbar[0].get_color())

  return x_data, results[y], errors

# ==================================================================================================

if __name__ == "__main__":

  if len(sys.argv) < 4:
    print("Usage: python3 plot_trend.py <var_x> <var_y> [dir_name(s)]")
    quit()

  x, y = sys.argv[1], sys.argv[2]
  filenames = sys.argv[3:]

  for filename in filenames:
    plot_trend(x, y, filename, label = filename)

  style.xlabel(const.info[x].format_label() if x in const.info.keys() else x)
  style.ylabel(const.info[y].format_label() if y in const.info.keys() else y)
  plt.legend()

  plt.show()
