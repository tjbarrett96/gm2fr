import numpy as np
import matplotlib.pyplot as plt
import gm2fr.style as style
style.setStyle()
import sys
import gm2fr.io as io
import gm2fr.constants as const

# ==================================================================================================

def plot_trend(x, y, filename, label = None, ls = "-", color = None):
  # results = np.load(f"{io.gm2fr_path}/analysis/results/{filename}/results.npy")
  results = np.load(filename, allow_pickle = True)
  style.errorbar(
    results[x] if x in results.dtype.names else x,
    results[y],
    results[f"err_{y}"] if f"err_{y}" in results.dtype.names else None,
    ls = ls,
    label = label,
    color = color
  )

# ==================================================================================================

if __name__ == "__main__":

  if len(sys.argv) < 4:
    print("Usage: python3 plot_trend.py <var_x> <var_y> [dir_name(s)]")
    quit()

  x, y = sys.argv[1], sys.argv[2]
  filenames = sys.argv[3:]

  for filename in filenames:
    plot_trend(x, y, filename, label = filename)

  style.xlabel(const.info[x].formatLabel() if x in const.info.keys() else x)
  style.ylabel(const.info[y].formatLabel() if y in const.info.keys() else y)
  plt.legend()

  plt.show()
