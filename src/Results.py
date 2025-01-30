import numpy as np
import pandas as pd
import gm2fr.src.style as style
style.set_style()
import matplotlib.pyplot as plt
import gm2fr.src.constants as const

# ==============================================================================

class Results:

  def __init__(self, data = None):
    self.table = pd.DataFrame(data, index = [0] if isinstance(data, dict) else None)

  # ============================================================================

  def merge(self, *results):

    merged = pd.concat(
      (self.table, *[result.table for result in results]),
      axis = "columns"
    )

    # Take a slice which removes any duplicated columns.
    self.table = merged.loc[:, ~merged.columns.duplicated()].copy()

  # ============================================================================

  def append(self, result):
    # self.table = self.table.append(result.table)
    self.table = pd.concat((self.table, result.table), ignore_index = True)

  # ============================================================================

  def array(self):
    return self.table.to_records(index = False)

  # ============================================================================

  def copy(self):
    copy = Results()
    copy.table = self.table.copy(deep = True)
    return copy

  # ============================================================================

  # Get the entry from the specified column and row. By default, return the whole column.
  def get(self, label, row = None):
    row = 0 if len(self.table) == 1 else row
    if row is not None:
      return self.table.iloc[row][label]
    else:
      return self.table[label]

  # ============================================================================

  def save(self, path, filename = "results", decimals = 5):

    array = self.array()
    np.save(f"{path}/{filename}.npy", array, allow_pickle = True)

    longestName = max([len(name) for name in array.dtype.names])
    numeric_columns = [name for name in array.dtype.names if np.issubdtype(array[name].dtype, np.number)]
    numeric_array = self.table[numeric_columns].to_numpy()
    onesPlaces = max(int(np.log10(np.nanmax(numeric_array[np.isfinite(numeric_array)]))), 0) + 1
    maxLength = max(longestName, onesPlaces + 1 + decimals)
    separator = "  "

    formats = []
    for name in array.dtype.names:
      if np.issubdtype(array[name].dtype, np.floating):
        formats.append(f"%{maxLength}.{decimals}f")
      elif np.issubdtype(array[name].dtype, np.integer):
        formats.append(f"%{maxLength}d")
      else:
        formats.append(f"%{maxLength}s")

    np.savetxt(
      f"{path}/{filename}.txt",
      array,
      fmt = formats,
      header = separator.join(f"{{:>{maxLength}}}".format(name) for name in array.dtype.names),
      delimiter = separator,
      comments = ""
    )

  # ============================================================================

  @staticmethod
  def load(path):
    return Results(np.load(path, allow_pickle = True))

  # ================================================================================================

  def plot(self, x, y, z = None, label = None, ls = "-", color = None, skip = 1):

    results = self.array()
    # mask = (results["c_e"] > 0) & (results["c_e"] < 1000) & (results["err_c_e"] < results["c_e"])
    mask = (~np.isnan(results["sig_x"])) & (results["bg_space"] < 100) & (results["bg_width"] < 100)
    results = results[mask]

    x_data = results[x]
    y_data = results[y]

    style.xlabel(const.info[x].format_symbol() if x in const.info else x)
    style.ylabel(const.info[y].format_symbol() if y in const.info else y)

    if z is None:

      if f"err_{y}" in results.dtype.names:
        errors = results[f"err_{y}"]
        errors[errors > 10 * (max(results[y]) - min(results[y]))] = 0
      else:
        errors = None

      if y == "t0": # patch the output for t0 being in microseconds
        y_data = y_data * 1E3
        errors = errors * 1E3

      skip = int(skip)

      errorbar = style.errorbar(
        x_data[::skip],
        y_data[::skip],
        errors[::skip],
        ls = ls,
        label = label,
        color = color
      )

      if f"ref_{y}" in results.dtype.names:
        plt.axhline(results[f"ref_{y}"][0], ls = "--", color = errorbar[0].get_color(), label = "Toy MC Truth")

      return errorbar

    else:

      z_data = results[z]
      heatmap = plt.tricontourf(x_data, y_data, z_data)
      style.colorbar(const.info[z].format_symbol() if z in const.info else z)

      return heatmap
