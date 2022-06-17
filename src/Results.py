import numpy as np
import pandas as pd

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
    self.result = merged.loc[:, ~merged.columns.duplicated()].copy()

  # ============================================================================

  def append(self, result):
    self.table = self.table.append(result.table)

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
