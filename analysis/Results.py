import numpy as np
import pandas as pd

# ==============================================================================

class Results:

  def __init__(self, data = None):
    self.table = pd.DataFrame(
      data,
      index = [0] if data is not None else None
    )
    # self.table = pd.DataFrame(data)

  # ============================================================================

  def merge(self, *results):
    self.table = pd.concat(
      (self.table, *[result.table for result in results]),
      axis = "columns"
    )

  # ============================================================================

  def append(self, result, index = None):

    # If an index was specified and there's only one row, set the new index.
    # if index is not None and len(result.table) == 1:
    #   result.table.index = [index]

    # Append the other table to this table.
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

  def save(self, path, filename = "results", decimals = 5):

    array = self.array()
    np.save(f"{path}/{filename}.npy", array)

    longestName = max([len(name) for name in array.dtype.names])
    numeric_columns = [name for name in array.dtype.names if np.issubdtype(array[name].dtype, np.number)]
    onesPlaces = max(int(np.log10(np.nanmax(self.table[numeric_columns].to_numpy()))), 0) + 1
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
    return Results(np.load(path))
