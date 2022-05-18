import os
import re
import gm2fr
import numpy as np

# ==================================================================================================

# Path to the top of the gm2fr project directory.
gm2fr_path = os.path.dirname(gm2fr.__file__)
results_path = f"{gm2fr_path}/analysis/results"

# ==================================================================================================

def find_index(string):
  """Returns the first sequence of digits within a string as an integer, or else None."""
  match = re.search(r"\D*(\d+)", string)
  return int(match.group(1)) if match else None

# ==================================================================================================

def find_indices(sequence):
  """Returns a list of numerical indices extracted from a sequence of strings, skipping non-matches."""
  return [find_index(item) for item in sequence]

# ==================================================================================================

def make_if_absent(path):
  """Creates a directory at the given path if it does not already exist."""
  if not os.path.isdir(path):
    print(f"\nCreating output directory '{path}'.")
    os.makedirs(path)

# ==================================================================================================

def force_list(obj):
  """Returns a single-element list containing the object, unless already a list."""
  return [obj] if type(obj) is not list else obj

# ==================================================================================================

def is_pair(obj):
  """Checks if an object matches the form of a 2-tuple (x, y)."""
  return isinstance(obj, tuple) and len(obj) == 2

# ==================================================================================================

def is_number(obj):
  """Checks if an object is an integer or float."""
  return isinstance(obj, (int, float)) or isinstance(obj, np.number)

# ==================================================================================================

def is_integer(obj):
  """Checks if an object is an integer."""
  return isinstance(obj, int) or isinstance(obj, np.integer)

# ==================================================================================================

def check_all(obj, function):
  """Checks the given boolean condition for all items in the given container object."""
  return all(function(x) for x in obj)

# ==================================================================================================

def is_numeric_pair(obj):
  """Checks if an object is a 2-tuple of numbers."""
  return is_pair(obj) and check_all(obj, is_number)

# ==================================================================================================

def is_array(obj, d = None):
  """Checks if an object is a d-dimensional NumPy array."""
  return isinstance(obj, np.ndarray) and (obj.ndim == d if d is not None else True)

# ==================================================================================================

def list_run_datasets(run):
  return [dataset for dataset in os.listdir(results_path) if re.match(f"{run}[A-Z]\Z", dataset)]
