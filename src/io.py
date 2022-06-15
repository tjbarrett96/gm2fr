import os
import re
import gm2fr
import numpy as np

# ==================================================================================================

# Paths to common locations in the gm2fr project directory.
gm2fr_path = os.path.dirname(gm2fr.__file__)
results_path = f"{gm2fr_path}/analysis/results"
data_path = f"{gm2fr_path}/data"
sim_path = f"{gm2fr_path}/simulation/data"

# ==================================================================================================

def find_index(string):
  """Returns the first sequence of digits within a string as an integer, or else None."""
  match = re.search(r"\D*(\d+)", string)
  return int(match.group(1)) if match else None

# ==================================================================================================

def find_indices(sequence):
  """Returns a list of numerical indices extracted from a sequence of strings."""
  return [find_index(item) for item in sequence]

# ==================================================================================================

def make_if_absent(path):
  """Creates a directory at the given path if it does not already exist."""
  if not os.path.isdir(path):
    os.makedirs(path)

# ==================================================================================================

def is_iterable(obj):
  """Checks whether or not the given object is an iterable container, not including strings."""
  try:
    iterator = iter(obj)
    return (not isinstance(obj, str))
  except TypeError:
    return False

# ==================================================================================================

def force_list(obj):
  """Returns a single-element list containing the object, unless already iterable."""
  return [obj] if not is_iterable(obj) else obj

# ==================================================================================================

def is_pair(obj):
  """Checks if an object matches the form of a 2-tuple (x, y)."""
  return isinstance(obj, tuple) and len(obj) == 2

# ==================================================================================================

def is_number(obj):
  """Checks if an object is an integer or float."""
  return np.issubdtype(type(obj), np.number)

# ==================================================================================================

def is_integer(obj):
  """Checks if an object is an integer."""
  return np.issubdtype(type(obj), np.integer)

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
  """Lists all datasets from the given run number with results in the standard directory."""
  return sorted([dataset for dataset in os.listdir(results_path) if re.match(f"{run}[A-Z]\Z", dataset)])

# ==================================================================================================

def list_subdirectories(directory):
  return [item for item in os.listdir(directory) if os.path.isdir(f"{directory}/{item}")]

# ==================================================================================================

def matches_number(string):
  return re.match(r"[-.\d]+$", string)

# ==================================================================================================

def parse_parameter(arg):

  # Check that parameter_spec has the form <name>:<value>.
  if re.match(r"\w+:[^:]+", arg):

    # Get the parameter name and specification, separated by the colon.
    name, parameter_spec = arg.split(":")

    # Check if it matches the range format (start,stop,end).
    if re.match(r"\([-.\d]+,[-.\d]+,[-.\d]+\)$", parameter_spec):
      start, end, step = parameter_spec[1:-1].split(",")
      value = np.arange(float(start), float(end), float(step))

    # Check if it matches the list format [item1,item2,...].
    elif re.match(r"\[(?:[-.\w]+,?)+\]$", parameter_spec):
      value = [float(item) if matches_number(item) else item for item in parameter_spec[1:-1].split(",")]

    # Otherwise, treat it as a single value: either a number, None, or string.
    else:
      if matches_number(parameter_spec):
        value = float(parameter_spec)
      elif parameter_spec.lower() == "none":
        value = None
      else:
        value = parameter_spec

  else:
    print(f"Parameter specification '{arg}' not understood. Valid formats:")
    print("  name:(start,end,step)")
    print("  name:[value1,value2,...]")
    print("  name:value")
    exit()

  return name, value
