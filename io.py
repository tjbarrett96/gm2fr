import os
import re
import gm2fr
import numpy as np

# ==================================================================================================

# Path to the top of the gm2fr project directory.
path = os.path.dirname(gm2fr.__file__)

# ==================================================================================================

# Find the first set of integers within a string.
def findIndex(string):
  match = re.search(r"\D+(\d+)", string)
  return int(match.group(1)) if match else None

# Return a list of numerical indices extracted from a sequence of strings.
def findIndices(sequence):
  return [index for index in (findIndex(item) for item in sequence) if index is not None]

# ==================================================================================================

def makeIfAbsent(path):
  if not os.path.isdir(path):
    print(f"\nCreating output directory '{path}'.")
    os.mkdir(path)

def forceList(obj):
  return [obj] if type(obj) is not list else obj

# Checks if an object matches the form of a 2-tuple (x, y).
def isPair(obj):
  return isinstance(obj, tuple) and len(obj) == 2

# Checks if an object is an integer or float.
def isNumber(obj):
  return isinstance(obj, (int, float)) or isinstance(obj, np.number)

# Checks if an object is an integer.
def isInteger(obj):
  return isinstance(obj, int) or isinstance(obj, np.integer)

# Checks a boolean condition for each item in an iterable.
def checkAll(obj, function):
  return all(function(x) for x in obj)

# Checks if an object is a 2-tuple of numbers.
def isNumericPair(obj):
  return isPair(obj) and checkAll(obj, isNumber)

# Checks if an object is a d-dimensional NumPy array.
def isArray(obj, d = None):
  return isinstance(obj, np.ndarray) and (obj.ndim == d if d is not None else True)
