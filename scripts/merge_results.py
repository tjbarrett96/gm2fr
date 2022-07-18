import sys
import os
import re
import gm2fr.src.io as io
from gm2fr.src.Results import Results
import argparse

# ==================================================================================================

def merge_results(results, output_dir = None, output_name = None, indices = None, sort_column = "index"):

  # Look for numerical indices in each folder name, e.g. "Calo12" -> 12. Default to -1 if none found.
  if indices is None:
    indices = [
      (index if index is not None else -1)
      for index in io.find_indices([os.path.basename(os.path.dirname(result)) for result in results])
    ]

  # Loop through the folders, load each results file, and merge them into a cumulative array.
  merged_results = Results()
  for i, result in enumerate(results):
    if os.path.isfile(result):
      merged_results.append(Results.load(result))
    else:
      del indices[i]

  merged_results.table["index"] = indices

  # Put the new 'index' column first, and sort the rows by the values in the specified column.
  column_order = ["index"] + [col for col in merged_results.table.columns if col != "index"]
  merged_results.table = merged_results.table[column_order].sort_values(by = sort_column)

  # Save (if output specified) and return merged Results object.
  if output_dir is not None:
    merged_results.save(output_dir, output_name if output_name is not None else "merged_results")
  return merged_results

# ==================================================================================================

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--output", "-o", required = True)
  parser.add_argument("--name", "-n", default = "merged_results")
  parser.add_argument("--results", "-r", nargs = "+")
  parser.add_argument("--indices", "-i", default = None, choices = [None, "parent"])
  parser.add_argument("--sort", "-s", default = "index")
  args = parser.parse_args()

  if args.indices == "parent":
    indices = [os.path.basename(os.path.dirname(os.path.dirname(result))) for result in args.results]
  else:
    indices = args.indices

  merge_results(args.results, args.output, args.name, indices, args.sort)
