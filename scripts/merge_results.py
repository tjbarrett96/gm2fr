import sys
import os
import re
import gm2fr.src.io as io
from gm2fr.src.Results import Results
import argparse
import numpy as np

# ==================================================================================================

def merge_results(results, output_dir = None, output_name = None, indices = None, sort_column = "index", transforms = False):

  # Look for numerical indices in each folder name, e.g. "Calo12" -> 12. Default to -1 if none found.
  if indices is None:
    indices = [
      (index if index is not None else -1)
      for index in io.find_indices([os.path.basename(os.path.dirname(result)) for result in results])
    ]

  if output_name is None:
    output_name = "merged_results"

  merged_results = Results()
  merged_transforms = {}

  unique_indices = [*indices] if len(np.unique(indices)) == len(indices) else np.arange(len(indices))

  # Loop through the folders, load each results file, and merge them into a cumulative array.
  for i, result in enumerate(results):

    if os.path.isfile(result):
      merged_results.append(Results.load(result))

      if transforms:
        parent_dir = os.path.dirname(result)
        transform_file = f"{parent_dir}/transform.npz"
        if os.path.isfile(transform_file):
          transform_data = np.load(transform_file, allow_pickle = True)
          for element in ("edges", "heights", "cov"):
            merged_transforms[f"{unique_indices[i]}/{element}"] = transform_data[f"transform_x/{element}"]

    else:
      del indices[i]

  merged_results.table["index"] = indices

  # Put the new 'index' column first, and sort the rows by the values in the specified column.
  column_order = ["index"] + [col for col in merged_results.table.columns if col != "index"]
  merged_results.table = merged_results.table[column_order].sort_values(by = sort_column)

  # Save (if output specified) and return merged Results object.
  if output_dir is not None:
    merged_results.save(output_dir, output_name)
    if transforms:
      np.savez(f"{output_dir}/{output_name.replace('_results', '') + '_transforms'}", **merged_transforms)

  if not transforms:
    return merged_results
  else:
    return merged_results, merged_transforms

# ==================================================================================================

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--output", "-o", required = True)
  parser.add_argument("--name", "-n", default = "merged_results")
  parser.add_argument("--results", "-r", nargs = "+")
  parser.add_argument("--indices", "-i", default = None, choices = [None, "parent"])
  parser.add_argument("--sort", "-s", default = "index")
  parser.add_argument("--transforms", "-t", action = "store_true")
  args = parser.parse_args()

  if args.indices == "parent":
    indices = [os.path.basename(os.path.dirname(os.path.dirname(result))) for result in args.results]
  else:
    indices = args.indices

  merge_results(args.results, args.output, args.name, indices, args.sort, args.transforms)
