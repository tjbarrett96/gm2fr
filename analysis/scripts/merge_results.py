import sys
import os
import gm2fr.io as io
from gm2fr.analysis.Results import Results

# ==================================================================================================

def merge_results(parent_dir, filename = "results"):

  # Get a list of sub-directories contained in the parent directory.
  folders = [folder for folder in os.listdir(parent_dir) if os.path.isdir(f"{parent_dir}/{folder}")]

  # Loop through the folders, load each results file, and merge them into a cumulative array.
  merged_results = Results()
  for folder in folders:
    results_path = f"{parent_dir}/{folder}/results.npy"
    if os.path.isfile(results_path):
      merged_results.append(Results.load(results_path))

  # Look for numerical indices in each folder name, e.g. "Calo12" -> 12. Default to -1 if none found.
  indices = [(index if index is not None else -1) for index in io.find_indices(folders)]
  merged_results.table["index"] = indices

  # Put the new 'index' column first, and sort the rows by index value.
  column_order = ["index"] + [col for col in merged_results.table.columns if col != "index"]
  merged_results.table = merged_results.table[column_order].sort_values(by = "index")

  merged_results.save(parent_dir, filename = filename)

# ==================================================================================================

if __name__ == "__main__":

  # Take the command line argument as the top-level directory of the group.
  parent_dir = None
  if len(sys.argv) == 2:
    parent_dir = sys.argv[1]
  else:
    print("Usage: python3 merge_results.py <directory>")
    exit()

  merge_results(parent_dir)
