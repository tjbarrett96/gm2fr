import sys
import os
import re
import gm2fr.io as io
from gm2fr.analysis.Results import Results

# ==================================================================================================

def merge_results(folders, output_dir = None, output_name = None, indices = None):

  # Loop through the folders, load each results file, and merge them into a cumulative array.
  merged_results = Results()
  for folder in folders:
    results_path = f"{folder}/results.npy"
    if os.path.isfile(results_path):
      merged_results.append(Results.load(results_path))

  # Look for numerical indices in each folder name, e.g. "Calo12" -> 12. Default to -1 if none found.
  if indices is None:
    indices = [(index if index is not None else -1) for index in io.find_indices(folders)]
  merged_results.table["index"] = indices

  # Put the new 'index' column first, and sort the rows by index value.
  column_order = ["index"] + [col for col in merged_results.table.columns if col != "index"]
  merged_results.table = merged_results.table[column_order].sort_values(by = "index")

  if output_dir is not None:
    merged_results.save(output_dir, output_name if output_name is not None else "merged_results")
  return merged_results

# ==================================================================================================

if __name__ == "__main__":

  if len(sys.argv) != 2:

    print("python3 merge_results.py <parent_directory / run_number>")
    exit()

  else:

    folders, output_dir, output_name, indices = None, None, None, None
    if re.match(r"\d+\Z", sys.argv[1]):
      run_number = int(sys.argv[1])
      indices = io.list_run_datasets(run_number)
      folders = [f"{io.results_path}/{dataset}/Nominal" for dataset in indices]
      output_dir = io.results_path
      output_name = f"Run{run_number}_nominal_results"
    else:
      parent_dir = sys.argv[1]
      folders = [f"{parent_dir}/{folder}" for folder in os.listdir(parent_dir) if os.path.isdir(f"{parent_dir}/{folder}")]
      output_dir = parent_dir

    merge_results(folders, output_dir, output_name, indices)
