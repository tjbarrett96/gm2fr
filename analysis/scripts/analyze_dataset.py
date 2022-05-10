import sys
import ROOT as root
import re
import numpy as np
import os
import gm2fr.io as io
from merge_results import merge_results

from gm2fr.analysis.Analyzer import Analyzer

# ==================================================================================================

# Dictionary mapping subset names to the name of the parent directory in the ROOT data file.
subset_dir = {
  "calo": "IndividualCalos",
  "bunch": "BunchNumber",
  "run": "RunNumber",
  "energy": "EnergyBins",
  "threshold": "EnergyThreshold",
  "row": "CrystalRow",
  "column": "CrystalColumn"
}

# ==================================================================================================

def analyze_dataset(dataset, subset = "nominal"):

  # Validate the requested subset to analyze.
  if subset not in ("nominal", *subset_dir.keys()):
    print(f"Data subset type '{subset}' not recognized.")
    return

  # Construct the standard path to the dataset ROOT file.
  input_path = f"{io.gm2fr_path}/data/FastRotation_{dataset}.root"

  if subset == "nominal":

    # For the nominal analysis, there are no input subdirectories and no output group folder.
    input_folders = ["FastRotation/AllCalos"]
    subset_indices = []
    output_group = None
    output_folders = ["Nominal"]

  else:

    # Get a list of subdirectories inside the input parent directory for this subset.
    input_file = root.TFile(input_path)
    input_folders = [
      f"FastRotation/{subset_dir[subset]}/{item.GetName()}"
      for item in input_file.Get(f"FastRotation/{subset_dir[subset]}").GetListOfKeys()
    ]
    input_file.Close()

    # Look in each subdirectory name for an identifying numerical index (e.g. calo number).
    subset_indices = io.find_indices(input_folders)

    # Construct the output group name (e.g. ByCalo) and output folders (e.g. [Calo1, Calo2, ...]).
    output_group = f"By{subset.capitalize()}"
    output_folders = [f"{subset.capitalize()}{index}" for index in subset_indices]

  # Run the analysis on each part of the subset (e.g. each calo).
  for input_folder, subset_index, output_folder in zip(input_folders, subset_indices, output_folders):

    # Special exclusions of subsets which don't behave well.
    if subset in ("energy", "threshold") and subset_index < 500:
      continue

    analyzer = Analyzer(
      filename = input_path,
      signal_label = f"{input_folder}/hHitTime",
      pileup_label = f"{input_folder}/hPileupTime",
      output_label = f"{dataset}/{(output_group + '/') if output_group is not None else ''}{output_folder}",
      fr_method = "nine" if subset == "nominal" else "five",
      n = 0.108 if dataset not in ("1B", "1C") else 0.120,
      units = "ns"
    )

    analyzer.analyze(
      start = 4,
      end = 250 if subset != "run" else 150,
      bg_model = "sinc",
      plots = 2 if subset == "nominal" else 1
    )

  # Concatenate the results over the subset into a single group results file.
  if output_group is not None:
    merge_results(
      parent_dir = f"{io.gm2fr_path}/analysis/results/{dataset}/{output_group}",
      filename = f"{dataset}_{subset}_results"
    )

# ==================================================================================================

if __name__ == "__main__":

  # Check for the dataset and (optional) scan arguments.
  if len(sys.argv) < 2:
    print("Arguments not recognized. Usage:")
    print("python3 analyze_dataset.py <dataset> [subset: nominal (default) / calo / bunch / run / energy / threshold / row / column / all]")
    exit()

  # Parse the dataset argument.
  dataset = sys.argv[1]

  # Parse the subset argument(s).
  if len(sys.argv) == 2:
    # Nominal (all calos) analysis if no scan specified.
    subsets = ["nominal"]
  else:
    # Take all remaining parameters as subset types.
    subsets = [subset.lower() for subset in sys.argv[2:]]

  # If one of the arguments was "all", then analyze all supported subset types.
  if "all" in subsets:
    subsets = ["nominal"] + list(subset_dir.keys())

  # Run the analysis on all requested subsets.
  for subset in subsets:
    analyze_dataset(dataset, subset)
