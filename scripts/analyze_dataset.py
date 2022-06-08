import sys
import ROOT as root
import numpy as np
import gm2fr.io as io
from merge_results import merge_results
from gm2fr.analysis.Analyzer import Analyzer
import argparse, re

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

def analyze_dataset(dataset, subset = "nominal", label = None, **analyze_args):

  # Validate the requested subset to analyze.
  if subset not in ("nominal", "sim", *subset_dir.keys()):
    print(f"Data subset type '{subset}' not recognized.")
    return

  if subset != "sim":

    # Construct the standard path to the dataset ROOT file.
    input_path = f"{io.gm2fr_path}/data/FastRotation_{dataset}.root"
    ref_filename = None

    if subset == "nominal":

      # For the nominal analysis, there are no input subdirectories and no output group folder.
      input_folders = ["FastRotation/AllCalos"]
      subset_indices = [None]
      output_group = None
      output_folders = [label if label is not None else "Nominal"]

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

  else:

    input_path = f"{io.sim_path}/{dataset}_sim/data.npz"
    input_folders = [None]
    subset_indices = [None]
    output_group = None
    output_folders = [label if label is not None else "Simulation"]
    ref_filename = "same"

  # Run the analysis on each part of the subset (e.g. each calo).
  for input_folder, subset_index, output_folder in zip(input_folders, subset_indices, output_folders):

    # Special exclusions of subsets which don't behave well.
    if subset in ("energy", "threshold") and subset_index < 500:
      continue

    if "fr_method" not in analyze_args:
      fr_method = "nine" if subset == "nominal" else ("five" if subset != "sim" else None)

    analyzer = Analyzer(
      filename = input_path,
      signal_label = f"{input_folder}/hHitTime" if subset != "sim" else "signal",
      pileup_label = f"{input_folder}/hPileupTime" if subset != "sim" else None,
      output_label = f"{dataset}/{(output_group + '/') if output_group is not None else ''}{output_folder}",
      fr_method = fr_method,
      n = 0.108 if dataset not in ("1B", "1C") else 0.120,
      time_units = 1E-9 if subset != "sim" else 1E-6,
      ref_filename = ref_filename
    )

    # Assume default analysis parameters, and pass any extra keyword arguments.
    analyzer.analyze(plot_level = 2 if subset in ("nominal", "sim") else 1, **analyze_args)

  # Concatenate the results over the subset into a single group results file.
  if output_group is not None:
    merge_results(
      folders = [f"{io.results_path}/{dataset}/{output_group}/{output_folder}" for output_folder in output_folders],
      output_dir = f"{io.results_path}/{dataset}/{output_group}",
      output_name = f"{dataset}_{subset}_results",
      indices = subset_indices
    )

# ==================================================================================================

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset", "-d", required = True)
  parser.add_argument("--subsets", "-s", nargs = "*", default = [])
  parser.add_argument("--label", "-l", default = None)
  parser.add_argument("--parameters", "-p", nargs = "*", default = [])
  args = parser.parse_args()

  # If one of the arguments was "all", then analyze all supported subset types.
  if len(args.subsets) == 0:
    args.subsets = ["nominal"]
  elif "all" in args.subsets:
    args.subsets = ["nominal", "sim"] + list(subset_dir.keys())

  if (args.label is not None) and (len(args.subsets) > 1 or (args.subsets[0] not in ("nominal", "sim"))):
    print("Can only use 'label' with subset 'nominal' or 'sim'.")
    exit()

  parameter_dict = dict()
  for parameter_spec in args.parameters:
    if re.match(r"\w+:[\d\.]+", parameter_spec):
      name, value = parameter_spec.split(":")
      parameter_dict[name] = float(value)
    elif re.match(r"\w+:\w+", parameter_spec):
      name, value = parameter_spec.split(":")
      parameter_dict[name] = value
    else:
      print(f"Parameter specification '{parameter_spec}' not understood.")
      exit()

  # Run the analysis on all requested subsets.
  for subset in args.subsets:
    analyze_dataset(args.dataset, subset, label = args.label, **parameter_dict)
