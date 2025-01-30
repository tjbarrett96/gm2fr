import sys
# import ROOT as root
import uproot
import numpy as np
import gm2fr.src.io as io
from merge_results import merge_results
from gm2fr.src.Analyzer import Analyzer
from gm2fr.src.Results import Results
import argparse, re, inspect

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

def analyze_dataset(dataset, subset = "nominal", label = None, constructor_arg_dict = None, analyze_arg_dict = None):

  # Validate the requested subset to analyze.
  if subset not in ("nominal", "sim", *subset_dir.keys()):
    print(f"Data subset type '{subset}' not recognized.")
    return

  if constructor_arg_dict is None:
    constructor_arg_dict = dict()
  if analyze_arg_dict is None:
    analyze_arg_dict = dict()

  if subset != "sim":

    # Construct the standard path to the dataset ROOT file.
    input_path = f"{io.gm2fr_path}/data/FastRotation_{dataset}.root"

    if subset == "nominal":

      # For the nominal analysis, there are no input subdirectories and no output group folder.
      input_folders = ["FastRotation/AllCalos"]
      subset_indices = [None]
      output_group = None
      output_folders = [label if label is not None else "Nominal"]

    else:

      # Get a list of subdirectories inside the input parent directory for this subset.
      # input_file = root.TFile(input_path)
      # input_folders = [
      #   f"FastRotation/{subset_dir[subset]}/{item.GetName()}"
      #   for item in input_file.Get(f"FastRotation/{subset_dir[subset]}").GetListOfKeys()
      # ]
      # input_file.Close()
      with uproot.open(input_path) as root_file:
        input_folders = [f"FastRotation/{subset_dir[subset]}/{dir_name}" for dir_name in root_file[f"FastRotation/{subset_dir[subset]}"].keys(recursive = False, cycle = False)]

      # Look in each subdirectory name for an identifying numerical index (e.g. calo number).
      subset_indices = io.find_indices(input_folders)

      # Construct the output group name (e.g. ByCalo) and output folders (e.g. [Calo1, Calo2, ...]).
      output_group = f"By{subset.capitalize()}{label if label is not None else ''}"
      output_folders = [f"{subset.capitalize()}{index}" for index in subset_indices]

  else:

    input_path = f"{io.sim_path}/{dataset}_sim/data.npz"
    input_folders = [None]
    subset_indices = [None]
    output_group = None
    output_folders = [label if label is not None else "Simulation"]
    constructor_arg_dict["ref_filename"] = "same"
    # constructor_arg_dict["ref_t0"] = None

  if "fr_method" not in analyze_arg_dict:
    analyze_arg_dict["fr_method"] = "five" if subset != "sim" else None

  # must save copy of argument conditions here, since constructor_arg_dict will change during loop!
  rob_bunch = ("ref_filename" in constructor_arg_dict) and (constructor_arg_dict["ref_filename"] == "rob_bunch")
  james_bunch = ("ref_filename" in constructor_arg_dict) and (constructor_arg_dict["ref_filename"] == "james_bunch")
  rob_noSmooth = ("ref_filename" in constructor_arg_dict) and (constructor_arg_dict["ref_filename"] == "rob_noSmooth")
  elia_bunch = ("ref_filename" in constructor_arg_dict) and (constructor_arg_dict["ref_filename"] == "elia_bunch")

  print(input_folders)

  # Run the analysis on each part of the subset (e.g. each calo).
  for input_folder, subset_index, output_folder in zip(input_folders, subset_indices, output_folders):

    # Special exclusions of subsets which don't behave well.
    if subset in ("energy", "threshold") and subset_index < 500:
      continue

    # special case for correcting bunch-by-bunch in runs 2 and 3
    if subset == "bunch" or (subset == "nominal" and elia_bunch):
      if rob_bunch:
        constructor_arg_dict["ref_filename"] = f"{io.data_path}/correlation/rob_bunch/{dataset}_Bunch{subset_index % 8}_smooth.root"
        constructor_arg_dict["ref_t0"] = 0
      elif elia_bunch:
        bunch_index = 8 if subset == "nominal" else (subset_index % 8)
        constructor_arg_dict["ref_filename"] = f"{io.data_path}/correlation/elia_bunch/{dataset}_Bunch{bunch_index}_smooth.root"
        constructor_arg_dict["ref_t0"] = 0
      elif rob_noSmooth:
        constructor_arg_dict["ref_filename"] = f"{io.data_path}/correlation/rob_bunch/{dataset}_Bunch{subset_index % 8}_noSmooth.root"
        constructor_arg_dict["ref_t0"] = 0
      elif james_bunch:
        constructor_arg_dict["ref_filename"] = f"{io.data_path}/correlation/james_bunch/{dataset}_Bunch{subset_index % 8}_smooth.root"
        constructor_arg_dict["ref_t0"] = 0
      elif constructor_arg_dict["ref_filename"]:
        constructor_arg_dict["ref_filename"] = re.sub(r"bunch\d+", f"bunch{subset_index % 8}", constructor_arg_dict["ref_filename"])

    analyzer = Analyzer(
      filename = input_path,
      signal_label = f"{input_folder}/hHitTime" if subset != "sim" else "signal",
      pileup_label = f"{input_folder}/hPileupTime" if subset != "sim" else None,
      output_label = f"{dataset}/{(output_group + '/') if output_group is not None else ''}{output_folder}",
      n = 0.108 if dataset not in ("1B", "1C") else 0.120,
      time_units = 1E-9 if subset != "sim" else 1E-6,
      **constructor_arg_dict
    )

    # Assume default analysis parameters, and pass any extra keyword arguments.
    analyzer.analyze(plot_level = 2 if subset in ("nominal", "sim") else 1, **analyze_arg_dict)

  # Concatenate the results over the subset into a single group results file.
  if output_group is not None:
    merge_results(
      results = [f"{io.results_path}/{dataset}/{output_group}/{output_folder}/results.npy" for output_folder in output_folders],
      output_dir = f"{io.results_path}/{dataset}/{output_group}",
      output_name = f"{dataset}_{subset}_results",
      indices = subset_indices,
      transforms = True
    )

# ==================================================================================================

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset", "-d", required = True)
  parser.add_argument("--subsets", "-s", nargs = "*", default = [])
  parser.add_argument("--label", "-l", default = None)
  parser.add_argument("--ref", "-r", default = None)
  parser.add_argument("--parameters", "-p", nargs = "*", default = [])
  args = parser.parse_args()

  if args.ref is not None:
    if args.ref not in ("rob_bunch", "rob_noSmooth", "james_bunch", "elia_bunch") and not args.ref.endswith(".root"):
      ref_filename = f"{io.sim_path}/{args.ref}_sim/data.npz"
      ref_t0 = float(Results.load(f"{io.results_path}/{args.ref}/Simulation/results.npy").get("t0"))
    else:
      ref_filename = args.ref
      ref_t0 = 0
    print(ref_filename, ref_t0)
    args.parameters += [f"ref_filename:{ref_filename}", f"ref_t0:{ref_t0:.10f}"]

  # If one of the arguments was "all", then analyze all supported subset types.
  if len(args.subsets) == 0:
    args.subsets = ["nominal"]
  elif "all" in args.subsets:
    args.subsets = ["nominal", "sim"] + list(subset_dir.keys())

  # if (args.label is not None) and (len(args.subsets) > 1 or (args.subsets[0] not in ("nominal", "sim"))):
  #   print("Can only use 'label' with subset 'nominal' or 'sim'.")
  #   exit()

  allowed_constructor_args = inspect.getfullargspec(Analyzer.__init__).args
  allowed_analyze_args = inspect.getfullargspec(Analyzer.analyze).args

  constructor_arg_dict = dict()
  analyze_arg_dict = dict()

  for arg in args.parameters:

    name, value = io.parse_parameter(arg)

    if io.is_iterable(value):
      raise NotImplementedError("Multiple values for analyze_dataset parameter.")

    if name in allowed_constructor_args:
      constructor_arg_dict[name] = value
    elif name in allowed_analyze_args:
      analyze_arg_dict[name] = value
    else:
      raise ValueError(f"Parameter '{name}' not recognized.")

  # Run the analysis on all requested subsets.
  for subset in args.subsets:
    analyze_dataset(args.dataset, subset, args.label, constructor_arg_dict, analyze_arg_dict)
