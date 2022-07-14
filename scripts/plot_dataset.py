import sys
import re
import os
import numpy as np

import gm2fr.src.io as io
import gm2fr.src.constants as const
from plot_trend import plot_trend
import analyze_dataset
from gm2fr.src.Results import Results

import matplotlib.pyplot as plt
import gm2fr.src.style as style
style.set_style()

import argparse

# ==================================================================================================

# Dictionary mapping subset names to axis labels.
subset_labels = {
  "nominal": "Dataset",
  "calo": "Calorimeter",
  "bunch": "Bunch Number",
  "run": "Run Number",
  "energy": "Energy (MeV)",
  "threshold": "Energy Threshold (MeV)",
  "row": "Crystal Row",
  "column": "Crystal Column"
}

# ==================================================================================================

def plot_dataset(dataset, subset, variable):

  # Validate the requested subset and variable.
  if subset not in subset_labels.keys():
    print(f"Data subset type '{subset}' not recognized.")
    return
  if variable not in const.info.keys():
    print(f"Variable '{variable}' not recognized.")
    return

  # TODO: add optional dashed line of same color for nominal result (no extra label)
  if subset == "nominal":

    results = np.load(f"{io.results_path}/{dataset}/Nominal/results.npy", allow_pickle = True)

    plot_trend(
      x = [dataset],
      y = variable,
      results = results,
      label = f"Run {dataset[0]}",
      color = f"C{int(dataset[0]) - 1}"
    )

  else:

    results = np.load(f"{io.results_path}/{dataset}/By{subset.capitalize()}/{dataset}_{subset}_results.npy", allow_pickle = True)
    nominal_results = np.load(f"{io.results_path}/{dataset}/Nominal/results.npy", allow_pickle = True)

    errorbar = plot_trend(
      x = "index",
      y = variable,
      results = results,
      label = dataset,
      ls = "-" if subset != "run" else ""
    )

    nominal_value = nominal_results[variable][0]
    if variable == "t0":
      nominal_value *= 1E3
    # style.draw_horizontal(nominal_value, c = errorbar[0].get_color())

    if subset != "threshold":
      weights = np.where(results["wg_N"] > 5000, results["wg_N"], 0)
      if subset == "energy":
        weights = np.where(results["index"] > 1700, results["wg_N"] * results["wg_A"], 0)
      avg = np.average(results[variable], weights = weights)
      std = np.sqrt(np.average((results[variable] - avg)**2, weights = weights))
      if variable == "t0":
        avg *= 1E3
        std *= 1E3
      # style.draw_horizontal(avg, ls = "--", c = errorbar[0].get_color())
      # style.horizontal_spread(std, avg, color = errorbar[0].get_color())

      return Results({
        "dataset": dataset,
        "subset": subset,
        f"avg_{variable}": avg,
        f"std_{variable}": std,
        f"diff_{variable}": avg - nominal_value
      })

# ==================================================================================================

def parse_dataset_arg(tokens):
  datasets = []
  for token in tokens:
    if re.match("[1-9]\*", token):
      datasets += io.list_run_datasets(token[0])
    else:
      datasets.append(token)
  return sorted(list(set(datasets)))

# ==================================================================================================

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--datasets", "-d", nargs = "+", required = True)
  parser.add_argument("--subsets", "-s", nargs = "+", default = list(subset_labels.keys()))
  parser.add_argument("--label", "-l", default = None)
  parser.add_argument("--variables", "-v", nargs = "*", default = ["x", "sig_x", "c_e", "t0", "bg_chi2_ndf"])
  args = parser.parse_args()

  datasets = parse_dataset_arg(args.datasets)
  label = ",".join(args.datasets).replace("*", "-") if args.label is None else args.label
  io.make_if_absent(f"{io.plot_path}/{label}")

  if len(args.subsets) == 1 and args.subsets[0] == "nominal":
    plt.rcParams["legend.fontsize"] = 9
    plt.rcParams["xtick.labelsize"] = 12

  for subset in args.subsets:
    pdf = style.make_pdf(f"{io.plot_path}/{label}/{label}_{subset}_plots.pdf")
    subset_results = Results()
    for variable in args.variables:
      for dataset in datasets:
        results = plot_dataset(dataset, subset, variable)
        if results is not None:
          if dataset not in subset_results["dataset"]:
            # need to add a new row for this dataset
            subset_results.append(results)
          elif f"avg_{variable}" in subset_results.table.columns:
            # column already added for this variable from previous dataset; update null values for this dataset
            subset_results.table.combine_first(results.table)
          else:
            # add new columns for this variable in the "dataset" row, excluding the "subset" column
            subset_results.table.join(results.table.loc[:, results.table.columns != "subset"], on = "dataset")
      style.label_and_save(
        subset_labels[subset],
        const.info[variable].format_label(),
        pdf,
        extend_x = 0.1 if subset != "nominal" else 0,
        loc = "center right" if subset != "nominal" else None
      )
    pdf.close()
    subset_results.save(f"{io.plot_path}/{label}", f"{label}_{subset}_results")
