import sys
import re
import os

import gm2fr.io as io
import gm2fr.constants as const
from plot_trend import plot_trend
import analyze_dataset

import matplotlib.pyplot as plt
import gm2fr.style as style
style.setStyle()

# ==================================================================================================

# Dictionary mapping subset names to axis labels.
subset_labels = {
  # "nominal": "Dataset", TODO: nominal results across datasets needs dedicated results file for simplicity
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
  plot_trend(
    x = "index",
    y = variable,
    filename = f"{io.gm2fr_path}/analysis/results/{dataset}/By{subset.capitalize()}/{dataset}_{subset}_results.npy",
    label = dataset
  )

# ==================================================================================================

def parse_dataset_arg(arg):
  tokens = arg.split(",")
  datasets = []
  for token in tokens:
    if re.match("[1-9][A-Z*]{0,1}", token):
      number = token[0]
      letter = token[1] if len(token) > 1 else None
      if letter == "*":
        datasets += io.list_run_datasets(number)
      else:
        datasets.append(token)
  return sorted(list(set(datasets)))

# ==================================================================================================

if __name__ == "__main__":

  if len(sys.argv) not in (2, 3, 4):
    print("Arguments not recognized. Usage:")
    print("python3 plotting.py <dataset(s)> [subset(s): calo,bunch,run,energy,threshold,row,column] [variable(s):x,sig_x,c_e,...]")
    exit()

  datasets = parse_dataset_arg(sys.argv[1])
  subsets = sys.argv[2].split(",") if len(sys.argv) > 2 else list(subset_labels.keys())
  variables = sys.argv[3].split(",") if len(sys.argv) > 3 else ["x", "sig_x", "c_e"]

  for subset in subsets:
    for variable in variables:
      for dataset in datasets:
        plot_dataset(dataset, subset, variable)
      style.xlabel(subset_labels[subset])
      style.ylabel(const.info[variable].formatLabel())

      # Extend the x-limits rightward by 10% and place the legend vertically.
      xLow, xHigh = plt.xlim()
      plt.xlim(xLow, xHigh + 0.1 * (xHigh - xLow))
      plt.legend(loc = "center right")

      plt.show()
