import gm2fr.src.style as style
style.set_style()
import gm2fr.src.io as io
import gm2fr.src.constants as const
from gm2fr.src.Results import Results

import argparse
import numpy as np

from gm2fr.scripts.plot_trend import plot_trend

# ==================================================================================================

# default labels for folders where systematic scan results are expected
systematics_folder = {
  "start": "StartTimeScan",
  "end": "EndTimeScan",
  "df": "FrequencySpacingScan",
  "freq_width": "FrequencyWidthScan",
  "inner_width": "InnerWidthScan",
  "fr_method": "FRMethodScan",
  "bg_model": "BGModelScan",
  "fr_rebin": "FRRebinScan"
}

# allowed ranges for parameters when computing std. dev. of systematic scan results
# we may want to plot outside this range to show what happens at boundaries, but not use for computations
limit_range = {
  "start": (4, 30),
  "end": (200, 400),
  "df": (0, 4.5),
  "freq_width": (150, 400)
}

# ==================================================================================================

def process_systematic(dataset, systematic, variable, output = None, folder = None, skip = 1):

  # get the default folder name for the systematic scan results, if none supplied
  folder = systematics_folder[systematic] if folder is None else folder

  # read the systematic scan results for data from the results folder
  data_results = np.load(f"{io.results_path}/{dataset}/{folder}/results.npy", allow_pickle = True)

  # remove misbehaved data points based on wildly unlikely C_E
  data_mask = (data_results["c_e"] > 0) & (data_results["c_e"] < 1000)
  data_results = data_results.loc[data_mask, :]

  data_x, data_y, data_err_y = data_results[systematic], data_results[variable], data_results[f"err_{variable}"]

  # plot the data trend
  data_errorbar = plot_trend(
    systematic,
    variable,
    data_results,
    label = "Data",
    skip = skip
  )

  # read the systematic scan results for toy MC from the results folder
  sim_results = np.load(f"{io.results_path}/{dataset}/{folder}/sim_results.npy", allow_pickle = True)

  # remove misbehaved data points based on wildly unlikely C_E
  sim_mask = (sim_results["c_e"] > 0) & (sim_results["c_e"] < 1000)
  sim_results = sim_results.loc[sim_mask, :]

  sim_x, sim_y, sim_err_y = sim_results[systematic], sim_results[variable], sim_results[f"err_{variable}"]

  # plot the MC trend
  sim_errorbar = plot_trend(
    systematic,
    variable,
    sim_results,
    label = "Toy MC",
    skip = skip
  )

  # mask the trends within the appropriate range for this parameter, before computing std. dev.
  if systematic in limit_range:
    range_mask = (data_x >= limit_range[systematic][0]) & (data_x <= limit_range[systematic][1])
    data_x, data_y, data_err_y = data_x[range_mask], data_y[range_mask], data_err_y[range_mask]
    sim_x, sim_y, sim_err_y = sim_x[range_mask], sim_y[range_mask], sim_err_y[range_mask]

  # get the default output name, if path or PDF not specified
  if output is None:
    output = f"{io.plot_path}/{dataset}_{systematic}_plots.pdf"

  # get the axis labels and units for the plot
  x_label = const.info[systematic].format_label() if systematic in const.info.keys() else systematic
  y_label = const.info[variable].format_symbol() if variable in const.info.keys() else variable
  y_unit = const.info[variable].units if variable in const.info.keys() else ""
  if y_unit is None:
    y_unit = ""

  # apply labels, legend, and save to output
  style.label_and_save(x_label, y_label, output)

  # iterate over two modes to plot: difference between data and simulation ("diff"), and raw data trend alone ("normal")
  for syst_y, syst_err_y, mode in ((data_y - sim_y, np.sqrt(data_err_y**2 + sim_err_y**2), "diff"), (data_y, data_err_y, "normal")):

    # get the mean and standard deviation of the trend
    syst_mean = np.mean(syst_y)
    syst_std = np.std(syst_y)

    # plot the trend, std. dev. band, and mean line
    style.errorbar(data_x, syst_y, syst_err_y)
    style.horizontal_spread(syst_std, syst_mean, label = f"std. dev. = {syst_std:.2f} {y_unit}")
    style.draw_horizontal(syst_mean, label = f"mean = {syst_mean:.2f} {y_unit}")

    # apply labels, legend, and save to output
    style.label_and_save(x_label, f"{y_label} Difference from Toy MC" if mode == "diff" else y_label, output)

  # assemble results object
  return Results({
    f"mean_{variable}": np.mean(data_y),
    f"std_{variable}": np.std(data_y),
    f"range_{variable}": np.max(data_y) - np.min(data_y),
    f"diff_mean_{variable}": np.mean(data_y - sim_y),
    f"diff_std_{variable}": np.std(data_y - sim_y),
    f"diff_range_{variable}": np.max(data_y - sim_y) - np.min(data_y - sim_y)
  })

# ==================================================================================================

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset", "-d", required = True)
  parser.add_argument("--systematic", "-s", required = True)
  parser.add_argument("--label", "-l", default = None)
  parser.add_argument("--skip", default = 1)
  args = parser.parse_args()

  pdf = style.make_pdf(f"{io.plot_path}/{args.dataset}/{args.dataset}_{args.systematic}_plots.pdf")
  results = Results({"dataset": args.dataset, "systematic": args.systematic})
  for variable in ("x", "sig_x", "c_e", "t0", "bg_chi2_ndf"):
    var_results = process_systematic(args.dataset, args.systematic, variable, output = pdf, skip = args.skip)
    results.merge(var_results)
  pdf.close()
  results.save(f"{io.plot_path}/{args.dataset}", f"{args.dataset}_{args.systematic}_results")
