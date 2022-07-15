import gm2fr.src.style as style
style.set_style()
import gm2fr.src.io as io
import gm2fr.src.constants as const
from gm2fr.src.Results import Results

import argparse
import numpy as np

from gm2fr.scripts.plot_trend import plot_trend

# ==================================================================================================

systematics_folder = {
  "start": "StartTimeScan",
  "end": "EndTimeScan",
  "df": "FrequencySpacingScan",
  "freq_width": "FrequencyWidthScan"
}

limit_range = {
  "start": (4, 30),
  "end": (200, 400),
  "freq_width": (150, np.inf)
}

# ==================================================================================================

def process_systematic(dataset, systematic, variable, output = None):

  data_results = np.load(f"{io.results_path}/{dataset}/{systematics_folder[systematic]}/results.npy", allow_pickle = True)
  data_errorbar = plot_trend(
    systematic,
    variable,
    data_results,
    label = "Data"
  )

  data_x, data_y = data_results[systematic], data_results[variable]

  sim_results = np.load(f"{io.results_path}/{dataset}/{systematics_folder[systematic]}/sim_results.npy", allow_pickle = True)
  sim_errorbar = plot_trend(
    systematic,
    variable,
    sim_results,
    label = "Toy MC"
  )

  sim_x, sim_y = sim_results[systematic], sim_results[variable]

  if systematic in limit_range:
    mask = (data_x >= limit_range[systematic][0]) & (data_x <= limit_range[systematic][1])
    data_x, data_y = data_x[mask], data_y[mask]
    sim_x, sim_y = sim_x[mask], sim_y[mask]

  if output is None:
    output = f"{io.results_path}/{dataset}/{systematics_folder[systematic]}/{systematic}_plot.pdf"

  x_label = const.info[systematic].format_label() if systematic in const.info.keys() else systematic
  y_label = const.info[variable].format_symbol() if variable in const.info.keys() else variable
  y_unit = const.info[variable].units if variable in const.info.keys() else ""

  style.label_and_save(x_label, y_label, output)

  for syst_y, mode in ((data_y - sim_y, "diff"), (data_y, "normal")):
    syst_mean = np.mean(syst_y)
    syst_std = np.std(syst_y)

    style.errorbar(data_x, syst_y, None)
    style.horizontal_spread(syst_std, syst_mean, label = f"spread = {syst_std:.2f} {y_unit}")
    style.draw_horizontal(syst_mean, label = f"mean = {syst_mean:.2f} {y_unit}")

    style.label_and_save(x_label, f"Difference in {y_label}" if mode == "diff" else y_label, output)

  return Results({
    f"mean_{variable}": np.mean(data_y),
    f"std_{variable}": np.std(data_y),
    f"diff_mean_{variable}": np.mean(data_y - sim_y),
    f"diff_std_{variable}": np.std(data_y - sim_y)
  })

# ==================================================================================================

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset", "-d", required = True)
  parser.add_argument("--systematic", "-s", required = True)
  args = parser.parse_args()

  pdf = style.make_pdf(f"{io.results_path}/{args.dataset}/{systematics_folder[args.systematic]}/{args.dataset}_{args.systematic}_plots.pdf")
  results = Results({"dataset": args.dataset, "systematic": args.systematic})
  for variable in ("x", "sig_x", "c_e", "t0", "bg_chi2_ndf"):
    var_results = process_systematic(args.dataset, args.systematic, variable, output = pdf)
    results.merge(var_results)
  pdf.close()
  results.save(f"{io.results_path}/{args.dataset}/{systematics_folder[args.systematic]}", f"{args.dataset}_{args.systematic}_results")
