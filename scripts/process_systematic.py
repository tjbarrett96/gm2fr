import gm2fr.style as style
style.set_style()
import gm2fr.io as io
import gm2fr.constants as const

import argparse
import numpy as np

from gm2fr.scripts.plot_trend import plot_trend

# ==================================================================================================

systematics_folder = {
  "start": "StartTimeScan"
}

# ==================================================================================================

def process_systematic(dataset, systematic, variable, output = None):

  data_x, data_y, data_error = plot_trend(
    systematic,
    variable,
    f"{io.results_path}/{dataset}/{systematics_folder[systematic]}/results.npy",
    label = "Data"
  )

  sim_x, sim_y, sim_error = plot_trend(
    systematic,
    variable,
    f"{io.results_path}/{dataset}/{systematics_folder[systematic]}/sim_results.npy",
    label = "Toy MC"
  )

  if output is None:
    output = f"{io.results_path}/{dataset}/{systematics_folder[systematic]}/{systematic}_plot.pdf"

  x_label = const.info[systematic].format_label() if systematic in const.info.keys() else systematic
  y_label = const.info[variable].format_label() if variable in const.info.keys() else variable
  y_unit = const.info[variable].units if variable in const.info.keys() else ""

  style.label_and_save(x_label, y_label, output)

  diff_y = data_y - sim_y
  diff_mean = np.mean(diff_y)
  diff_std = np.std(diff_y)

  style.errorbar(data_x, diff_y, None)
  style.horizontal_spread(diff_std, diff_mean, label = f"spread = {diff_std:.2f} {y_unit}")
  style.draw_horizontal(diff_mean, label = f"mean = {diff_mean:.2f} {y_unit}")

  style.label_and_save(x_label, f"Error in {y_label}", output)

# ==================================================================================================

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset", "-d", required = True)
  parser.add_argument("--systematic", "-s", required = True)
  args = parser.parse_args()

  pdf = style.make_pdf(f"{io.results_path}/{args.dataset}/{systematics_folder[args.systematic]}/{args.systematic}_plots.pdf")
  for variable in ("x", "sig_x", "c_e", "t0"):
    process_systematic(args.dataset, args.systematic, variable, output = pdf)
  pdf.close()
