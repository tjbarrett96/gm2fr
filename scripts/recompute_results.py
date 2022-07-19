from gm2fr.src.Results import Results
from gm2fr.src.Histogram1D import Histogram1D
import gm2fr.src.constants as const
import numpy as np
import argparse

def recompute_results(directory, transform_name, results_name):

  output_variables = ["f", "x", "dp_p0", "T", "tau", "gamma"]

  results = Results.load(f"{directory}/{results_name}.npy")
  transform_data = np.load(f"{directory}/{transform_name}.npz", allow_pickle = True)

  if "index_0/edges" in list(transform_data):
    num_histograms = int(len(transform_data) / 4)
  else:
    num_histograms = 1

  for i in range(num_histograms):

    label = "transform_f" if num_histograms == 1 else f"index_{i}"

    transform = Histogram1D(
      transform_data[f"{label}/edges"],
      heights = transform_data[f"{label}/heights"],
      cov = transform_data[f"{label}/cov"]
    )

    for unit in output_variables:

      conv_transform = transform.copy().map(const.info[unit].from_frequency)

      mean, mean_err = conv_transform.mean(error = True)
      std, std_err = conv_transform.std(error = True)
      results.table.iloc[i, results.table.columns.get_loc(f"{unit}")] = mean,
      results.table.iloc[i, results.table.columns.get_loc(f"err_{unit}")] = mean_err,
      results.table.iloc[i, results.table.columns.get_loc(f"sig_{unit}")] = std,
      results.table.iloc[i, results.table.columns.get_loc(f"err_sig_{unit}")] = std_err
      if unit == "x":
        avg_x2, err_avg_x2 = conv_transform.moment(2, central = False, error = True)
        c_e = 2*0.108*(1-0.108)*(const.info["beta"].magic/const.info["r"].magic)**2*avg_x2*1E9
        results.table.iloc[i, results.table.columns.get_loc("c_e")] = c_e,
        results.table.iloc[i, results.table.columns.get_loc("err_c_e")] = (c_e / avg_x2) * err_avg_x2

  results.save(directory, results_name)

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--directory", "-d", required = True)
  parser.add_argument("--transform", "-t", default = "transform")
  parser.add_argument("--results", "-r", default = "results")
  args = parser.parse_args()

  recompute_results(args.directory, args.transform, args.results)
