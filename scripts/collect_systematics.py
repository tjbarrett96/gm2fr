from gm2fr.src.Results import Results
import gm2fr.src.io as io
import argparse

systematics = [
  "start",
  "end",
  "df",
  "bg_width",
  "bg_space",
  "bg_model",
  "fr_method"
]

diff_mode = [
  "start",
  "bg_width"
]

variables = [
  "x",
  "sig_x",
  "c_e",
  "t0"
]

# ==================================================================================================

def collect_systematics():

  results = Results()
  for i, systematic in enumerate(systematics):

    syst_results = Results.load(f"{io.results_path}/systematics/systematic_{systematic}_results.npy")

    if i == 0:
      # copy the dataset column
      results.table["dataset"] = syst_results.table["dataset"]

    for variable in variables:
      prefix = "diff_" if systematic in diff_mode else ""
      results.table[f"{systematic}_std_{variable}"] = syst_results.table[f"{prefix}std_{variable}"]
      results.table[f"{systematic}_range_{variable}"] = syst_results.table[f"{prefix}range_{variable}"]

  results.save(f"{io.results_path}/systematics", "all_systematics")

# ==================================================================================================

if __name__ == "__main__":
  collect_systematics()
