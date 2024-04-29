import ROOT as root
import argparse
import gm2fr.src.io as io

def hadd_run_numbers(dataset, start, end):

  # Open input ROOT file.
  input_filename = f"{io.data_path}/FastRotation_{dataset}.root"
  input_file = root.TFile(input_filename)
  
  # Get list of run-number directories inside file.
  run_dirs = [key.GetName() for key in input_file.Get("FastRotation/RunNumber").GetListOfKeys()]

  # Define list of histograms to hadd within each run-number directory. 
  labels = ["hHitTime", "hHitTime_Num", "hHitTime_Den", "hHitEnergy", "hHitEnergyAfter4us", "hCaloNum", "hBunchNum"]
  histograms = {}

  # Create dictionary of histograms and cloned TH1s, initially reset to empty.
  for label in labels:
    histogram = input_file.Get(f"FastRotation/RunNumber/{run_dirs[0]}/{label}").Clone()
    histogram.Reset()
    histogram.SetDirectory(0) # decouple cloned TH1 object from open input file (so stupid)
    histograms[label] = histogram
 
  # Iterate through run-number directories and add each histogram to global histograms.
  for run_dir in run_dirs:
    run_number = run_dir[3:]
    if not (int(start) <= int(run_number) <= int(end)):
      continue
    print(f"Including {run_dir} in hadded result.")
    for label in labels:
      histograms[label].Add(input_file.Get(f"FastRotation/RunNumber/{run_dir}/{label}"))

  input_file.Close()

  # Open output ROOT file.
  output_filename = f"{io.data_path}/FastRotation_{dataset}_{start}_{end}.root"
  output_file = root.TFile(output_filename, "RECREATE")
 
  # Create directory structure.
  output_file.mkdir("FastRotation")
  output_file.mkdir("FastRotation/AllCalos")
  output_file.cd("FastRotation/AllCalos")

  for label in labels:
    histograms[label].Write()
  output_file.Close()

if __name__ == "__main__":
  
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset", "-d", required = True)
  parser.add_argument("--start", "-s", required = True)
  parser.add_argument("--end", "-e", required = True)
  args = parser.parse_args()

  hadd_run_numbers(args.dataset, args.start, args.end) 
