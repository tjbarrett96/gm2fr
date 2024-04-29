import ROOT as root
import argparse
import gm2fr.src.io as io

histogram_names = [
  "hHitTime",
  "hHitTime_Num",
  "hHitTime_Den",
  "hHitEnergy",
  "hHitEnergyAfter4us",
  "hCaloNum",
  "hBunchNum"
]

def merge_bunches(input_filename):
  
  input_file = root.TFile(input_filename, "UPDATE")
  print(f"Working on {input_filename}.")

  bunch_directory = input_file.Get("FastRotation/BunchNumber")

  bunch_prefix = "FastRotation/BunchNumber"
  bunch_folders = [
    f"{bunch_prefix}/{item.GetName()}"
    for item in input_file.Get(bunch_prefix).GetListOfKeys()
  ]

  bunch_indices = [io.find_index(bunch_folder) for bunch_folder in bunch_folders]
  
  for bunch_index in range(8, 16):
    if bunch_index in bunch_indices:
      
      print(f"Found bunch {bunch_index}.")
      nominal_directory = input_file.Get(f"{bunch_prefix}/Bunch{bunch_index % 8}")
      extra_directory = input_file.Get(f"{bunch_prefix}/Bunch{bunch_index}")
      
      for histogram in histogram_names:
        
        nominal_histogram = input_file.Get(f"{bunch_prefix}/Bunch{bunch_index % 8}/{histogram}")
        extra_histogram = input_file.Get(f"{bunch_prefix}/Bunch{bunch_index}/{histogram}")
        print(f"{histogram}: {nominal_histogram.GetEntries()}, {extra_histogram.GetEntries()}")
        nominal_histogram.Add(extra_histogram)
        print(f"  -> {nominal_histogram.GetEntries()}")
        
        nominal_directory.cd()
        nominal_histogram.Write(nominal_histogram.GetName(), root.TObject.kOverwrite)
      
      extra_directory.Delete(f"*;*")
      bunch_directory.Delete(f"Bunch{bunch_index};*")

  print()
  input_file.Close()

if __name__ == "__main__":
  
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", "-i", required = True)
  args = parser.parse_args()

  merge_bunches(args.input)
