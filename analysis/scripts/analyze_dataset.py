import sys
import ROOT as root
import re
import numpy as np
import os
import gm2fr.io as io

from gm2fr.analysis.Analyzer import Analyzer

# ==================================================================================================

# Check for the dataset and (optional) scan arguments.
if len(sys.argv) < 2 or len(sys.argv) > 9:
  print("Arguments not recognized. Usage:")
  print("python3 analyze_dataset.py [dataset] [all / calo / bunch / run / energy / threshold / row / column]")
  exit()

# Parse the dataset argument.
dataset = sys.argv[1]

# Create the output directory for the dataset if it doesn't already exist.
if not os.path.isdir(f"{io.gm2fr_path}/analysis/results/{dataset}"):
  os.mkdir(f"{io.gm2fr_path}/analysis/results/{dataset}")

# The path to the dataset ROOT file.
inputPath = f"{io.gm2fr_path}/data/FastRotation_{dataset}.root"

# ==================================================================================================

# Parse the scan type argument(s).
if len(sys.argv) == 2:
  # Nominal (all calos) analysis if no scan specified.
  scanTypes = ["nominal"]
else:
  # Take all remaining parameters as scan types.
  scanTypes = [scan.lower() for scan in sys.argv[2:]]

# If "all" is specified as a scan type, do them all.
if "all" in scanTypes:
  scanTypes = ["nominal", "calo", "bunch", "run", "energy", "threshold", "row", "column"]

# ==================================================================================================

for scanType in scanTypes:

  # Configure the paths to the histograms within the ROOT file, and the output directory tags.
  if scanType == "nominal":
    tags = ["Nominal"]
    directories = ["FastRotation/AllCalos"]

  elif scanType == "calo":
    # Assuming there are 24 calorimeters in the file.
    tags = [f"Calo{i}" for i in range(1, 25)]
    directories = [f"FastRotation/IndividualCalos/{tag}" for tag in tags]

  elif scanType == "bunch":
    # Assuming there are 16 bunches in the file.
    tags = [f"Bunch{i}" for i in range(0, 16)]
    directories = [f"FastRotation/BunchNumber/{tag}" for tag in tags]

  elif scanType == "run":
    # Find which run numbers are in the file.
    inputFile = root.TFile(inputPath)
    numbers = util.findIndices([item.GetName() for item in inputFile.Get("FastRotation/RunNumber").GetListOfKeys()])
    inputFile.Close()
    tags = [f"Run{i}" for i in numbers]
    directories = [f"FastRotation/RunNumber/{tag}" for tag in tags]

  elif scanType == "energy":
    # Find which energy bins are in the file.
    inputFile = root.TFile(inputPath)
    energies = util.findIndices([item.GetName() for item in inputFile.Get("FastRotation/EnergyBins").GetListOfKeys()])
    inputFile.Close()
    tags = [f"Energy{i}" for i in energies]
    directories = [f"FastRotation/EnergyBins/Energy_{i}_to_{int(i) + 200}_MeV" for i in energies]

  elif scanType == "threshold":
    # Find which energy thresholds are in the file.
    inputFile = root.TFile(inputPath)
    energies = util.findIndices([item.GetName() for item in inputFile.Get("FastRotation/EnergyThreshold").GetListOfKeys()])
    inputFile.Close()
    tags = [f"Threshold{i}" for i in energies]
    directories = [f"FastRotation/EnergyThreshold/Energy_gt_{i}_MeV" for i in energies]

  elif scanType == "row":
    # Assuming there are 6 crystal rows in the file.
    tags = [f"Row{i}" for i in range(0, 6)]
    directories = [f"FastRotation/CrystalRow/{tag}" for tag in tags]

  elif scanType == "column":
    # Assuming there are 9 crystal columns in the file.
    tags = [f"Column{i}" for i in range(0, 9)]
    directories = [f"FastRotation/CrystalColumn/{tag}" for tag in tags]

  else:
    print(f"Scan type {scanType} not recognized.")
    exit()

  # Assemble the lists of histogram paths for each signal in the scan.
  signals = [f"{inputDir}/hHitTime" for inputDir in directories]
  pileupSignals = [f"{inputDir}/hPileupTime" for inputDir in directories]

  for signal, pileup_signal, tag in zip(signals, pileupSignals, tags):

    if scanType != "nominal":
      tag = f"{dataset}/By{scanType.capitalize()}/{tag}"
    else:
      tag = f"{dataset}/Nominal"

    analyzer = Analyzer(
      inputPath,
      signal,
      pileup_signal,
      tag,
      # group = f"{dataset}/By{scanType.capitalize()}" if scanType != "nominal" else f"{dataset}",
      units = "ns",
      fr_method = "nine" if scanType == "nominal" else "five",
      n = 0.108 if dataset not in ["1B", "1C"] else 0.120
    )

    analyzer.analyze(
      start = 4,
      end = 250 if scanType != "run" else 150,
      bg_model = "sinc",
      plots = 2 if scanType == "nominal" else 1
    )
