from gm2fr.simulation.mixture import GaussianMixture
from gm2fr.simulation.simulator import Simulator

import numpy as np
import ROOT as root
import root_numpy as rnp
import matplotlib.pyplot as plt

# ==============================================================================

directory = "2C/Nominal"

inFile = root.TFile(f"../../analysis/results/{directory}/transform.root")
results = np.load(f"../../analysis/results/{directory}/results.npy")

frequencies = inFile.Get("transform")
for i in range(frequencies.GetNbinsX()):
  frequencies.SetBinContent(i, abs(frequencies.GetBinContent(i)))

# Must update this for random sampling to work after fixing negative bins.
# Annoying!
frequencies.ComputeIntegral()

times = root.TH1F("", "", 150, -75, 75)
fn = root.TF1("", "TMath::Gaus(x, 0, 25)", -75, 75)
for i in range(times.GetNbinsX()):
  times.SetBinContent(i, fn.Eval(times.GetBinCenter(i)))

simulation = Simulator(
  "../data/2C_sim",
  overwrite = True,
  kinematics_dist = frequencies,
  kinematics_type = "f",
  time_dist = times,
  time_units = 1E-9
)

simulation.simulate(muons = 1E9, decay = "uniform", detector = results["t0"] / 149.14)
simulation.save()
simulation.plot()

inFile.Close()
