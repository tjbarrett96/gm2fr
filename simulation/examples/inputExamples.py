from gm2fr.simulation.mixture import GaussianMixture
from gm2fr.simulation.simulator import Simulator
import gm2fr.utilities as util

import matplotlib.pyplot as plt
import gm2fr.style as style
style.setStyle()

import ROOT as root

# ==================================================================================================
# Gaussian mixture input.
# ==================================================================================================

# Create a Gaussian mixture for the muon kinematics.
# This can be cyclotron frequency (kHz), momentum (GeV), or momentum offset, specified later below.
# This example uses cyclotron frequency.
frequency = GaussianMixture(
  weights = [1, 0.5],
  means = [6705, 6702],
  widths = [10, 6]
)

# Create a Gaussian mixture for the injection time distribution.
# Units can be nanoseconds, microseconds, or seconds, specified later below.
# This example uses nanoseconds.
injection = GaussianMixture(
  weights = [1, 0.5, 0.3],
  means = [0, 30, -20],
  widths = [25, 15, 15]
)

# Choose poly. coeffs. (decreasing order) which shift the mean frequency over injection time.
# This example makes a linear shift of +10 kHz over 50 ns.
correlation = [10 / 50, 0]

# Create the simulation object, specifying the output directory name.
simulation = Simulator("exampleMixture", overwrite = True)

# Specify the source distributions (and optionally, correlation polynomial).
simulation.useMixture(
  frequency,     # GaussianMixture object for muon kinematics
  "frequency",   # kinematics variable type: "frequency", "momentum", or "offset"
  injection,     # GaussianMixture object for injection time
  "nanoseconds", # injection time units: "nanoseconds", "microseconds", or "seconds"
  correlation    # (optional) correlation polynomial coefficients
)

# Run, save, and plot the simulation.
simulation.simulate(muons = 1E6, decay = "uniform")
simulation.save()
simulation.plot()

# ==================================================================================================
# ROOT TH1 histogram input.
# ==================================================================================================

# Create/load the ROOT TH1 for muon kinematics.
# This can be cyclotron frequency (kHz), momentum (GeV), or momentum offset, specified later below.
# This example uses cyclotron frequency.
frequency = root.TH1F("", "", 150, 6630, 6780)
freqFunction = root.TF1("freqFunction", "gaus(0)", 6630, 6780)
freqFunction.SetParameters(1, 6705, 10)
frequency.FillRandom("freqFunction", 100_000)

# Create/load the ROOT TH1 for the injection times.
# Units can be nanoseconds, microseconds, or seconds, specified later below.
# This example uses nanoseconds.
injection = root.TH1F("", "", 150, -80, 80)
timeFunction = root.TF1("timeFunction", "gaus(0)", -80, 80)
timeFunction.SetParameters(1, 0, 25)
injection.FillRandom("timeFunction", 100_000)

# Choose poly. coeffs. (decreasing order) which shift the mean frequency over injection time.
# This example makes a quadratic shift of -10 kHz over the first 50 ns.
correlation = [-10 / 50**2, 0, 0]

# Create the simulation object, specifying the output directory name.
simulation = Simulator("exampleHistogram1D", overwrite = True)

# Specify the source histograms (and optionally, correlation polynomial).
simulation.useHistogram1D(
  frequency,     # ROOT TH1 for muon kinematics
  "frequency",   # kinematics variable type: "frequency", "momentum", or "offset"
  injection,     # ROOT TH1 for injection time
  "nanoseconds", # injection time units: "nanoseconds", "microseconds", or "seconds"
  correlation    # (optional) correlation polynomial coefficients
)

# Run, save, and plot the simulation.
simulation.simulate(muons = 1E6, decay = "uniform")
simulation.save()
simulation.plot()

# ==================================================================================================
# ROOT TH2 histogram input.
# ==================================================================================================

# Create/load a ROOT TH2 for the joint distribution of muon kinematics (x) and injection time (y).
# Kinematics can be cyclotron frequency (kHz), momentum (GeV), or momentum offset.
# Time units can be nanoseconds, microseconds, or seconds.
# This example uses momentum offset for kinematics and nanoseconds for injection time.
joint = root.TH2F("", "", 150, -80, 80, 150, -0.005, 0.005)
function = root.TF2("f", "xygaus", -80, 80, -0.005, 0.005)
function.SetParameters(1, 0, 25, 0, 0.0015)
joint.FillRandom("f", 1_000_000)

# Create the simulation object, specifying the output directory name.
simulation = Simulator("exampleHistogram2D", overwrite = True)

# Specify the source histogram.
simulation.useHistogram2D(
  joint,        # ROOT TH2 for joint distribution of injection time (x) and muon kinematics (y)
  "offset",     # kinematics variable type: "frequency", "momentum", or "offset"
  "nanoseconds" # injection time units: "nanoseconds", "microseconds", or "seconds"
)

# Run, save, and plot the simulation.
simulation.simulate(muons = 1E6, decay = "uniform")
simulation.save()
simulation.plot()
