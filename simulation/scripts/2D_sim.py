from gm2fr.simulation.mixture import GaussianMixture
from gm2fr.simulation.simulator import Simulator
import gm2fr.io as io
import root_numpy as rnp
from scipy.ndimage import gaussian_filter as smooth

import matplotlib.pyplot as plt
import gm2fr.style as style
style.setStyle()

import ROOT as root

inFile = root.TFile(f"{io.gm2fr_path}/../archived/realistic-correlation/tarazona_v2.root")
joint = inFile.Get("dp_dt_v1")

data = rnp.hist2array(joint)
smoothData = smooth(data, sigma = 1, truncate = 20)
rnp.array2hist(smoothData, joint)

# Create the simulation object, specifying the output directory name.
# The specified folder will be created in your current directory.
simulation = Simulator(
  "../data/cosy_smooth",
  overwrite = True,
  joint_dist = joint,
  kinematics_type = "dp_p0",
  time_units = 1
)

# Run the simulation, using the specified number of muons.
simulation.simulate(muons = 1E7, end = 200)

# Save and plot the results.
simulation.save()
simulation.plot()
