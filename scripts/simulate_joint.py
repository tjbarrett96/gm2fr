from gm2fr.src.mixture import GaussianMixture
from gm2fr.src.simulator import Simulator
import gm2fr.src.io as io
import gm2fr.src.constants as const
import root_numpy as rnp
from scipy.ndimage import gaussian_filter
import numpy as np
from scipy.interpolate import RectBivariateSpline

import matplotlib.pyplot as plt
import gm2fr.src.style as style
style.set_style()

import ROOT as root

import argparse

# ==================================================================================================

def simulate_joint(filename, label, kinematics_type, time_units, smooth, kinematics_scale):

  input_file = root.TFile(f"{io.data_path}/correlation/{filename}.root")
  joint_dist = input_file.Get(label)

  time_bin_width = joint_dist.GetXaxis().GetBinWidth(1)
  freq_bin_width = abs(
    const.info[kinematics_type].to_frequency(joint_dist.GetYaxis().GetBinCenter(2) * kinematics_scale) \
    - const.info[kinematics_type].to_frequency(joint_dist.GetYaxis().GetBinCenter(1) * kinematics_scale)
  )

  min_time_bin = 1E-9
  time_rebin_factor = int(np.ceil(min_time_bin / (time_bin_width * time_units)))

  min_freq_bin = 1
  kinematics_rebin_factor = int(np.ceil(min_freq_bin / freq_bin_width))

  joint_dist.Rebin2D(time_rebin_factor, kinematics_rebin_factor)

  if smooth:
    data, (x_edges, y_edges) = rnp.hist2array(joint_dist, return_edges = True)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    spline = RectBivariateSpline(x_centers, y_centers, data)
    interp_factor = 2
    new_x_edges = np.linspace(x_edges[0], x_edges[-1], len(x_edges) * interp_factor)
    new_y_edges = np.linspace(y_edges[0], y_edges[-1], len(y_edges) * interp_factor)
    new_x_centers = (new_x_edges[:-1] + new_x_edges[1:]) / 2
    new_y_centers = (new_y_edges[:-1] + new_y_edges[1:]) / 2
    new_data = spline.ev(new_x_centers, new_y_centers)
    joint_dist = root.TH2D(
      joint_dist.GetName(),
      joint_dist.GetTitle(),
      new_x_edges[0],
      new_x_edges[-1],
      len(new_x_centers),
      new_y_edges[0],
      new_y_edges[-1],
      len(new_y_centers)
    )
    rnp.array2hist(new_data, joint_dist)
    # Ensure the frequency distribution goes to zero at the boundaries.
    # edge_mean = (np.mean(data[:, 0]) + np.mean(data[:, -1])) / 2
    # data -= edge_mean
    # Smooth the distribution with Gaussian smearing.
    # smooth_data = gaussian_filter(data, sigma = (5, 3), truncate = 1)
    # rnp.array2hist(np.where(smooth_data < 0, 0, smooth_data), joint_dist)

  simulation = Simulator(
    f"{filename}_sim",
    overwrite = True,
    joint_dist = joint_dist,
    kinematics_type = kinematics_type,
    time_units = time_units,
    kinematics_scale = kinematics_scale
  )

  simulation.simulate(muons = 1E9, end = 250)
  simulation.save()
  simulation.plot()

# ==================================================================================================

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--file")
  parser.add_argument("--label")
  parser.add_argument("--type")
  parser.add_argument("--time", type = float)
  parser.add_argument("--scale", type = float, default = 1)
  parser.add_argument("--smooth", action = "store_true")
  args = parser.parse_args()

  simulate_joint(args.file, args.label, args.type, args.time, args.smooth, args.scale)

# ==================================================================================================

# inFile = root.TFile(f"{io.gm2fr_path}/../archived/realistic-correlation/tarazona_v2.root")
# joint = inFile.Get("dp_dt_v1")
#
# data = rnp.hist2array(joint)
# smoothData = smooth(data, sigma = 1, truncate = 20)
# rnp.array2hist(smoothData, joint)
#
# # Create the simulation object, specifying the output directory name.
# # The specified folder will be created in your current directory.
# simulation = Simulator(
#   "../data/cosy_smooth",
#   overwrite = True,
#   joint_dist = joint,
#   kinematics_type = "dp_p0",
#   time_units = 1
# )
#
# # Run the simulation, using the specified number of muons.
# simulation.simulate(muons = 1E7, end = 200)
#
# # Save and plot the results.
# simulation.save()
# simulation.plot()
