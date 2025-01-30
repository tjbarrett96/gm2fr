from gm2fr.src.mixture import GaussianMixture
#from gm2fr.src.simulator import Simulator
from gm2fr.src.Histogram2D import Histogram2D
import gm2fr.src.io as io
import gm2fr.src.constants as const
# import root_numpy as rnp
from scipy.ndimage import gaussian_filter
import numpy as np
from scipy.interpolate import RectBivariateSpline
#from scipy.interpolate import SmoothBivariateSpline

import matplotlib.pyplot as plt
import gm2fr.src.style as style
style.set_style()

# import ROOT as root
import uproot

import argparse

# ==================================================================================================

def smooth_joint(filename, label, output_filename, kinematics_type, time_units, kinematics_scale, interp_factor, transpose):

  # # open ROOT file and get joint histogram
  # input_file = root.TFile(f"{io.data_path}/correlation/{filename}.root")
  # joint_root = input_file.Get(label)
  pdf = style.make_pdf(f"{io.data_path}/correlation/{output_filename}.pdf")

  # load joint distribution from file
  joint = Histogram2D.load(f"{io.data_path}/correlation/{filename}", label, transpose = transpose)

  # convert kinematics (y) axis to frequency, and time (x) axis to nanoseconds
  joint.map(
    y = lambda y: const.info[kinematics_type].to_frequency(y * kinematics_scale),
    x = lambda t: t * time_units / 1E-9
  )

  if interp_factor > 1:

    # get the time and frequency bin widths
    time_bin_width = joint.x_width
    freq_bin_width = joint.y_width[0] if io.is_iterable(joint.y_width) else joint.y_width

    # check if we should rebin the time axis
    min_time_bin = 6E-9
    time_rebin_factor = max(1, int(np.floor(min_time_bin / (time_bin_width * time_units))))

    # check if we should rebin the frequency axis
    min_freq_bin = 4
    kinematics_rebin_factor = max(1, int(np.floor(min_freq_bin / freq_bin_width)))

    # rebin the distribution
    if (time_rebin_factor > 1) or (kinematics_rebin_factor > 1):
      joint.rebin(xStep = time_rebin_factor, yStep = kinematics_rebin_factor)
  
    # create a spline out of the data
    spline = RectBivariateSpline(joint.x_centers, joint.y_centers, joint.heights)
    
    # interpolate the x and y bin boundaries
    #interp_factor = 3
    new_x_edges = np.linspace(joint.x_edges[0], joint.x_edges[-1], len(joint.x_edges) * interp_factor)
    new_y_edges = np.linspace(joint.y_edges[0], joint.y_edges[-1], len(joint.y_edges) * interp_factor)
    new_x_centers = (new_x_edges[:-1] + new_x_edges[1:]) / 2
    new_y_centers = (new_y_edges[:-1] + new_y_edges[1:]) / 2
    
    # flattened 1d arrays of matching x and y coordinates in grid
    new_x_centers_1d = np.repeat(new_x_centers, len(new_y_centers))
    new_y_centers_1d = np.tile(new_y_centers, len(new_x_centers))

    # evaluate spline at interpolated bin centers
    new_heights = spline.ev(new_x_centers_1d, new_y_centers_1d).reshape((len(new_x_centers), len(new_y_centers)))
    new_joint = Histogram2D(x_bins = new_x_edges, y_bins = new_y_edges, heights = new_heights)
    # new_joint.heights = np.where(new_joint.heights < 0.001 * np.max(new_joint.heights), 0, new_joint.heights)
  
  else:

    new_joint = Histogram2D(x_bins = joint.x_edges, y_bins = joint.y_edges, heights = joint.heights)
  
  #joint.map(y = lambda y: const.info["dp_p0"].from_frequency(y))
  #new_joint.map(y = lambda y: const.info["dp_p0"].from_frequency(y)) 

  # plot the original distribution
  joint.plot()
  style.ylabel("Frequency (kHz)")
  #style.ylabel("Relative Momentum Offset")
  style.xlabel("Injection Time (ns)")
  pdf.savefig()
  
  plt.figure()
  new_joint.plot()
  plt.xlim(-75, 75)
  style.ylabel("Frequency (kHz)")
  #style.ylabel("Relative Momentum Offset")
  style.xlabel("Injection Time (ns)")
  pdf.savefig()
  
  new_joint_heights, new_joint_x, new_joint_y = new_joint.to_root("joint")
  # time_profile = new_joint.ProjectionX("time")
  #freq_profile = new_joint.ProjectionY("frequencies")
  # freq_profile = new_joint.ProjectionY("dp_p0")
  time_profile = (np.sum(new_joint_heights, axis = 1), new_joint_x)
  freq_profile = (np.sum(new_joint_heights, axis = 0), new_joint_y)

  # output_file = root.TFile(f"{io.data_path}/correlation/{output_filename}.root", "RECREATE")
  # new_joint.Write()
  # time_profile.Write()
  # freq_profile.Write()
  # output_file.Close()
  with uproot.recreate(f"{io.data_path}/correlation/{output_filename}.root") as root_file:
    root_file["joint"] = (new_joint_heights, new_joint_x, new_joint_y)
    root_file["profile"] = time_profile
    root_file["frequencies"] = freq_profile
    #root_file["time"] = time_profile
    #root_file["dp_p0"] = freq_profile

  pdf.close()

# ==================================================================================================

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--file")
  parser.add_argument("--label")
  parser.add_argument("--output")
  parser.add_argument("--type")
  parser.add_argument("--transpose", action="store_true")
  parser.add_argument("--factor", type = int, default = 3)
  parser.add_argument("--time", type = float)
  parser.add_argument("--scale", type = float, default = 1)
  args = parser.parse_args()

  smooth_joint(args.file, args.label, args.output, args.type, args.time, args.scale, args.factor, args.transpose)

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
