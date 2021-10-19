from gm2fr.simulation.mixture import GaussianMixture
from gm2fr.simulation.simulator import Simulator
import gm2fr.utilities as util

import matplotlib.pyplot as plt
import gm2fr.style as style
style.setStyle()

# Create a Gaussian mixture distribution for the muon revolution frequencies.
# This is a probability distribution modeled as a sum of individual Gaussians.
# This allows us to model interesting and asymmetric distributions for testing.
# Each Gaussian has a relative weight (i.e. scale/amplitude), mean, and width.
# Input format is a list for each parameter.
# Reference: magic frequency is 6705 kHz, with spread about ~10 kHz.
# For this example, I'll keep it simple, to check the analysis results.
frequency = GaussianMixture(
  weights = [1],
  means = [6705],
  widths = [8.5]
)

# Create a Gaussian mixture distribution for the injection time profile, in ns.
# Same format as above.
# Reference: generally centered near zero, with spread about ~30 ns.
# Realistic profiles often contain 3+ peaks, with 2 on either side of zero.
injection = GaussianMixture(
  # weights = [1, 0.5, 0.3],
  # means = [0, 30, -20],
  # widths = [25, 15, 15]
  weights = [1],
  means = [0],
  widths = [25]
)

# To model a toy correlation between frequency and injection time, we can define a function which
# shifts the mean of the frequency distribution as a function of injection time.
# Choose poly. coeffs. (decreasing order) which shift the mean frequency over injection time.
# For simple testing with no correlation, make this [0], or omit from the simulation object below.
# e.g. correlation = [10 / 50, 0] makes a linear shift of +/- 10 kHz over +/- 50 ns.
correlation = [10/50**2, 0, 0]

# Create the simulation object, specifying the output directory name.
# The specified folder will be created in your current directory.
simulation = Simulator(
  "gaussian_quadratic",
  overwrite = True,
  kinematicsDistribution = frequency,
  timeDistribution = injection,
  kinematicsUnits = "frequency",
  timeUnits = "nanoseconds",
  correlation = correlation
)

# Run the simulation, using the specified number of muons.
simulation.simulate(muons = 1E8, end = 200)

# Save and plot the results.
simulation.save()
simulation.plot()
