from gm2fr.simulation.mixture import GaussianMixture
from gm2fr.simulation.simulator import Simulator
import gm2fr.utilities as util

import matplotlib.pyplot as plt
import gm2fr.style as style
style.set_style()

import ROOT as root

# ==================================================================================================
# No muon decay. Each muon counted again on every turn.
# Fully correlated statistics in each turn.
# Makes the smoothest signal from the fewest muons, if bin errors aren't important.
# ==================================================================================================

frequency = GaussianMixture([1], [6705], [10])
injection = GaussianMixture([1], [0], [25])

simulation = Simulator(
  "exampleNoDecay",
  overwrite = True,
  kinematicsDistribution = frequency,
  kinematicsUnits = "frequency",
  timeDistribution = injection,
  timeUnits = "nanoseconds"
)

simulation.simulate(muons = 1E6, decay = "none", normalize = False)
simulation.save()
simulation.plot()

# ==================================================================================================
# Exponential muon decay. Each muon counted once, based on a random exponential decay time.
# Mostly uncorrelated statistics, except for dividing out the exponential envelope.
# Requires many more muons to get a smooth signal, but bin errors are more meaningful.
# ==================================================================================================

simulation = Simulator(
  "exampleExponential",
  overwrite = True,
  kinematicsDistribution = frequency,
  kinematicsUnits = "frequency",
  timeDistribution = injection,
  timeUnits = "nanoseconds"
)

simulation.simulate(muons = 1E7, decay = "exponential", normalize = False)
simulation.save()
simulation.plot()

# ==================================================================================================
# Uniform muon decay. Each muon counted once, and the turn number is chosen uniformly at random.
# Entirely uncorrelated statistics.
# Requires many more muons to get a smooth signal, but ideal choice when bin errors are important.
# ==================================================================================================

simulation = Simulator(
  "exampleUniform",
  overwrite = True,
  kinematicsDistribution = frequency,
  kinematicsUnits = "frequency",
  timeDistribution = injection,
  timeUnits = "nanoseconds"
)

simulation.simulate(muons = 1E7, decay = "uniform", normalize = False)
simulation.save()
simulation.plot()
