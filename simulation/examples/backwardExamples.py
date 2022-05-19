from gm2fr.simulation.mixture import GaussianMixture
from gm2fr.simulation.simulator import Simulator
import gm2fr.utilities as util

import matplotlib.pyplot as plt
import gm2fr.style as style
style.set_style()

import ROOT as root

# ==================================================================================================
# No muon decay, both forward and backward in time.
# ==================================================================================================

frequency = GaussianMixture([1], [6705], [10])
injection = GaussianMixture([1], [0], [25])

simulation = Simulator("exampleBackwardNoDecay", overwrite = True)
simulation.useMixture(frequency, "frequency", injection, "nanoseconds")

simulation.simulate(muons = 1E6, decay = "none", backward = True)
simulation.save()
simulation.plot()

# ==================================================================================================
# Exponential muon decay, both forward and backward in time.
# ==================================================================================================

simulation = Simulator("exampleBackwardExponential", overwrite = True)
simulation.useMixture(frequency, "frequency", injection, "nanoseconds")

simulation.simulate(muons = 1E7, decay = "exponential", backward = True)
simulation.save()
simulation.plot()

# ==================================================================================================
# Uniform muon decay, both forward and backward in time.
# ==================================================================================================

simulation = Simulator("exampleBackwardUniform", overwrite = True)
simulation.useMixture(frequency, "frequency", injection, "nanoseconds")

simulation.simulate(muons = 1E7, decay = "uniform", backward = True)
simulation.save()
simulation.plot()
