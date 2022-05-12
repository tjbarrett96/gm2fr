import numpy as np
import matplotlib.pyplot as plt

# ==================================================================================================

# Speed of light (m/s).
c = 299792458

# Anomalous magnetic moment.
a_mu = 11659208.9E-10

# Muon charge (C).
q = 1.602176565E-19

# Mean muon lifetime at rest (us).
lifetime = 2.1969811

# Mass conversion factor from GeV to kg.
GeV_to_kg = 1E9 * q / c**2

# Muon mass (GeV).
m_mu_GeV = 0.1056583715

# Muon mass (kg).
m_mu_kg = m_mu_GeV * GeV_to_kg

# Nominal dipole magnetic field (T).
B = 1.4513

# Conversion from (kHz * us) to standard units (Hz * s = 1).
kHz_us = 1E-3

# ==================================================================================================

# Class that groups information pertaining to different quantities, e.g. momentum, frequency, etc.
class Quantity:

  def __init__(
    self,
    label, # readable label string to appear in plots, e.g. "Equilibrium Radius"
    symbol, # mathematical symbol in LaTeX format (assuming math mode already), e.g. r"\sigma_{x_e}"
    units, # string for nominal units of this quantity (NOT assuming math mode already), e.g. r"$\mu$s"
    magic = None, # nominal ("magic") value for this quantity
    min = None, # minimum storable value for this quantity
    max = None, # maximum storable value for this quantity
    toF = None, # function converting this quantity (in nominal units) to frequency (in kHz)
    fromF = None # function converting frequency (in kHz) to this quantity (in nominal units)
  ):
    self.label, self.symbol, self.units = label, symbol, units
    self.magic, self.min, self.max = magic, min, max
    self.toF, self.fromF = toF, fromF

  def formatLabel(self):
    return self.label + (f" ({self.units})" if self.units is not None else "")

# ==================================================================================================

# Dictionary containing information (e.g. labels, magic values, etc.) for various types of quantities.
# Intended for use in converting cyclotron frequency to other quantities.
# For now, just initialize with labels, and set the magic/min/max values and conversions below.
info = {
  "f": Quantity("Frequency", "f", "kHz"),
  "r": Quantity("Equilibrium Radius", "r", "mm"),
  "x": Quantity("Equilibrium Radial Offset", "x", "mm"),
  "p": Quantity("Momentum", "p", "GeV"),
  "gamma": Quantity("Boost Factor", r"\gamma", None),
  "beta": Quantity("Speed", r"\beta", "$c$"),
  "dp_p0": Quantity("Fractional Momentum Offset", r"\Delta p/p_0", None),
  "T": Quantity("Period", "T", "ns"),
  "tau": Quantity("Lifetime", r"\tau", r"$\mu$s"),
  "c_e": Quantity("Electric Field Correction", "C_E", "ppb")
}

# Set the magic values for all the quantities.
info["p"].magic = m_mu_GeV / np.sqrt(a_mu) # momentum value that zeroes electric field term
info["gamma"].magic = np.sqrt(1 / a_mu + 1) # gamma value that zeroes electric field term
info["beta"].magic = info["p"].magic / (m_mu_GeV * info["gamma"].magic) # v = p/(gamma*m)
info["r"].magic = (info["p"].magic * GeV_to_kg * c) / (q * B) * 1E3 # r = p/(qB), in mm
info["f"].magic = q * B / (2 * np.pi * info["gamma"].magic * m_mu_kg) * 1E-3 # w = qB/(gamma*m), in kHz
info["T"].magic = 1 / (info["f"].magic * 1E3) * 1E9 # T = 1/f, in ns
info["tau"].magic = info["gamma"].magic * lifetime # tau = gamma * tau_0, in us
info["x"].magic = 0
info["c_e"].magic = 0
info["dp_p0"].magic = 0

# Set the conversion functions from frequency.
info["f"].fromF = lambda f: f
info["r"].fromF = lambda f: info["beta"].magic * c / (2 * np.pi * f) # r = v/w, in mm for w in kHz
info["x"].fromF = lambda f: info["r"].fromF(f) - info["r"].magic # x = r - r_0
info["T"].fromF = lambda f: 1 / (f * 1E3) * 1E9 # T = 1/f, in ns
info["p"].fromF = lambda f, n = 0.108: info["p"].magic * (1 + (1 - n) * (1 - f / info["f"].magic))
info["gamma"].fromF = lambda f, n = 0.108: info["p"].fromF(f, n) / (m_mu_GeV * info["beta"].magic)
info["tau"].fromF = lambda f, n = 0.108: info["gamma"].fromF(f, n) * lifetime
info["dp_p0"].fromF = lambda f, n = 0.108: (info["p"].fromF(f, n) - info["p"].magic) / info["p"].magic
info["beta"].fromF = lambda f, n = 0.108: np.sqrt(1 - 1 / info["gamma"].fromF(f, n)**2)
info["c_e"].fromF = lambda f, n = 0.108: 2 * n * (1 - n) * (info["beta"].magic * info["x"].fromF(f) / info["r"].magic)**2 * 1E9

# Set the conversion functions to frequency (only useful for some variables).
info["f"].toF = lambda f: f
info["r"].toF = lambda r: info["beta"].magic * c / (2 * np.pi * r) # f = v/r, in kHz for r in mm
info["p"].toF = lambda p, n = 0.108: info["f"].magic * (1 - (p / info["p"].magic - 1) / (1 - n))
info["dp_p0"].toF = lambda dp_p0, n = 0.108: info["p"].toF(info["p"].magic * (1 + dp_p0), n)

# Set the minimum and maximum values for stored radii, and convert to frequency.
info["r"].min, info["r"].max = info["r"].magic - 45, info["r"].magic + 45
info["f"].min, info["f"].max = info["r"].toF(info["r"].max), info["r"].toF(info["r"].min)
# Everything else is inversely proportional to frequency, so quantity_min ~ f_max and vice versa.
for quantity in info.keys():
  if quantity in ("r", "f"):
    continue
  info[quantity].min = info[quantity].fromF(info["f"].max)
  info[quantity].max = info[quantity].fromF(info["f"].min)

# ==================================================================================================

# Given a NumPy array of a quantity, find the index mask for physical (storable) values.
def physical(array, unit = "f"):
  return (array >= info[unit].min) & (array <= info[unit].max)

# Given an array of a quantity, find the index mask for unphysical (unstorable) values.
def unphysical(array, unit = "f"):
  return ~physical(array, unit)
