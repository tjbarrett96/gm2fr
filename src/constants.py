import numpy as np

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
    to_frequency = None, # function converting this quantity (in nominal units) to frequency (in kHz)
    from_frequency = None # function converting frequency (in kHz) to this quantity (in nominal units)
  ):
    self.label, self.symbol, self.units = label, symbol, units
    self.magic, self.min, self.max = magic, min, max
    self.to_frequency, self.from_frequency = to_frequency, from_frequency

  def format_label(self):
    return self.label + (f" ({self.units})" if self.units is not None else "")

  def format_symbol(self):
    return f"${self.symbol}$" + (f" ({self.units})" if self.units is not None else "")

# ==================================================================================================

# Dictionary containing information (e.g. labels, magic values, etc.) for various types of quantities.
# Intended for use in converting cyclotron frequency to other quantities.
# For now, just initialize with labels, and set the magic/min/max values and conversions below.
info = {
  "f": Quantity("Frequency", "f", "kHz"),
  "r": Quantity("Equilibrium Radius", "r", "mm"),
  "x": Quantity("Equilibrium Radial Offset", "x_e", "mm"),
  "p": Quantity("Momentum", "p", "GeV"),
  "gamma": Quantity("Boost Factor", r"\gamma", None),
  "beta": Quantity("Speed", r"\beta", "$c$"),
  "dp_p0": Quantity("Fractional Momentum Offset", r"\Delta p/p_0", None),
  "dp_p0_%": Quantity("Fractional Momentum Offset", r"\Delta p/p_0", "%"),
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
info["dp_p0_%"].magic = 0

# Set the conversion functions from frequency.
info["f"].from_frequency = lambda f: f
info["r"].from_frequency = lambda f: info["beta"].magic * c / (2 * np.pi * f) # r = v/w, in mm for w in kHz
info["x"].from_frequency = lambda f: info["r"].from_frequency(f) - info["r"].magic # x = r - r_0
info["T"].from_frequency = lambda f: 1 / (f * 1E3) * 1E9 # T = 1/f, in ns
info["p"].from_frequency = lambda f, n = 0.108: info["p"].magic * (1 + (1 - n) * (1 - f / info["f"].magic))
info["gamma"].from_frequency = lambda f, n = 0.108: info["p"].from_frequency(f, n) / (m_mu_GeV * info["beta"].magic)
info["tau"].from_frequency = lambda f, n = 0.108: info["gamma"].from_frequency(f, n) * lifetime
info["dp_p0"].from_frequency = lambda f, n = 0.108: (info["p"].from_frequency(f, n) - info["p"].magic) / info["p"].magic
info["dp_p0_%"].from_frequency = lambda f, n = 0.108: info["dp_p0"].from_frequency(f, n) * 100
info["beta"].from_frequency = lambda f, n = 0.108: np.sqrt(1 - 1 / info["gamma"].from_frequency(f, n)**2)
info["c_e"].from_frequency = lambda f, n = 0.108: 2*n*(1-n)*(info["beta"].magic * info["x"].from_frequency(f) / info["r"].magic)**2 * 1E9

# Set the conversion functions to frequency (only useful for some variables).
info["f"].to_frequency = lambda f: f
info["r"].to_frequency = lambda r: info["beta"].magic * c / (2 * np.pi * r) # f = v/r, in kHz for r in mm
info["p"].to_frequency = lambda p, n = 0.108: info["f"].magic * (1 - (p / info["p"].magic - 1) / (1 - n))
info["dp_p0"].to_frequency = lambda dp_p0, n = 0.108: info["p"].to_frequency(info["p"].magic * (1 + dp_p0), n)
info["dp_p0_%"].to_frequency = lambda dp_p0, n = 0.108: info["dp_p0"].to_frequency(dp_p0 / 100, n)

# Set the minimum and maximum values for stored radii, and convert to frequency.
info["r"].min, info["r"].max = info["r"].magic - 45, info["r"].magic + 45
info["f"].min, info["f"].max = info["r"].to_frequency(info["r"].max), info["r"].to_frequency(info["r"].min)

# Everything else is inversely proportional to frequency, so quantity_min ~ f_max and vice versa.
for quantity in info.keys():
  if quantity in ("r", "f"):
    continue
  info[quantity].min = info[quantity].from_frequency(info["f"].max)
  info[quantity].max = info[quantity].from_frequency(info["f"].min)

# Add corresponding width quantities.
for quantity in list(info.keys()):
  info[f"sig_{quantity}"] = Quantity(
    f"{info[quantity].label} Width",
    rf"\sigma_{{{info[quantity].symbol}}}",
    info[quantity].units
  )

# Add extra variables.
info["start"] = Quantity("Start Time", "t_s", r"$\mu$s")
info["end"] = Quantity("End Time", "t_m", r"$\mu$s")
info["df"] = Quantity("Frequency Spacing", r"\Delta f", "kHz")
info["bg_width"] = Quantity("Background Width", r"\mathrm{Background\;Width}", "kHz")
info["bg_space"] = Quantity("Background Space", r"\mathrm{Background\;Space}", "kHz")
info["bg_model"] = Quantity("Background Model", r"\mathrm{Background\;Model}", None)
info["fr_method"] = Quantity("FR Signal Method", r"\mathrm{FR\;Signal\;Method}", None)
info["dt"] = Quantity("FR Signal Bin Width", r"\Delta t", "ns")
info["t0"] = Quantity("Reference Time", "t_0", "ns")
info["bg_chi2_ndf"] = Quantity(r"Background Fit $\chi^2$/ndf", r"\mathrm{Background} \; \chi^2/\mathrm{ndf}", None)
info["bg_pval"] = Quantity("Background Fit $p$-value", "p", None)
info["wg_chi2_ndf"] = Quantity(r"Wiggle Fit $\chi^2$/ndf", r"\mathrm{Wiggle} \; \chi^2/\mathrm{ndf}", None)
info["wg_pval"] = Quantity("Wiggle Fit $p$-value", "p", None)
info["wg_N"] = Quantity("Wiggle Normalization", "N", None)
info["wg_tau"] = Quantity("Wiggle Lifetime", r"\tau_\mu", r"$\mu$s")
info["wg_A"] = Quantity("Wiggle Asymmetry", "A", None)
info["wg_phi_a"] = Quantity("Wiggle Phase", r"\phi_a", "rad")
info["wg_tau_cbo"] = Quantity("Wiggle CBO Lifetime", r"\tau_{CBO}", r"$\mu$s")
info["wg_A_cbo"] = Quantity("Wiggle CBO Amplitude", "A_{CBO}", None)
info["wg_f_cbo"] = Quantity("Wiggle CBO Frequency", "f_{CBO}", "MHz")
info["wg_phi_cbo"] = Quantity("Wiggle CBO Phase", r"\phi_{CBO}", "rad")

# ==================================================================================================

# Given a NumPy array of a quantity, find the index mask for physical (storable) values.
def physical(array, unit = "f"):
  return (array >= info[unit].min) & (array <= info[unit].max)

# Given an array of a quantity, find the index mask for unphysical (unstorable) values.
def unphysical(array, unit = "f"):
  return ~physical(array, unit)
