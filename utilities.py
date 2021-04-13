import numpy as np
import pandas as pd
import scipy.stats

# ==============================================================================

# Speed of light (m/s).
c = 299792458

# Anomalous magnetic moment.
a_mu = 11659208.9E-10

# Muon charge (C).
q_mu = 1.602176565E-19

# Mean muon lifetime at rest (ns).
# TODO: convert this to microseconds. probably also requires making the simulation code all in terms of microseconds... should be done anyway.
lifetime = 2.1969811E3

# Mass conversion factor from GeV to kg.
GeV_to_kg = 1E9 * q_mu / c**2

# Muon mass (GeV).
m_mu_GeV = 0.1056583715

# Muon mass (kg).
m_mu_kg = m_mu_GeV * GeV_to_kg

# Nominal dipole magnetic field (T).
b = 1.4513

# ==============================================================================

magic = {
  "x": 0,
  "dp/p0": 0,
  "c_e": 0,
  "gamma": np.sqrt(1 / a_mu + 1)
}

magic["beta"] = np.sqrt(1 - 1 / magic["gamma"]**2)
magic["p"] = magic["gamma"] * m_mu_GeV * magic["beta"]
magic["r"] = (magic["p"] * GeV_to_kg * c) / (q_mu * b) * 1E3
magic["tau"] = magic["gamma"] * lifetime * 1E-3

# ==============================================================================

# Cyclotron radius (mm) to cyclotron frequency (kHz).
def radiusToFrequency(r):
  return (magic["beta"] * c / r) / (2 * np.pi)

# Cyclotron frequency (kHz) to cyclotron radius (mm).
def frequencyToRadius(f):
  return magic["beta"] * c / (2 * np.pi * f)

# Magic cyclotron frequency (kHz).
magic["f"] = radiusToFrequency(magic["r"])

# Conversion from cyclotron frequency (kHz) to cyclotron period (ns).
def frequencyToPeriod(f):
  return 1 / (f * 1E3) * 1E9

# Magic cyclotron period (ns).
magic["T"] = frequencyToPeriod(magic["f"])

# Momentum (GeV) to radius (mm).
def momentumToRadius(p, n = 0.108):
  return magic["r"] * (1 + 1 / (1 - n) * (p - magic["p"]) / magic["p"])

# Radius (mm) to momentum (GeV).
def radiusToMomentum(r, n = 0.108):
  return magic["p"] * (1 + (1 - n) * (r - magic["r"]) / magic["r"])

# Momentum (GeV) to cyclotron frequency (kHz).
def momentumToFrequency(p, n = 0.108):
  return magic["f"] * (1 - 1 / (1 - n) * (p - magic["p"]) / magic["p"])

# Cyclotron frequency (kHz) to momentum (GeV).
def frequencyToMomentum(f, n = 0.108):
  return magic["p"] * (1 + (1 - n) * (1 - f / magic["f"]))

# Cyclotron frequency (kHz) to gamma.
def frequencyToGamma(f, n = 0.108):
  return frequencyToMomentum(f, n) / (m_mu_GeV * magic["beta"])

# Conversion from fractional momentum offset to momentum (GeV).
def offsetToMomentum(dpp0):
  return (1 + dpp0) * magic["p"]

# Conversion from momentum (GeV) to fractional momentum offset.
def momentumToOffset(p):
  return (p - magic["p"]) / magic["p"]

# Conversion from fractional momentum offset to cyclotron frequency (kHz).
def offsetToFrequency(dpp0):
  return momentumToFrequency(offsetToMomentum(dpp0))

# Conversion from cyclotron frequency (kHz) to cyclotron radial offset (mm).
def frequencyToRadialOffset(f):
  return frequencyToRadius(f) - magic["r"]

def correction(mean, width, n = 0.108):
  return 2 * n * (1 - n) * magic["beta"]**2 * (mean**2 + width**2) / magic["r"]**2 * 1E9

# Conversion from cyclotron radial offsets (mm) to electric field correction (ppb).
def radialOffsetToCorrection(radii, heights, n = 0.108):
  mean = np.average(radii, weights = heights)
  std = np.sqrt(np.average((radii - mean)**2, weights = heights))
  return correction(mean, std)

# Conversion from cyclotron frequencies (kHz) to electric field correction (ppb).
def frequencyToCorrection(frequencies, heights, n = 0.108):
  return radialOffsetToCorrection(frequencyToRadialOffset(frequencies), heights, n)

# ==============================================================================

min = {
  "x": -45
}

max = {
  "x": +45
}

min["r"] = min["x"] + magic["r"]
max["r"] = max["x"] + magic["r"]

min["f"] = radiusToFrequency(max["r"])
max["f"] = radiusToFrequency(min["r"])

min["T"] = frequencyToPeriod(max["f"])
max["T"] = frequencyToPeriod(min["f"])

min["gamma"] = frequencyToGamma(max["f"])
max["gamma"] = frequencyToGamma(min["f"])

min["p"] = radiusToMomentum(min["r"])
max["p"] = radiusToMomentum(max["r"])

min["dp/p0"] = momentumToOffset(min["p"]) * 100
max["dp/p0"] = momentumToOffset(max["p"]) * 100

min["tau"] = min["gamma"] * lifetime * 1E-3
max["tau"] = max["gamma"] * lifetime * 1E-3

min["beta"] = np.sqrt(1 - 1/min["gamma"]**2)
max["beta"] = np.sqrt(1 - 1/max["gamma"]**2)

# ==============================================================================

# String labels for each variable: math mode, units, plot axes, and filename.
labels = {

  "f": {
    "math": "f",
    "units": "kHz",
    "plot": "Revolution Frequency",
    "file": "frequency"
  },

  "T": {
    "math": "T",
    "units": "ns",
    "plot": "Revolution Period",
    "file": "period"
  },

  "x": {
    "math": "x_e",
    "units": "mm",
    "plot": "Equilibrium Radius",
    "file": "radius"
  },

  "p": {
    "math": "p",
    "units": "GeV",
    "plot": "Momentum",
    "file": "momentum"
  },

  "gamma": {
    "math": r"\gamma",
    "units": "",
    "plot": "Gamma Factor",
    "file": "gamma"
  },

  "tau": {
    "math": r"\tau",
    "units": r"$\mu$s",
    "plot": "Muon Lifetime",
    "file": "lifetime"
  },

  "dp/p0": {
    "math": r"\delta p/p_0",
    "units": r"\%",
    "plot": "Fractional Momentum Offset",
    "file": "offset"
  },

  "c_e": {
    "math": "C_E",
    "units": "ppb",
    "plot": "Electric Field Correction",
    "file": "correction"
  },

  "t0": {
    "math": "t_0",
    "units": "ns",
    "plot": "$t_0$",
    "file": "t0"
  },

  "sig_x": {
    "math": r"\sigma_{x_e}",
    "units": "mm",
    "plot": "Equilibrium Radial Width",
    "file": "width"
  }

}

# ==============================================================================

# Conversion from Antoine's parameter labels to LaTeX format.
symbols = {
  "t0": r"t_0",
  "eq_radius": r"x_e",
  "std": r"\sigma",
  "c_e": r"C_E",
  "tS": r"t_s",
  "tM": r"t_m"
}

# Conversion from Antoine's parameter labels to standard units in LaTeX format.
units = {
  "t0": "ns",
  "eq_radius": "mm",
  "std": "mm",
  "c_e": "ppb",
  "tS": r"$\mu$s",
  "tM": r"$\mu$s"
}

# ==============================================================================

# Load Antoine's output parameter format into a Python dictionary.
def loadResults(filename):

  with open(filename, "r") as file:

    # Read input data.
    items = [line.replace(" \n", "").split(" ") for line in file.readlines()]

    # Read every other entry from the first line to get the header labels.
    labels = items[0][::2]

    # Read every other entry from each line to get the values.
    values = np.array([line[1::2] for line in items], dtype = np.float64)

    # Package as pandas table.
    table = {labels[i]: values[:, i] for i in range(len(labels))}
    table["t0"] *= 1000

    return table

# ==============================================================================

# Invert the matrix A, checking for success.
def invert(A, atol = 1e-2):

  # Check if the matrix is safely invertible.
  if np.linalg.cond(A) * np.finfo(A.dtype).eps > 0.1:
    raise RuntimeError("Matrix ill-conditioned for inversion.")

  # Evaluate the inverse.
  result = np.linalg.inv(A)

  # Furthermore, check that the result is sensible: inv(A) * A ~ A * inv(A) ~ I.
  id = np.eye(A.shape[0])
  firstCheck = np.allclose(result @ A, id, atol = atol)
  secondCheck = np.allclose(A @ result, id, atol = atol)
  if not firstCheck or not secondCheck:
    print("Determinant:", np.linalg.det(A))
    raise RuntimeError("Matrix inversion unsuccessful.")

  return result

# ==============================================================================

# Calculate the two-sided p-value from the specified chi-squared distribution.
def pval(chi2, ndf):

  # Difference between the observed chi-squared and expectation value.
  diff = abs(chi2 - ndf)

  # Probability of a more extreme value on the left side of the mean.
  left = scipy.stats.chi2.cdf(ndf - diff, ndf)

  # Probability of a more extreme value on the right side of the mean.
  # (sf = 1 - cdf)
  right = scipy.stats.chi2.sf(ndf + diff, ndf)

  # Total probability of a more extreme value than what was observed.
  return left + right

# ==============================================================================

# One-sided FFT magnitude and frequencies.
def spectrum(time, data):

  # Calculate the sampling frequency.
  samplingFrequency = time[1] - time[0]

  # Get the DFT.
  transform = np.fft.fft(data)

  # Get the size of the signal/transform.
  n = len(transform)

  # Take the magnitude.
  transform = np.abs(transform)

  # Fold the negative-frequency bins onto the positive-frequency bins.
  for i in range(1, int(np.ceil(n / 2 - 1)) + 1):
    transform[i] += transform[n - i]

  # Discard the negative-frequency bins.
  transform = transform[0:(int(n / 2) + 1)]

  # Get the frequencies corresponding to the DFT bins.
  frequencies = np.arange(len(transform)) * samplingFrequency / n

  return frequencies, transform

# ==============================================================================

# One-sided FFT and frequencies (kHz), assuming time in microseconds.
def fft(time, data):

  # Calculate the FFT.
  transform = np.fft.rfft(data)

  # Calculate the frequencies corresponding to the FFT bins, in kHz.
  frequencies = np.fft.rfftfreq(len(time), d = 1E-9) * 1E-3

  return frequencies, transform

# ==============================================================================

# Correlation weight factors from a joint distribution of injection time and cyclotron frequency.
def weights(times, frequencies, heights):

  weights = np.zeros(len(frequencies))

  for i in range(len(frequencies)):
    if np.sum(heights[:, i]) > 0:
      weights[i] = np.average(
        np.cos(2 * np.pi * (frequencies[i] * 1E3) * (times * 1E-9)),
        weights = heights[:, i]
      )
    else:
      weights[i] = 1

  return weights
