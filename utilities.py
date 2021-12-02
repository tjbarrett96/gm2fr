import numpy as np
import scipy.linalg as linalg
import pandas as pd
import scipy.stats

import os
import re
import gm2fr
import matplotlib.pyplot as plt
import gm2fr.style as style
# import gm2fr.Histogram1D

# ==============================================================================

path = os.path.dirname(gm2fr.__file__)

def findIndex(string):
  match = re.search(r"\w+(\d+)$", string)
  return match.group(1) if match else None

def findIndices(sequence):
  return [index for index in (findIndex(item) for item in sequence) if index is not None]

# ==============================================================================

# Speed of light (m/s).
c = 299792458

# Anomalous magnetic moment.
a_mu = 11659208.9E-10

# Muon charge (C).
q_mu = 1.602176565E-19

# Mean muon lifetime at rest (us).
lifetime = 2.1969811

# Mass conversion factor from GeV to kg.
GeV_to_kg = 1E9 * q_mu / c**2

# Muon mass (GeV).
m_mu_GeV = 0.1056583715

# Muon mass (kg).
m_mu_kg = m_mu_GeV * GeV_to_kg

# Nominal dipole magnetic field (T).
b = 1.4513

# Conversion from (kHz * us) to standard units (Hz * s = 1).
kHz_us = 1E-3

# ==============================================================================

magic = {
  "x": 0,
  "dp_p0": 0,
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

# Conversion from momentum (GeV) to fractional momentum offset (%).
def momentumToOffset(p):
  return (p - magic["p"]) / magic["p"] * 100

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

frequencyTo = {
  "f": lambda f: f,
  "r": frequencyToRadius,
  "x": frequencyToRadialOffset,
  "T": frequencyToPeriod,
  "gamma": frequencyToGamma,
  "p": frequencyToMomentum,
  "dp_p0": lambda f, n = 0.108: momentumToOffset(frequencyToMomentum(f, n)),
  "tau": lambda f, n = 0.108: frequencyToGamma(f, n) * lifetime * 1E-3,
  "beta": lambda f, n = 0.108: np.sqrt(1 - 1/frequencyToGamma(f, n)**2),
  "c_e": lambda f, n = 0.108: 1E9 * 2 * n * (1 - n) * (magic["beta"] / magic["r"] * frequencyToRadialOffset(f))**2
}

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

min["dp_p0"] = momentumToOffset(min["p"]) * 100
max["dp_p0"] = momentumToOffset(max["p"]) * 100

min["tau"] = min["gamma"] * lifetime * 1E-3
max["tau"] = max["gamma"] * lifetime * 1E-3

min["beta"] = np.sqrt(1 - 1/min["gamma"]**2)
max["beta"] = np.sqrt(1 - 1/max["gamma"]**2)

min["c_e"] = 0
max["c_e"] = None

# ==============================================================================

# String labels for each variable: math mode, units, plot axes, and filename.
labels = {

  "beta": {
    "math": r"\beta",
    "units": "c",
    "plot": "Speed",
    "file": "beta"
  },

  "r": {
    "math": "r",
    "units": "mm",
    "plot": "Equilibrium Radius",
    "file": "fullRadius"
  },

  "f": {
    "math": "f",
    "units": "kHz",
    "plot": "Frequency",
    "file": "frequency"
  },

  "T": {
    "math": "T",
    "units": "ns",
    "plot": "Period",
    "file": "period"
  },

  "x": {
    "math": "x_e",
    "units": "mm",
    "plot": "Equilibrium Radial Offset",
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

  "dp_p0": {
    "math": r"\delta p/p_0",
    "units": "%",
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

# Plot an FFT of the raw positron signal.
def plotFFT(t, y, output):

  # Calculate the FFT magnitude.
  f, transform = fft(t, y)
  mag = np.abs(transform)

  # Plot the FFT magnitude.
  plt.plot(f, mag)

  # Axis labels.
  style.xlabel("Frequency (kHz)")
  style.ylabel("Arbitrary Units")

  # Use a logarithmic vertical scale, and set the limits appropriately.
  plt.yscale("log")
  plt.xlim(0, 8000)
  plt.ylim(np.min(mag[(f < 8000)]), None)

  # Save and clear.
  plt.savefig(output)
  plt.clf()

# ==============================================================================

# Helper function that checks if an object matches the form of a 2-tuple (x, y).
def isPair(obj):
  return isinstance(obj, tuple) and len(obj) == 2

# Helper function that checks if an object is an integer or float.
def isNumber(obj):
  return isinstance(obj, (int, float)) or isinstance(obj, np.number)

# Helper function that checks if an object is an integer.
def isInteger(obj):
  return isinstance(obj, int) or isinstance(obj, np.integer)

# Helper function that checks a boolean condition for each item in an iterable.
def checkAll(obj, function):
  return all(function(x) for x in obj)

def isNumericPair(obj):
  return isPair(obj) and checkAll(obj, isNumber)

# Helper function that checks if an object is a d-dimensional NumPy array.
def isArray(obj, d = None):
  return isinstance(obj, np.ndarray) and (obj.ndim == d if d is not None else True)

# Helper function that returns a mask where the given array is inside the given range.
# def isInside(arr, range):
#   if isNumericPair(range):
#     if isArray(arr):
#       return (arr >= range[0]) & (arr <= range[1])
#     else:
#       raise ValueError(f"Array data '{arr}' not understood.")
#   else:
#     raise ValueError(f"Range '{range}' not understood.")

# ==============================================================================

# Un-normalized sinc function, with optional scale in numerator: sin(ax)/x.
def sinc(x, a = 1):
  return a * np.sinc(a * x / np.pi)

# s(omega)
def sine(f, ts, tm, t0):
  return sinc(2*np.pi*f, (tm - t0)*kHz_us) - sinc(2*np.pi*f, (ts - t0)*kHz_us)

# c(omega)
def cosine(f, ts, tm, t0):
  # use trig identity for cos(a) - cos(b) to avoid indeterminate form
  return -np.sin(np.pi*f*(tm + ts - 2*t0)*kHz_us) * sinc(np.pi*f, (tm-ts)*kHz_us)

def physical(array, unit = "f"):
  return (array >= min[unit]) & (array <= max[unit])

def unphysical(array, unit = "f"):
  return ~physical(array, unit)

# # A(omega)
# # TODO: add errorbars
# def coefficient(rho, t0, function):
#
#   f, tau, heights, errors = rho.yCenters, rho.xCenters * 1E-3, rho.heights.T, rho.errors.T
#   result, error = np.zeros(len(f)), np.zeros(len(f))
#
#   for i in range(len(f)):
#
#     total = np.sum(heights[i])
#     if total == 0:
#       result[i], error[i] = np.nan, np.nan
#       continue
#
#     functionValues = function(2*np.pi*f[i]*(tau - t0)*kHz_us)
#     result[i] = np.average(functionValues, weights = heights[i])
#     error[i] = 1/total * np.sqrt(np.sum((functionValues - result[i])**2 * errors[i]**2))
#
#   return result, error

def A(t, t0, f = magic["f"]):
  return np.cos(2 * np.pi * f * (t - t0) * kHz_us)

def B(t, t0, f = magic["f"]):
  return np.sin(2 * np.pi * f * (t - t0) * kHz_us)

def cosineTransform(frequency, signal, time, t0, wiggle = True):
  result = np.einsum("i, ki -> k", signal, np.cos(2 * np.pi * np.outer(frequency, time - t0) * kHz_us))
  if wiggle:
    dt = time[1] - time[0]
    return result - 1 / (dt * kHz_us) * sine(frequency, time[0], time[-1], t0)
  else:
    return result

def sineTransform(frequency, signal, time, t0, wiggle = True):
  result = np.einsum("i, ki -> k", signal, np.sin(2 * np.pi * np.outer(frequency, time - t0) * kHz_us))
  if wiggle:
    dt = time[1] - time[0]
    return result + 1 / (dt * kHz_us) * cosine(frequency, time[0], time[-1], t0)
  else:
    return result

# def transform(signal, frequencies, t0, type = "cosine", errors = True, wiggle = True):
#
#   if isinstance(frequencies, gm2fr.Histogram1D.Histogram1D):
#     result = frequencies
#   else:
#     df = frequencies[1] - frequencies[0]
#     result = gm2fr.Histogram1D.Histogram1D(np.arange(frequencies[0] - df/2, frequencies[-1] + df, df))
#
#   differences = np.arange(result.length) * result.width
#   cov = None
#
#   # Note: to first order, cosine and sine transforms have the same covariance. (Not a typo.)
#   if type == "cosine":
#
#     heights = cosineTransform(result.centers, signal.heights, signal.centers, t0, wiggle)
#     if errors:
#       cov = 0.5 * linalg.toeplitz(cosineTransform(differences, signal.errors**2, signal.centers, t0, False))
#
#   elif type == "sine":
#
#     heights = sineTransform(result.centers, signal.heights, signal.centers, t0, wiggle)
#     if errors:
#       cov = 0.5 * linalg.toeplitz(cosineTransform(differences, signal.errors**2, signal.centers, t0, False))
#
#   elif type == "magnitude":
#
#     cosine = cosineTransform(result.centers, signal.heights, signal.centers, t0, wiggle)
#     sine = sineTransform(result.centers, signal.heights, signal.centers, t0, wiggle)
#
#     tempCos = linalg.toeplitz(cosineTransform(differences, signal.errors**2, signal.centers, t0, False))
#     tempSin = linalg.toeplitz(sineTransform(differences, signal.errors**2, signal.centers, t0, False))
#
#     heights = np.sqrt(cosine**2 + sine**2)
#     if errors:
#       cov = 0.5 / np.outer(heights, heights) * (
#         (np.outer(cosine, cosine) + np.outer(sine, sine)) * tempCos \
#         + (np.outer(sine, cosine) - np.outer(cosine, sine)) * tempSin
#       )
#
#   else:
#     raise ValueError(f"Frequency transform type '{type}' not recognized.")
#
#   result.setHeights(heights)
#   if errors:
#     result.setCov(cov)
#   return result
