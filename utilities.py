import numpy as np
import pandas as pd

# ==============================================================================

# Speed of light (m/s).
c = 299792458

# Anomalous magnetic moment.
a_mu = 11659208.9E-10

# Muon charge (C).
q_mu = 1.602176565E-19

# Mean muon lifetime at rest (ns).
# TODO: convert this to micronseconds. probably also requires making the simulation code all in terms of microseconds... should be done anyway.
lifetime = 2.1969811E3

# Mass conversion factor from GeV to kg.
GeV_to_kg = 1E9 * q_mu / c**2

# Muon mass (GeV).
m_mu_GeV = 0.1056583715

# Muon mass (kg).
m_mu_kg = m_mu_GeV * GeV_to_kg

# Nominal dipole magnetic field (T).
b = 1.4513

# Magic gamma factor.
magicGamma = np.sqrt(1 / a_mu + 1)

# Magic beta factor.
magicBeta = np.sqrt(1 - 1 / magicGamma**2)

# Magic momentum (GeV).
magicMomentum = magicGamma * m_mu_GeV * magicBeta

# Magic radius (mm). (Note: this formula only works at magic momentum!)
magicRadius = (magicMomentum * GeV_to_kg * c) / (q_mu * b) * 1E3

# ==============================================================================

# Cyclotron radius (mm) to cyclotron frequency (kHz).
def radiusToFrequency(radius):
  return (magicBeta * c / radius) / (2 * np.pi)
  
# Cyclotron frequency (kHz) to cyclotron radius (mm).
def frequencyToRadius(frequency):
  return magicBeta * c / (2 * np.pi * frequency)

# Magic cyclotron frequency (kHz).
magicFrequency = radiusToFrequency(magicRadius)

# Conversion from cyclotron frequency (kHz) to cyclotron period (ns).
def frequencyToPeriod(frequency):
  return 1 / (frequency * 1E3) * 1E9
  
# Magic cyclotron period (ns).
magicPeriod = frequencyToPeriod(magicFrequency)

# Momentum (GeV) to radius (mm).
def momentumToRadius(momentum, n = 0.108):
  return magicRadius * (1 + 1 / (1 - n) * (momentum - magicMomentum) / magicMomentum)

# Radius (mm) to momentum (GeV).
def radiusToMomentum(radius, n = 0.108):
  return magicMomentum * (1 + (1 - n) * (radius - magicRadius) / magicRadius)

# Momentum (GeV) to cyclotron frequency (kHz).
def momentumToFrequency(momentum, n = 0.108):
  return magicFrequency * (1 - 1 / (1 - n) * (momentum - magicMomentum) . magicMomentum)

# Cyclotron frequency (kHz) to momentum (GeV).
def frequencyToMomentum(frequency, n = 0.108):  
  return magicMomentum * (1 + (1 - n) * (1 - frequency / magicFrequency))

# Cyclotron frequency (kHz) to gamma.
def frequencyToGamma(frequency, n = 0.108):
  return frequencyToMomentum(frequency, n) / (m_mu_GeV * magicBeta)

# Conversion from fractional momentum offset to momentum (GeV).
def offsetToMomentum(offset):
  return (1 + offset) * magicMomentum

# Conversion from momentum (GeV) to fractional momentum offset.
def momentumToOffset(momentum):
  return (momentum - magicMomentum) / magicMomentum
  
# Conversion from fractional momentum offset to cyclotron frequency (kHz).
def offsetToFrequency(offset):
  return momentumToFrequency(offsetToMomentum(offset))

# Conversion from cyclotron frequency (kHz) to cyclotron radial offset (mm).
def frequencyToRadialOffset(frequency):
  return frequencyToRadius(frequency) - magicRadius

def correction(mean, width, n = 0.108):
  return 2 * n * (1 - n) * magicBeta**2 * (mean**2 + width**2) / magicRadius**2 * 1E9

# Conversion from cyclotron radial offsets (mm) to electric field correction (ppb).
def radialOffsetToCorrection(radii, heights, n = 0.108):
  mean = np.average(radii, weights = heights)
  std = np.sqrt(np.average((radii - mean)**2, weights = heights))
  return correction(mean, std)
  
# Conversion from cyclotron frequencies (kHz) to electric field correction (ppb).
def frequencyToCorrection(frequencies, heights, n = 0.108):
  return radialOffsetToCorrection(frequencyToRadialOffset(frequencies), heights, n)

# ==============================================================================

minimum = {
  "radius": -45
}

maximum = {
  "radius": +45
}

minimum["frequency"] = radiusToFrequency(maximum["radius"] + magicRadius)
maximum["frequency"] = radiusToFrequency(minimum["radius"] + magicRadius)

minimum["period"] = frequencyToPeriod(maximum["frequency"])
maximum["period"] = frequencyToPeriod(minimum["frequency"])

minimum["gamma"] = frequencyToGamma(maximum["frequency"])
maximum["gamma"] = frequencyToGamma(minimum["frequency"])

minimum["momentum"] = radiusToMomentum(minimum["radius"] + magicRadius)
maximum["momentum"] = radiusToMomentum(maximum["radius"] + magicRadius)

minimum["offset"] = momentumToOffset(minimum["momentum"]) * 100
maximum["offset"] = momentumToOffset(maximum["momentum"]) * 100

minimum["lifetime"] = minimum["gamma"] * lifetime * 1E-3
maximum["lifetime"] = maximum["gamma"] * lifetime * 1E-3

minRadialOffset = -45
maxRadialOffset = +45

minRadius = magicRadius + minRadialOffset
maxRadius = magicRadius + maxRadialOffset

minFrequency = radiusToFrequency(maxRadius)
maxFrequency = radiusToFrequency(minRadius)

# old names for backward compatibility
collimatorLow = minFrequency
collimatorHigh = maxFrequency

minPeriod = frequencyToPeriod(maxFrequency)
maxPeriod = frequencyToPeriod(minFrequency)

# assuming n = 0.108
minGamma = frequencyToGamma(maxFrequency)
maxGamma = frequencyToGamma(minFrequency)

# assuming n = 0.108
minMomentum = radiusToMomentum(minRadius)
maxMomentum = radiusToMomentum(maxRadius)

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

# ==================================================================================================

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

# Calculate the kth harmonic of the cosine transform, with symmetry time "t0".
def cosineTransform(time, signal, t0, full = False, k = 1):

  f = np.arange(6630.5, 6780, 1)
  result = np.zeros(len(f))
  
  if not full:
    mask = (time > t0)
    time = time[mask]
    signal = signal[mask]
  
  for i in range(len(f)):
    result[i] = np.sum(
      signal * np.cos(2 * np.pi * (k * f[i] * 1E3) * (time - t0) * 1E-9)
    )
    
  return f, result
  
# ==============================================================================

# Calculate the kth harmonic of the sine transform, with symmetry time "t0".
def sineTransform(time, signal, t0, full = False, k = 1):

  f = np.arange(6630.5, 6780, 1)
  result = np.zeros(len(f))
  
  if not full:
    mask = (time > t0)
    time = time[mask]
    signal = signal[mask]
  
  for i in range(len(f)):
    result[i] = np.sum(
      signal * np.sin(2 * np.pi * (k * f[i] * 1E3) * (time - t0) * 1E-9)
    )
    
  return f, result
  
# ==================================================================================================

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
  
# ==================================================================================================

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
    
