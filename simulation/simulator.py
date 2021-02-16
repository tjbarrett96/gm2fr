import numpy as np
import os

import ROOT as root
import root_numpy as rnp

from gm2fr.simulation.mixture import GaussianMixture
from gm2fr.simulation.histogram import Histogram
import gm2fr.utilities as utilities

import matplotlib.pyplot as plt
import gm2fr.style as style

# ==================================================================================================

class Simulator:

  # ================================================================================================

  def __init__(self, name, overwrite = False):

    if not os.path.isdir(name):
      os.mkdir(name)
      print(f"Creating simulation directory '{name}'.")
    elif overwrite:
      print(f"Overwriting simulation directory '{name}'.")
    else:
      raise RuntimeError(f"Simulation directory '{name}' already exists.")

    if not os.path.isdir(f"{name}/numpy"):
      os.mkdir(f"{name}/numpy")

    self.directory = name

  # ================================================================================================

  # Specify Gaussian mixtures for the frequency and time distributions, with optional correlation.
  def useMixture(
    self,
    kinematicsDistribution, # GaussianMixture object for muon frequency/momentum/offset
    kinematicsVariable,     # one of "frequency", "momentum", or "offset"
    timeDistribution,       # GaussianMixture object for muon injection time
    timeUnits,              # one of "nanoseconds", "microseconds", or "seconds"
    correlation = [0]       # poly. coeffs. (decreasing) which shift mean frequency df(t)
  ):

    self.sourceMode = "Mixture"
    self.sourceKinematicsDistribution = kinematicsDistribution
    self.sourceKinematicsVariable = kinematicsVariable
    self.sourceTimeDistribution = timeDistribution
    self.sourceTimeUnits = timeUnits
    self.sourceCorrelation = correlation

  # ================================================================================================

  # Specify a TH1 for the "kinetic" and time distributions, with optional correlation.
  # The "kinetic" distribution can be cyclotron frequency, momentum, etc. (see below).
  def useHistogram1D(
    self,
    kinematicsHistogram, # root.TH1 object for muon frequency/momentum/offset histogram
    kinematicsVariable,  # one of "frequency", "momentum", or "offset"
    timeHistogram,       # root.TH1 object for muon injection time histogram (ROOT TH1)
    timeUnits,           # one of "nanoseconds", "microseconds", or "seconds"
    correlation = [0]    # poly. coeffs. (decreasing) which shift mean frequency, i.e. df(t)
  ):

    self.sourceMode = "Histogram1D"
    self.sourceKinematicsHistogram = kinematicsHistogram
    self.sourceKinematicsHistogram.SetName("sourceKinematicsHistogram")
    self.sourceKinematicsVariable = kinematicsVariable
    self.sourceTimeHistogram = timeHistogram
    self.sourceTimeHistogram.SetName("sourceTimeHistogram")
    self.sourceTimeUnits = timeUnits
    self.sourceCorrelation = correlation

  # ================================================================================================

  # Specify a TH2 for the joint (i.e. correlated) "kinetic" and time distribution.
  # The "kinetic" variable can be cyclotron frequency, momentum, etc. (see below).
  def useHistogram2D(
    self,
    jointHistogram,     # root.TH2 object for joint kinematics and injection time histogram
    kinematicsVariable, # one of "frequency", "momentum", or "offset"
    timeUnits           # one of "nanoseconds", "microseconds", or "seconds"
  ):

    self.sourceMode = "Histogram2D"
    self.sourceJointHistogram = jointHistogram
    self.sourceJointHistogram.SetName("sourceJointHistogram")
    self.sourceKinematicsVariable = kinematicsVariable
    self.sourceTimeUnits = timeUnits

  # ================================================================================================

  # (For internal use.) Draw injection times and cyclotron frequencies for each muon.
  def __populate(self, muons):

    # For each muon in the batch, draw injection times and kinematics variables.
    if self.sourceMode == "Mixture":
      offsets = self.sourceTimeDistribution.draw(choices = muons)
      kinematics = self.sourceKinematicsDistribution.draw(choices = muons)
    elif self.sourceMode == "Histogram1D":
      offsets = rnp.random_sample(self.sourceTimeHistogram, muons)
      kinematics = rnp.random_sample(self.sourceKinematicsHistogram, muons)
    elif self.sourceMode == "Histogram2D":
      samples = rnp.random_sample(self.sourceJointHistogram, muons)
      offsets = samples[:, 0]
      kinematics = samples[:, 1]
    else:
      raise ValueError(f"Input mode '{self.sourceMode}' not recognized.")

    # Apply the correlation polynomial to the kinematics variables.
    if self.sourceMode in ["Mixture", "Histogram1D"]:
      kinematics += np.polyval(self.sourceCorrelation, offsets)

    # Convert injection times to nanoseconds.
    if self.sourceTimeUnits == "nanoseconds":
      pass
    elif self.sourceTimeUnits == "microseconds":
      offsets *= 1E3
    elif self.sourceTimeUnits == "seconds":
      offsets *= 1E9
    else:
      raise ValueError(f"Time unit '{self.sourceTimeUnits}' not recognized.")

    # Center the mean injection time to zero (as a definition).
    offsets -= np.average(offsets)

    # Convert kinematics parameters to cyclotron frequencies.
    if self.sourceKinematicsVariable == "frequency":
      frequencies = kinematics
    elif self.sourceKinematicsVariable == "momentum":
      frequencies = utilities.momentumToFrequency(kinematics)
    elif self.sourceKinematicsVariable == "offset":
      frequencies = utilities.offsetToFrequency(kinematics)
    else:
      raise ValueError(f"Kinematics variable '{self.sourceKinematicsVariable}' not recognized.")

    return offsets, frequencies

  # ================================================================================================

  def __normalize(self):

    if self.decay == "exponential":

      # Divide out the exponential decay using the ensemble-averaged lifetime.
      scale = self.muons if not self.backward else self.muons / 2
      self.signal *= 1 / (
        (scale / self.meanLifetime) * np.exp(-np.abs(self.signal.xCenters) / self.meanLifetime)
      )

    elif self.decay == "uniform":

      # Normalize the debunched current (muons per one nanosecond) to 1.
      totalTurns = self.maxTurns - self.minTurns + 1
      self.signal *= 1 / (self.muons / totalTurns * (self.frequencies.mean() * 1E3) * 1E-9)

    else:

      # Normalize the debunched current (muons per one nanosecond) to 1.
      self.signal *= 1 / (self.muons * (self.frequencies.mean() * 1E3) * 1E-9)

  # ================================================================================================

#  def __transform(self):
#
#    periods = utilities.frequencyToPeriod(self.joint.yCenters)
#    offsets = self.joint.xCenters

#    heights_T = np.sum(self.joint.heights, axis = 0)
#    heights_tau = np.sum(self.joint.heights, axis = 1)

#    mean_T = np.average(periods, weights = heights_T)
#    mean_tau = np.average(offsets, weights = heights_tau)

#    sigma_T = np.sqrt(np.average((periods - mean_T)**2, weights = heights_T))
#    sigma_tau = np.sqrt(np.average((offsets - mean_tau)**2, weights = heights_tau))

#    rho = np.average(np.outer(offsets - mean_tau, periods - mean_T), weights = self.joint.heights)
#    rho /= sigma_T * sigma_tau

#    n0 = - sigma_tau / sigma_T * rho
#    t0 = round(n0 + self.detector) * mean_T + mean_tau

#    _, transform = utilities.cosineTransform(self.signal.xCenters, self.signal.heights, t0)
#    weights = utilities.weights(self.joint.xCenters, self.joint.yCenters, self.joint.heights)

  # ================================================================================================

  def simulate(
    self,
    muons,                  # number of muons
    end = 150_000,          # end time (nanoseconds)
    detector = 0.74,        # position of reference detector, as a fraction of turns
    decay = "none",         # muon decay option: "exponential", "uniform", or "none"
    backward = False,       # turn on/off backward signal
    normalize = True,       # turn on/off signal normalization
    batchSize = 10_000_000  # split the total number of muons into batches of this size
  ):

    # Store simulation parameters.
    self.muons = int(muons)
    self.end = int(end)
    self.start = 0 if not backward else -self.end
    self.detector = detector
    self.decay = decay
    self.backward = backward
    self.normalize = normalize

    # Prepare the fast rotation histogram.
    self.signal = Histogram((self.start, self.end, 1))

    # Prepare the cyclotron frequency and injection time histograms.
    self.frequencies = Histogram((6630, 6780, 1))
    self.profile = Histogram((-80, 80, 1))

    # Prepare the joint histogram of cyclotron frequency vs. injection time.
    self.joint = Histogram(
      (self.profile.xEdges[0], self.profile.xEdges[-1], self.profile.xWidth),
      (self.frequencies.xEdges[0], self.frequencies.xEdges[-1], self.frequencies.xWidth)
    )

    # Maximum number of turns up to the chosen end time.
    self.maxTurns = self.end // (1 / utilities.collimatorHigh * 1E6) + 1
    self.minTurns = -self.maxTurns if self.backward else 0

    # Break up the total number of muons into batches, for memory efficiency.
    numberOfBatches = self.muons // batchSize
    remainder = self.muons % batchSize
    batches = [batchSize] * numberOfBatches
    if remainder > 0:
      numberOfBatches += 1
      batches += [remainder]

    # Initialize the ensemble-averaged lifetime, for the exponential decay removal later.
    if self.decay == "exponential":
      self.meanLifetime = 0

    # Process each batch of muons.
    for i in range(len(batches)):

      print(f"Working on batch {i + 1} of {numberOfBatches}...")

      # Draw injection times and cyclotron frequencies for each muon in the batch.
      offsets, frequencies = self.__populate(batches[i])

      # Update the truth-level histograms.
      self.profile.fill(offsets)
      self.frequencies.fill(frequencies)
      self.joint.fill(offsets, frequencies)

      # Calculate the cyclotron periods, in nanoseconds.
      periods = utilities.frequencyToPeriod(frequencies)

      if decay == "exponential":

        # Draw a random decay time for each muon.
        lifetimes = utilities.frequencyToGamma(frequencies) * utilities.lifetime
        decays = np.random.exponential(scale = lifetimes)

        # Update the mean lifetime with this batch of muons.
        self.meanLifetime += np.sum(lifetimes)

        # Make half of the random decay times negative.
        if backward:
          decays[int(len(decays) / 2):] *= -1

        # Calculate the turn number of the last detector crossing for each decay time.
        turns = (decays - (detector * periods + offsets)) // periods

        # Calculate the time of detection for each muon, using its decay turn number.
        detections = (detector + turns) * periods + offsets
        self.signal.fill(detections)

      elif decay == "uniform":

        # Select a random turn number for each muon.
        turns = np.random.randint(self.minTurns, self.maxTurns + 1, batches[i])

        # Calculate the time of detection for each muon, using its decay turn number.
        detections = (detector + turns) * periods + offsets
        self.signal.fill(detections)

      elif decay == "none":

        progress = -1
        for turn in np.arange(self.minTurns, self.maxTurns + 1):

          # Calculate the time of detection for each muon, on the current turn number.
          detections = (detector + turn) * periods + offsets
          self.signal.fill(detections)

          # Print a progress update.
          fraction = int((turn - self.minTurns) / (self.maxTurns - self.minTurns) * 100)
          if fraction > progress:
            progress = fraction
            print(f"{progress}% complete.", end = "\r")
        print()

      else:

        raise ValueError(f"Decay mode '{decay}' not recognized.")

    # Finish calculating the mean lifetime, now that the batches are done.
    if self.decay == "exponential":
      self.meanLifetime /= self.muons

    # Normalize the signal.
    if self.normalize:
      self.__normalize()

#    self.__transform()

    print("Finished!")

  # ================================================================================================

  def save(self):

    # Save the toy-model source distributions in custom NumPy-oriented format.
    if self.sourceMode == "Mixture":

      self.sourceKinematicsDistribution.save(f"{self.directory}/numpy/sourceKinematicsDistribution")
      self.sourceTimeDistribution.save(f"{self.directory}/numpy/sourceTimeDistribution")
      np.savez(f"{self.directory}/numpy/sourceCorrelation", sourceCorrelation = self.sourceCorrelation)

    # Save the source distributions in ROOT format.
    else:

      # Open a ROOT file to contain the source information.
      sourceFile = root.TFile(f"{self.directory}/source.root", "RECREATE")

      if self.sourceMode == "Histogram1D":

        # Write the kinematics and injection time histograms.
        self.sourceKinematicsHistogram.Write()
        self.sourceTimeHistogram.Write()

        # Write the correlation polynomial coefficients as entries in a single-branched TTree.
        rnp.array2tree(
          np.array(self.sourceCorrelation, dtype = [("sourceCorrelation", np.float32)]),
          name = "sourceCorrelation"
        ).Write()

      else:

        # Write the joint kinematics and injection time histogram.
        self.sourceJointHistogram.Write()

      sourceFile.Close()

    # Save simulation histograms in NumPy formats.
    self.frequencies.save(f"{self.directory}/numpy/frequencies")
    self.profile.save(f"{self.directory}/numpy/profile")
    self.joint.save(f"{self.directory}/numpy/joint")
    self.signal.save(f"{self.directory}/numpy/signal")

    # Prepare the truth-level radial TGraph.
    radial = root.TGraph()
    radial.SetName("radial")
    rnp.fill_graph(
      radial,
      np.stack(
        (utilities.frequencyToRadius(self.frequencies.xCenters), self.frequencies.heights),
        axis = -1
      )
    )

    # Save simulation histograms in ROOT format.
    rootFile = root.TFile(f"{self.directory}/simulation.root", "RECREATE")
    self.frequencies.toRoot("frequencies", ";Cyclotron Frequency (kHz);Entries").Write()
    self.profile.toRoot("profile", ";Injection Time (ns);Entries").Write()
    self.joint.toRoot("joint", ";Injection Time (ns);Cyclotron Frequency (kHz)").Write()
    self.signal.toRoot("signal", ";Time (us);Arbitrary Units", xRescale = 1E-3).Write()
    radial.Write()
    rootFile.Close()

  # ================================================================================================

  def plot(self, times = []):

    style.setStyle()

    self.frequencies.plot()
    style.xlabel("Cyclotron Frequency (kHz)")
    style.ylabel("Entries / 1 kHz")
    plt.savefig(f"{self.directory}/frequencies.pdf")
    plt.close()

    self.profile.plot()
    style.xlabel("Injection Time (ns)")
    style.ylabel("Entries / 1 ns")
    plt.savefig(f"{self.directory}/profile.pdf")
    plt.close()

    self.joint.plot()
    style.xlabel("Injection Time (ns)")
    style.ylabel("Cyclotron Frequency (kHz)")
    plt.savefig(f"{self.directory}/joint.pdf")
    plt.close()

    self.signal.plot()
    style.xlabel(r"Time (ns)")
    style.ylabel("Intensity / 1 ns")
    plt.savefig(f"{self.directory}/signal.pdf")
    for i in range(len(times)):
      if self.backward:
        plt.xlim(-times[i], times[i])
      else:
        plt.xlim(0, times[i])
      plt.savefig(f"{self.directory}/signal_{i}.pdf")
    plt.close()
