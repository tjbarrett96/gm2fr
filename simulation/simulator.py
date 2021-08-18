import numpy as np
import os

import ROOT as root
import root_numpy as rnp

from gm2fr.simulation.mixture import GaussianMixture
from gm2fr.simulation.histogram import Histogram
import gm2fr.utilities as utilities

import matplotlib.pyplot as plt
import gm2fr.style as style

# ==============================================================================

class Simulator:

  # ============================================================================

  def __init__(
    self,
    name,
    overwrite = False,
    kinematicsDistribution = None,
    timeDistribution = None,
    jointDistribution = None,
    correlation = [0],
    kinematicsUnits = "frequency",
    timeUnits = "nanoseconds"
  ):

    self.kinematicsDistribution = kinematicsDistribution
    self.timeDistribution = timeDistribution
    self.jointDistribution = jointDistribution
    self.correlation = correlation
    self.kinematicsUnits = kinematicsUnits
    self.timeUnits = timeUnits

    if jointDistribution is None and (kinematicsDistribution is None or timeDistribution is None):
      raise ValueError("Invalid input distributions.")

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

  # ============================================================================

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

  # ============================================================================

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

  # ============================================================================

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

  # ============================================================================

  # (For internal use.) Draw injection times and cyclotron frequencies for each muon.
  def __populate(self, muons):

    def draw(distribution, choices = 1):
      if isinstance(distribution, GaussianMixture):
        return distribution.draw(choices)
      elif isinstance(distribution, (root.TH1, root.TH2)):
        return rnp.random_sample(distribution, choices)

    if self.jointDistribution is not None:
      samples = draw(self.jointDistribution, muons)
      offsets = samples[:, 0]
      kinematics = samples[:, 1]
    else:
      offsets = draw(self.timeDistribution, muons)
      kinematics = draw(self.kinematicsDistribution, muons)
      # Apply the correlation polynomial to the kinematics variables.
      kinematics += np.polyval(self.correlation, offsets)

    # For each muon in the batch, draw injection times and kinematics variables.
    # if self.sourceMode == "Mixture":
    #   offsets = self.sourceTimeDistribution.draw(choices = muons)
    #   kinematics = self.sourceKinematicsDistribution.draw(choices = muons)
    # elif self.sourceMode == "Histogram1D":
    #   offsets = rnp.random_sample(self.sourceTimeHistogram, muons)
    #   kinematics = rnp.random_sample(self.sourceKinematicsHistogram, muons)
    # elif self.sourceMode == "Histogram2D":
    #   samples = rnp.random_sample(self.sourceJointHistogram, muons)
    #   offsets = samples[:, 0]
    #   kinematics = samples[:, 1]
    # else:
    #   raise ValueError(f"Input mode '{self.sourceMode}' not recognized.")

    # Apply the correlation polynomial to the kinematics variables.
    # if self.sourceMode in ["Mixture", "Histogram1D"]:
    #   kinematics += np.polyval(self.sourceCorrelation, offsets)

    # Convert injection times to microseconds.
    if self.timeUnits == "nanoseconds":
      offsets *= 1E-3
    elif self.timeUnits == "microseconds":
      pass
    elif self.timeUnits == "seconds":
      offsets *= 1E6
    else:
      raise ValueError(f"Time unit '{self.timeUnits}' not recognized.")

    # Center the mean injection time to zero (as a definition).
    offsets -= np.average(offsets)

    # Convert kinematics parameters to cyclotron frequencies.
    if self.kinematicsUnits == "frequency":
      frequencies = kinematics
    elif self.kinematicsUnits == "momentum":
      frequencies = utilities.momentumToFrequency(kinematics)
    elif self.kinematicsUnits == "offset":
      frequencies = utilities.offsetToFrequency(kinematics)
    else:
      raise ValueError(f"Kinematics variable '{self.kinematicsUnits}' not recognized.")

    # Mask unphysical frequencies.
    mask = (frequencies >= utilities.min["f"]) & (frequencies <= utilities.max["f"])

    return offsets[mask], frequencies[mask]

  # ============================================================================

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

  # ============================================================================

  def simulate(
    self,
    muons,                  # number of muons
    end = 200,          # end time (nanoseconds)
    detector = 0,        # position of reference detector, as a fraction of turns
    decay = "uniform",         # muon decay option: "exponential", "uniform", or "none"
    backward = False,       # turn on/off backward signal
    normalize = True,       # turn on/off signal normalization
    batchSize = 10_000_000  # split the total number of muons into batches of this size
  ):

    # Store simulation parameters.
    self.muons = int(muons)
    self.end = end
    self.start = 0 if not backward else -self.end
    self.detector = detector
    self.decay = decay
    self.backward = backward
    self.normalize = normalize

    # Prepare the fast rotation histogram.
    self.signal = Histogram((self.start, self.end, 0.001))

    # Prepare the cyclotron frequency and injection time histograms.
    self.frequencies = Histogram((6630, 6780, 2))
    self.profile = Histogram((-80, 80, 1))

    # Prepare the joint histogram of cyclotron frequency vs. injection time.
    self.joint = Histogram(
      (self.profile.xEdges[0], self.profile.xEdges[-1], self.profile.xWidth),
      (self.frequencies.xEdges[0], self.frequencies.xEdges[-1], self.frequencies.xWidth)
    )

    # Maximum number of turns up to the chosen end time.
    self.maxTurns = self.end // (1 / utilities.max["f"] * 1E3) + 1
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

    # Reset the total number, to be updated below. (Some muons may be vetoed.)
    self.muons = 0

    # Process each batch of muons.
    for i in range(len(batches)):

      print(f"Working on batch {i + 1} of {numberOfBatches}...")

      # Draw injection times and cyclotron frequencies for each muon in the batch.
      offsets, frequencies = self.__populate(batches[i])
      self.muons += len(frequencies)

      # Update the truth-level histograms (injection times in nanoseconds).
      self.profile.fill(offsets * 1E3)
      self.frequencies.fill(frequencies)
      self.joint.fill(offsets * 1E3, frequencies)

      # Calculate the cyclotron periods, in microseconds.
      periods = utilities.frequencyToPeriod(frequencies) * 1E-3

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
        turns = np.random.randint(self.minTurns, self.maxTurns + 1, len(frequencies))

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

    print("Finished!")

  # ============================================================================

  def save(self):

    # Save the toy-model source distributions in custom NumPy-oriented format.
    # if self.sourceMode == "Mixture":
    #
    #   self.sourceKinematicsDistribution.save(f"{self.directory}/numpy/sourceKinematicsDistribution")
    #   self.sourceTimeDistribution.save(f"{self.directory}/numpy/sourceTimeDistribution")
    #   np.savez(f"{self.directory}/numpy/sourceCorrelation", sourceCorrelation = self.sourceCorrelation)
    #
    # # Save the source distributions in ROOT format.
    # else:
    #
    #   # Open a ROOT file to contain the source information.
    #   sourceFile = root.TFile(f"{self.directory}/source.root", "RECREATE")
    #
    #   if self.sourceMode == "Histogram1D":
    #
    #     # Write the kinematics and injection time histograms.
    #     self.sourceKinematicsHistogram.Write()
    #     self.sourceTimeHistogram.Write()
    #
    #     # Write the correlation polynomial coefficients as entries in a single-branched TTree.
    #     rnp.array2tree(
    #       np.array(self.sourceCorrelation, dtype = [("sourceCorrelation", np.float32)]),
    #       name = "sourceCorrelation"
    #     ).Write()
    #
    #   else:
    #
    #     # Write the joint kinematics and injection time histogram.
    #     self.sourceJointHistogram.Write()
    #
    #   sourceFile.Close()

    # Save simulation histograms in NumPy format.
    np.savez(
      f"{self.directory}/data.npz",
      **self.frequencies.collect("frequencies"),
      **self.profile.collect("profile"),
      **self.joint.collect("joint"),
      **self.signal.collect("signal")
    )

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
    self.signal.toRoot("signal", ";Time (us);Arbitrary Units").Write()
    radial.Write()
    rootFile.Close()

  # ============================================================================

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
