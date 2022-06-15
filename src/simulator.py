import numpy as np
import os

import ROOT as root
import root_numpy as rnp

from gm2fr.src.mixture import GaussianMixture
from gm2fr.src.Histogram1D import Histogram1D
from gm2fr.src.Histogram2D import Histogram2D
import gm2fr.src.io as io
import gm2fr.src.constants as const

import matplotlib.pyplot as plt
import gm2fr.src.style as style

import time

# ==============================================================================

class Simulator:

  # ============================================================================

  def __init__(
    self,
    name,
    overwrite = False,
    kinematics_dist = None,
    time_dist = None,
    joint_dist = None,
    correlation = [0],
    kinematics_type = "frequency",
    time_units = 1E-9 # nanoseconds by default
  ):

    self.kinematics_dist = kinematics_dist
    self.time_dist = time_dist
    self.joint_dist = joint_dist
    self.correlation = correlation
    self.kinematics_type = kinematics_type
    self.time_units = time_units

    if joint_dist is None and (kinematics_dist is None or time_dist is None):
      raise ValueError("Invalid input distributions.")

    path = f"{io.sim_path}/{name}"
    if not os.path.isdir(path):
      os.mkdir(path)
      print(f"Creating simulation directory '{name}'.")
    elif overwrite:
      print(f"Overwriting simulation directory '{name}'.")
    else:
      raise RuntimeError(f"Simulation directory '{name}' already exists.")

    self.directory = path

  # ============================================================================

  # Draw injection times and cyclotron frequencies for each muon.
  def draw_times_frequencies(self, muons):

    def draw(distribution, choices = 1):
      if isinstance(distribution, (GaussianMixture, Histogram1D)):
        return distribution.draw(choices)
      elif isinstance(distribution, (root.TH1, root.TH2)):
        return rnp.random_sample(distribution, choices)

    if self.joint_dist is not None:
      samples = draw(self.joint_dist, muons)
      offsets = samples[:, 0]
      kinematics = samples[:, 1]
    else:
      offsets = draw(self.time_dist, muons)
      kinematics = draw(self.kinematics_dist, muons)
      # Apply the correlation polynomial to the kinematics variables.
      kinematics += np.polyval(self.correlation, offsets)

    # Convert injection times to microseconds.
    offsets *= self.time_units / 1E-6

    # Center the mean injection time to zero (as a definition).
    offsets -= np.average(offsets)

    # Convert kinematics parameters to cyclotron frequencies.
    if self.kinematics_type in ("f", "p", "dp_p0", "dp_p0_%"):
      frequencies = const.info[self.kinematics_type].to_frequency(kinematics)
    else:
      raise ValueError(f"Kinematics variable '{self.kinematics_type}' not recognized.")

    # Mask unphysical frequencies.
    mask = const.physical(frequencies)
    return offsets[mask], frequencies[mask]

  # ============================================================================

  def normalize(self):

    if self.decay == "exponential":

      # Divide out the exponential decay using the ensemble-averaged lifetime.
      scale = self.muons if not self.backward else self.muons / 2
      self.signal = self.signal.divide(
        (scale / self.mean_lifetime) * np.exp(-np.abs(self.signal.centers) / self.mean_lifetime)
      )

    elif self.decay == "uniform":

      # Normalize the debunched current (muons per one nanosecond) to 1.
      total_turns = self.max_turns - self.min_turns + 1
      self.signal = self.signal.divide(self.muons / total_turns * (self.frequencies.mean() * 1E3) * 1E-9)

    else:

      # Normalize the debunched current (muons per one nanosecond) to 1.
      self.signal = self.signal.divide(self.muons * (self.frequencies.mean() * 1E3) * 1E-9)

  # ============================================================================

  def simulate(
    self,
    # number of muons
    muons,
    # end time (microseconds)
    end = 200,
    # position of reference detector, as a fraction of turns
    detector = 0,
    # muon decay option: "exponential", "uniform", or None
    decay = "uniform",
    # turn on/off backward signal
    backward = False,
    # turn on/off signal normalization
    normalize = True,
    # split the total number of muons into batches of this size
    batch_size = 10_000_000
  ):

    begin = time.time()

    # Store simulation parameters.
    self.muons = int(muons)
    self.end = end
    self.start = 0 if not backward else -self.end
    self.detector = detector
    self.decay = decay
    self.backward = backward

    # Prepare the fast rotation histogram.
    self.signal = Histogram1D(np.arange(self.start, self.end + 0.001, 0.001))

    # Prepare the cyclotron frequency and injection time histograms.
    self.frequencies = Histogram1D(np.arange(6630, 6781, 1))
    self.profile = Histogram1D(np.arange(-80, 81, 1))

    # Prepare the joint histogram of cyclotron frequency vs. injection time.
    self.joint = Histogram2D(self.profile.edges, self.frequencies.edges)

    # Maximum number of turns up to the chosen end time.
    self.max_turns = self.end // (1 / const.info["f"].max * 1E3) + 1
    self.min_turns = -self.max_turns if self.backward else 0

    # Break up the total number of muons into batches, for memory efficiency.
    number_of_batches = self.muons // batch_size
    remainder = self.muons % batch_size
    batches = [batch_size] * number_of_batches
    if remainder > 0:
      number_of_batches += 1
      batches.append(remainder)

    # Initialize the ensemble-averaged lifetime, for the exponential decay removal later.
    if self.decay == "exponential":
      self.mean_lifetime = 0

    # Reset the total number, to be updated below. (Some muons may be vetoed.)
    self.muons = 0

    # Process each batch of muons.
    for i in range(len(batches)):

      print(f"Working on batch {i + 1} of {number_of_batches}...")

      # Draw injection times and cyclotron frequencies for each muon in the batch.
      offsets, frequencies = self.draw_times_frequencies(batches[i])
      self.muons += len(frequencies)

      # Update the truth-level histograms (injection times in nanoseconds).
      self.profile.fill(offsets * 1E3)
      self.frequencies.fill(frequencies)
      self.joint.fill(offsets * 1E3, frequencies)

      # Calculate the cyclotron periods, in microseconds.
      periods = 1 / frequencies * 1E3

      if decay == "exponential":

        # Draw a random decay time for each muon.
        lifetimes = const.info["tau"].fromF(frequencies)
        decays = np.random.exponential(scale = lifetimes)

        # Update the mean lifetime with this batch of muons.
        self.mean_lifetime += np.sum(lifetimes)

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
        turns = np.random.randint(self.min_turns, self.max_turns + 1, len(frequencies))

        # Calculate the time of detection for each muon, using its decay turn number.
        detections = (detector + turns) * periods + offsets
        self.signal.fill(detections)

      elif decay is None:

        progress = -1
        for turn in np.arange(self.min_turns, self.max_turns + 1):

          # Calculate the time of detection for each muon, on the current turn number.
          detections = (detector + turn) * periods + offsets
          self.signal.fill(detections)

          # Print a progress update.
          fraction = int((turn - self.min_turns) / (self.max_turns - self.min_turns) * 100)
          if fraction > progress:
            progress = fraction
            print(f"{progress}% complete.", end = "\r")
        print()

      else:

        raise ValueError(f"Decay mode '{decay}' not recognized.")

    # Finish calculating the mean lifetime, now that the batches are done.
    if self.decay == "exponential":
      self.mean_lifetime /= self.muons

    # Normalize the signal.
    if normalize:
      self.normalize()

    # Move the injection times to the first detection.
    time_map = lambda tau: tau + 1E6 / self.frequencies.mean() * self.detector
    self.joint.map(x = time_map)
    self.profile.map(time_map)

    print(f"Finished in {(time.time() - begin):.4f} seconds.")

  # ============================================================================

  def save(self):

    # Save simulation histograms in NumPy format.
    np.savez(
      f"{self.directory}/data.npz",
      **self.frequencies.collect("frequencies"),
      **self.profile.collect("profile"),
      **self.joint.collect("joint"),
      **self.signal.collect("signal")
    )

    # Save simulation histograms in ROOT format.
    rootFile = root.TFile(f"{self.directory}/simulation.root", "RECREATE")
    self.frequencies.to_root("frequencies", ";Frequency (kHz);Entries").Write()
    self.profile.to_root("profile", ";Injection Time (ns);Entries").Write()
    self.joint.to_root("joint", ";Injection Time (ns);Frequency (kHz)").Write()
    self.signal.to_root("signal", ";Time (us);Arbitrary Units").Write()
    rootFile.Close()

  # ============================================================================

  def plot(self):

    style.set_style()

    self.frequencies.plot()
    style.label_and_save("Frequency (kHz)", "Entries / 1 kHz", f"{self.directory}/frequencies.pdf")

    self.profile.plot()
    style.label_and_save("Injection Time (ns)", "Entries / 1 ns", f"{self.directory}/profile.pdf")

    self.joint.transpose().plot()
    plt.xlim(const.info["f"].min, const.info["f"].max)
    style.label_and_save("Frequency (kHz)", "Injection Time (ns)", f"{self.directory}/joint.pdf")

    pdf = style.make_pdf(f"{self.directory}/signal.pdf")
    endTimes = [5, 100, 300]
    for endTime in endTimes:
      start_time = 0 if not self.backward else -endTime
      self.signal.plot(errors = False, start = start_time, end = endTime, skip = int(np.clip(endTime - start_time, 1, 10)))
      style.label_and_save(r"Time ($\mu$s)", "Intensity / 1 ns", pdf)
    pdf.close()
