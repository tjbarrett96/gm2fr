import gm2fr.src.io as io
import gm2fr.src.constants as const
from gm2fr.src.Histogram1D import Histogram1D
from gm2fr.src.Analyzer import Analyzer

import numpy as np
import matplotlib.pyplot as plt
import gm2fr.src.style as style
style.set_style()

# ==================================================================================================

# Function that takes NumPy arrays of FR signal times and heights, and runs the FR analysis.
def run_fr_analysis(
  time, # NumPy array of signal times
  signal, # NumPy array of signal heights
  errors = None, # NumPy array of signal error bars
  output_label = "fr_analysis", # label for output folder containing results
  time_units = 1E-6 # units for signal times relative to seconds
):

  # Construct a Histogram1D object from the signal times and heights.
  dt = time[1] - time[0]
  time_edges = np.append(time - dt/2, time[-1] + dt/2)
  fr_signal = Histogram1D(time_edges, heights = signal, cov = errors**2 if errors is not None else (0.001 * np.abs(signal))**2)
  # non-zero uncertainties are required, so assign small 'cov' (covariance diagonal sqrts) as toy placeholder if no errors provided

  # convert time units to microseconds
  fr_signal.map(lambda t: t * time_units / 1E-6)

  # Construct an Analyzer object and assign the fast rotation signal we created above.
  # Usually Analyzer's constructor takes filenames to load from disk, so this is a workaround.
  analyzer = Analyzer(output_label = output_label)
  analyzer.fr_signal = fr_signal

  # Run the analysis, supplying optional arguments for various parameter choices.
  analyzer.analyze(
    start = 4, # in microseconds
    end = 250 # in microseconds
    # see gm2fr.src.Analyzer for more fine-tuning options
  )

  return analyzer

# ==================================================================================================

# When executing this file from the terminal.
if __name__ == "__main__":

  # Running a test.
  fr_signal = Histogram1D.load(f"{io.sim_path}/sample/simulation.root", "signal")
  time, signal, errors = fr_signal.centers, fr_signal.heights, fr_signal.errors
  analyzer = run_fr_analysis(time, signal, errors)

  # Plot the optimized cosine transform.
  analyzer.converted_transforms["f"].normalize().plot()
  style.draw_horizontal(0)
  style.draw_vertical(const.info["f"].magic)
  style.xlabel("Frequency (kHz)")
  style.ylabel("Arbitrary Units")
  plt.show()
