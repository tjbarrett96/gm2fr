from gm2fr.analysis.Analyzer import Analyzer

# Create the analyzer object.
# Specify the input fast rotation signal file, and desired output directory name.
# (Output will be placed inside gm2fr/analysis/results/{yourFolderName}).
analyzer = Analyzer(
  "../simulation/testing/numpy/signal.npz",
  "testing",
  units = "ns" # analyzer works in units of microseconds, but simulation was in nanoseconds -- I'll fix this soon
)

# Run the analysis.
analyzer.analyze(
  fit = None, # wiggle fit model: None / "two" / "five" / "nine"
  t0 = 0.110, # supply an initial t0 guess (in us), which will be optimized within +/- 15 ns (default from simulation is 0.74*T_magic ~ 110 ns)
  start = 4, # start time for cosine transform (us)
  end = 200, # end time for cosine transform (us)
  model = "sinc" # background fit model: "parabola" / "sinc" / "error"
)
