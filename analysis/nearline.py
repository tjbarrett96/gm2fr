import gm2fr.Analyzer as Analyzer
import sys
import re

def run(filename, tag):
  analyzer = Analyzer(filename, signal = "hSignalRatio", tag = tag)
  analyzer.analyze(fit = None, end = 150)

if __name__ == "__main__":

  if len(sys.argv) != 2:
    print("Usage: python3 nearline.py [ROOT filename]")
    quit()

  filename = sys.argv[1]

  runNumbers = re.findall(r"run(\d{5})", filename)
  if len(runNumbers) == 1:
    tag = f"Run{runNumbers[0]}"
  else:
    tag = re.findall(r"(\w+).root", filename)[0]

  run(filename, tag)
