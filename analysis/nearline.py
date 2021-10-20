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

  try:
    tag = re.findall("run(\d{5})", filename)[0]
  except:
    tag = filename.split(".")[0]

  run(filename, tag)
