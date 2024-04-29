DATASET=${1}
FILENAME="${gm2fr}/data/correlation/gm2ringsim_${DATASET}.root"
echo ${FILENAME}
if [ -f ${FILENAME} ]; then
  echo ${DATASET}
  python3 $gm2fr/scripts/simulate_joint.py --file $(basename ${FILENAME} .root) --label FastRotationAnalyzer/hPlane0DpAveThr0_FR --type dp_p0 --time 1E-9 --smooth
  #python3 $gm2fr/scripts/simulate_joint.py --file $(basename ${FILENAME} .root) --label FastRotationAnalyzer/hPlane0DpAve --type dp_p0 --time 1E-9 --smooth
fi
