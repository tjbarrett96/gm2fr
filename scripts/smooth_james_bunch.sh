DATASET=${1}
for BUNCH in {0..7}; do
  FILENAME="${gm2fr}/data/correlation/james_bunch/TRDists.root"
  FILELABEL=$(basename ${FILENAME} .root)
  HISTLABEL="hTR_${DATASET}B${BUNCH}"
  if [ -f ${FILENAME} ]; then
    echo ${DATASET}/${BUNCH}
    python3 $gm2fr/scripts/smooth_joint.py --file james_bunch/${FILELABEL} --label ${HISTLABEL} --type x --time 1E-9 --scale 10 --output james_bunch/${DATASET}_Bunch${BUNCH}_smooth
  fi
done
