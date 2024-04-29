DATASET=${1}
for BUNCH in {0..7}; do
  FILENAME="${gm2fr}/data/correlation/bunch/${DATASET}_Bunch${BUNCH}.root"
  FILELABEL=$(basename ${FILENAME} .root)
  if [ -f ${FILENAME} ]; then
    echo ${DATASET}/${BUNCH}
    python3 $gm2fr/scripts/smooth_joint.py --file bunch/${FILELABEL} --label hJointDistMinuitC009 --type x --time 1E-9 --scale 10 --output bunch/${FILELABEL}_noSmooth --factor 1
  fi
done
