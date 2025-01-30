DATASET=${1}
for BUNCH in {0..7}; do
  FILENAME="${gm2fr}/data/correlation/for_brynn/${DATASET}_bunch${BUNCH}.npz"
  FILELABEL=$(basename ${FILENAME} .npz)
  if [ -f ${FILENAME} ]; then
    echo ${DATASET}/${BUNCH}
    python3 $gm2fr/scripts/smooth_joint.py --file for_brynn/$(basename ${FILENAME}) --label joint --transpose --type f --time 1E-9 --output for_brynn/${FILELABEL}_smooth
  fi
done
