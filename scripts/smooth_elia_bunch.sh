DATASET=${1}
RUNYEAR=${DATASET:0:1}
for BUNCH in {0..8}; do
  FILENAME="${gm2fr}/data/correlation/elia_bunch/radialDist_Run${RUNYEAR}_byBunch.root"
  FILELABEL=$(basename ${FILENAME} .root)
  HISTLABEL="RT_dist_bunch${BUNCH}_Run${DATASET}"
  if [[ -f ${FILENAME} ]]; then
    echo ${DATASET}/${BUNCH}
    python $gm2fr/scripts/smooth_joint.py --file elia_bunch/${FILELABEL}.root --label ${HISTLABEL} --type x --time 1E-9 --scale 10 --output elia_bunch/${DATASET}_Bunch${BUNCH}_smooth
  fi
done
