DATASET=${1}
#for DATASET in ${RUN23}; do
  FILENAME="${gm2fr}/data/correlation/david_${DATASET}.root"
  if [ -f ${FILENAME} ]; then
    echo ${DATASET}
    python3 $gm2fr/scripts/simulate_joint.py --file $(basename ${FILENAME} .root) --label h2_dp_dt_optim --type dp_p0 --time 1E-9 --scale 0.01 --smooth #--flip
  fi
#done
