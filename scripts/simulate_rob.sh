DATASET=${1}
#for DATASET in ${RUN23}; do
  FILENAME="${DATA}/FastRotation/correlation/rob_${DATASET}_12p.root"
  if [ -f ${FILENAME} ]; then
    echo ${DATASET}
    python3 $gm2fr/scripts/simulate_joint.py --file $(basename ${FILENAME} .root) --label hJointDistMinuitC000 --type x --time 1E-9 --scale 10 --smooth #--flip
  fi
#done
