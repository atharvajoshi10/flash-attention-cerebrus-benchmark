#!/usr/bin/bash
#SBATCH --gres=cs:cerebras:1
#SBATCH --ntasks=7
#SBATCH --cpus-per-task=14
#SBATCH --account=cis240032p

newgrp cis240032p 

YOUR_DATA_DIR=/ocean/neocortex/cerebras/data/
YOUR_MODEL_ROOT_DIR=${PROJECT}/modelzoo/modelzoo
YOUR_ENTRY_SCRIPT_LOCATION=${YOUR_MODEL_ROOT_DIR}/transformers/pytorch/bert
BIND_LOCATIONS=/local1/cerebras/data,/local2/cerebras/data,/local3/cerebras/data,/local4/cerebras/data,${YOUR_DATA_DIR},${YOUR_MODEL_ROOT_DIR}
CEREBRAS_CONTAINER=/ocean/neocortex/cerebras/cbcore_latest.sif
cd ${YOUR_ENTRY_SCRIPT_LOCATION}
#PARAMS=/ocean/projects/cis240032p/shared/p4/modelzoo/modelzoo/transformers/pytorch/bert/configs/bert_base_msl128.yaml
PARAMS=configs/bert_base_msl128.yaml
srun --kill-on-bad-exit singularity exec --bind ${BIND_LOCATIONS} ${CEREBRAS_CONTAINER} python-pt run.py --mode train --model_dir training --params $PARAMS --cs_ip ${CS_IP_ADDR}
