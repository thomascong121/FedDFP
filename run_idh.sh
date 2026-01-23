#!/bin/bash
#PBS -P iq24
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -q gpuvolta
#PBS -l mem=64GB
#PBS -l storage=gdata/iq24+scratch/iq24
#PBS -l walltime=18:00:00
#PBS -l jobfs=10GB

PROJECT_ID=iq24
MODEL_NAME=ResNet50
FEATURE_TYPE=R50_features
MODEL_NAME_MIL=CLAM_SB
MODEL_NAME_FED=fed_prfed_promptox
INITIALIZATION=random
EXP_CODE=1_prompt_100local
DP_NOISE=1.0
G_EPOCHS=100
N_PROMPTS=10
LOCAL_EPOCHS=50
REPEAT=5
OPTIMIZER=adam
OPTIMIZER_IMAGE=adam
DATA_NAME=IDH
FT_ROOT=/g/data/$PROJECT_ID/IDH
CODE_ROOT=/scratch/iq24/cc0395/FedDFP

cd $CODE_ROOT
source /g/data/$PROJECT_ID/mmcv_env/bin/activate
echo "Current Working Directory: $(pwd)"

#--heter_model \
python3 main.py \
--feature_type $FEATURE_TYPE \
--ft_model $MODEL_NAME \
--mil_method $MODEL_NAME_MIL \
--fed_method $MODEL_NAME_FED \
--opt $OPTIMIZER \
--repeat $REPEAT \
--n_classes 2 \
--drop_out \
--lr 2e-4 \
--B 1 \
--accumulate_grad_batches 1 \
--task $DATA_NAME \
--exp_code $EXP_CODE \
--global_epochs $G_EPOCHS \
--local_epochs $LOCAL_EPOCHS \
--bag_loss ce \
--inst_loss svm \
--results_dir $CODE_ROOT/exp \
--data_root_dir $FT_ROOT \
--prompt_lr 3e-4 \
--prompt_initialisation $INITIALIZATION \
--prompt_aggregation multiply \
--number_prompts $N_PROMPTS \
# --top_k 5000 \
# --debug
