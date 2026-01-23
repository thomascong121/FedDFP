#!/bin/bash
#PBS -P iq24
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -q gpuvolta
#PBS -l mem=64GB
#PBS -l storage=gdata/iq24+scratch/iq24
#PBS -l walltime=12:00:00
#PBS -l jobfs=10GB

PROJECT_ID=iq24
MODEL_NAME=ResNet50
FEATURE_TYPE=R50_features
MODEL_NAME_MIL=CLAM_SB
MODEL_NAME_FED=fed_prompt
INITIALIZATION=random
EXP_CODE=1_prompt_100local
DP_NOISE=1.0
OPTIMIZER=adamw
OPTIMIZER_IMAGE=sgd
G_EPOCHS=100
N_PROMPTS=1
LOCAL_EPOCHS=50
REPEAT=1
LR=0.001
DATA_NAME=CAMELYON17
FT_ROOT=/g/data/$PROJECT_ID/CAMELYON17_patches/centers
CODE_ROOT=/scratch/iq24/cc0395/FedDFP

cd $CODE_ROOT
source /g/data/$PROJECT_ID/mmcv_env/bin/activate
echo "Current Working Directory: $(pwd)"

# --heter_model \
python3 main.py \
--feature_type $FEATURE_TYPE \
--ft_model $MODEL_NAME \
--mil_method $MODEL_NAME_MIL \
--fed_method $MODEL_NAME_FED \
--opt $OPTIMIZER \
--contrast_mu 2 \
--repeat $REPEAT \
--n_classes 4 \
--drop_out \
--lr $LR \
--B 8 \
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
# --dp_average \
# --dp_noise $DP_NOISE \
# --debug
