#! /usr/bin/bash

if [ -n "$WORLD_SIZE" ]; then
    echo "multi-nodes activates..."
    nproc_per_node=$CGPU_COUNT; master_addr=$MASTER_ADDR; node_rank=$RANK
    master_port=$MASTER_PORT; nnodes=$WORLD_SIZE
else
    nproc_per_node=4; master_addr="127.0.0.1"; node_rank=0
    master_port=1227; nnodes=1
fi

export PYTHONPATH="$(pwd -P):$PYTHONPATH"
export OMP_NUM_THREADS=8

conda info

fs_root=""

# DATA
video_root="${fs_root}/Datasets"
label_root="${fs_root}/data_list"

k400_video="${video_root}/k400"; k400_label="${label_root}/kinetics"
ucf_video="${video_root}/ucf101"; ucf_label="${label_root}/ucf101"
diving_video="${video_root}/diving48"; diving_label="${label_root}/diving"
ssthv2_video="${video_root}/ssv2"; ssthv2_label="${label_root}/something"

# Pretrain
task_root=""
pt_log="${fs_root}/logs/${task_root}"
pt_mdl_path="${fs_root}/data/${task_root}"

ep=800; res=224;
bs_per_gpu=8; ttl_bs=$(($bs_per_gpu * $nproc_per_node * $nnodes)); nclip=4

exp_name="PRP_bs${ttl_bs}_nclip${nclip}_${res}_U101_ep${ep}_1x8"
echo "EXP LOGPATH: ${pt_log}/${exp_name}"
torchrun --nproc_per_node=${nproc_per_node} --master_addr=${master_addr} --nnodes=${nnodes} \
    --master_port=${master_port} --node_rank=${node_rank} tools/run.py \
    --cfg "configs/pretrain/SpeedPro_R50.yaml" --output "${pt_log}/${exp_name}" \
    --opts MODELDATA "${pt_mdl_path}/${exp_name}" \
    SOLVER.MAX_EPOCH ${ep} \
    NUM_GPUS ${nproc_per_node} BN.NUM_SYNC_DEVICES ${nproc_per_node} DATA.BATCHSIZE_PER_GPU ${bs_per_gpu} DATA.NUM_CLIP ${nclip} \
    DATA.DATASET "kinetics" DATA.DATADIR "${ucf_video}" DATA.LABELDIR "${ucf_label}" \

