ulimit -n 50000

export TF_CPP_MIN_LOG_LEVEL=2
export META_DATASET_ROOT=#YOUR_PATH
export RECORDS=#YOUR_PATH

GPU_ID=$1
echo "Running IFR with seed $SEED on GPU $GPU_ID"

MODEL_DIR=#YOUR_PATH
model_name=url
CUDA_VISIBLE_DEVICES=$GPU_ID python run_ifr.py --model.name=$model_name --model.dir $MODEL_DIR \
                 --weight_decay=0.1 \
                 --setting_name=train_on_all_datasets \
                 --scale_reconst=10.0 \
                 --inner_iter=40 \
                 --experiment_name=ifr