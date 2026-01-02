################### Test URL Model with PA ###################
ulimit -n 50000

export TF_CPP_MIN_LOG_LEVEL=2
export META_DATASET_ROOT=#YOUR_PATH
export RECORDS=#YOUR_PATH

GPU_ID=$1
echo "Running with seed $SEED on GPU $GPU_ID"

MODEL_DIR=#YOUR_PATH
model_name=url
CUDA_VISIBLE_DEVICES=$GPU_ID python test_origin_pa.py --model.name=$model_name --model.dir $MODEL_DIR \
                                         --setting_name=train_on_all_datasets \
                                         --experiment_name=pa_standard_${model_name}
