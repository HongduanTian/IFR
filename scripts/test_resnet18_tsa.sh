#! /bin/bash
ulimit -n 50000
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/htian/anaconda3/envs/pytorch/lib
export META_DATASET_ROOT=./meta_dataset
export RECORDS=/data/cshdtian/meta_dataset
CUDA_VISIBLE_DEVICES=0 python test_extractor_tsa.py --model.name=url --model.dir ../../pretrained_models/URL_pretrained_models/url --test.tsa-ad-type residual --test.tsa-ad-form matrix --test.tsa-opt alpha+beta --test.tsa-init eye --test.mode mdl --seed=42