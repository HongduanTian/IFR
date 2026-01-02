from math import dist
import os
import datetime
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import sys
from socket import EAI_SOCKTYPE
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
import numpy as np
import pandas as pd

from tqdm import tqdm
from tabulate import tabulate
from utils import check_dir, device, set_determ, Recorder
from config import args

from models.cka import kernel_HSIC
from models.augmentation import DataAugmentation
from models.model_helpers import get_model
from models.losses import cross_entropy_loss
from models.model_utils import (CheckPointer)
from models.ifr import AttentionHead

from data.meta_dataset_reader import (MetaDatasetEpisodeReader, TRAIN_METADATASET_NAMES, ALL_METADATASET_NAMES)

tf.compat.v1.disable_eager_execution()


def get_backbone():
    # Load pretrained backbone
    backbone = get_model(None, args)
    backbone_checkpointer = CheckPointer(args, backbone, optimizer=None)
    backbone_checkpointer.restore_model(ckpt='best', strict=False)
    backbone.eval()
    return backbone


def get_optimizer(model, atten_lr, head_lr, weight_decay):

    return torch.optim.Adadelta(
        [{'params': model.query_head.parameters(), 'lr':atten_lr},
         {'params': model.key_head.parameters(), 'lr':atten_lr},
         {'params': model.value_head.parameters(), 'lr':atten_lr},
         {'params': model.transform_head.parameters()}], 
        lr=head_lr, weight_decay = weight_decay)



if __name__ == '__main__':
    SAVE_PATH = os.path.join("./tmp/ifr", args['setting_name'], args['experiment_name']) # dir for experiment results analysis

    # Prepare data
    if args['data.train_source'] == 'singlesource':
        trainsets = args['data.train']
    elif args['data.train_source'] == 'multisource':
        trainsets = TRAIN_METADATASET_NAMES
    else:
        raise ValueError("Unrecognized key word.")
    
    testsets = ALL_METADATASET_NAMES
    testdataloader = MetaDatasetEpisodeReader(mode='test', test_set=testsets,
                                              test_type=args['test.type'])
    
    # Initialize models & objects of classes
    backbone = get_backbone()
    attention_head = AttentionHead(args)
    data_aug_generator = DataAugmentation(args['num_aug'])

    accs_names = ['NCC']

    train_var_accs = dict()
    var_accs = dict()

    res_recorder = Recorder(saveroot=SAVE_PATH, datasets=ALL_METADATASET_NAMES, 
                            key_wd_list=['train_losses', 'train_accs', 'val_losses', 'val_accs'])

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth=False
    
    print(f"================= Evaluation on {args['experiment_name']} Starts! =================")
    
    with tf.compat.v1.Session(config=config) as session:
        for dataset in testsets:
            if dataset in ["traffic_sign", "mnist"]:
                atten_lr = 1.0
                head_lr = 1.0
            else:
                atten_lr = 0.1
                head_lr = 0.1

            weight_decay = args['weight_decay']
            max_inner_iter = args['inner_iter']

            print(dataset)
            train_var_accs[dataset] = {name:[] for name in accs_names}
            var_accs[dataset] = {name:[] for name in accs_names}

            for i in tqdm(range(600)):
                with torch.no_grad():
                    sample = testdataloader.get_test_task(session, dataset)
                    context_features = backbone.embed(sample['context_images'], is_pooling=False)
                    aug_context_features = backbone.embed(data_aug_generator.generate_augmentations(sample['context_images']), is_pooling=False)
                    target_features = backbone.embed(sample['target_images'], is_pooling=False)
                    context_labels = sample['context_labels']
                    target_labels = sample['target_labels']

                # reset the parameters and send them to cuda
                attention_head.reset_params()
                attention_head.to(device)

                # renew optimizer
                optimizer = get_optimizer(model=attention_head, 
                                          atten_lr=atten_lr, 
                                          head_lr=head_lr, 
                                          weight_decay=weight_decay)
                
                tmp_recorder = {
                    'train_losses': [],
                    'train_accs': [],
                    'val_losses': [],
                    'val_accs': []
                }

                for j in range(max_inner_iter):

                    # ----------- validation res record --------------------
                    attention_head.eval()
                    with torch.no_grad():
                        val_res = attention_head.pred(target_x=target_features,
                                                      aug_context=aug_context_features,
                                                      context_x=context_features,
                                                      context_y=context_labels)
                        _, tmp_val_stats, _ = cross_entropy_loss(logits=val_res, targets=target_labels)
                        
                        tmp_recorder['val_losses'].append(tmp_val_stats['loss'])
                        tmp_recorder['val_accs'].append(tmp_val_stats['acc'])
                    # ------------------------------------------------------
                    attention_head.train()
                    optimizer.zero_grad()

                    logits = attention_head.forward_pass(context_x=context_features,
                                                         context_y=context_labels,
                                                         aug_context=aug_context_features)
                    loss, train_stats, _ = cross_entropy_loss(logits=logits, targets=context_labels)

                    tmp_recorder['train_losses'].append(train_stats['loss'])
                    tmp_recorder['train_accs'].append(train_stats['acc'])

                    total_losses = loss
                    total_losses.backward()
                    optimizer.step()

                    if j == max_inner_iter - 1:
                        attention_head.eval()
                        with torch.no_grad():
                            val_res = attention_head.pred(target_x=target_features,
                                                          aug_context=aug_context_features,
                                                          context_x=context_features,
                                                          context_y=context_labels)
                            _, tmp_val_stats, _ = cross_entropy_loss(logits=val_res, targets=target_labels)

                            tmp_recorder['val_losses'].append(tmp_val_stats['loss'])
                            tmp_recorder['val_accs'].append(tmp_val_stats['acc'])

                # eval query data
                train_var_accs[dataset]['NCC'].append(train_stats['acc'])
                var_accs[dataset]['NCC'].append(tmp_val_stats['acc'])

                res_recorder.update_records(dataset, tmp_recorder)

            train_acc = np.array(train_var_accs[dataset]['NCC'])*100
            dataset_acc = np.array(var_accs[dataset]['NCC'])*100
            print(f"{dataset}: train_acc {train_acc.mean():.2f}%; test_acc {dataset_acc.mean():.2f} +/- {(1.96*dataset_acc.std()) / np.sqrt(len(dataset_acc)):.2f}%")

    timestamp = datetime.datetime.now().strftime("%Y%m%d")
    res_recorder.save(filename=args['experiment_name'] + "_" + timestamp)

    #print('results of {} with P%{}'.format(args['model.name'], args['headmodel.name']))
    print('results of {}'.format(args['experiment_name']))
    rows = []
    for dataset_name in testsets:
        row = [dataset_name]
        for model_name in accs_names:
            acc = np.array(var_accs[dataset_name][model_name])*100
            mean_acc = acc.mean()
            conf = (1.96*acc.std()) / np.sqrt(len(acc))
            row.append(f"{mean_acc:0.2f} +/- {conf:0.2f}%")
        rows.append(row)
    outpath = os.path.join(args['out.dir'], 'weights')
    outpath = check_dir(outpath, True)
    outpath = os.path.join(outpath, '{}-sslattention-{}-test_results.npy'.format(args['model.name'], args['headmodel.name']))
    np.save(outpath, {'rows':rows})

    table = tabulate(rows, headers=['model \\ data'] + accs_names, floatfmt=".2f")
    print(table)
    print("\n")
    print(f"{args['experiment_name']} Done!")