<h1 align="center"> Cross-domain Few-shot Classification via Invariant-content Feature Reconstruction </h1>

<p align="center">
<!-- <a href=""><img src="https://img.shields.io/badge/arXiv-2512.18786-b31b1b.svg" alt="Paper"></a>  -->
<a href="https://link.springer.com/journal/11263"><img src="https://img.shields.io/badge/Pub-IJCV'25-blue" alt="Conf"></a> 
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="Liscence"></a> 
<a href=""><img src="https://img.shields.io/badge/Slides%20-D76364" alt="Slides"></a> 
<a href=""><img src="https://img.shields.io/badge/Poster%20-Ffa500" alt="Poster"></a> 
<!-- <a href="https://www.bilibili.com/video/BV1k4421X7zK/?spm_id_from=333.1007.top_right_bar_window_dynamic.content.click&vd_source=a1aae47e2835186f922fa2e1c94933c9"><img src="https://img.shields.io/badge/CN_Video%20-54b345" alt="CN_Video"></a> 
<a href="https://www.youtube.com/watch?v=uWMM63Sv0ZI&t=110s"><img src="https://img.shields.io/badge/EN_Video%20-54b345" alt="EN_Video"></a></p> -->

This repository contains the source codes for reproducing the results of IJCV'25 paper: [**Cross-domain Few-shot Classification via Invariant-content Feature Reconstruction**](). 

**Author List**: Hongduan Tian, Feng Liu, Ka Chun Cheung, Zhen Fang, Simon See, Tongliang Liu, Bo Han.

## Introduction

<p align="center">
  <img src="./assets/ifr_framework.png" style="width:80%">
</p>

In *cross-domain few-shot classification* (CFC), mainstream studies aim to train a simple module (e.g. a linear transformation head) to select or transform features~(a.k.a., the high-level semantic features) for previously unseen domains with a few labeled training data available on top of a powerful pre-trained model. These studies usually assume that high-level semantic features are shared across these domains, and just simple feature selection or transformations are enough to adapt features to previously unseen domains. However, in this paper, we find that the simply transformed features are too general to fully cover the key content features regarding each class. Thus, we propose an effective method, *invariant-content feature reconstruction* (IFR), to train a simple module that simultaneously considers both high-level and fine-grained invariant-content features for the previously unseen domains. Specifically, the fine-grained invariant-content features are considered as a set of *informative* and *discriminative* features learned from a few labeled training data of tasks sampled from unseen domains and are extracted by retrieving features that are invariant to style modifications from a set of content-preserving augmented data in pixel level with an attention module. Extensive experiments on the Meta-Dataset benchmark show that IFR achieves good generalization performance on unseen domains, which demonstrates the effectiveness of the fusion of the high-level features and the fine-grained invariant-content features. Specifically, IFR improves the average accuracy on unseen domains by 1.6\% and 6.5\% respectively under two different cross-domain few-shot classification settings.

## Preparation
### Dependencies
In our experiments, the main dependences required are the following libraries:
```
Python 3.6 or greater (Ours: Python 3.8)
PyTorch 1.0 or greater (Ours: torch=1.7.1, torchvision=0.8.2)
TensorFlow 1.14 or greater (Ours: TensorFlow=2.10)
tqdm (Ours: 4.64.1)
tabulate (0.8.10)
```

### Dataset
- Follow [Meta-Dataset repository](https://github.com/google-research/meta-dataset) to prepare `ILSVRC_2012`, `Omniglot`, `Aircraft`, `CU_Birds`, `Textures (DTD)`, `Quick Draw`, `Fungi`, `VGG_Flower`, `Traffic_Sign` and `MSCOCO` datasets.

- Follow [CNAPs repository](https://github.com/cambridge-mlg/cnaps) to prepare `MNIST`, `CIFAR-10` and `CIFAR-100` datasets.



### Backbone Pretraining
In this paper, we follow [URL](https://arxiv.org/pdf/2103.13841.pdf) and use ResNet-18 as the frozen backbone in all our experiments. For reproduction, two ways are provided:

__Train your own backbone.__ You can train the ResNet-18 backbone from scratch by yourself. The pretraining mainly contains two phases: domain-specific pretraining and universal backbone distillation.

To train the single domain-specific learning backbones (on 8 seen domains), run:
```
./scripts/train_resnet18_sdl.sh
```

Then, distill the model by running:
```
./scripts/train_resnet18_url.sh
```

__Use the released backbones.__ URL repository has released both universal backbone and single domain backbone. For simplicity, you can directly use the released model.
- [Single-domain networks (one for each dataset)](https://drive.google.com/file/d/1MvUcvQ8OQtoOk1MIiJmK6_G8p4h8cbY9/view?usp=sharing)

- [A single universal network (URL) learned from 8 training datasets](https://drive.google.com/file/d/1Dv8TX6iQ-BE2NMpfd0sQmH2q4mShmo1A/view?usp=sharing)

The backbones can be downloaded with the above links. To download the pretrained URL model, one can use `gdown` (installed by ```pip install gdown```) and execute the following command in the root directory of this project:
```
gdown https://drive.google.com/uc?id=1MvUcvQ8OQtoOk1MIiJmK6_G8p4h8cbY9 && md5sum sdl.zip && unzip sdl.zip -d ./saved_results/ && rm sdl.zip  # Universal backbone
gdown https://drive.google.com/uc?id=1Dv8TX6iQ-BE2NMpfd0sQmH2q4mShmo1A && md5sum url.zip && unzip url.zip -d ./saved_results/ && rm url.zip  # Domain specific backbones
```
In this way, the backbones are donwnloaded. Please create the ```./saved_results``` directory and place the backbone weights in it. 

### Evaluate IFR
To evaluate the IFR, you can run:
```
./scripts/run_ifr.sh
```
Specifically, the running command is:
```
python run_ifr.py --model.name=url \
                 --model.dir ./url \
                 --weight_decay=0.1 \
                 --setting_name=train_on_all_datasets \
                 --scale_reconst=10.0 \
                 --inner_iter=40 \
                 --experiment_name=ifr
```
The hyperparameters can be modified for different experiments:
- `model_name: ['sdl', 'url']`: `sdl` means using single domain backbone; `url` means using universal backbone.
- `model.dir`: Path to the backbone weights.
- `scale_const (float)`: Hyperparameter for the fusion of original and reconstructed features.

### Evaluate Pre-classifier Alignment (PA)
To evaluate Pre-classifier Alignment (PA), which is the typical case of URL, run:

```
./scripts/test_resnet18_pa.sh
```


## Acknowledgement
 
 The repository is built mainly upon these repositories:
 
- [VICO-UoE/URL [1]](https://github.com/VICO-UoE/URL);
- [google-research/meta-dataset [2]](https://github.com/google-research/meta-dataset)

[1] Li et al. [Universal representation learning from multiple domains for few-shot classification](https://arxiv.org/pdf/2103.13841), ICCV 2021.

[2] Triantafillou et al. [Meta-dataset: A dataset of datasets for learning to learn from few examples](https://arxiv.org/pdf/1903.03096), ICLR 2020.

## Citation
```
@inproceedings{tian2025cross,
    title={cross-domain few-shot classification via invariant-content feature reconstruction},
    author={Hongduan Tian and Ka Chun Cheung and Zhen Fang and Simon See and Tongliang Liu and Bo Han},
    booktitle={International Journal of Computer Vision (IJCV)},
    year={2025}
}
```