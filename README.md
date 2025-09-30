
# Beyond Simple Edits: Composed Video Retrieval with Dense Modifications (🔥ICCV-2025)

#### [Omkar Thawakar*](https://scholar.google.com/citations?user=flvl5YQAAAAJ&hl=en), [Dmitry Demidov*](https://scholar.google.com/citations?user=k3euI0sAAAAJ&hl=en), [Ritesh Thawkar](#), , [Rao Muhammad Anwer](https://scholar.google.com/citations?hl=en&authuser=1&user=_KlvMVoAAAAJ), [Mubarak Shah](https://scholar.google.com/citations?user=p8gsO3gAAAAJ&hl=en), [Fahad Khan](https://sites.google.com/view/fahadkhans/home) and [Salman Khan](https://salman-h-khan.github.io/)


#### **Mohamed Bin Zayed University of Artificial Intelligence (MBZUAI), UAE** and Linköping University, Sweden

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2508.14039)
🤗 [![HuggingFace Dataset](https://img.shields.io/badge/HuggingFace-Page-F9D371)](https://huggingface.co/datasets/omkarthawakar/Dense-WebVid-CoVR)


---

## 📢 Latest Updates
- **August-21-2025**- Training code is released on github.
- **August-20-2025**- Arxiv Preprint is released.
- **July-15-2025**- BSE-CoVR accepted to ICCV-2025


## Overview
Composed video retrieval is a challenging task that strives to retrieve a target video based on a query video and a textual description detailing specific modifications. Standard retrieval frameworks typically struggle to handle the complexity of fine-grained compositional queries and variations in temporal understanding limiting their retrieval ability in the fine-grained setting. 
To address this issue, we introduce a novel dataset that captures both fine-grained and composed actions across diverse video segments, enabling more detailed compositional changes in retrieved video content.
The proposed dataset, named Dense-WebVid-CoVR, consists of 1.6 million samples with dense modification text that is around seven times more than its existing counterpart. We further develop a new model that integrates visual and textual information through Cross-Attention (CA) fusion using grounded text encoder, enabling precise alignment between dense query modifications and target videos. The proposed model achieves state-of-the-art results surpassing existing methods on all metrics. Notably, it achieves 71.3\% Recall@1 in visual+text setting and outperforms the state-of-the-art by 3.4\%, highlighting its efficacy in terms of leveraging detailed video descriptions and dense modification texts.

## Repository Overview

This repository contains the code for model implementation used for BSE-CoVR.

The repository structure: 

```markdown
📦 covr
 ┣ 📂 configs                 # hydra config files
 ┣ 📂 src                     # Pytorch datamodules
 ┣ 📂 tools                   # scripts and notebooks
 ┣ 📜 .gitignore
 ┣ 📜 README.md
 ┣ 📜 test.py                 # test script
 ┣ 📜 evaluate_scores.py      # recall score evaluation script
 ```

## Installation

### Create environment

```bash
conda create --name covr-env
conda activate covr-env
```

To install the necessary packages, use requirements.txt file:
```bash
python -m pip install -r requirements.txt
```

The code was tested on Python 3.10 and PyTorch 2.4.


### (Optional) Download pre-trained models

To download the checkpoints, run:
```bash
bash tools/scripts/download_pretrained_models.sh
```

### Download dataset : Dense-WebVid-CoVR

To download the dataset, refer to our 🤗 [![HuggingFace](https://img.shields.io/badge/HuggingFace-Page-F9D371)]([#](https://huggingface.co/datasets/omkarthawakar/Dense-WebVid-CoVR))

## Usage

### Computing BLIP embeddings

Before evaluating, you will need to compute the BLIP embeddings for the videos/images. To do so, run:
```bash
# This will compute the BLIP embeddings for the WebVid-CoVR videos. 
# Note that you can use multiple GPUs with --num_shards and --shard_id
python tools/embs/save_blip_embs_vids.py --video_dir datasets/WebVid/2M/train --todo_ids annotation/webvid-covr/webvid2m-covr_train.csv 

# This will compute the BLIP embeddings for the WebVid-CoVR-Test videos.
python tools/embs/save_blip_embs_vids.py --video_dir datasets/WebVid/8M/train --todo_ids annotation/webvid-covr/webvid8m-covr_test.csv 

# This will compute the BLIP embeddings for the WebVid-CoVR modifications text. Only needed if using the caption retrieval loss (model/loss_terms=si_ti+si_tc).
python tools/embs/save_blip_embs_txts.py annotation/webvid-covr/webvid2m-covr_train.csv datasets/WebVid/2M/blip-vid-embs-large-all
```

### Training

The command to launch a training experiment is the folowing:
```bash
python train.py [OPTIONS]
```
The parsing is done by using the powerful [Hydra](https://github.com/facebookresearch/hydra) library. You can override anything in the configuration by passing arguments like ``foo=value`` or ``foo.bar=value``. See *Options parameters* section at the end of this README for more details.


### Evaluation

#### Calculating Query Features

The command to calculate the query feature results for Image/Video + descriptions:
```bash
python test.py test=webvid-covr
```

The command to calculate the query feature description for Image/Video descriptions only:
```bash
python test.py test=webvid-covr_text
```

The results will be saved in a numpy array file `query_feat.npy` and `query_feat_txt_only.npy` in the output folder for Image/Video + Description and Descriptions only respectively.

#### Calculating Recalls for evaluation

To calculate the recalls for the query features results for Image/Video + descriptions, execute the following command:
```bash
python evaluate_scores.py evaluate=webvid-covr
```

And, to calculate the recalls for the query features results for descriptions only, execute the following command:
```bash
python evaluate_scores.py evaluate=webvid-covr_text
```

The recalls will be saved in a json file `recalls.json` and `recalls_txt_only.json` in the output folder for Image/Video + description and descriptions only respectively.

The Format of the `recalls.json` is as following:
```json
{
  "R1": 45.00,
  "R5": 68.40,
  "R10": 78.50,
  "R50": 92.00,
  "meanR3": 63.97,
  "meanR4": 70.97,
  "annotation": "webvid8m-covr_test.csv"
}
```

<details><summary>Options parameters</summary>


#### Datasets:
- ``data=webvid-covr``: WebVid-CoVR datasets.
- ``data=cirr``: CIRR dataset.
- ``data=fashioniq``: FashionIQ dataset.


#### Models:
- ``model=blip-large``: BLIP model.

#### Tests:
- ``test=all``: Test on WebVid-CoVR, CIRR and all three Fashion-IQ test sets.
- ``test=webvid-covr``: Test on WebVid-CoVR.
- ``test=cirr``: Test on CIRR.
- ``test=fashioniq``: Test on all three Fashion-IQ test sets (``dress``, ``shirt`` and ``toptee``).

#### Checkpoints:
- ``model/ckpt=blip-l-coco``: Default checkpoint for BLIP-L finetuned on COCO.
- ``model/ckpt=webvid-covr``: Default checkpoint for CoVR finetuned on WebVid-CoVR.

#### Training
- ``trainer=gpu``: training with CUDA, change ``devices`` to the number of GPUs you want to use.
- ``trainer=ddp``: training with Distributed Data Parallel (DDP), change ``devices`` and ``num_nodes`` to the number of GPUs and number of nodes you want to use.
- ``trainer=cpu``: training on the CPU (not recommended).

#### Logging
- ``trainer/logger=csv``: log the results in a csv file. Very basic functionality.


#### Machine
- ``machine=server``: You can change the default path to the dataset folder and the batch size. You can create your own machine configuration by adding a new file in ``configs/machine``.

#### Experiment
There are many pre-defined experiments from the paper in ``configs/experiment`` and ``configs/experiment2``. Simply add ``experiment=<experiment>`` or ``experiment2=<experiment>`` to the command line to use them. 

&emsp; 

</details>



## Acknowledgements
Based on [CoVR](https://github.com/lucas-ventura/CoVR), [BLIP](https://github.com/salesforce/BLIP/) and [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template/tree/main).

