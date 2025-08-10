# CoVR-VidLLM-CVPR25

## Description

This repository contains the code for baseline model implementation used for CoVR-VidLLM workshop CVPR-2025.

Please visit our [Workshop Page](https://eval.ai/web/challenges/challenge-page/2443/overview) for more details.

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

### Evaluating

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


## Acknowledgements
Based on [CoVR](https://github.com/lucas-ventura/CoVR), [BLIP](https://github.com/salesforce/BLIP/) and [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template/tree/main).

