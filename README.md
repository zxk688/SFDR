# SFDR

This repository contains the implementation for the following paper: ***Semantic-Spatial Feature Fusion with Dynamic Graph Refinement for Remote Sensing Image Captioning***, which has been accpeted by IEEE JSTARS.

### Requirements

All the environments and dependencies are in the **requirements.txt**, and you can install them through the command `pip install` , for example:

```bash
pip install absl-py==1.4.0
pip install *** == ***
......
......
```

### Datasets

In this paper, all the experimental results and comparisons are conducted on Sydney-Captions, UCM-Captions and RSICD. The dataset is available at [RSICD_optimal](https://github.com/201528014227051/RSICD_optimal).

### Feature Extraction and Processing

Detailed feature extraction and processing are introduced in detail in the ***Implementation Details*** section of the paper, which you can read from the paper.

- CLIP Feature Extraction and Embedding: `process_features/CLIP_feat_extract.py`


- Grid Feature Extraction: `process_features/Grid_feat_extract.py`


- ROI Feature Extraction: `process_features/ROI_feat_extract.py`


- CLIP Feature Resize: `process_features/resize.py`


- Feature Fusion: `process_features/fusion.py`

After the above operations, place different features in different folders, such as datasets/Sydney_Captions/features/fused_feature and datasets/Sydney_Captions/features/roi_feature.

### Training

If you want to train from scratch, please:

```bash
python train.py
```

After training, checkpoints will be saved to the checkpoint folder, such as **Sydney_best.pth** and **Sydney_last.pth**

Besides, if you want to continue training the model from different checkpoint, you can:

```
python train.py --resume_last
```

or

```
python train.py --resume.best
```

### Testing

```
python test.py
```

### Acknowledgements

The codes are heavily borrowed from [PKG-Transformer](https://github.com/One-paper-luck/PKG-Transformer). We'd like to thank the authors for their excellent work.
