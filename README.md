# SFDR

This repository contains the implementation for the following paper: [***Semantic-Spatial Feature Fusion with Dynamic Graph Refinement for Remote Sensing Image Captioning***](https://ieeexplore.ieee.org/document/11039674), which has been accepted by IEEE JSTARS.
If you find this research or dataset useful for your research, please cite our paper:
```
@ARTICLE{11039674,
  author={Liu, Maofu and Liu, Jiahui and Zhang, Xiaokang},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={Semantic-Spatial Feature Fusion with Dynamic Graph Refinement for Remote Sensing Image Captioning}, 
  year={2025},
  volume={},
  number={},
  pages={1-14},
  keywords={Feature extraction;Visualization;Semantics;Remote sensing;Attention mechanisms;Decoding;Transformers;Vectors;Sensors;Convolutional neural networks;Remote sensing image captioning;feature fusion;graph attention;dynamic weighting mechanism},
  doi={10.1109/JSTARS.2025.3580686}}
```
### Requirements

All the environments and dependencies are in the **requirements.txt**, and you can install them through the command `pip install -r requirements.txt`.


### Datasets

In this paper, all the experimental results and comparisons are conducted on Sydney-Captions, UCM-Captions and RSICD. The dataset is available at [RSICD_optimal](https://github.com/201528014227051/RSICD_optimal).

### Feature Extraction and Processing

Detailed feature extraction and processing are introduced in the ***Implementation Details*** section of the paper.

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

Besides, if you want to continue training the model from a different checkpoint, you can:

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
