# Multi-Context Temporal Consistent Modeling for Referring Video Object Segmentation

[![PyTorch](https://img.shields.io/badge/PyTorch-1.13.0-%23EE4C2C.svg?style=&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.7%20|%203.8%20|%203.9-blue.svg?style=&logo=python&logoColor=ffdd54)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b?style=flat)](https://arxiv.org/abs/2501.04939)
[![PDF](https://img.shields.io/badge/PDF-Download-blue?style=flat)](https://ieeexplore.ieee.org/document/10888377)

This repository contains code for **ICASSP2025** paper:

### [Multi-Context Temporal Consistent Modeling for Referring Video Object Segmentation](https://arxiv.org/abs/2501.04939)  
Sun-Hyuk Choi, Hayoung Jo, Seong-Whan Lee  
ICASSP 2025

## Installation:
Please see [INSTALL.md](https://github.com/henghuiding/MeViS/blob/main/INSTALL.md). Then
```
pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
```
## Inference
Obtain the output masks of Val set for [CodaLab](https://codalab.lisn.upsaclay.fr/competitions/15094) online evaluation:
```
cd scripts
sh train_mevis.sh
```
## Inference
```
cd scripts
sh test_mevis.sh
```
## Models
☁️ [Google Drive](https://drive.google.com/file/d/1M4CZY3xKSg6qbwiU8BECcUHmCWeLtUHr/view?usp=sharing)
## Acknowledgement
This project is based on [DsHmp](https://github.com/heshuting555/DsHmp) and [DVIS](https://github.com/zhang-tao-whu/DVIS). Many thanks to the authors for their great works!
