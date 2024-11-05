# Multi-Context Temporal Consistent Modeling for Referring Video Object Segmentation
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13.0-%23EE4C2C.svg?style=&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.7%20|%203.8%20|%203.9-blue.svg?style=&logo=python&logoColor=ffdd54)](https://www.python.org/downloads/)

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
sh refiner_test.sh
```
## Models
☁️ [Google Drive](https://drive.google.com/file/d/1YLnRUsANuPVfLo1jrgK05EGUJglrwA9H/view?usp=drive_link)
## Acknowledgement
This project is based on [DsHmp](https://github.com/heshuting555/DsHmp) and [DVIS](https://github.com/zhang-tao-whu/DVIS). Many thanks to the authors for their great works!
