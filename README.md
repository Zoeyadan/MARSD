# MARSD


### Installation

Manual install pytorch 2.6 from https://pytorch.org/get-started/previous-versions/
Install  python  environment.
```
pip install -r requirements.txt
```
## Data Preparation
Please follow the instructions to prepare all datasets.
Datasets list:
- [Office-Home](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view?pli=1&resourcekey=0-2SNWq0CDAuWOBRRBL7ZZsw)
- [ImageClef](https://github.com/jindongwang/transferlearning/blob/master/data/dataset.md#imageclef)
- [DomainNet](https://ai.bu.edu/DomainNet/)
- [VisDA-2017](http://ai.bu.edu/visda-2017/#download)



### Training
Using the `MARSD_Home.py` for training on Office-Home dataset can be found below.
```
python MARSD_Home.py data/office-home -d OfficeHome -s A -t C -a resnet50 --epochs 20 -i 500 --seed 1 --log logs/MARSD/OfficeHome_A2C
python MARSD_Home.py data/office-home -d OfficeHome -s A -t P -a resnet50 --epochs 20 -i 500 --seed 1 --log logs/MARSD/OfficeHome_A2P
python MARSD_Home.py data/office-home -d OfficeHome -s A -t R -a resnet50 --epochs 20 -i 500 --seed 1 --log logs/MARSD/OfficeHome_A2R
python MARSD_Home.py data/office-home -d OfficeHome -s C -t A -a resnet50 --epochs 20 -i 500 --seed 1 --log logs/MARSD/OfficeHome_C2A
python MARSD_Home.py data/office-home -d OfficeHome -s C -t P -a resnet50 --epochs 20 -i 500 --seed 1 --log logs/MARSD/OfficeHome_C2P
python MARSD_Home.py data/office-home -d OfficeHome -s C -t R -a resnet50 --epochs 20 -i 500 --seed 1 --log logs/MARSD/OfficeHome_C2R
python MARSD_Home.py data/office-home -d OfficeHome -s P -t A -a resnet50 --epochs 20 -i 500 --seed 1 --log logs/MARSD/OfficeHome_P2A
python MARSD_Home.py data/office-home -d OfficeHome -s P -t C -a resnet50 --epochs 20 -i 500 --seed 1 --log logs/MARSD/OfficeHome_P2C
python MARSD_Home.py data/office-home -d OfficeHome -s P -t R -a resnet50 --epochs 20 -i 500 --seed 1 --log logs/MARSD/OfficeHome_P2R
python MARSD_Home.py data/office-home -d OfficeHome -s R -t A -a resnet50 --epochs 20 -i 500 --seed 1 --log logs/MARSD/OfficeHome_R2A
python MARSD_Home.py data/office-home -d OfficeHome -s R -t C -a resnet50 --epochs 20 -i 500 --seed 1 --log logs/MARSD/OfficeHome_R2C
python MARSD_Home.py data/office-home -d OfficeHome -s R -t P -a resnet50 --epochs 20 -i 500 --seed 1 --log logs/MARSD/OfficeHome_R2P
```



Using the `MARSD_DomainNet.py` file for training. Sample commands to execute the training of the MARSD methods on DomainNet dataset  can be found below.
```
python MARSD_DomainNet.py data/DomainNet -d DomainNet -s R -t C -a  --epochs 20 -i 500 --seed 1 --log logs/MARSD/DomainNetP_R2C
python MARSD_DomainNet.py data/DomainNet -d DomainNet -s R -t P -a  --epochs 20 -i 500 --seed 1 --log logs/MARSD/DomainNetP_R2P
python MARSD_DomainNet.py data/DomainNet -d DomainNet -s R -t S -a  --epochs 20 -i 500 --seed 1 --log logs/MARSD/DomainNetP_R2S
python MARSD_DomainNet.py data/DomainNet -d DomainNet -s C -t R -a  --epochs 20 -i 500 --seed 1 --log logs/MARSD/DomainNetP_C2R
python MARSD_DomainNet.py data/DomainNet -d DomainNet -s C -t P -a  --epochs 20 -i 500 --seed 1 --log logs/MARSD/DomainNetP_C2P
python MARSD_DomainNet.py data/DomainNet -d DomainNet -s C -t S -a  --epochs 20 -i 500 --seed 1 --log logs/MARSD/DomainNetP_C2S
python MARSD_DomainNet.py data/DomainNet -d DomainNet -s P -t R -a  --epochs 20 -i 500 --seed 1 --log logs/MARSD/DomainNetP_P2R
python MARSD_DomainNet.py data/DomainNet -d DomainNet -s P -t C -a  --epochs 20 -i 500 --seed 1 --log logs/MARSD/DomainNetP_P2C
python MARSD_DomainNet.py data/DomainNet -d DomainNet -s P -t S -a  --epochs 20 -i 500 --seed 1 --log logs/MARSD/DomainNetP_P2S
python MARSD_DomainNet.py data/DomainNet -d DomainNet -s S -t R -a  --epochs 20 -i 500 --seed 1 --log logs/MARSD/DomainNetP_S2R
python MARSD_DomainNet.py data/DomainNet -d DomainNet -s S -t C -a  --epochs 20 -i 500 --seed 1 --log logs/MARSD/DomainNetP_S2C
python MARSD_DomainNet.py data/DomainNet -d DomainNet -s S -t P -a  --epochs 20 -i 500 --seed 1 --log logs/MARSD/DomainNetP_S2P
```

### Visualization
Using the `MARSD_DomainNet.py` file for DomainNet visualization. Sample command to execute the MARSD methods on DomainNet dataset (with real as source domain and clipart as the target domain) can be found below. Checkpoint is required for visualization

```
python MARSD_DomainNet.py data/DomainNet -d DomainNet -s R -t C -a resnet50  --log logs/MARSD/DoaminNet_R2C  --phase analysis
```

---
### Overview of the arguments
Generally, all scripts in the project take the following flags
- `DIR`: dataset path(data/|OfficeHome|DomainNet|ImageCLEF|)
- `-a`: Architecture of the backbone. (resnet50|resnet101)
- `-d`: Dataset (||OfficeHome|DomainNet|VisDA2017) 
- `-s`: Source Domain
- `-t`: Target Domain
- `--epochs`: Number of Epochs to be trained for.
- `--log`: path of the run log.
- `-i`: iterations per epoch
---

### Results
| Method | DomainNet | OfficeHome | ImageCLEF |
| :-----:| :-----:| :----: | :----: | 
| ResNet | 62.5 | 46.1 | 80.8 |
| DANN | 74.5 | 57.6 | 85.0 |
| BIWAA-I | 79.4 | 71.5 | - |
| GSDE | 83.1 |  73.6 | -|
| LUHP | 82.0| 75.4 | - |
| DLRE | - | 75.4 | 91.0 |
| MARSD (ours)| 86.0 | 77.7 | 91.7 |

### Acknowledgement
Our implementation is based on the Transfer Learning Library.

