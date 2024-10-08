# myProject
## Requirements
we use a single RTX3090 24G GPU for training and evaluation. 

**Basic Environment**
```
Python 3.9.16
PyTorch 1.12.1
torch-vision 0.13.1
```
**Installation**
```
pip install -r requirements.txt
```
## Datasets and Pre-trained Models 

**Download from BaiduYun:**

* **MARS Dataset**:
```
链接：https://pan.baidu.com/s/16Krv3AAlBhB9JPa1EKDbLw 提取码：zi08
```

* **Pre-trained Models (VTF-Pretrain.pth)**:
```
链接：https://pan.baidu.com/s/150t_zCW35YQHViKxsRIVzQ  提取码：glbd
```

**Download from DropBox:**
```
https://www.dropbox.com/scl/fo/h70nbcuj4gsmi4txhq1i0/h?rlkey=rwn1gbqbjpak6d7zhp46o3rnb&dl=0
``` 




## Training and Testing 
Use the following code to learn a model for MARS Dataset:

Training
```
python ./dataset/preprocess/mars.py
python train.py MARS
```
Testing
```
python eval.py MARS
```
