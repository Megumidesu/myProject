## Requirements
**Environment**
```
Python 3.9.20
PyTorch 1.12.1
CUDA 11.3
torch-vision 0.13.1
```
**Installation**
```
conda create -n my_env python=3.9
conda activate my_env
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

## Pre-trained Models Download
Please put the pre-trained models in the vit_model folder. The folder should look like this:
```
├── vit_model
│   ├── jx_vit_base_p16_224-80ecf9dd.pth
│   │   ViT-B-16.pt
│   │   all-mpnet-base-v2
│   │   │   ├── ...
```

* **Sentence-Transformers:**

```
链接: https://pan.baidu.com/s/1umvg1MAlWeGmTdEMAO-arw 提取码: wqt3
```
or
```
git clone https://huggingface.co/sentence-transformers/all-mpnet-base-v2
```

* **Vit-base-16**:
```
链接: https://pan.baidu.com/s/1niZyaO9i10nRNX5xG7RI9g 提取码: qdn8 
```

## Training and Testing 
* **Training**
1. Please put your dataset in the dataset folder. The folder should look like this:
```
├── dataset
│   ├── images
│   │   ├── B-2
│   │   │   ├── ...
│   workspace
│   ...
```
2. Start training by the following command:
```
cd workspace
python dataset/preprocess/airplane.py
python train.py
```
* **Testing**
Please first modify the parameters img_path and checkpoint_path in test.py
```
python test.py
```
