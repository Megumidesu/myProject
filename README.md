**Environment**
```
Python 3.9.16
PyTorch 1.12.1
CUDA 11.3
torch-vision 0.13.1
```
**Installation**
```
conda create -n my_env python=3.9
conda activate my_env
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```
## Training and Testing 
Training
```
cd workspace
python dataset/preprocess/airplane.py
python train.py
```
Testing
```
python test.py
```
