import os
from collections import OrderedDict, defaultdict
import numpy as np
import torch
import time
from config import argument_parser
from loss.CE_loss import *
from models.base_block import *
from tools.utils import set_seed
from CLIP.clip import clip
from CLIP.clip.model import *
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

set_seed(605)
device = "cuda" if torch.cuda.is_available() else "cpu"

ViT_model, ViT_preprocess = clip.load("ViT-B/16", device=device, download_root='../vit_model/') 
attr_words = [
    'B-2',
    'B-52H', 
    'E2D',
    'F18',
    'F35'
]

def convert2jpg(img):
    if img.mode in ('RGBA', 'LA'):
        bg = Image.new(img.mode[:-1], img.size, (255, 255, 255))
        bg.paste(img, img.split()[-1])
        img = bg
    else:
        img = img.convert('RGB')
    return img

def test_picture(img_path, checkpoint_path, ViT_model):
    parser = argument_parser()
    args = parser.parse_args()
    normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        normalize
    ])


    imgs=[]
    if not os.path.isfile(img_path):
        img_list = os.listdir(img_path)
        for i in img_list:
            pil=Image.open(img_path + i)
            pil = convert2jpg(pil)
            imgs.append(transform(pil))
    else:
        img_list = [img_path]
        pil=Image.open(img_path)
        pil = convert2jpg(pil)
        imgs.append(transform(pil))
    img_tensor=torch.stack(imgs).to(device)
    model = TransformerClassifier(attr_num=5,attr_words=attr_words)
    model = model.to(device)
    model.eval()
    ViT_model.eval()
    ViT_model = ViT_model.to(device)
    checkpoint=torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'],strict=False)
    start = time.time()
    with torch.no_grad():
        logits = model(img_tensor, ViT_model=ViT_model)
        probs = torch.sigmoid(logits).cpu().numpy()

        index_list=[0, 5]
        group=['B-2', 'B-52H', 'E2D', 'F18', 'F35']
        for idx in range(len(index_list)-1):
            if index_list[idx+1]-index_list[idx] >1 :
                pred_result = np.argmax(probs, axis=1)
                categories = [group[i] for i in pred_result]
        print('--'*70+'\n')
        for i, c in zip(img_list, categories):
            print(f"img:{os.path.basename(i)}---------------------------------pred:{c}, label:{i.split('/')[-2]}")
        print('\n'+'--'*70)
    end = time.time()
    print(f"test time:{(end-start):.4f}s")
        

if __name__ == '__main__':
    #要预测的图片路径或者文件夹路径
    img_path = ''
    #模型权重文件路径
    checkpoint_path = ''
    test_picture(img_path, checkpoint_path, ViT_model)

