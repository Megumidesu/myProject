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
from tqdm import tqdm

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

    print('*' * 50 + "加载数据" + '*' * 50)
    imgs=[]
    if not os.path.isfile(img_path):
        file_list = os.listdir(img_path)
        img_list = []
        for i in file_list:
            if i.endswith('.png'):
                pil=Image.open(os.path.join(img_path, i))
                pil = convert2jpg(pil)
                imgs.append(transform(pil))
                img_list.append(os.path.join(img_path, i))
        
    else:
        img_list = [img_path]
        pil=Image.open(img_path)
        pil = convert2jpg(pil)
        imgs.append(transform(pil))
    img_tensor=torch.stack(imgs).to(device)
    print('*' * 10 + "数据加载完成" + '*' * 10)
    print('\n' + '*' * 10 + "加载模型" + '*' * 10)
    model = TransformerClassifier(attr_num=5,attr_words=attr_words)
    model = model.to(device)
    model.eval()
    ViT_model.eval()
    ViT_model = ViT_model.to(device)
    checkpoint=torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'],strict=False)
    print('*' * 10 + "模型加载完成" + '*' * 10)

    print('\n' + '*' * 10 + "开始预测" + '*' * 10)    
    start = time.time()
    with torch.no_grad():
        logits = model(img_tensor, ViT_model=ViT_model)
        probs = torch.sigmoid(logits).cpu().numpy()

        index_list=[0, 5]
        group=['B-2', 'B-52H', 'E2D', 'F18', 'F35']
        for idx in tqdm(range(len(index_list)-1), desc='模型预测'):
            if index_list[idx+1]-index_list[idx] >1 :
                pred_result = np.argmax(probs, axis=1)
                categories = [group[i] for i in pred_result]
        print('-'*10 + '预测结果' + '-'*10 + '\n')
        for i, (img, cls) in enumerate(zip(img_list, categories)):
            label = img.split('/')[-2]
            print(f"sample:{i}------pred:{cls}, label:{label}, {True if cls==label else False}")
        print('\n'+'-'*10)
    end = time.time()
    print(f"test time:{(end-start):.4f}s")
    
if __name__ == '__main__':
    #要预测的图片路径或者文件夹路径
    img_path = 'test_imgs/B-52H'
    #模型权重文件路径
    checkpoint_path = '../vit_model/2024-08-25_23_28_48_epoch37_mf172.22.pth'
    test_picture(img_path, checkpoint_path, ViT_model)
