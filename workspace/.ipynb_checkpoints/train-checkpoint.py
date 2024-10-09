import os
import pprint
from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data import DataLoader
import time
from torch import optim
from batch_engine import valid_trainer, batch_trainer
from config import argument_parser
from dataset.AttrDataset import MultiModalAttrDataset, get_transform
from loss.CE_loss import *
from models.base_block import *
from tools.function import get_multiAttr_metrics,get_signle_metrics
from tools.utils import time_str, ReDirectSTD, set_seed, select_gpus, create_directory
from solver.scheduler_factory import create_scheduler

from CLIP.clip import clip
from CLIP.clip.model import *

set_seed(605)
device = "cuda" if torch.cuda.is_available() else "cpu"
ViT_model, ViT_preprocess = clip.load("ViT-B/16", device=device, download_root='../vit_model/') 

def main(args):
    log_dir = os.path.join('../logs', 'exp')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    else:
        log_dir = create_directory('../logs', 'exp')
    stdout_file = os.path.join(log_dir, f'stdout_{time_str()}.txt')

    if args.redirector:
        print('redirector stdout')
        ReDirectSTD(stdout_file, 'stdout', False)

    pprint.pprint(OrderedDict(args.__dict__))

    print('-' * 60)
    select_gpus(args.gpus)

    print(f'train set: {args.dataset} {args.train_split}, test set: {args.valid_split}')

    train_tsfm, valid_tsfm = get_transform(args) 

    train_set = MultiModalAttrDataset(args=args, split=args.train_split , transform=train_tsfm) 

    train_loader = DataLoader(
        dataset=train_set, 
        batch_size=args.batchsize, 
        shuffle=True,
        num_workers=32,
        pin_memory=True, 
        drop_last=True
    )
    
    valid_set = MultiModalAttrDataset(args=args, split=args.valid_split , transform=valid_tsfm) 

    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=32,
        pin_memory=True,
    )

    labels = train_set.label
    sample_weight = labels.mean(0)
    model = TransformerClassifier(train_set.attr_num,attr_words=train_set.attributes)
    if torch.cuda.is_available():
        model = model.cuda()

    criterion = CEL_Sigmoid(sample_weight,attr_idx=train_set.attr_num)
    lr = args.lr
    epoch_num = args.epoch
    start_epoch=1
    optimizer = optim.Adam(model.parameters(),lr=lr)
    scheduler = create_scheduler(optimizer, num_epochs=epoch_num, lr=lr, warmup_t=5)


    trainer(args=args,
            epoch=epoch_num,
            model=model,
            ViT_model=ViT_model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            path=log_dir,
            start_epoch=start_epoch)
    
def trainer(args,epoch, model,ViT_model, train_loader, valid_loader, criterion, optimizer, scheduler,path,start_epoch):
    best_map,best_epoch=0,0
    start=time.time()
    for i in range(start_epoch, epoch+1):
        scheduler.step(i)
        train_loss, train_gt, train_probs = batch_trainer(
            epoch=i,
            model=model,
            ViT_model=ViT_model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer
        )
        valid_loss, valid_gt, valid_probs, _ = valid_trainer(
            epoch=epoch,
            model=model,
            ViT_model=ViT_model,
            valid_loader=valid_loader,
            criterion=criterion,
        )
        if args.dataset =='AIRPLANE' : 
            #AIRPLANE
            index_list=[0, 5]
            group="airplane-type"
            plane_name=['B-2', 'B-52H', 'E2D', 'F18', 'F35']
        group_mAP=[]
        group_f1=[]
        group_acc=[]
        group_prec=[]
        group_recall=[]
        for idx in range(len(index_list)-1):
            if index_list[idx+1]-index_list[idx] >1 :
                result=get_multiAttr_metrics(valid_gt[:,index_list[idx]:index_list[idx+1]], valid_probs[:,index_list[idx]:index_list[idx+1]])
            elif idx < 9  :
                result=get_signle_metrics(valid_gt[:,index_list[idx]], valid_probs[:,index_list[idx]])
            group_mAP.append(result.instance_mAP)
            group_f1.append(result.instance_f1) 
            group_acc.append(result.instance_acc)  
            group_prec.append(result.instance_prec)
            group_recall.append(result.instance_recall)

        group_all= [group_mAP,group_f1,group_acc,group_prec,group_recall]
        average_mAP = np.mean(group_mAP)
        average_instance_f1 = np.mean(group_f1)
        average_acc = np.mean(group_acc)
        average_prec = np.mean(group_prec)    
        average_recall = np.mean(group_recall)
        average_all=[average_mAP,average_instance_f1,average_acc,average_prec,average_recall]    
        if best_map < average_mAP:
            best_map = average_mAP
            best_epoch = i
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'average_all': average_all,
                'valid_loss': valid_loss,
                'group_all':group_all,
                }, os.path.join(path, "best.pth"))

        print(f'{time_str()}||Evaluation on test set, valid_loss:{valid_loss:.4f}\n',
              f"Attr      :{group} \n",
              f"Acc       :",','.join(str(elem)[:6] for elem in group_acc),'\n',
              f"Prec      :",','.join(str(elem)[:6] for elem in group_prec),'\n',
              f"Recall    :",','.join(str(elem)[:6] for elem in group_recall),'\n',
              f"F1        :",','.join(str(elem)[:6] for elem in group_f1),'\n',
              f"airplane  :",',  '.join(str(elem) for elem in plane_name),'\n',
              f"cls_AP    :",','.join(str(elem)[:6] for elem in result.instance_per_AP),'\n',
              'average_mAP: {:.4f}, average_acc: {:.4f}, average_prec: {:.4f}, average_recall: {:.4f},  average_f1: {:.4f}'.format(average_mAP, average_acc, average_prec, average_recall, average_instance_f1)                 
            )
        print('-' * 60)     

        if i % args.epoch_save_ckpt == 0:
            torch.save({
                        'epoch': i,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'average_all': average_all,
                        'valid_loss': valid_loss,
                        'group_all':group_all,
                        }, os.path.join(path, f"epoch_{i}.pth"))             
    print(f"The best model at epoch{best_epoch}, mAP: {best_map}")

if __name__ == '__main__':
    #在config.py更改训练参数
    parser = argument_parser()
    args = parser.parse_args()
    main(args)
