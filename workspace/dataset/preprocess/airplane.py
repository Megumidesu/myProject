import os
import numpy as np
import random
import pickle
import csv
from easydict import EasyDict
import glob
from sentence_transformers import SentenceTransformer
from PIL import Image

np.random.seed(0)
random.seed(0)

attr_words = [
    'B-2', 'B-52H', 'E2D', 'F18', 'F35'#plane type -0
]

def resize_and_pad_to_square(image_path, output_path, size=224, fill=(0, 0, 0)):
    # 打开原始图片
    with Image.open(image_path) as img:
        # 检查图像是否有透明通道
        if img.mode == 'RGBA':
            # 创建一个黑色背景的图像
            background = Image.new('RGBA', img.size, (0, 0, 0, 255))
            # 将原始图像粘贴到背景图像上，忽略透明部分
            background.paste(img, mask=img.split()[3])  # 3 是 alpha 通道的索引
            img = background

        # 计算原始图片的宽高比
        original_width, original_height = img.size
        ratio = min(size / original_width, size / original_height)
        
        # 根据宽高比缩放图片
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)
        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 创建一个新的正方形图片，背景为指定颜色
        new_img = Image.new("RGB", (size, size), fill)
        
        # 计算粘贴的位置使图片居中
        paste_x = (size - new_width) // 2
        paste_y = (size - new_height) // 2
        new_img.paste(resized_img, (paste_x, paste_y))
        
        # 保存图片
        new_img.save(output_path, 'JPEG')

def clear_file(file_path):
    if os.path.exists(file_path):
        if os.path.getsize(file_path) > 0:
            # 文件不为空，清空文件
            open(file_path, 'w').close()

def preprocess_dataset(dataset_root, save_padimg_root):
    # 数据集图片路径名
    imgpath = os.path.join(dataset_root, 'images')
    # 生成文件路径名
    track_name_txt = os.path.join(dataset_root, 'track_name.txt')
    label_csv = os.path.join(dataset_root, 'label.csv')
    train_set_txt = os.path.join(dataset_root, 'train_set.txt')
    test_set_txt = os.path.join(dataset_root, 'test_set.txt')

    # 判断生成文件是否存在，是否为空
    clear_file(track_name_txt)
    clear_file(label_csv)
    clear_file(train_set_txt)
    clear_file(test_set_txt)


    # 建立类别与索引的映射
    for root, dirs, files in os.walk(imgpath, topdown=False):
        if root == imgpath:
            break
        files = sorted(files)
        track_dict = {}
        label_name = root.split('/')[-1]
        label_id = attr_words.index(label_name) + 1
        indexes_of_start = [index for index, file_name in enumerate(files) if file_name.endswith('_1.png')]
        # 初始化分隔点列表
        split_points = [indexes_of_start[0]]

        # 遍历indexes_of_start，检查索引之间的差距
        for i in range(1, len(indexes_of_start)):
            if indexes_of_start[i] - indexes_of_start[i - 1] > 60:
                # 如果两个索引之间的差距大于60，将当前索引添加到分隔点列表
                split=list(range(indexes_of_start[i-1], indexes_of_start[i] + 1, 60))
                for s in split[1:]:
                    split_points.append(s)
                if split[-1] != indexes_of_start[i]:
                    split_points.append(indexes_of_start[i])
            else:
                # 否则，继续检查下一个索引
                split_points.append(indexes_of_start[i])

        # 添加最后一个分隔点
        # split_points.append(indexes_of_start[-1])
        for i, index in enumerate(split_points):
            # 每个track路径名
            track_path = f"{int(label_id):04d}T{int(i+1):04d}"

            # track_name.txt开始写入
            with open(track_name_txt, 'a+') as f:
                f.write(track_path+"\n")
            if i == len(split_points)-1:
                track_dict[i] = files[index:]
            else:
                track_dict[i] = files[index:split_points[i+1]]
            os.makedirs(os.path.join(save_padimg_root + track_path), exist_ok=True)
            for frame in track_dict[i]:
                resize_and_pad_to_square(os.path.join(root, frame), os.path.join(save_padimg_root, track_path, frame.replace('png', 'jpg')))

        # 读取TXT文件
        with open(track_name_txt, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        # 创建CSV文件并写入表头
        with open(label_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            header = ['tracklet_id'] + attr_words
            writer.writerow(header)
    
            # 遍历每一行数据
            for line in lines:
                tracklet_id = line.strip()  # 移除尾随的换行符
                row = [tracklet_id]  # 初始化行数据，第一列为tracklet_id
                
                # 创建一个全0的列表，长度为类别数
                category_row = [0] * len(attr_words)
                
                # 根据tracklet_id的前四个数字确定类别
                category_row[int(tracklet_id[:4]) - 1] = 1  # 将对应类别的位置标记为1
                
                # 将类别行数据添加到行数据中
                row.extend(category_row)
                
                # 写入CSV文件
                writer.writerow(row)
    
    # 读取track_name.txt
    with open(track_name_txt, 'r') as file:
        lines = file.readlines()

    # 计算80%的行数
    split_point = int(len(lines) * 0.8)

    # 随机打乱行的顺序
    random.shuffle(lines)

    # 选择前80%的行
    train_name = lines[:split_point]
    # 选择剩余的20%的行
    test_name = lines[split_point:]

    # 写入新的文本文件
    with open(train_set_txt, 'w') as file1:
        file1.writelines(train_name)

    with open(test_set_txt, 'w') as file2:
        file2.writelines(test_name)

def track_imgpath(track_path):#返回为每个tracklet里照片路径的列表
    images_path=glob.glob(os.path.join(track_path,"*.jpg"))
    for count,i in enumerate(images_path) :
        images_path[count]=r'/'.join(i.split('\\'))
    return images_path

def generate_imgs(path,track_name):#形成一个字典，key=track_name，value=imgs_path
    imgs_path={}
    for i in track_name:
        tracklet_path=path+'/'+str(i)+'/'
        result=track_imgpath(tracklet_path)
        imgs_path[i]=result
    return imgs_path

def get_label_embeds(labels):
    model = SentenceTransformer('/root/autodl-tmp/vit_model/all-mpnet-base-v2')
    embeddings = model.encode(labels)
    return embeddings


#生成一个标签的字典，其中键代表tracklets_id,值为一个list（为样本的标签值）
def generate_label(filename):
    with open(filename, "r") as csvfile:
        csv_reader = csv.reader(csvfile)
        header = next(csv_reader)  # 获取CSV文件的表头
        result_dict = {}
        for row in csv_reader:
            row_buf=[int(i) for i in row[1:]]
            result_dict[str(row[0])] = np.array(row_buf)
    return result_dict

def count_lines(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()  # 读取所有行到一个列表中

    # 检查最后一行是否为空
    if lines and lines[-1].strip() == '':
        # 如果最后一行为空，则排除它
        return len(lines) - 1
    else:
        # 如果最后一行不为空，则返回总行数
        return len(lines)

# 生成预处理好的dataset.pkl
def generate_data_description(dataset_root, save_dir):    
    dataset = EasyDict()
    dataset.description = 'airplane'
    dataset.root=os.path.join(save_padimg_root)
    dataset.attr_name=attr_words
    dataset.words=np.array(attr_words)
    dataset.attr_vectors=get_label_embeds(attr_words)
    result_dict=generate_label(os.path.join(dataset_root, 'label.csv')) #每个tracklet的属性字典（标签）
    trainval_name=[]
    test_name=[]
    trainval_gt_list=[]
    test_gt_list=[]
    track_name=[]
    track_gt_list=[]

    track_name_file=open(os.path.join(dataset_root, 'track_name.txt'),'r',encoding='utf8').readlines()
    for name in track_name_file :
        curLine=name.strip('\n')
        track_name.append(curLine)  #整个数据集tracklet名字典

    trainval_name_file=open(os.path.join(dataset_root, 'train_set.txt'),'r',encoding='utf8').readlines()
    for name in trainval_name_file :
        curLine=name.strip('\n')
        trainval_name.append(curLine)  #train数据集trackle名字典
    test_name_file=open(os.path.join(dataset_root, 'test_set.txt'),'r',encoding='utf8').readlines()
    for name in test_name_file :
        curLine=name.strip('\n')
        test_name.append(curLine)  #test数据集trackle名字典

    for gt in track_name:
        curLine=name.strip('\n')
        track_gt_list.append(result_dict[curLine])  #整个数据集tracklet的标签列表

    for gt in trainval_name:
        curLine=name.strip('\n')
        trainval_gt_list.append(result_dict[curLine])  #train数据集tracklet的标签列表

    for gt in test_name_file:
        curLine=name.strip('\n')
        test_gt_list.append(result_dict[curLine])  #test数据集tracklet的标签列表
    dataset.test_name=test_name
    dataset.trainval_name=trainval_name

    dataset.track_name=dataset.trainval_name+dataset.test_name

    dataset.trainval_gt_list=trainval_gt_list
    dataset.test_gt_list=test_gt_list

    dataset.track_gt_list=track_gt_list
    dataset.result_dict = result_dict
    
    train_num = count_lines(os.path.join(dataset_root, 'train_set.txt')) #训练集track数
    test_num = count_lines(os.path.join(dataset_root, 'test_set.txt')) #测试集track数

    dataset.label = np.concatenate((np.array(trainval_gt_list),np.array(test_gt_list)), axis=0)
    assert dataset.label.shape == (train_num + test_num, 5)

    dataset.partition = EasyDict()
    dataset.attr_name = attr_words
    dataset.partition.test = np.arange(train_num, train_num + test_num)  # 测试集索引
    dataset.partition.trainval = np.arange(0, train_num)  # 训练集索引
    # 包含每个tracklet中图片的地址
    dataset.track_imgs_path=generate_imgs(os.path.join(dataset_root, 'pad_dataset'), track_name)

    pkl_path = os.path.join(save_dir, 'ppc_dataset.pkl')
    clear_file(pkl_path)
    with open(pkl_path, 'wb+') as f:
        pickle.dump(dataset, f)

if __name__ == "__main__":
    dataset_root = '../dataset'
    save_dir = './dataset/AIRPLANE/'
    save_padimg_root = '../dataset/pad_dataset/'
    
    preprocess_dataset(dataset_root, save_padimg_root)
    generate_data_description(dataset_root, save_dir)
