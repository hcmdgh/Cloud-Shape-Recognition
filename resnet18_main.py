# coding=gbk

from dataset import * 
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings("ignore")


BATCH_SIZE = 64

data_arr = prepare_dataset()

train_set, val_set, test_set = split_train_val_test_set(
    arr = data_arr,
    train_ratio = 0.6,
    val_ratio = 0.2,
    test_ratio = 0.2, 
 )
    
train_dataloader = DataLoader(
        dataset = ImageDataset(train_set),
        batch_size = BATCH_SIZE,
        shuffle = True,
        drop_last = False, 
)
val_dataloader = DataLoader(
        dataset = ImageDataset(val_set),
        batch_size = BATCH_SIZE,
        shuffle = False,
        drop_last = False, 
)
test_dataloader = DataLoader(
        dataset = ImageDataset(test_set),
        batch_size = BATCH_SIZE,
        shuffle = False,
        drop_last = False, 
)

'''
    for batch_img, batch_label in train_dataloader:
        print(batch_img.shape, batch_label.shape)
'''

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

#只微调训练模型的最后一层（全连接分类层
model=models.resnet18(pretrained=True)
model.fc=nn.Linear(model.fc.in_features,28)
#输出与数据集类别对应，为28
#由于数据集中标签从1开始，但pytorch默认从0开始
#因此我在dataset.py中将label改成label-1了，也就是data_list.append((path, label-1))，之后将预测结果写成csv时需要label+1
#也可以不改label，那就要将28改为29，否则会报错
optimizer = optim.Adam(model.fc.parameters())

model = model.to(device)
criterion = nn.CrossEntropyLoss()#交叉熵损失函数

EPOCHS=20 #每个EPOCH大概要运行六七分钟

def train_one_batch(images,labels):
    '''运行一个batch的训练，返回当前batch的训练日志'''

    #获得一个batch的数据和标注
    images=images.to(device)
    labels=labels.to(device)

    outputs = model(images)
    loss = criterion(outputs,labels)# 计算当前 batch 中，每个样本的平均交叉熵损失函数值
    
    #优化更新权重
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    _,preds = torch.max(outputs,1)#获得当前batch所有图像的预测类别
    preds=preds.cpu().numpy()
    loss=loss.detach().cpu().numpy()
    outputs = outputs.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    log_train={}
    log_train['epoch']=epoch
    log_train['batch']=batch_idx
    log_train['train_loss']=loss
    log_train['train_accuracy']=accuracy_score(labels,preds)
    log_train['train_precision'] = precision_score(labels, preds, average='macro')
    log_train['train_recall'] = recall_score(labels, preds, average='macro')
    log_train['train_f1-score'] = f1_score(labels, preds, average='macro')

    return log_train

def evaluate_testset():
    '''在整个测试集上评估，返回分类评估指标日志'''

    loss_list=[]
    labels_list=[]
    preds_list=[]

    with torch.no_grad():
        for images,labels in val_dataloader: #生成一个batch的数据和标注
            images=images.to(device)
            labels=labels.to(device)
            outputs=model(images)

            
            _,preds=torch.max(outputs,1)#获得当前batch所有图像的预测类别
            preds = preds.cpu().numpy()
            loss = criterion(outputs, labels) # 由 logit，计算当前 batch 中，每个样本的平均交叉熵损失函数值
            loss = loss.detach().cpu().numpy()
            outputs = outputs.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            loss_list.append(loss)
            labels_list.extend(labels)
            preds_list.extend(preds)

    log_test = {}
    log_test['epoch'] = epoch
    
    # 计算分类评估指标
    log_test['test_loss'] = np.mean(loss)
    log_test['test_accuracy'] = accuracy_score(labels_list, preds_list)
    log_test['test_precision'] = precision_score(labels_list, preds_list, average='macro')
    log_test['test_recall'] = recall_score(labels_list, preds_list, average='macro')
    log_test['test_f1-score'] = f1_score(labels_list, preds_list, average='macro')
    
    return log_test
       
        
if __name__ == '__main__':
 ###开始训练，记录日志
    epoch = 0
    batch_idx = 0
    best_test_f1 = 0

    # 训练日志-训练集
    df_train_log = pd.DataFrame()
    log_train = {}
    log_train['epoch'] = 0
    log_train['batch'] = 0
    images, labels = next(iter(train_dataloader))
    log_train.update(train_one_batch(images, labels))
    df_train_log = df_train_log.append(log_train, ignore_index=True)

    # 训练日志-测试集
    df_test_log = pd.DataFrame()
    log_test = {}
    log_test['epoch'] = 0
    log_test.update(evaluate_testset())
    df_test_log = df_test_log.append(log_test, ignore_index=True)

    for epoch in range(1,EPOCHS+1):
        print(f'Epoch {epoch}/{EPOCHS}')

        ##训练阶段
        model.train()
        for images,labels in tqdm(train_dataloader):
            batch_idx += 1
            log_train = train_one_batch(images,labels)
            df_train_log = df_train_log.append(log_train, ignore_index=True)
        
        #lr_scheduler.step()

        ##测试阶段
        model.eval()
        log_test = evaluate_testset()
        df_test_log = df_test_log.append(log_test, ignore_index=True)
        
        # 保存最新的最佳模型文件
        # 把f1分数作为文件名了，比较直观
        if log_test['test_f1-score'] > best_test_f1: 
            # 删除旧的最佳模型文件(如有)
            old_best_checkpoint_path = './model/checkpoints/best-{:.3f}.pth'.format(best_test_f1)
            if os.path.exists(old_best_checkpoint_path):
                os.remove(old_best_checkpoint_path)
            # 保存新的最佳模型文件
            new_best_checkpoint_path = './model/checkpoints/best-{:.3f}.pth'.format(log_test['test_f1-score'])
            torch.save(model, new_best_checkpoint_path)
            print('保存新的最佳模型', './model/checkpoints/best-{:.3f}.pth'.format(best_test_f1))
            best_test_f1 = log_test['test_f1-score']   
    df_train_log.to_csv('训练日志-训练集.csv', index=False)
    df_test_log.to_csv('训练日志-测试集.csv', index=False)

    # 载入最佳模型作为当前模型
    model = torch.load('./model/checkpoints/best-{:.3f}.pth'.format(best_test_f1))
    model.eval()
    print(evaluate_testset())

