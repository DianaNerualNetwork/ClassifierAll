import torch 
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning import seed_everything
# pip install pytorch_lightning
import numpy as np 
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

from configs import Configs
from models import *
from datasets import catdogDataset
from utils import logger,utils

def validate(model,val_loader,loss_fn,epoch,device):
    model.eval()
    correct_val = 0.
    total_val = 0.
    loss_val = 0.
    valid_curve=[]
    with torch.no_grad(): # !避免模型内存泄露
        for j, data in enumerate(val_loader):
            
            inputs, labels = data
            inputs=inputs.to(device)
            labels=labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted.cpu() == labels.cpu()).squeeze().sum().numpy()

            loss_val += loss.item()

            valid_curve.append(loss.item())
            logger.info("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                epoch, Configs.epochs[0], j+1, len(val_loader), loss.item(), correct_val / total_val))
    
    logger.info("[EVAL] Epoch:{:0>3}  Loss: {:.4f} Acc:{:.2%} ".format(
                epoch, loss_val/len(val_loader), correct_val / total_val
            ))
    logger.info("EVALUATE Done.....Ready to start next loop")
    model.train()
    return valid_curve

def train(model,train_loader,val_loader,optim,loss_fn,lr_scheduler,device):
    model.train()
    train_curve=[]
    valid_curve=[]
    for epoch in range(Configs.epochs[0]):
        
        curve_trian=train_one_epoch(model,train_loader,optim,loss_fn,lr_scheduler,epoch,device)
        curve_val=validate(model,val_loader,loss_fn,epoch,device)
        train_curve.append(curve_trian)
        valid_curve.append(curve_val)

def train_one_epoch(model,train_loader,optim,loss_fn,lr_scheduler,epoch,device):
    loss_mean = 0.
    correct = 0.
    total = 0.
    train_curve=[]
    iter_count = 0
    for i, data in enumerate(train_loader):
        iter_count += 1
        # forward
        inputs, labels = data
        inputs =inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)

        # backward
        optim.zero_grad()
        loss = loss_fn(outputs, labels)
        loss.backward()

        # update weights
        optim.step()
         # 统计分类情况
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels.cpu()).squeeze().sum().numpy()

        # 打印训练信息
        loss_mean += loss.item()
        train_curve.append(loss.item())
        if (i+1) % Configs.log_interval[0] == 0:
            loss_mean = loss_mean / Configs.log_interval[0]
            logger.info("[TRAIN] Epoch[{:0>3}/{:0>3}]  Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%} ".format(
                epoch,Configs.epochs[0],i+1, len(train_loader), loss_mean, correct / total
            ))
           
        lr_scheduler.step()  # 更新学习率 每个batch更新一次
    return train_curve
    

def  main():
    # get train valid dataset
    # get_mean_std_transform=transforms.Compose([
    #     transforms.ToTensor(),
    # ])
    # traindataset=catdogDataset.CatDogDataset(data_dir="/root/classfier/data/train",mode="train",transform=get_mean_std_transform)
    # train_mean,train_std=utils.get_mean_and_std(traindataset)
    # valdataset=catdogDataset.CatDogDataset(data_dir="/root/classfier/data/train",mode="valid",transform=get_mean_std_transform)
    # val_mean,val_std=utils.get_mean_and_std(valdataset)
    # build transforms
    trian_transforms=transforms.Compose([
        transforms.Resize((256)),      
        transforms.CenterCrop(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        #$transforms.Normalize(train_mean,train_std) # ? Why and how to analysis?  
        
    ])
    val_transforms=transforms.Compose([
        transforms.Resize((256)),      
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        #transforms.Normalize(val_mean,val_std) # ? Why and how to analysis?  [0.485, 0.456, 0.406],[0.229, 0.224, 0.225]
        
    ])
    # build device
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # build dataset
    train_dataset=catdogDataset.CatDogDataset(data_dir="/root/classfier/data/train",mode="train",transform=trian_transforms)
    val_dataset=catdogDataset.CatDogDataset(data_dir="/root/classfier/data/train",mode="valid",transform=trian_transforms)
    trainloader=torch.utils.data.DataLoader(train_dataset,batch_size=Configs.bs[0],shuffle=True,drop_last=True)
    valloader=torch.utils.data.DataLoader(val_dataset,batch_size=1,shuffle=False,drop_last=True)
    # build model
    model=ResNet18_vd(num_classes=2).to(device)
    # test model 
    from torchsummary import summary
    summary(model,(3,256,256),device='cuda')
    logger.info("Model build done....")
    # build optimizer
    optim=torch.optim.SGD(params=model.parameters(),lr=Configs.lr[0])
    # build lr_sheduler
    sampler = torch.utils.data.SequentialSampler(train_dataset)
    batch_sampler=torch.utils.data.sampler.BatchSampler(sampler,batch_size=Configs.bs[0],drop_last=True)
    iters_per_epoch=len(batch_sampler)
    lr_sheduler=torch.optim.lr_scheduler.OneCycleLR(optim,max_lr=Configs.lr[0],total_steps=(Configs.epochs[0]*iters_per_epoch),epochs=Configs.epochs[0])
    # build loss funciton
    loss_fn=torch.nn.CrossEntropyLoss()
    # train loop
    logger.info("********************Ready to train**********************")
    # get train and valid curve
    train_curve,valid_curve=train(model,trainloader,valloader,optim,loss_fn,lr_sheduler,device)
    # TODO draw it
if __name__ == "__main__":
    print("Start to set programing seed")
    seed_everything(Configs.seed[0])
    main()
