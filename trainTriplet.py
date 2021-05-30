import os
import glob
import time
import random
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import PIL.ImageOps
from sklearn.model_selection import train_test_split

from DensenetSiamese import DensenetSiameseNetworkTriplet
model = DensenetSiameseNetworkTriplet()

import torch
# import torchvision
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchsummary import summary
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
from torch.utils.data import DataLoader,Dataset
from torch.autograd import Variable
import torchvision.utils

from losses import ContrastiveLoss, TripletLoss
from utils import prep_dataset, prep_datasetV2, check_dataset, show_plot, TrainDatasetV5




os.environ["CUDA_VISIBLE_DEVICES"]="2"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Args:
    height_img, width_img = 224, 224
    batch_size = 32
    lr = 1e-3 
    seed = 3
    split_ratio = 0.3
    split_val_ratio = 0.2
    num_epochs = 20
    image_file_folder="../data/train_patch_224_norm_images_rad"
    save_folder="./models"
    save_details_folder="./model_details"
    dataset_used = "./storage/dataset_used_rad.txt"
    expt = "19"
    dataframe_path = '../data/radboud.csv'

args = Args()


model = model.to(device)

criterion = TripletLoss()
optimizer = optim.Adam(model.parameters(), lr = args.lr, betas = (0.9, 0.999))

def create_dir(args):
    # make directory if doesn't exist
    if not os.path.exists('./{}/expt{}'.format(args.save_folder,args.expt)):
        os.makedirs('./{}/expt{}'.format(args.save_details_folder,args.expt))
        os.makedirs('./{}/expt{}'.format(args.save_folder,args.expt))

def train_fn(X_train,X_val, args):

    transformV1 = transforms.Compose([
        transforms.ColorJitter(brightness = 0.075, saturation = 0.075, hue = 0.075), # TRAINDATASETV2
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset=TrainDatasetV5(X_train, args.image_file_folder, transform = transformV1)
    train_loader=DataLoader(train_dataset,batch_size=args.batch_size)
    
    val_dataset=TrainDatasetV5(X_val, args.image_file_folder, transform = transformV1)
    val_loader=DataLoader(val_dataset,batch_size=args.batch_size)
    train_losses = []
    val_losses = []
    counter = []

    for epoch in range(args.num_epochs):
        print('Epoch {}/{}'.format(epoch, args.num_epochs - 1))
        print('-' * 10)
        
        since = time.time()
    
        model.train()
        running_loss = 0
        batch = tqdm(enumerate(train_loader), total=len(train_loader))
        for idx, (img0, img1, img2) in batch:
            img0 = img0.to(device)
            img1 = img1.to(device)
            img2 = img2.to(device)
            
            optimizer.zero_grad()
            
            img0.requires_grad_()
            img1.requires_grad_()
            img2.requires_grad_()
            output1, output2, output3 = model(img0, img1, img2)

            loss = criterion(output1, output2 , output3)
            loss.backward()
            optimizer.step()
            
            if idx%25 ==0:
                print('loss:{}'.format(loss.item()))
            

            running_loss += loss.item()         
        
        running_loss /= len(train_loader)
        train_losses.append(running_loss)

        model.eval()
        running_val_loss = 0 
        val_batch = tqdm(enumerate(val_loader), total=len(val_loader))
        for idx, (val_img0, val_img1, val_img2) in val_batch:
            val_img0 = val_img0.to(device)
            val_img1 = val_img1.to(device)
            val_img2 = val_img2.to(device)
            
            with torch.no_grad():
                val_output1, val_output2, val_output3 = model(val_img0, val_img1, val_img2)
                
            val_loss = criterion(val_output1, val_output2, val_output3)
            if idx%100 ==0:
                print('val loss:{}'.format(val_loss.item()))
            
            
            running_val_loss += val_loss.item()
        
        running_val_loss /= len(val_loader)
        val_losses.append(running_val_loss)
        
        print('Epoch {} -- train loss: {:.4f} val loss: {:.4f} '.format(epoch + 1, running_loss, running_val_loss) )
        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))        
        counter.append(epoch+1)
        torch.save(model.state_dict(), os.path.join('{}/expt{}/checkpoint_{}.pt'.format(args.save_folder,args.expt,epoch))) # save only final models
    average_train_loss = np.sum(train_losses) / args.num_epochs
    average_val_loss = np.sum(val_losses) / args.num_epochs
    print('end of epoch:','average_train_loss: ',average_train_loss,'average_val_loss: ',average_val_loss)
    show_plot(counter,args.save_details_folder, train_losses, val_losses,args.expt)
    return average_train_loss, average_val_loss, train_losses, val_losses
create_dir(args)
train_data, test_data = prep_datasetV2(args)
train_data, deleted_data = check_dataset(train_data)
f= open("{}/expt{}/deleted_dataset.txt".format(args.save_details_folder,args.expt),"w+")
for i in range(len(deleted_data)):
    f.write("{}\n".format(deleted_data[i]))
f.close() 
X_train, X_val = train_test_split(train_data, test_size = args.split_val_ratio, random_state= args.seed)
average_train_loss, average_val_loss, train_losses, val_losses = train_fn(X_train,X_val, args)

def save_model_details(average_train_loss, average_val_loss, train_losses, val_losses, expt, save_details_folder):

    np.save('{}/expt{}/average_val_loss.npy'.format(save_details_folder, expt),average_val_loss)
    np.save('{}/expt{}/val_losses.npy'.format(save_details_folder, expt),np.array(val_losses))

    np.save('{}/expt{}/average_train_loss.npy'.format(save_details_folder, expt),average_train_loss)
    np.save('{}/expt{}/train_loss.npy'.format(save_details_folder,expt),np.array(train_losses))

save_model_details(average_train_loss, average_val_loss, train_losses, val_losses, args.expt, args.save_details_folder)