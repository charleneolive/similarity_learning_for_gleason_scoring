import os
import glob
import random
from random import sample
import numpy as np
import pandas as pd
from PIL import Image
import PIL.ImageOps
import matplotlib.pyplot as plt
from matplotlib import gridspec

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
from torch.utils.data import DataLoader,Dataset
from sklearn.model_selection import train_test_split

#%% DATA PREPARATION



def prep_dataset(args):
    list_of_cases= [directory for root,dirs,files in os.walk(args.image_file_folder) for directory in dirs]
    list_of_cases = list_of_cases[:args.data_size]
    training_dataset, test_dataset = train_test_split(list_of_cases,test_size=args.split_ratio, random_state=args.seed)

    f= open("{}/expt{}/training_dataset.txt".format(args.save_details_folder,args.expt),"w+")
    for i in range(len(training_dataset)):
        f.write("{}\n".format(training_dataset[i]))
    f.close() 

    f= open("{}/expt{}/test_dataset.txt".format(args.save_details_folder,args.expt),"w+")
    for i in range(len(test_dataset)):
        f.write("{}\n".format(test_dataset[i]))
    f.close() 

    train_data_list=[glob.glob(os.path.join(args.image_file_folder,directory,'*.png')) for directory in training_dataset]
    train_data=[item for sublist in train_data_list for item in sublist]

    test_data_list=[glob.glob(os.path.join(args.image_file_folder,directory,'*.png')) for directory in test_dataset]
    test_data=[item for sublist in test_data_list for item in sublist]

    return train_data, test_data

def prep_datasetV2(args):
    my_file = open(args.dataset_used,"r+")  
    dataset_dir = my_file.read()
    dataset_dir = dataset_dir.split("\n")
    my_file.close()
    while "" in dataset_dir:
        dataset_dir.remove("")
    
    training_dataset, test_dataset = train_test_split(dataset_dir,test_size=args.split_ratio, random_state=args.seed)

    f= open("{}/expt{}/training_dataset.txt".format(args.save_details_folder,args.expt),"w+")
    for i in range(len(training_dataset)):
        f.write("{}\n".format(training_dataset[i]))
    f.close() 

    f= open("{}/expt{}/test_dataset.txt".format(args.save_details_folder,args.expt),"w+")
    for i in range(len(test_dataset)):
        f.write("{}\n".format(test_dataset[i]))
    f.close() 

    train_data_list=[glob.glob(os.path.join(args.image_file_folder,directory,'*.png')) for directory in training_dataset]
    train_data=[item for sublist in train_data_list for item in sublist]

    test_data_list=[glob.glob(os.path.join(args.image_file_folder,directory,'*.png')) for directory in test_dataset]
    test_data=[item for sublist in test_data_list for item in sublist]

    return train_data, test_data

def check_dataset(dataset):
    new_dataset = []
    deleted_data = []
    for data in dataset:
        slide_name = os.path.dirname(data)
        tile_name = os.path.splitext(os.path.basename(data))[0]
        similarity_csv = pd.read_csv('{}/similarity_test_{}.csv'.format(slide_name, tile_name))
        if len(similarity_csv[similarity_csv["similar"]==1])>0 and len(similarity_csv[similarity_csv["similar"]==2]) >0:
            new_dataset.append(data)
        else:
            deleted_data.append(data)

    return new_dataset, deleted_data

#%% Pytorch train dataset 3 - for contrastive loss
'''this should be done after checking dataset (def check_dataset) to make sure that there are negative and positive examples'''
'''similar to V2 but corrected to make sure that there are both positive & negative samples'''

class TrainDatasetV3(Dataset):
    def __init__(self, image_list, image_file_path, transform=None):
        self.image_file_path=image_file_path
        self.image_list=image_list
        self.transform=transform
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self,idx):
        img0_tuple = self.image_list[idx]
        
        img0 = PIL.Image.open(img0_tuple)
        slide_filename = os.path.basename(os.path.dirname(img0_tuple))
        img0_filename = os.path.splitext(os.path.basename(img0_tuple))[0]
        
        # check if similar/ dissimilar tiles exist
        df = pd.read_csv(os.path.join(self.image_file_path, slide_filename,'similarity_test_'+str(img0_filename)+'.csv'))
        similar_df = df[df['similar']==1]
        dissimilar_df = df[df['similar']==2]
        get_same_class = random.randint(0,1)

        if get_same_class==1:
            sample = similar_df.sample()
            idx = int(sample['Unnamed: 0'])
            img1 = PIL.Image.open(os.path.join(self.image_file_path, slide_filename, str(idx)+'.png'))      
                
        elif get_same_class==0:
            sample = dissimilar_df.sample()
            idx = int(sample['Unnamed: 0'])
            img1 = PIL.Image.open(os.path.join(self.image_file_path, slide_filename, str(idx)+'.png'))
        
        if self.transform:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        # img0 = TF.to_tensor(img0) # this will scale data to [0,1], and transpose
        # img1 = TF.to_tensor(img1) 
    
        return img0, img1, torch.from_numpy(np.array([int(get_same_class==0)],dtype=np.float32))

#%% Pytorch train dataset - for contrastive loss
'''this should be done after checking dataset (def check_dataset) to make sure that there are negative and positive examples'''
'''Utilise weak labels to pick tiles from images with different isup score'''

class TrainDatasetV4(Dataset):
    def __init__(self, image_list, image_file_path, dataframe_path, transform=None):
        self.image_file_path=image_file_path
        self.image_list=image_list
        self.dataframe_path = dataframe_path
        self.transform=transform
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self,idx): 
        img0_tuple = self.image_list[idx]
        
        img0 = PIL.Image.open(img0_tuple)
        slide_filename = os.path.basename(os.path.dirname(img0_tuple))
        img0_filename = os.path.splitext(os.path.basename(img0_tuple))[0]
        
        # check if similar/ dissimilar tiles exist
        df = pd.read_csv(os.path.join(self.image_file_path, slide_filename,'similarity_test_'+str(img0_filename)+'.csv'))
        
        get_same_class = random.randint(0,1)
        
        if get_same_class: # get nearby location
            similar_df = df[df['similar']==1]
            one_sample = similar_df.sample()
            idx = int(one_sample['Unnamed: 0'])
            img1 = PIL.Image.open(os.path.join(self.image_file_path, slide_filename, str(idx)+'.png'))      
                
        else: # get a tile from another slide with a different grade
            dataframe = pd.read_csv(self.dataframe_path)
            current_grade = dataframe[dataframe['image_id']==slide_filename]['isup_grade']
            diff = dataframe[dataframe['isup_grade']!=int(current_grade)]['image_id'].values.tolist()
            random_list = []
            while len(random_list)==0:
                case = sample(diff,1)[0]
                random_list=glob.glob(os.path.join(self.image_file_path,case,'*.png'))
            another_img_path = random.choice(random_list)
            img1 = PIL.Image.open(another_img_path)


        if self.transform:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
    
        return img0, img1, torch.from_numpy(np.array([int(get_same_class==0)],dtype=np.float32))

#%% Pytorch train dataset 5 - for triplet loss
'''this should be done after checking dataset (def check_dataset) to make sure that there are negative and positive examples'''

class TrainDatasetV5(Dataset):
    def __init__(self, image_list, image_file_path, transform=None):
        self.image_file_path=image_file_path
        self.image_list=image_list
        self.transform=transform
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self,idx):
        img0_tuple = self.image_list[idx]
        
        img0 = PIL.Image.open(img0_tuple)
        slide_filename = os.path.basename(os.path.dirname(img0_tuple))
        img0_filename = os.path.splitext(os.path.basename(img0_tuple))[0]
        
        # check if similar/ dissimilar tiles exist
        df = pd.read_csv(os.path.join(self.image_file_path, slide_filename,'similarity_test_'+str(img0_filename)+'.csv'))
        similar_df = df[df['similar']==1]
        dissimilar_df = df[df['similar']==2]

        similar_sample = similar_df.sample()
        similar_idx = int(similar_sample['Unnamed: 0'])
        img1 = PIL.Image.open(os.path.join(self.image_file_path, slide_filename, str(similar_idx)+'.png'))      
                
        dissimilar_sample = dissimilar_df.sample()
        dissimilar_idx = int(dissimilar_sample['Unnamed: 0'])
        img2 = PIL.Image.open(os.path.join(self.image_file_path, slide_filename, str(dissimilar_idx)+'.png'))
        if self.transform:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        # img0 = TF.to_tensor(img0) # this will scale data to [0,1], and transpose
        # img1 = TF.to_tensor(img1) 
        # img2 = TF.to_tensor(img2)
    
        return img0, img1, img2
#%% Pytorch Test Dataset
''' generate similar and non-similar tiles by pairing tiles with the same ground truth annotation together'''
class TestDataset(Dataset):
    def __init__(self, image_list, image_file_path, dataframe_path, version, transform=None):
        self.image_file_path=image_file_path
        self.image_list=image_list
        self.version = version
        self.dataframe_path = dataframe_path
        
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self,idx):
        img0_tuple = self.image_list[idx]
        
        img0 = PIL.Image.open(img0_tuple)
        slide_name0 = os.path.basename(os.path.dirname(img0_tuple))
        img0_filename = os.path.splitext(os.path.basename(img0_tuple))[0]
        dataframe = pd.read_csv(self.dataframe_path)
        current_grade = dataframe[dataframe['image_id']==slide_name0]['isup_grade']

        if self.version == "V1":
            self.version = ""
        mask_label=np.load("{}/{}/masklab{}_{}.npy".format(self.image_file_path, slide_name0, self.version, img0_filename))
#         img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
#         img0 = img0/255
#         img0 = TF.to_pil_image(np.float32(img0))
        get_same_class = random.randint(0,1)
        if get_same_class:
            while True:
                # keep looping till a patch with same label (further away in same WSI) is found
                img1_tuple = random.choice(self.image_list) 
                img1_filename = os.path.splitext(os.path.basename(img1_tuple))[0]
                slide_name1 = os.path.basename(os.path.dirname(img1_tuple))
                mask_label1=np.load("{}/{}/masklab{}_{}.npy".format(self.image_file_path, slide_name1, self.version, img1_filename))
                if mask_label1==mask_label and img1_tuple!= img0_tuple: # make sure that it is not the same image 
                    img1 = PIL.Image.open(img1_tuple)
                    break                  
                
        else:
            while True:
                # keep looping till a patch with a different label is found
                img1_tuple = random.choice(self.image_list) 
                img1_filename = os.path.splitext(os.path.basename(img1_tuple))[0]
                slide_name1 = os.path.basename(os.path.dirname(img1_tuple))
                mask_label1=np.load("{}/{}/masklab{}_{}.npy".format(self.image_file_path, slide_name1, self.version, img1_filename))
                if mask_label1!=mask_label: 
                    img1 = PIL.Image.open(img1_tuple)
                    break   
        img0 = TF.to_tensor(img0) # this will scale data to [0,1], and transpose
    
        img1 = TF.to_tensor(img1) 
    
        return img0, img1, np.array([int(get_same_class==0)],dtype=np.float32), mask_label, mask_label1, int(current_grade)


#%% DISPLAY

def imshow(img,save_folder,text, iter):
    npimg = img.numpy()
    fig, ax = plt.subplots( nrows=1, ncols=1 )
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    fig.savefig(os.path.join(save_folder,'visualise_results_{}.png'.format(iter)))  
    plt.close(fig)

def show_plot(iteration,save_folder, train_losses, val_losses,expt):
    fig, ax = plt.subplots( nrows=1, ncols=1 )
    plt.plot(train_losses, label="train losses")
    plt.plot(val_losses, label="val losses")
    plt.legend()
    fig.savefig('{}/expt{}/running_loss.png'.format(save_folder,expt))
    plt.close(fig)

def image_retrieval(rank, labels, data, ref_image, ref_label, save_folder,expt):
    rank = np.array(rank)
    fig = plt.figure()
    gs = gridspec.GridSpec(1, len(rank)+1)
    ax = fig.add_subplot(gs[0,0])
    ax.imshow(np.transpose(np.squeeze(ref_image),(1,2,0)))
    ax.text(1,1,str(ref_label)+ " ref",bbox=dict(facecolor='white', alpha=0.5))
    ax.axis('off')
    for i in range(len(rank)):
        ax = fig.add_subplot(gs[0,i+1])
        retrieved_img_path = data[rank[i]]
        retrieved_img = PIL.Image.open(retrieved_img_path)
        ax.imshow(retrieved_img)
        ax.text(1,1,labels[i],bbox=dict(facecolor='white', alpha=0.5))
        ax.axis('off')

    fig.savefig('{}/expt{}/image_retrieval_{}.png'.format(save_folder,expt, len(rank)))
    plt.close(fig)
