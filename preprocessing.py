# REFRENCES:
# For overall preprocessing: https://github.com/Mansi-khemka/Cancer-Metastases-Segmentation-in-Gigapixel-Pathology-Images/blob/master/Preprocessing%20File.ipynb
# For stain normalisation: https://github.com/schaugf/HEnorm_python 


# General packages
import os
import shutil
import glob
# import openslide
import skimage.io
import random
# import seaborn as sns
import cv2

import pandas as pd
import numpy as np

import PIL
from tqdm import tqdm

from skimage.filters import threshold_otsu
from skimage.color import rgb2gray

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import sklearn.metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold



TRAIN = '../data/train_images'
MASKS = '../data/train_label_masks'
SAVE_IMAGES = '../data/train_patch_224_norm_images_rad'
SAVE_MASKS = '../data/train_patch_224_norm_masks_rad'
train_df=pd.read_csv('../data/train.csv')
test_df=pd.read_csv('../data/test.csv')
discard_threshold=0.9
level = 0
sz = 224
N = 300


# step 1: get image patches
def read_chips(img, mask, x_required, y_required):
      
    row_iters = int(img.shape[1]//x_required) # Number of cuts in X dimension
    col_iters = int(img.shape[0]//y_required) # Number of cuts in Y dimension

    slide_images = [] # array of all the cuts of slides
    mask_images = [] #array of all the cuts of tumor masks
    x_corr=[]
    y_corr=[]
    # loop through the coordinates
    for i in range(0, row_iters):
        # coordinates wrt level 0 
        x = i*x_required 
        for j in range(0, col_iters):
            y = j*y_required

            slide_image = img[y:y+y_required,x:x+x_required]
            slide_images.append(slide_image)

            mask_image = mask[y:y+y_required,x:x+x_required]
            mask_images.append(mask_image)

            x_corr.append(x)
            y_corr.append(y)

    return slide_images, mask_images, x_corr, y_corr
        
# stain normalisation
def normalizeStaining(img, Io=240, alpha=1, beta=0.15):
    ''' Normalize staining appearence of H&E stained images
    
    Example use:
        see test.py
        
    Input:
        I: RGB input image
        Io: (optional) transmitted light intensity
        
    Output:
        Inorm: normalized image
        H: hematoxylin image
        E: eosin image
    
    Reference: 
        A method for normalizing histology slides for quantitative analysis. M.
        Macenko et al., ISBI 2009
    '''
             
    HERef = np.array([[0.5626, 0.2159],
                      [0.7201, 0.8012],
                      [0.4062, 0.5581]])
        
    maxCRef = np.array([1.9705, 1.0308])
    
    # define height and width of image
    h, w, c = img.shape
    
    # reshape image
    img = img.reshape((-1,3))

    # calculate optical density
    OD = -np.log((img.astype(np.float)+1)/Io)
    
    # remove transparent pixels
    ODhat = OD[~np.any(OD<beta, axis=1)]
        
    # compute eigenvectors
    eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
    
    #eigvecs *= -1
    
    #project on the plane spanned by the eigenvectors corresponding to the two 
    # largest eigenvalues    
    That = ODhat.dot(eigvecs[:,1:3])
    
    phi = np.arctan2(That[:,1],That[:,0])
    
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100-alpha)
    
    vMin = eigvecs[:,1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:,1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)
    
    # a heuristic to make the vector corresponding to hematoxylin first and the 
    # one corresponding to eosin second
    if vMin[0] > vMax[0]:
        HE = np.array((vMin[:,0], vMax[:,0])).T
    else:
        HE = np.array((vMax[:,0], vMin[:,0])).T
    
    # rows correspond to channels (RGB), columns to OD values
    Y = np.reshape(OD, (-1, 3)).T
    
    # determine concentrations of the individual stains
    C = np.linalg.lstsq(HE,Y, rcond=None)[0]
    
    # normalize stain concentrations
    maxC = np.array([np.percentile(C[0,:], 99), np.percentile(C[1,:],99)])
    tmp = np.divide(maxC,maxCRef)
    C2 = np.divide(C,tmp[:, np.newaxis])
    
    # recreate the image using reference mixing matrix
    Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
    Inorm[Inorm>255] = 254
    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8) 

    return Inorm

# step 2: filter out patches according to discard threshold
def find_tissue_pixels(image, intensity=0.8):
    im_gray = rgb2gray(image)
    assert im_gray.shape == (image.shape[0], image.shape[1])
    indices = np.where(im_gray <= intensity)
    return list(zip(indices[0], indices[1]))

def get_concentrated_slides(slide_images, mask_images,discard_threshold,x_corr,y_corr):
    concentrated_slides = []
    concentrated_tissue_mask= []
    final_x_corr=[]
    final_y_corr=[]
    count = 0
    
    for slide_index in range(len(slide_images)):
        slide_image = slide_images[slide_index]
        tissue_pixels = find_tissue_pixels(slide_image)
        percent_tissue = len(tissue_pixels) / float(slide_image.shape[0] * slide_image.shape[0])
        
        if(percent_tissue>=discard_threshold):
            
            slide_image = normalizeStaining(slide_image)
            concentrated_slides.append(slide_image)
            concentrated_tissue_mask.append(mask_images[slide_index])
            final_slide_index=[]
            final_x_corr.append(x_corr[slide_index])
            final_y_corr.append(y_corr[slide_index])
            count +=1
    
    return concentrated_slides, concentrated_tissue_mask, final_x_corr, final_y_corr, count

# step 3: save images
def write_to_file(data, file_path, type):
    for i in range(len(data)):
        if type=="mask":
            with open(os.path.join(file_path,str(i)+'.npy'),'wb') as f:
                np.save(f,data[i])
        else:
            cv2.imwrite(os.path.join(file_path,str(i)+".png"),cv2.cvtColor(data[i], cv2.COLOR_RGB2BGR))
    
def preprocess(slide_path,mask_path,image):
    img_height,img_width=sz,sz
    valid_file=True
    
    # # get dimensions of openslide
    try:

        # get patches
        img = skimage.io.MultiImage(slide_path)[level] # to load the last level??
        mask = skimage.io.MultiImage(mask_path)[level][:,:,0]  
        slide_images, mask_images, x_corr, y_corr = read_chips(img, mask,img_height,img_width)
        # process patches 
        
        concentrated_slides, concentrated_tissue_mask, final_x_corr, final_y_corr, count = get_concentrated_slides(slide_images, mask_images,discard_threshold,x_corr, y_corr) 
        save_folder_i= SAVE_IMAGES
        save_folder_m= SAVE_MASKS
        slide_save_folder = save_folder_i + str(image) +"_"
        mask_save_folder = save_folder_m + str(image) +"_"
        if os.path.exists(os.path.join(save_folder_i,image)):
            shutil.rmtree(os.path.join(save_folder_i,image))
        os.mkdir(os.path.join(save_folder_i,image))

        if os.path.exists(os.path.join(save_folder_m,image)):
            shutil.rmtree(os.path.join(save_folder_m,image))
        os.mkdir(os.path.join(save_folder_m,image))
        write_to_file(concentrated_slides, os.path.join(save_folder_i,image),"slide")
        write_to_file(concentrated_tissue_mask, os.path.join(save_folder_m, image), "mask")
        d = {'x_corr':final_x_corr,'y_corr':final_y_corr}
        df = pd.DataFrame(data=d)
        df.to_csv(os.path.join(save_folder_i,image,'image_coordinates.csv'))
        return valid_file
    except:
        valid_file=False
        return valid_file

def main():
    mask_names=(glob.glob(os.path.join(MASKS,'*.tiff')))
    mask_names=[os.path.basename(x) for x in mask_names]
    updated_mask_names=[name.split("_")[0] for name in mask_names]

    train_no_masks=train_df[~train_df['image_id'].isin(updated_mask_names)]
    train_masks=train_df[train_df['image_id'].isin(updated_mask_names)]

    karolinska_df=train_masks[train_masks['data_provider']=="karolinska"]
    radboud_df=train_masks[train_masks['data_provider']=="radboud"]

    karolinska_df.to_csv('../data/karolinska.csv',index=False)
    radboud_df.to_csv('../data/radboud.csv',index=False)

    #len(teacher_train_df)
    invalid_files=[]
    for i in range(300,500):
        image = radboud_df["image_id"].values[i]
        image_path=os.path.join(TRAIN,image+".tiff")
        mask_path=os.path.join(MASKS,image+"_mask.tiff")

        valid_file=preprocess(image_path,mask_path,image)
        if valid_file==False:
            invalid_files.append(image)
            
        print('image:'+image+': iteration: '+str(i))
        
    df = pd.DataFrame(data=invalid_files)
    df.to_csv(os.path.join(SAVE_IMAGES,'invalid_files.csv'))

if __name__=="__main__":
    main()