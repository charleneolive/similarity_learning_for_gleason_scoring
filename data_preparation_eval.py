from PIL import Image
import matplotlib.pyplot as plt
import PIL.ImageOps
import numpy as np
import glob
import os

class Args:
    image_file_folder = '../data/train_patch_224_norm_images_rad'
    mask_file_folder = '../data/train_patch_224_norm_masks_rad'
    dataset_used = './storage/dataset_used_rad.txt'
args = Args()

my_file = open(args.dataset_used,"r+")  
dataset_dir = my_file.read()
dataset_dir = dataset_dir.split("\n")
my_file.close()
while "" in dataset_dir:
    dataset_dir.remove("")
def find_mask_label (main_folder, case, tile_name): # LABEL BY MOST FREQUENT

    mask_path = os.path.join(main_folder, case, tile_name+'.npy')
    # get majority label
    with open(mask_path,'rb') as f:
        mask = np.load(f)
    mask_flat = mask.flatten()
    mask_count = np.bincount(mask_flat)
    most_freq = np.argmax(mask_count)
    if most_freq in [0, 1, 2]: # group to non-cancerous
        most_freq = 0 
    elif most_freq in [3, 4, 5]: # group to cancerous
        most_freq = 1
    return most_freq

def find_mask_label2 (main_folder, case, tile_name): # LABEL BY MOST SEVERE AND AT LEAST 5%

    mask_path = os.path.join(main_folder, case, tile_name+'.npy')
    # get majority label
    with open(mask_path,'rb') as f:
        mask = np.load(f)
    mask_flat = mask.flatten()
    mask_count = np.bincount(mask_flat)
    total_count = np.sum(mask_count)
    percent_count = mask_count/total_count> 0.05
    true_indices = [i for i,x in enumerate(percent_count) if x]
    max_true = np.max(true_indices)
#     most_freq = np.argmax(mask_count)
    if max_true in [0, 1, 2]: # group to non-cancerous
        max_true = 0
    elif max_true in [3, 4, 5]: # group to cancerous
        max_true = 1
    return max_true

for case in dataset_dir:
    train_data_list=glob.glob(os.path.join(args.mask_file_folder, case,'*.npy'))
    for tile in train_data_list:
        tile_name = os.path.splitext(os.path.basename(tile))[0]
        mask_label = find_mask_label (args.mask_file_folder, case, tile_name)
        np.save("{}/{}/masklabV3_{}.npy".format(args.image_file_folder, case, tile_name), mask_label)
        mask_label2 = find_mask_label2 (args.mask_file_folder, case, tile_name)
        np.save("{}/{}/masklabV4_{}.npy".format(args.image_file_folder, case, tile_name), mask_label2)