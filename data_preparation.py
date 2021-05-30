# CHEK SIMILAIRITY OF TILES BASED ON SPATIAL SIMILARITY AND SAVE

import os
import pandas as pd
import numpy as np


class Args:
    data_size = 298
    image_file_folder = '../data/train_patch_224_norm_images_rad'
    dataset_used = './storage/dataset_used_rad.txt'
    similar_dist = 448 #1792
    dissimilar_dist = 6720
args = Args()

my_file = open(args.dataset_used,"r+")  
dataset_dir = my_file.read()
dataset_dir = dataset_dir.split("\n")
my_file.close()
while "" in dataset_dir:
    dataset_dir.remove("")

def compute_euclidean(x_corr, y_corr, x_ref, y_ref):
    dist = np.sqrt((x_corr-x_ref)**2 + (y_corr - y_ref)**2)
    if dist < args.similar_dist and dist > 0:
        similar = 1
    elif dist > args.dissimilar_dist:
        similar = 2
    else: # if in between
        similar = 0
    return similar

def compute_euclidean_all(coordinates_df,index, tile_name, x_corr, y_corr, save_folder):
    
    coordinates_df['similar']= coordinates_df.apply(lambda row: compute_euclidean(row['x_corr'], row['y_corr'], x_corr, y_corr),axis =1)
    coordinates_df.to_csv('{}/similarity_testS448_{}.csv'.format(save_folder,index))           

for case in dataset_dir:
    coordinates_df = pd.read_csv(os.path.join(args.image_file_folder, case, 'image_coordinates.csv'))
    save_folder = os.path.join(args.image_file_folder, case)
    for index, row in coordinates_df.iterrows():
        tile_name = row['Unnamed: 0']
        x_corr = row['x_corr']
        y_corr = row['y_corr']
        compute_euclidean_all(coordinates_df,index, tile_name, x_corr, y_corr, save_folder)

    
