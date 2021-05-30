
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
import sklearn.cluster
import sklearn.metrics.cluster
from sklearn.model_selection import train_test_split
from DensenetSiamese import DensenetSiameseNetwork, DensenetSiameseNetworkTriplet

import torch
# import torchvision
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.data import DataLoader,Dataset
from torch.autograd import Variable

os.environ["CUDA_VISIBLE_DEVICES"]="2"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Args:
    image_file_folder="../data/train_patch_224_norm_images_rad"
    save_folder="./model_details"
    models_folder='./models'
    checkpoint='checkpoint_16.pt'
    test_dataset="test_dataset.txt"
    expt = "21"
    version = "V4" #version of mask label
    bs = 32
    num_classes = 2 # choose 2 or 6 
    loss = "Contrastive" # choose Triplet or Contrastive
    dataframe_path = '../data/radboud.csv'

args = Args()
if args.loss == "Triplet":
    model = DensenetSiameseNetworkTriplet()
elif args.loss == "Contrastive":
    model = DensenetSiameseNetwork()


from losses import ContrastiveLoss
from sklearn.metrics.cluster import normalized_mutual_info_score
from utils import imshow, prep_dataset, TestDataset, image_retrieval


checkpoint_path = os.path.join(args.models_folder,'expt'+args.expt, args.checkpoint)
test_dataset_path = os.path.join(args.save_folder,'expt'+args.expt, args.test_dataset)

model = model.to(device)
model.load_state_dict(torch.load(checkpoint_path))
my_file = open(test_dataset_path,"r+")  
test_dir = my_file.read()
test_dir = test_dir.split("\n")
my_file.close()
while "" in test_dir:
    test_dir.remove("")

if not os.path.exists('./{}/expt{}/{}/{}'.format(args.save_folder,args.expt,os.path.splitext(args.checkpoint)[0], args.version)):
    os.makedirs('./{}/expt{}/{}/{}'.format(args.save_folder,args.expt,os.path.splitext(args.checkpoint)[0], args.version))

test_data_list=[glob.glob(os.path.join(args.image_file_folder,directory,'*.png')) for directory in test_dir]
test_data=[item for sublist in test_data_list for item in sublist]
image_list = [os.path.basename(os.path.dirname(i)) for i in test_data]
patch_list = [os.path.basename(i) for i in test_data]
test_dataset=TestDataset(test_data,args.image_file_folder, args.dataframe_path, args.version)
test_loader=DataLoader(test_dataset,batch_size=args.bs)



#%%Evaluation

def cluster_by_kmeans(X, nb_clusters):
    """
    xs : embeddings with shape [nb_samples, nb_features]
    nb_clusters : in this case, must be equal to number of classes
    """
    return sklearn.cluster.KMeans(nb_clusters).fit(X).labels_

def calc_normalized_mutual_information(ys, xs_clustered):
    return sklearn.metrics.cluster.normalized_mutual_info_score(xs_clustered, ys)

def find_recall(X,T,num_classes, Kset):
    distances = torch.cdist(X,X)
    k = int(np.max(Kset))
    
    indices = distances.topk(k + 1, largest=False)[1][:, 1: k + 1]
    Y =  np.array([[T[i] for i in ii] for ii in indices])
    Y = torch.from_numpy(Y)
    recall = []
    recall_list = [[] for _ in range(num_classes)]
    
    for k in [1, 2, 4, 8]:
        # t is the actual label, y are its neighbours
        s = sum([1 for t, y in zip(T, Y) if t in y[:k]])
        for one_class in range(num_classes):
            recall_list[one_class].append(sum([1 for t, y in zip(T, Y) if t in y[:k] and t == one_class]) / (T == one_class).sum().numpy())
        r_at_k = s / (1. * len(T))
        recall.append(r_at_k)

    return recall, recall_list

#%% Test Mode
testdata = torch.Tensor()
testlabel = torch.LongTensor()
similar_distance = torch.Tensor()
dissimilar_distance = torch.Tensor()
similar_pairs = torch.Tensor()
dissimilar_pairs = torch.Tensor()
all_isup_grade = torch.LongTensor()

test_batch = tqdm(enumerate(test_loader), total=len(test_loader))

model.eval()
for i, (img0, img1, label, label0, label1, current_grade) in test_batch: #label0 and label1 are the actual labels
    img0 = img0.to(device)
    img1 = img1.to(device)
    
    with torch.no_grad():
        if args.loss == "Triplet":
            output1, output2, output3 = model(img0, img1, img0) # only 2 embeddings are required, the first one because it follows an order, so it is repeated
        elif args.loss == "Contrastive":
            output1, output2 = model(img0, img1) # only 2 embeddings are required, the first one because it follows an order, so it is repeated

    euclidean_distance = F.pairwise_distance(output1.cpu(), output2.cpu())
    dissimilar_distance_batch = label.view(-1) * euclidean_distance
    temp = label.clone()
    temp[label!=0]=0
    temp[label==0]=1
    similar_distance_batch = temp.view(-1) * euclidean_distance 
    similar_distance = torch.cat((similar_distance, similar_distance_batch))
    dissimilar_distance = torch.cat((dissimilar_distance, dissimilar_distance_batch))
    similar_pairs = torch.cat((similar_pairs, temp.view(-1)))
    dissimilar_pairs = torch.cat((dissimilar_pairs, label.view(-1)))
    all_isup_grade = torch.cat((all_isup_grade, current_grade))

    testdata = torch.cat((testdata, output1.cpu()), 0)
    testlabel = torch.cat((testlabel, label0))
    
recall, recall_list = find_recall(testdata, testlabel, args.num_classes, [1, 2, 4, 8])

if args.num_classes == 2:
    recall_names = ['RecallK0', 'RecallK1']
elif args.num_classes == 6:
    recall_names = ['RecallK0', 'RecallK1', 'RecallK2', 'RecallK3', 'RecallK4', 'RecallK5']

nmi = calc_normalized_mutual_information(
            testlabel,
            cluster_by_kmeans(
                testdata, args.num_classes
            )
        )
L2_similar = np.sum(similar_distance.numpy())/np.sum(similar_pairs.numpy())
L2_dissimilar = np.sum(dissimilar_distance.numpy())/np.sum(dissimilar_pairs.numpy())
print('Recall@1, 2, 4, 8: {recall[0]:.3f}, {recall[1]:.3f}, {recall[2]:.3f}, {recall[3]:.3f}; NMI: {nmi:.3f} \n'
                  .format(recall=recall, nmi=nmi))
print('L2_similar:{}, L2_dissimilar:{}'.format(L2_similar, L2_dissimilar))
for idx, x in enumerate(recall_list):
    print('{}@1, 2, 4, 8: {recall[0]:.3f}, {recall[1]:.3f}, {recall[2]:.3f}, {recall[3]:.3f}'
                .format(recall_names[idx], recall=x))

    np.save('{}/expt{}/{}/{}/{}.npy'.format(args.save_folder,args.expt, os.path.splitext(args.checkpoint)[0], args.version, recall_names[idx]), x)
np.save('{}/expt{}/{}/{}/nmi.npy'.format(args.save_folder,args.expt,os.path.splitext(args.checkpoint)[0], args.version), nmi)
np.save('{}/expt{}/{}/{}/L2_similar.npy'.format(args.save_folder,args.expt,os.path.splitext(args.checkpoint)[0], args.version), L2_similar)
np.save('{}/expt{}/{}/{}/L2_dissimilar.npy'.format(args.save_folder,args.expt,os.path.splitext(args.checkpoint)[0], args.version),L2_dissimilar)
np.save('{}/expt{}/{}/{}/test_label.npy'.format(args.save_folder,args.expt,os.path.splitext(args.checkpoint)[0], args.version), testlabel.numpy()) #label depends on Ver

if not os.path.exists('{}/expt{}/{}/test_data.npy'.format(args.save_folder,args.expt,os.path.splitext(args.checkpoint)[0])):
    np.save('{}/expt{}/{}/test_data.npy'.format(args.save_folder,args.expt,os.path.splitext(args.checkpoint)[0]), testdata.numpy()) #depends on checkpoint
    np.save('{}/expt{}/{}/isup_grades.npy'.format(args.save_folder,args.expt,os.path.splitext(args.checkpoint)[0]), all_isup_grade.numpy()) #fixed
    np.save('{}/expt{}/{}/image_list.npy'.format(args.save_folder,args.expt,os.path.splitext(args.checkpoint)[0]), np.array(image_list)) #fixed
    np.save('{}/expt{}/{}/patch_list.npy'.format(args.save_folder,args.expt,os.path.splitext(args.checkpoint)[0]), np.array(patch_list)) #fixed