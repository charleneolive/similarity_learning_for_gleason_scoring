# Similarity Learning: The Key to Unsupervised Gleason Scoring

## Report
View the pdf file regarding the details of this project. This project is inspired from this paper: https://arxiv.org/abs/1905.08139. 

## Code
<code> preprocessing.py </code> : Take in  WSIs, partition into patches, do background removal and stain normalisation
<code> data_preparation.py </code>: Check on similarity of tiles based on spatial similarity. For each WSI, tiles located within 1792 pixels are considered similar, tiles located more than 6720 pixels away are considered different 
<code> data_preparation_eval.py </code>: Prepare masks for model evaluation (four different mask variants used)
<code> train.py </code>: Training model for siamese neural network, contrastive loss. Choose weakly or unsupervised 
<code> trainTriplet.py </code> Training model for siamese neural network, triplet loss
<code> losses.py </code> Types of losses (Contrastive and Triplet)
<code> DensenetSiamese.py </code> Model used 
<code> utils.py </code> Functions and classes such as pytorch dataset, plotting graphs, loading datasets 
<code> analyse_results.py </code> Analysis of embeddings (scatter plots) and welch’s test 
<code> plot_visualisation.py </code> Overlay images and mask and save 

## References
- Overall preprocessing: https://github.com/Mansi-khemka/Cancer-Metastases-Segmentation-in-Gigapixel-Pathology-Images/blob/master/Preprocessing%20File.ipynb
- Stain normalisation: https://github.com/schaugf/HEnorm_python 
- Plot_visualisation: https://www.kaggle.com/wouterbulten/getting-started-with-the-panda-dataset
- Recall at K & NMI: https://github.com/dichotomies/proxy-nca
    
## Data Source
I got the data from https://www.kaggle.com/c/prostate-cancer-grade-assessment, if you would like to obtain the data, please get from there directly. 

## Folder description
1. data: data, raw and preprocessed, used in project
2. code/model_details: all the results from model training and model evaluation, as well as visualisation of results (scatterplots etc) for the four label variants
3. code/models: model checkpoints
4. code/storage: Miscellaneous information, such as mask label distribution and dataset used 

All code in code.

## Experimental Evaluation

1. Different Models Tested
- expt19: Unsupervised Siamese Network, Triplet Loss (trained for 20 epochs, picked checkpoint 6 - epoch 7)
- expt20: Unsupervised Siamese Network, Contrastive Loss (trained for 20 epochs, picked checkpoint 18 - epoch 19)
- expt21: Weakly supervised Siamese Network, Contrastive Loss (trained for 17 epochs, picked checkpoint 16 - epoch 17)

2. Different Labels
- V1: Labels according to the most frequent pattern (distinguishing all 6 classes) 
- V2: Labels according to the highest Gleason score present occupying at least 5% of the image tile (distinguishing all 6 classes) 
- V3: Labels according to the most frequent pattern (distinguishing just non-cancerous vs cancerous) 
- V4: Labels according to the highest Gleason score present occupying at least 5% of the image tile (distinguishing just cancerous vs non-cancerous) 