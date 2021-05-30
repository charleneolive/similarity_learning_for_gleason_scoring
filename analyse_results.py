import os
import pandas as pd
import numpy as np
import plotly
import plotly.express as px
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from scipy import stats
import statistics 
from scipy.stats import chi2

class Args:
    model_folders = ['./model_details/expt19/checkpoint_6',
                     './model_details/expt19/checkpoint_6',
                     './model_details/expt19/checkpoint_6',
                     './model_details/expt19/checkpoint_6',
                     './model_details/expt20/checkpoint_18',
                     './model_details/expt20/checkpoint_18', 
                     './model_details/expt20/checkpoint_18',
                     './model_details/expt20/checkpoint_18',
                     './model_details/expt21/checkpoint_16',
                     './model_details/expt21/checkpoint_16',
                     './model_details/expt21/checkpoint_16',
                     './model_details/expt21/checkpoint_16',
                     
                    ]
    ver = ['V1','V2','V3','V4','V1','V2','V3','V4','V1','V2','V3','V4']
    
args = Args()

for idx,folder in enumerate(args.model_folders):
    test_label_path = os.path.join(folder, args.ver[idx], 'test_label.npy')
    test_data_path = os.path.join(folder, 'test_data.npy')
    isup_grade_path = os.path.join(folder, 'isup_grades.npy')
    image_list_path = os.path.join(folder, 'image_list.npy')
    patch_list_path = os.path.join(folder, 'patch_list.npy') #names of the patches
    save_folder = os.path.join(folder, args.ver[idx], "plots")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)


    test_label=np.load(test_label_path)
    test_data=np.load(test_data_path)
    isup_grade=np.load(isup_grade_path)
    image_list=np.load(image_list_path)
    patch_list=np.load(patch_list_path)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(test_data) # never normalise because around same range
    PCA_components = pd.DataFrame(principalComponents)
    im = ax.scatter3D(PCA_components[0], PCA_components[1], PCA_components[2], alpha=.1, c=test_label)
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_zlabel('PCA 3')

    fig.colorbar(im, ax = ax)
    fig.savefig("{}/3D_scatter_tiles_patchlabels.jpg".format(save_folder), dpi = 300)

    PCA_components['grade'] = isup_grade
    PCA_components['slide_label'] = test_label
    PCA_components['image_names'] = image_list
    PCA_components['patch_names'] = patch_list

#%% exploration on patch-based labels
    # individual gleason score
    fig = px.scatter_3d(PCA_components, x =0, y = 1, z = 2, color="slide_label", hover_name = "image_names", hover_data = ["patch_names"])
    fig.update_layout(scene = dict(
                        xaxis_title="PCA 1",
                        yaxis_title="PCA 2",
                        zaxis_title="PCA 3"))

    fig.write_html("{}/3D_scatter_tiles_patchlabels.html".format(save_folder))

    # these are isup grades (slide based) of the tiles 
    fig = px.scatter_3d(PCA_components, x =0, y = 1, z = 2, color="grade", hover_name = "image_names", hover_data = ["patch_names"])
    fig.update_layout(scene = dict(
                        xaxis_title="PCA 1",
                        yaxis_title="PCA 2",
                        zaxis_title="PCA 3"))

    fig.write_html("{}/3D_scatter_tiles_isup.html".format(save_folder))

#%% exploration on slides
    all_image_names = np.unique(image_list)
    all_rep = np.zeros((len(all_image_names), 128))
    all_grades = np.zeros((len(all_image_names)))
    image_names = []

    for idx,image_name in enumerate(all_image_names):
        slide_position = image_list == image_name
        one_slide = test_data[slide_position]
        one_slide_grade = isup_grade[slide_position]
        one_slide_label = test_label[slide_position] # 0 - 2: non-cancerous tissue, 3-5: cancerous tissue
        all_grades[idx] = one_slide_grade[0] # grade for 1 slide should be the same regardless of label
        all_rep[idx,:] = np.mean(one_slide) # take mean of all the vectors 
        image_names.append(image_name)

    scaler = preprocessing.StandardScaler().fit(all_rep) # do standard normalisation on representations
    all_rep_scale = scaler.transform(all_rep)
    pca = PCA(n_components=3)
    principalComponents_oneslide = pca.fit_transform(all_rep_scale)
    PCA_components_slide = pd.DataFrame(principalComponents_oneslide)
    PCA_components_slide["grade"] = all_grades
    PCA_components_slide["image_names"] = image_names

    # let's check out the 3D scatter plot
    fig = px.scatter_3d(PCA_components_slide, x =0, y = 1, z = 2,color="grade", hover_name = "image_names")
    fig.update_layout(scene = dict(
                        xaxis_title="PCA 1",
                        yaxis_title="PCA 2",
                        zaxis_title="PCA 3"))
    fig.write_html("{}/3D_scatter_slide_isup.html".format(save_folder))

    # looks like we are seeing a pattern in 1st PCA component, let's explore that further
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=principalComponents_oneslide[:,0], y=np.zeros((principalComponents_oneslide.shape[0])), mode='markers', marker = dict(size=10, color = all_grades)
    ))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False, 
                    zeroline=True, zerolinecolor='black', zerolinewidth=3,
                    showticklabels=False)
    fig.update_layout(height=200, plot_bgcolor='white')
    
    fig.write_html("{}/1D_scatter_slide_isup.html".format(save_folder))

    PCA_components_slide = PCA_components_slide.sort_values(by=['grade'])
    fig = px.histogram(PCA_components_slide, x=0, y="grade", color="grade", marginal="rug",
                    hover_data=PCA_components_slide.columns)
    fig.update_layout(scene = dict(
                        xaxis_title="1st PCA Component of Mean Embedding(Normalised)",
                        yaxis_title="Count"))
    fig.write_html("{}/distplot_slide_isup.html".format(save_folder))

#%% STATISTICAL TESTING FOR 1ST PCA COMPONENT
    # 1. check for normality (Shapiro - Wilk Test)
    # 2. check for equal variances (Barlett's test)
    # 3. Welch's t-test - does not assume equal sample size, or equal variance


    # perform the shapiro -wilk test for normality for the 1st PCA component
    # Cannot reject null hypothesis that the data is drawn from a normal distribution

    alpha = 0.05
    deviate_normality = []
    grade_list = [ int(i) for i in PCA_components_slide['grade'].unique().tolist()]
    for grade in grade_list:
        _ , p_value = stats.shapiro(PCA_components_slide[PCA_components_slide['grade']==grade][0])
        if p_value < alpha:
            deviate_normality.append(grade)
    if len(deviate_normality)==0:
        print("Cannot reject null hypothesis that data is drawn from normal distribution")
    print(deviate_normality)
    for element in deviate_normality:
        grade_list.remove(element)
    # now use Barlett's test to test null hypothesis that the variances are equal for all samples - not necessary actually
#     k = PCA_components_slide['grade'].nunique()
#     n = len(PCA_components_slide)
#     group_n = np.array(PCA_components_slide.groupby(['grade']).count()[0].tolist())
#     group_var = np.array([statistics.variance(PCA_components_slide[PCA_components_slide["grade"]==i][0].tolist()) for i in grade_list])

#     pool_var = 1 / (n - k) * np.sum((group_n - 1) * group_var)
#     x2_num = (n - k) * np.log(pool_var) - np.sum((group_n - 1) * np.log(group_var))
#     x2_den = 1 + 1 / (3 * (k - 1)) * (np.sum(1 / (group_n - 1)) - 1 / (n - k))
#     x2 = x2_num / x2_den
#     p = 1 - chi2.cdf(x2, k - 1)
#     if p < alpha:
#         print("Can reject null hypothesis that variances are equal for all samples")

    # p is less than 0.05. Hence we reject the null hypothesis that the group variances are equal

    # run statistical test to see which pair of 
    statistical_matrix = np.ones((len(grade_list), len(grade_list)))
    for idx1, grade1 in enumerate(grade_list):
        for idx2, grade2 in enumerate(grade_list):
            if grade2 != grade1:
                _, p_value = stats.ttest_ind(PCA_components_slide[PCA_components_slide['grade']==grade1][0], PCA_components_slide[PCA_components_slide['grade']==grade2][0], equal_var = False)
                statistical_matrix[idx1, idx2] = p_value
            else:
                continue

    significant_matrix_5 = statistical_matrix < alpha
    significant_matrix_10 = statistical_matrix < alpha*2
    fig, ax = plt.subplots( nrows=1, ncols=1 )
    plt.imshow(significant_matrix_5, cmap='hot', interpolation='nearest')
    plt.xticks(range(len(grade_list)), grade_list, fontsize=12)
    plt.yticks(range(len(grade_list)), grade_list, fontsize=12)
    plt.title("Heatmap Comparing Means of Samples of \n Different Grades using Welch's T-Test (alpha=0.05)")
    fig.savefig("{}/welch_test_resuls_alpha05.jpg".format(save_folder), dpi = 300)
    plt.close(fig)

    fig, ax = plt.subplots( nrows=1, ncols=1 )
    plt.imshow(significant_matrix_10, cmap='hot', interpolation='nearest')
    plt.xticks(range(len(grade_list)), grade_list, fontsize=12)
    plt.yticks(range(len(grade_list)), grade_list, fontsize=12)
    plt.title("Heatmap Comparing Means of Samples of \n Different Grades using Welch's T-Test (alpha=0.10)")
    fig.savefig("{}/welch_test_resuls_alpha1.jpg".format(save_folder), dpi = 300)
    plt.close(fig)