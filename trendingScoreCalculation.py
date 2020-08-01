import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# function to divide given feature into bins such that each bin has equal number of features
# Store each bin in a dictionary along with the number of trending and nontrending videos in each bin
def makeBins(feature):
    dataFrame[feature+"_bin"] = (pd.cut(dataFrame[feature], bins=10))
    bin_dict = {}
    for i in range(dataFrame.shape[0]):
        if dataFrame[feature+"_bin"][i] not in bin_dict:
            bin_dict.update({dataFrame[feature+"_bin"][i]:[0,0]})
        if dataFrame["performance"][i] == 1:
            bin_dict[dataFrame[feature+"_bin"][i]][0] += 1
        else:
            bin_dict[dataFrame[feature+"_bin"][i]][1] += 1
    return (bin_dict)

# function to calculate WoE score for given bin
# formula for WoE score = log(number of trending videos/number of non trending videos)*100
def calculateWoE(bin_dataFrame,bin,bin_dict):
    bin_dataFrame[bin] = bin_dict.keys()
    WoE = []
    for key in bin_dict:
        print (bin_dict[key][1])
        if bin_dict[key][1]>=0 and bin_dict[key][1]<1:
            bin_dict[key][1] = 0.1
        if bin_dict[key][0]>=0 and bin_dict[key][0]<1:
            bin_dict[key][0] = 0.1
        WoE_val = (np.log(bin_dict[key][0]/bin_dict[key][1])*100)
        WoE.append(WoE_val)
        bin_dict[key] = WoE_val
    return (WoE,bin_dict)

# main

# read the merged dataset
dataFrame = pd.read_csv(r'youtube-new/dataset.csv', index_col=None, header=0)

likes_dict = {}
dislikes_dict = {}
views_dict = {}
comment_count_dict = {}

# divide the features likes, dislikes, views and comment_count into bins
bin_data = pd.DataFrame()
likes_dict = makeBins("likes")
dislikes_dict = makeBins("dislikes")
views_dict = makeBins("views")
comment_count_dict = makeBins("comment_count")

# calculate WoE score for each bin
bin_dataFrame  = pd.DataFrame()
bin_dataFrame["likes_WoE"],likes_dict = calculateWoE(bin_dataFrame,"likes_bin",likes_dict)
bin_dataFrame["dislikes_WoE"],dislikes_dict = calculateWoE(bin_dataFrame,"dislikes_bin",dislikes_dict)
bin_dataFrame["views_WoE"],views_dict = calculateWoE(bin_dataFrame,"views_bin",views_dict)
bin_dataFrame["comment_count_WoE"],comment_count_dict = calculateWoE(bin_dataFrame,"comment_count_bin",comment_count_dict)
bin_dataFrame.to_csv(path_or_buf=r'youtube-new/binnedDataset.csv',index = None, header=True)

likes_WoE_list = []
dislikes_WoE_list = []
views_WoE_list = []
comment_count_WoE_list = []

# store the WoE values of each feature as a separate feature in the dataset (stored in WoEDataset.csv)
for i in range(len(dataFrame)):
    likes_WoE_list.append(likes_dict[dataFrame["likes_bin"][i]])
    dislikes_WoE_list.append(dislikes_dict[dataFrame["dislikes_bin"][i]])
    views_WoE_list.append(views_dict[dataFrame["views_bin"][i]])
    comment_count_WoE_list.append(comment_count_dict[dataFrame["comment_count_bin"][i]])

dataFrame["likes_WoE"] = pd.DataFrame(likes_WoE_list)
dataFrame["dislikes_WoE"] = pd.DataFrame(dislikes_WoE_list)
dataFrame["views_WoE"] = pd.DataFrame(views_WoE_list)
dataFrame["comment_count_WoE"] = pd.DataFrame(comment_count_WoE_list)

print (dataFrame)
dataFrame.to_csv(path_or_buf=r'youtube-new/WoEDataset.csv',index = None, header=True)
