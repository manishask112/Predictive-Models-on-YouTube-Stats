import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# read the dataset which has the previosuly calculated WoE values
dataFrame = pd.read_csv(r'youtube-new/WoEDataset.csv', index_col=None, header=0)
# take each of the WoE values as a feature for logistic regression
x_col = ["likes_WoE","dislikes_WoE","views_WoE","comment_count_WoE"]
# take the performance feature as a target variable for logistic regression
y_col = ["performance"]
x = dataFrame[x_col]
y = dataFrame[y_col]
# split the dataset into train and test (70-30)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
# perform logistic regression (fit) and store the coefficients
lr = LogisticRegression()
lr.fit(x_train,y_train)
beta = (lr.coef_[0])
alpha = (lr.intercept_[0])
coefficients = np.append (lr.intercept_, lr.coef_)
print (coefficients)
# perform logistic regression (predict) and prediction probabilities
y_pred=lr.predict_proba(x_test)[:,1]
print('Predict proba: ' ,y_pred)
trend_prob = []
non_trend_prob = []
# if the probability value is less than 0.4, we consider it a nontrending video, else, a trending video
for i in y_pred:
    if i<0.4:
        non_trend_prob.append(i)
    else:
        trend_prob.append(i)
print ("trend_prob",len(trend_prob))
print ("non_trend_prob",len(non_trend_prob))

# to calculate trending score, we require offset and factor to scale the score values such that they are distributed sensible. 
# Offset and factor calculations have been tweaked to give a sensible score
score_top=50
non_trending=50
trending=1
odds_top=trending/non_trending 
pd_top=non_trending/(non_trending+trending)
pdo=20
# factor is a default constant value given as 20/log(2)
factor=pdo/np.log(2)
# offset is given by (score top [which is a base score, 50 in our case] - factor * log(odds top [which is ratio of trending is to nontrending, in our case, 1:50])
offset=score_top-factor*np.log(odds_top)

bin_dataframe = pd.read_csv(r'youtube-new/binnedDataset.csv', index_col=None, header=0)
likes_score = []
dislikes_score = []
views_score = []
comment_count_score = []
# calculate the trending score using the formula (offset - factor * [sum over of all datapoints in each bin (WoE score * logistic regression coefficient)])
for i in range(bin_dataframe.shape[0]):
    likes_score.append(offset-factor*sum(np.multiply(coefficients,bin_dataframe["likes_WoE"][i])))
    dislikes_score.append(offset-factor*sum(np.multiply(coefficients,bin_dataframe["dislikes_WoE"][i])))
    views_score.append(offset-factor*sum(np.multiply(coefficients,bin_dataframe["views_WoE"][i])))
    comment_count_score.append(offset-factor*sum(np.multiply(coefficients,bin_dataframe["comment_count_WoE"][i])))

# store the scores of each bin in binnedDataset.csv
bin_dataframe["likes_score"] = pd.DataFrame(likes_score)
bin_dataframe["dislikes_score"] = pd.DataFrame(dislikes_score)
bin_dataframe["views_score"] = pd.DataFrame(views_score)
bin_dataframe["comment_count_score"] = pd.DataFrame(comment_count_score)

print (bin_dataframe)
bin_dataframe.to_csv(path_or_buf=r'youtube-new/binnedDataset.csv',index = None, header=True)

# read the WoEDataset
WoEdataFrame = pd.read_csv(r'youtube-new/WoEDataset.csv', index_col=None, header=0)
# prepare a test set which has the features video ID and bins as feature
x_col = ["video_id","likes_bin","dislikes_bin","views_bin","comment_count_bin"]
x = WoEdataFrame[x_col]
x_train, x_test = train_test_split(x, test_size=0.3, random_state=42)
x_test_df = pd.DataFrame(x_test)
x_test_df["score"] = 0
print (x_test_df)
# calculate overall trending score
# depending on which bin each of the feature - likes, dislikes, views and comment_count fall into, sum up the corresponding score of each feature
# for example, if a video has 12 likes, 4 dislikes, 6 comments and 40 views, we see which bin each of this falls into and sum up the score associated with each bin
for index,row in x_test_df.iterrows():
    for j in range(len(bin_dataframe)):
        if x_test_df["likes_bin"][index] == bin_dataframe["likes_bin"][j]:
            x_test_df["score"][index] += (bin_dataframe["likes_score"][j])
        if x_test_df["dislikes_bin"][index] == bin_dataframe["dislikes_bin"][j]:
            x_test_df["score"][index] += (bin_dataframe["dislikes_score"][j])
        if x_test_df["views_bin"][index] == bin_dataframe["views_bin"][j]:
            x_test_df["score"][index] += (bin_dataframe["views_score"][j])
        if x_test_df["comment_count_bin"][index] == bin_dataframe["comment_count_bin"][j]:
            x_test_df["score"][index] += (bin_dataframe["comment_count_score"][j])

x_test_df["Chances_Of_Trending"] = ""
x_test_df["performance"] = ""
# if the overall trending score is greater than the median score, we tag its chances of trending as "high" and performance feature as 1
# otherwise, we tag its chances of trending as "low" and performance feature as 0
median_score = np.median(x_test_df["score"])
performance = []
for index,row in x_test_df.iterrows():
    if x_test_df["score"][index] > median_score:
        x_test_df["Chances_Of_Trending"][index] = "High"
        x_test_df["performance"][index] = 1
    else:
        x_test_df["Chances_Of_Trending"][index] = "Low"
        x_test_df["performance"][index] = 0
print (x_test_df)

# store the test results in testScore.csv
x_test_df.to_csv(path_or_buf=r'youtube-new/testScore.csv',index = None, header=True)

# we evaluate our model using accuracy based on the "performance" feature (1 if trending and 0 if non-trending)
# the groundtruth performance is taken from the dataset and the predicted perforamnce values have been calculated above using median score

# store the groundtruth video ID along with its performance value in a dictionary
true_dict = {}
for i in range(len(dataFrame)):
    true_dict[dataFrame['video_id'][i]] = dataFrame['performance'][i]

# store the predicted video ID along with its performance value in a dictionary
pred_dict = {}
for index,row in x_test_df.iterrows():
    pred_dict[x_test_df['video_id'][index]] = x_test_df['performance'][index]

# if keys in both dictionary matches, store the corresponding performance value (ordering the values)
pred_arr = []
true_arr = []
for key in pred_dict.keys():
    if key in true_dict.keys():
        pred_arr.append(pred_dict[key])
        true_arr.append(true_dict[key])

# calculate accuracy - currently observed accuracy is 65.53169469598966
print("Accuracy:", accuracy_score(true_arr, pred_arr)*100)
