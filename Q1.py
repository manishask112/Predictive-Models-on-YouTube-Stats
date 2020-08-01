
# coding: utf-8

# In[4]:


import datetime
import sys
import calendar
import glob
import math as m
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix

def timestamp_split(path):
    i=0
    new_list = []
    range_list=[]
    value_list =[]
    list_times = []
    list_days = []
    day =[]
    df = pd.read_csv(path)
    day_num = np.zeros((375942,1))
    
    #SPLIT DATETIME INTO DATES AND TIMES
    df['Publish_Dates'] = pd.to_datetime(df['publish_time']).dt.date
    df['Publish_Time'] =pd.to_datetime(df['publish_time']).dt.time
    
    #FIND THE DAY OF THE WEEK FROM THE DATE
    for index, date in df['Publish_Dates'].items():
        i+=1
        date = str(date)
        day_num=datetime.datetime.strptime(date, '%Y-%m-%d').weekday() 
        day.append(calendar.day_name[day_num])
    df['Days'] = day
    
    for index, days in df['Days'].items():
        if (days=='Friday'):
            list_days.append(int(1))
        if (days=='Saturday'):
            list_days.append(2)
        if (days=='Sunday'):
            list_days.append(3)
        if (days=='Monday'):
            list_days.append(4)
        if (days=='Tuesday'):
            list_days.append(5)
        if (days=='Wednesday'):
            list_days.append(6)  
        if (days=='Thursday'):
            list_days.append(7)       
    df['Day_Bins'] =  pd.DataFrame(list_days)   
    
    #CREATE BINS OF TIME
#     for t, times in df['Publish_Time'].items():
#         if (times >= datetime.time(00,00,00) and times<= datetime.time(3,59,59)):
#             list_times.append('12 am - 4 am')
#         if (times >= datetime.time(4,00,00) and times<= datetime.time(7,59,59)):
#             list_times.append('4 am - 8 am')
#         if (times >= datetime.time(8,00,00) and times<= datetime.time(11,59,59)):
#             list_times.append('8 am - 12 pm')
#         if (times >= datetime.time(12,00,00) and times<= datetime.time(15,59,59)):
#             list_times.append('12 pm - 4 pm')   
#         if (times >= datetime.time(16,00,00) and times<= datetime.time(19,59,59)):
#             list_times.append('4 pm - 8 pm')
#         if (times >= datetime.time(20,00,00) and times<= datetime.time(23,59,59)):
#             list_times.append('8 pm - 12 am') 

    for t, times in df['Publish_Time'].items():
        if (times >= datetime.time(00,00,00) and times<= datetime.time(5,59,59)):
            range_list.append('12 am - 6 am')
            list_times.append(int(1))
        if (times >= datetime.time(6,00,00) and times<= datetime.time(11,59,59)):
            range_list.append('6 am - 12 pm')
            list_times.append(2)
        if (times >= datetime.time(12,00,00) and times<= datetime.time(17,59,59)):
            range_list.append('12 pm - 6 pm')
            list_times.append(3)
        if (times >= datetime.time(18,00,00) and times<= datetime.time(23,59,59)):
            range_list.append('6 pm - 12 am')
            list_times.append(4)    
    df['Times'] =  pd.DataFrame(range_list)
    df['Time_Bins'] =  pd.DataFrame(list_times) 
    
    #CREATE FINAL BINS OF 7 DAYS AND 4 6-HOUR BINS    
    #https://thispointer.com/python-map-function-explained-with-examples/
    df['TimePeriod'] = df['Days'] + df['Times'].map(str)
    df['Bins'] = df['Day_Bins'].map(str) + df['Time_Bins'].map(str)
    df['Bins'] = df['Bins'].astype(int)
    
    #CHANGE TRENDING DATE FORMAT
    for val, value in df['trending_date'].items():
        value_list.append(datetime.datetime.strptime(value, "%y.%d.%m"))#.strftime("%Y-%m-%d"))
    df['trending_dates'] = pd.DataFrame(value_list)
    
    #CREATE PUBLISHED DATES WITH SAME TYPE AS TD
#     df['trending_dates'] = pd.to_datetime(df['trending_date']).dt.date
    for u, new in df['publish_time'].items():
        new =datetime.datetime.strptime(new,'%Y-%m-%dT%H:%M:%S.%fZ')
        new_list.append(new)
    df['Only_Published_Dates']=pd.DataFrame(new_list)
#     df['Only_Published_Dates'] = pd.to_datetime(pd.DataFrame(new_list).astype(str)).dt.date
    
    #FIND NUMBER OF DAYS IT TOOK FOR VIDEO TO TREND
    a = df['Only_Published_Dates']
    b = df['trending_dates']
    df['Days_To_Trend'] = -1*((a-b).dt.days)  
    #print("Days_To",df['Days_To_Trend'])
    #datetime.datetime.strptime(df['Publish_Dates'], "%Y-%m-%d") - datetime.datetime.strptime(df['trending_dates'], "%Y-%m-%d")
    
    #DROP DATA THAT DOES NOT TREND WITHIN THREE DAYS
    df = df.drop(df[df.Days_To_Trend > 3].index)
    
    #ONE-HOT ENCODING FOR COUNTRY
    df = pd.concat([df, pd.get_dummies(df['country'], prefix='country')], axis=1)
    df = pd.concat([df, pd.get_dummies(df['category_id'], prefix='category_id')], axis=1)
    
    #NORMALIZING VIEWS,LIKES,DISLIKES,COMMENT COUNT
    df['views']=(df['views']-df['views'].mean())/df['views'].std() 
    df['likes']=(df['likes']-df['likes'].mean())/df['likes'].std() 
    df['dislikes']=(df['dislikes']-df['dislikes'].mean())/df['dislikes'].std() 
    df['comment_count']=(df['comment_count']-df['comment_count'].mean())/df['comment_count'].std() 
    
    #LOGISTIC REGRESSION MODEL
    x_col = ["country_Canada","country_France","country_Germany","country_Great Britain","country_India","country_Japan","country_Mexico","country_Russia","country_USA","country_South Korea","category_id_1","category_id_2","category_id_10","category_id_15","category_id_17","category_id_19","category_id_20","category_id_22","category_id_23","category_id_24","category_id_25","category_id_26","category_id_27","category_id_28"]
    #y_col = ["TimePeriod"]
    #y_col = ["Day_Bins"]
    y_col = ["Time_Bins"]
    x = df[x_col]
    y = df[y_col]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    lr = LogisticRegression()
    lr.fit(x_train,y_train)
    y_pred=lr.predict(x_test)
    
#     dataf =pd.DataFrame()
#     dataf["Predicted"]=y_pred
#     range_list_2=[]
#     for t, times in dataf['Predicted'].items():
#         if (times == str(1)):
#             range_list_2.append('12 am - 6 am')
#         if (times == str(2)):
#             range_list_2.append('6 am - 12 pm')
#         if (times == str(3)):
#             range_list_2.append('12 pm - 6 pm')
#         if (times == str(4)):
#             range_list_2.append('6 pm - 12 am')
#     dataf["Predicted_Time"] =pd.DataFrame(range_list_2)  
#     print(dataf["Predicted_Time"])
    print("Predicted Output is ",y_pred)
    accuracy = lr.score(x_test,y_test)
    print("The accuracy for Logistic Regression is: ", accuracy*100)
    
    #NEURAL NETWORK MODEL
    print("Results for the Neural Network Model")
    clf = MLPClassifier(solver='lbfgs', alpha=0.01,hidden_layer_sizes=(15,15), random_state=1)
    clf.fit(x_train,y_train.values.ravel())
    y_pred_train=clf.predict(x_train)
    y_pred=clf.predict(x_test)
    print("Predicted output is",y_pred)
    #print(confusion_matrix(y_test,y_pred))
    #print(classification_report(y_test,y_pred))
#   print(classification_report(y_train,y_pred_train))

    return df, x_train, y_train, x_test, y_test

###############################################################################
# Please insert the path to your merged dataset here, in place of my path, incase the system input does not work
# (dataframe, x_train, y_train, x_test, y_test) = timestamp_split(r'C:\Users\umamg\Downloads\youtube-new\dataset.csv')
(dataframe, x_train, y_train, x_test, y_test) = timestamp_split(sys.argv[1])
###############################################################################
#Decision Tree Model
# Referred from https://www.geeksforgeeks.org/decision-tree-implementation-python/

def gini_index(x_train, x_test, y_train):  
    g = DecisionTreeClassifier(criterion = "gini", 
            random_state = 100,max_depth=3, min_samples_leaf=5) 
    g.fit(x_train, y_train) 
    return g 
      
# Function to perform training with entropy. 
def entropy(x_train, x_test, y_train): 
  
    # Decision tree with entropy 
    e = DecisionTreeClassifier( 
            criterion = "entropy", random_state = 100, 
            max_depth = 3, min_samples_leaf = 5) 
  
    # Performing training 
    e.fit(x_train, y_train) 
    return e 
  
  # Function to make predictions 
def prediction(x_test, clf_object): 
    # Predicton on test with giniIndex 
    y_pred = clf_object.predict(x_test) 
    print("Predicted values:") 
    print(y_pred) 
    return y_pred 
      
# Function to calculate accuracy 

def cal_accuracy(y_test, y_pred): 
      
    #print("Confusion Matrix: ", 
    #    confusion_matrix(y_test, y_pred))
    #     print("Report : ", 
    #classification_report(y_test, y_pred)) 
    print ("The Accuracy for Decision Tree is : ", accuracy_score(y_test,y_pred)*100) 

g = gini_index(x_train, x_test, y_train) 
e = entropy(x_train, x_test, y_train) 

print("Results Using Gini-Index:")
y_pred_gini = prediction(x_test, g) 
cal_accuracy(y_test, y_pred_gini) 

print("Results Using Entropy:") 
y_pred_entropy = prediction(x_test, e) 
cal_accuracy(y_test, y_pred_entropy)    

