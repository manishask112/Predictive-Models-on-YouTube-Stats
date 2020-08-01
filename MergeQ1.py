
# coding: utf-8

# In[9]:


import pandas as pd
import glob
import numpy as np
import sys
def addPerformanceFeature(filename):
	mylist = []
#Insert your path to the dataset here, instead of r'C:\Users\umamg\Downloads\youtube-new/, but keep the '/'    
	for chunk in  pd.read_csv(r'C:\Users\umamg\Downloads\youtube-new/'+filename, index_col=None, header=0,encoding="iso-8859-1",chunksize=200):

		mylist.append(chunk)
	us_df = pd.concat(mylist, axis= 0)
	del mylist
	# us_df = pd.read_csv("youtube-new/"+filename,encoding="iso-8859-1")
	us_df = us_df[["video_id","trending_date","category_id","views","likes","dislikes","comment_count","publish_time"]]
	if filename == 'CAvideos.csv':
		country_list = ['Canada']*us_df.shape[0]
	elif filename == 'DEvideos.csv':
		country_list = ['Germany']*us_df.shape[0]  
	elif filename == 'FRvideos.csv':
		country_list = ['France']*us_df.shape[0]
	if filename == 'GBvideos.csv':
		country_list = ['Great Britain']*us_df.shape[0]
	if filename == 'INvideos.csv':
		country_list = ['India']*us_df.shape[0]
	if filename == 'JPvideos.csv':
		country_list = ['Japan']*us_df.shape[0]
	if filename == 'KRvideos.csv':
		country_list = ['South Korea']*us_df.shape[0]
	if filename == 'MXvideos.csv':
		country_list = ['Mexico']*us_df.shape[0] 
	if filename == 'RUvideos.csv':
		country_list = ['Russia']*us_df.shape[0] 
	if filename == 'USvideos.csv':
		country_list = ['USA']*us_df.shape[0]     
        
	us_df["country"] = pd.DataFrame(country_list)
#Insert your path to the dataset here, instead of r'C:\Users\umamg\Downloads\youtube-new/, but keep the '/'    
	us_df.to_csv(path_or_buf=r'C:\Users\umamg\Downloads\youtube-new/'+filename,index = None, header=True)

addPerformanceFeature('CAvideos.csv')
addPerformanceFeature('DEvideos.csv')
addPerformanceFeature('FRvideos.csv')
addPerformanceFeature('GBvideos.csv')
addPerformanceFeature('INvideos.csv')
addPerformanceFeature('JPvideos.csv')
addPerformanceFeature('KRvideos.csv')
addPerformanceFeature('MXvideos.csv')
addPerformanceFeature('RUvideos.csv')
addPerformanceFeature('USvideos.csv')

#Insert your path to the dataset here, for each of the countries, instead of r'C:\Users\umamg\Downloads\youtube-new/, but keep the '/'
all_files = [r'C:\Users\umamg\Downloads\youtube-new/CAvideos.csv', r'C:\Users\umamg\Downloads\youtube-new/DEvideos.csv', r'C:\Users\umamg\Downloads\youtube-new/FRvideos.csv', r'C:\Users\umamg\Downloads\youtube-new/GBvideos.csv', r'C:\Users\umamg\Downloads\youtube-new/INvideos.csv', r'C:\Users\umamg\Downloads\youtube-new/JPvideos.csv',r'C:\Users\umamg\Downloads\youtube-new/KRvideos.csv',r'C:\Users\umamg\Downloads\youtube-new/MXvideos.csv',r'C:\Users\umamg\Downloads\youtube-new/RUvideos.csv',r'C:\Users\umamg\Downloads\youtube-new/USvideos.csv']

li = []

for filename in all_files:
	#print(filename)
	df = pd.read_csv(filename, index_col=None, header=0,encoding="iso-8859-1")
	# df = df.drop(['trending_date', 'publish_time','views', 'likes', 'dislikes', 'comment_count', 'thumbnail_link', 'comments_disabled', 'ratings_disabled', 'video_error_or_removed'], axis = 1)
	li.append(df)

frame = pd.concat(li, axis=0, ignore_index=True)

#Insert your path to the dataset here, instead of r'C:\Users\umamg\Downloads\youtube-new/, but keep the '/'
frame.to_csv(path_or_buf=r'C:\Users\umamg\Downloads\youtube-new/dataset.csv',index = None, header=True)

#Insert your path to the dataset here, instead of r'C:\Users\umamg\Downloads\youtube-new/, but keep the '/'
dataFrame = pd.read_csv(r'C:\Users\umamg\Downloads\youtube-new/dataset.csv', index_col=None, header=0)

print(dataFrame.shape)

