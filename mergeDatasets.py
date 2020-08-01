import pandas as pd
import glob
import numpy as np

# Add performance feature to all the country datasets
# Performance is 1 for trending video and 0 for non trending video 
def addPerformanceFeature(filename):
	mylist = []
	for chunk in  pd.read_csv("youtube-new/"+filename, index_col=None, header=0,encoding="iso-8859-1",chunksize=200):
		mylist.append(chunk)
	us_df = pd.concat(mylist, axis= 0)
	del mylist
	# us_df = pd.read_csv("youtube-new/"+filename,encoding="iso-8859-1")
	us_df = us_df[["video_id","trending_date","views","likes","dislikes","comment_count","publish_time"]]
	target_performance = []
	for i in range(0,us_df.shape[0],200):
		for j in range(i,i+75):
			target_performance.append(1)
		for j in range(i+75,i+200):
			target_performance.append(0)

	us_df["performance"] = pd.DataFrame(target_performance)
	us_df.to_csv(path_or_buf=r'youtube-new/'+filename,index = None, header=True)

# main

# Add performance feature to all the country datasets
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

# read each country dataset and concatinate it into a single file (dataset.csv)
all_files = [r'youtube-new/CAvideos.csv', r'youtube-new/DEvideos.csv', r'youtube-new/FRvideos.csv', r'youtube-new/GBvideos.csv', r'youtube-new/INvideos.csv', r'youtube-new/JPvideos.csv',r'youtube-new/KRvideos.csv',r'youtube-new/MXvideos.csv',r'youtube-new/RUvideos.csv',r'youtube-new/USvideos.csv']

li = []
for filename in all_files:
	df = pd.read_csv(filename, index_col=None, header=0,encoding="iso-8859-1")
	li.append(df)

frame = pd.concat(li, axis=0, ignore_index=True)
frame.to_csv(path_or_buf=r'youtube-new/dataset.csv',index = None, header=True)

dataFrame = pd.read_csv(r'youtube-new/dataset.csv', index_col=None, header=0)
print(dataFrame.shape)
