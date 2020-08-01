import pandas as pd
import sys

# We create two datasets- large_dataset which considers samples from all the 10 countires
# small_dataset- Countries CA and US
if sys.argv[1] == 'large_dataset':
	all_files = [sys.argv[2]+'/CAvideos.csv', sys.argv[2]+'/USvideos.csv', sys.argv[2]+'/DEvideos.csv', sys.argv[2]+'/FRvideos.csv', sys.argv[2]+'/GBvideos.csv', sys.argv[2]+'/INvideos.csv', sys.argv[2]+'/JPvideos.csv', sys.argv[2]+'/KRvideos.csv', sys.argv[2]+'/MXvideos.csv', sys.argv[2]+'/RUvideos.csv']
elif sys.argv[1] == 'small_dataset':
	all_files = [sys.argv[2]+'/CAvideos.csv', sys.argv[2]+'/USvideos.csv']

li = []

# Reading data from the files
# dropping null rows and duplicates from the files
for filename in all_files:
	print("Reading file:", filename)
	df = pd.read_csv(filename, index_col=None, header=0,  encoding = "ISO-8859-1")
	df = df.drop(['trending_date', 'publish_time','views', 'likes', 'dislikes', 'comment_count', 'thumbnail_link', 'comments_disabled', 'ratings_disabled', 'video_error_or_removed'], axis = 1)
	df.drop_duplicates(subset ="video_id", keep = "first", inplace = True)
	df = df[df['tags'] != '[none]']
	print("Dataset size after dropping duplicate entries from this file:", df.shape[0])
	df = df.dropna(axis = 0)
	print("Number of samples after dropping null value rows from this file", df.shape[0])
	li.append(df)

# concatinating all files in a frame
frame = pd.concat(li, axis=0, ignore_index=True)
print("Number of samples in the dataset:", frame.shape[0])
frame.to_csv(path_or_buf=r'data.csv',index = None, header=True)
frame = pd.read_csv(r'data.csv', index_col=None, header=0)
frame.drop_duplicates(subset ="video_id", keep = "first", inplace = True)
frame = frame[frame['tags'] != '[none]']
print("Dataset size after droping duplicate entries:", frame.shape[0])
frame = frame.dropna(axis = 0)

# Storing the dataframe to data.csv file
frame.to_csv(path_or_buf=r'data.csv',index = None, header=True)

print("Number of entries stored", frame.shape[0])
print("Number of Features stored", frame.shape[1])
print("Features selected for the tag generation problem from the dataset", frame.columns)

