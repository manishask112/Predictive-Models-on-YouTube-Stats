import pandas as pd
import re
import numpy as np
import langdetect
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import text


urls = []

list_my_words = ['follow', 'subscribe', 'https', 'like']
# To create our very own stop words list but we want to also use stop words present in English 
# Ref: https://stackoverflow.com/questions/26826002/adding-words-to-stop-words-list-in-tfidfvectorizer-in-sklearn
my_stop_words = text.ENGLISH_STOP_WORDS.union(list_my_words)

# Preprocessing of text-
# Removing urls from text
# Removing newlines from text
# stripping blank spaces in text
# stripping punctuation from text
# Converting text to lower case
# remove \, /, ~, `, @, !, #, $, %, ^, &, *, (, ), -, _, +, =, |, {, }, [, ], ;, :, ', ", <, >, ., ,, ?, 
def pre_process(text):
	text = text.strip()
	
	text = text.replace(r'\n', " ")
	# Converting text to lower case
	text = text.lower()

	text = re.sub(r"http://amzn\S+", " ", text)
	text = re.sub(r"https://amzn\S+", " ", text)
	text = re.sub(r"http://bit.ly\S+", " ", text)
	text = re.sub(r"https://bit.ly\S+", " ", text)
	text = re.sub(r"http://goo.gl\S+", " ", text)
	text = re.sub(r"https://goo.gl\S+", " ", text)
	text = text.replace(r'https://www.instagram.com/', " ")
	text = text.replace(r'http://www.instagram.com/', " ")
	text = text.replace(r'http://instagram.com', " ")
	text = text.replace(r'https://instagram.com/', " ")
	text = text.replace(r'https://www.youtube.com/', " ")
	text = text.replace(r'http://www.youtube.com', " ")
	text = text.replace(r'https://twitter.com/', " ")
	text = text.replace(r'http://twitter.com/', " ")
	text = text.replace(r'http://www.twitter.com/', " ")
	text = text.replace(r'https://mobile.twitter.com/', " ")	
	text = text.replace(r'https://soundcloud.com/', " ")
	text = text.replace(r'http://soundcloud.com/', " ")
	text = re.sub(r"http://tinyurl.com\S+", " ", text)
	text = re.sub(r"https://tinyurl.com\S+", " ", text)
	text = re.sub(r"http://open.spotify.com\S+", " ", text)
	text = re.sub(r"https://open.spotify.com\S+", " ", text)
	text = re.sub(r"http://youtu.be\S+", " ", text)
	text = re.sub(r"https://youtu.be\S+", " ", text)
	text = text.replace(r'http://youtube.com/', " ")
	text = text.replace(r'https://youtube.com/', " ")
	text = text.replace(r'http://youtube.com/c/', " ")
	text = text.replace(r'https://youtube.com/c/', " ")
	text = text.replace(r'https://m.youtube.com/', " ")
	text = text.replace(r'http://m.youtube.com/', " ")	
	text = re.sub(r"https://www.snapchat.com/add/\S+", " ", text)
	text = re.sub(r"http://www.snapchat.com/add/\S+", " ", text)
	text = re.sub(r"http://snapchat.com/add/\S+", " ", text)
	text = re.sub(r"https://snapchat.com/add/\S+", " ", text)
	text = re.sub(r"https://smarturl.it\S+", " ", text)
	text = re.sub(r"http://smarturl.it\S+", " ", text)
	text = re.sub(r"http://bitly.com/\S+", " ", text)
	text = re.sub(r"https://bitly.com/\S+", " ", text)
	text = re.sub(r"https://creativecommons.org/\S+", " ", text)
	text = re.sub(r"http://creativecommons.org/\S+", " ", text)	
	text = text.replace(r'http://facebook.com/', " ")
	text = text.replace(r'http://www.facebook.com/', " ")
	text = text.replace(r'https://www.facebook.com/', " ")
	text = text.replace(r'https://facebook.com/', " ")
	text = text.replace(r'https://m.facebook.com/', " ")
	text = text.replace(r'http://www.google.com/', " ")
	text = text.replace(r'https://plus.google.com/', " ")
	text = text.replace(r'http://plus.google.com/', " ")
	text = text.replace(r'http://google.com/', " ")
	text = text.replace(r'https://www.patreon.com/', " ")
	text = text.replace(r'http://www.patreon.com/', " ")
	text = re.sub(r"https://play.google.com\S+", " ", text)
	text = re.sub(r"http://play.google.com\S+", " ", text)
	text = re.sub(r"http://ebay\S+", " ", text)
	text = re.sub(r"https://ebay\S+", " ", text)
	text = re.sub(r"http://vevo.ly/\S+", " ", text)
	text = re.sub(r"https://vevo.ly/\S+", " ", text)
	text = re.sub(r"https://itunes.apple.com/us/app/arydigital/\S+", " ", text)
	text = re.sub(r"http://itunes.apple.com/us/app/arydigital/\S+", " ", text)

	text = re.sub(r'.com', " ", text)

	# Keeps only alphabets and numbers in text
	text = re.sub('[\\W_]+', ' ', text)
	text = re.sub(r'http', " ", text)
	text = re.sub(r'www', " ", text)
	text = text.strip()
	# Strips off extra space that is created after removing all the above words/ characters
	text = re.sub('(\\s+)',' ',text)
	return text
# Removes stop words from tags
def tag_process(text):

	tags = text.split()

	tagg  = [tag for tag in tags if tag not in my_stop_words]
	result = ' '.join(tagg)

	return result

# Removes unicode charaters from text
def stripUnicode(text):
	# To remove unicode characters
	# Ref: https://stackoverflow.com/questions/23680976/python-removing-non-latin-characters
	text = re.sub(r'[^\x00-\x7F\x80-\xFF\u0100-\u017F\u0180-\u024F\u1E00-\u1EFF]', '', text)
	return text


df = pd.read_csv(r'data.csv', index_col=None, header=0, low_memory=False)
print("Data size", df.shape)

# Creating a feature 'text' in out dataset which merges video titles, channel titles and description into one
df['text'] = df['title'] + " " + df['channel_title'] + " " + df['description']# add a space between

### Performing preprocessing on the text feature 
df['text'] = df['text'].apply(lambda x: pre_process(x))
df['text'] = df['text'].apply(lambda x: stripUnicode(x))
df = df.dropna(axis = 0)

# Detecting language for the samples in the dataset
# And removing samples which are in any lanuguage other than English
# https://stackoverflow.com/questions/48543032/fast-way-of-checking-for-language-in-csv
# https://stackoverflow.com/questions/40783383/error-using-langdetect-in-python-no-features-in-text
for row in df['text']:
    try:
        language = langdetect.detect(row)
    except:
        language = "error"
        ind = df.text[df.text == row].index.tolist()
        df = df.drop(axis = 0, index = ind)
print(df.shape)
df['Language'] = df['text'].apply(lambda x: langdetect.detect(x))
df = df.loc[df['Language'] == 'en']
print("Number of samples after selecting samples in English language", df.shape)

# Preprocess tags to strip off spaces, urls, punctuation
df['tags'] = df['tags'].apply(lambda x: pre_process(x))
# remove stop words
df['tags'] = df['tags'].apply(lambda x: tag_process(x))
# Remove Unicode characters
df['tags'] = df['tags'].apply(lambda x: stripUnicode(x))


df = df.dropna(axis = 0)
print("Drop null value rows")
print("Dataset size", df.shape)

# Storing the pre processed data to tags_gen_dataset.csv file
df.to_csv(path_or_buf=r'tags_gen_dataset.csv',index = None, header=True)
