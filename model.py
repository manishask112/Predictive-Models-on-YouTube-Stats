import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pylab as plt
from collections import OrderedDict
import statistics
import math

# Reference to the official documentation and example for TfidfVectorizer
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
# https://machinelearningmastery.com/prepare-text-data-machine-learning-scikit-learn/
def sorttfidfVector(vector, features, size):
	row, col = vector.nonzero()
	score = vector.data
	# sort in (col, score) in decreasing order of score 
	sub = [0] * len(col)
	for i in range(len(col)):
		sub[i]= (col[i], score[i])
	sub = sorted(sub, key=lambda x: (x[1], x[0]), reverse=True)
	order = [x[0] for x in sub]
	size = int(math.ceil(size))
	keywords = []
	# Extracting size number of Keywords 
	if len(order) < size:
		for i in range(len(order)):
			ind = order[i]
			keywords.append(features[ind])
	else:
		for i in range(size):
			ind = order[i]
			keywords.append(features[ind])
	return keywords

df = pd.read_csv(r'tags_gen_dataset.csv', index_col=None, header=0)
print("number of enteries", df.shape[0])

y = df['tags']
x = df.drop(['tags'], axis=1)
# split the dataset into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 6)
docs = x_train['text'].tolist()

# Fitting the set 
vectorizer = TfidfVectorizer(max_df = 0.80, stop_words = 'english')
vectorizer.fit(x_train['text'].values.astype('U'))
jaccards = []
docs_test= x_test['text'].tolist()
tags_list = y_test.tolist()
all_features = vectorizer.get_feature_names()

# train set
print("TRAIN SET")
x = x_train['text'].tolist()
counter = []
for i in range(x_train.shape[0]):
	row = x[i]
	row = [x for x in row.split(' ')]
	counter.append(len(row))
avg_words = statistics.mean(counter)
print("Average words", avg_words)
median_words = statistics.median(counter)
print("Median words", median_words)
mode_words = statistics.mode(counter)
print("Mode words", mode_words)
stddev_words = statistics.stdev(counter)
print("Standard Deviation", stddev_words)
print("range", avg_words - stddev_words, avg_words + stddev_words)

# test set
print("TEST SET")
x = x_test['text'].tolist()
counter = []
for i in range(x_test.shape[0]):
	row = x[i]
	row = [x for x in row.split(' ')]
	counter.append(len(row))
avg_wordst = statistics.mean(counter)
print("Average words", avg_wordst)
median_wordst = statistics.median(counter)
print("Median words", median_wordst)
mode_wordst = statistics.mode(counter)
print("Mode words", mode_wordst)
stddev_wordst = statistics.stdev(counter)
print("Standard Deviation", stddev_wordst)
print("range", avg_wordst - stddev_wordst, avg_wordst + stddev_wordst)


print("TAGS SET")
counter = []
for i in range(x_test.shape[0]):
	y_dict = set()
	for word in tags_list[i].split():
		if word in docs_test[i]:
			y_dict.add(word)
	counter.append(len(y_dict))
avg_words = sum(counter)/x_test.shape[0]
print("Average words", avg_words)
median_words = statistics.median(counter)
print("Median words", median_words)
mode_words = statistics.mode(counter)
print("Mode words", mode_words)
stddev_words = statistics.stdev(counter)
print("Standard Deviation", stddev_words)
print("range", avg_words - stddev_words, avg_words + stddev_words)

final_test = []
final_y = []
final_jacc = []
for i in range(x_test.shape[0]):
	
	# Count test set number of words
	test_word_counter = 0
	for word in docs_test[i].split():
		test_word_counter +=1
	if test_word_counter > avg_wordst - stddev_wordst and test_word_counter < avg_wordst + stddev_wordst:
		tfidf_vector = vectorizer.transform([docs_test[i]])
		y_dict = set()
		for word in tags_list[i].split():
			if word in docs_test[i]:
				y_dict.add(word)
		if len(y_dict) > avg_words - stddev_words and len(y_dict) < avg_words + stddev_words:
			keywords_dict = sorttfidfVector(tfidf_vector, all_features, len(y_dict))
			jaccard_numerator = 0
			for word in keywords_dict:
				if word in y_dict:
					jaccard_numerator +=1
			y_dict.update(keywords_dict)
			jaccard_denominator = len(y_dict)
			jaccards.append(jaccard_numerator/jaccard_denominator  * 100)
			final_test.append(docs_test[i])
			final_y.append(keywords_dict)
			final_jacc.append(jaccard_numerator/jaccard_denominator  * 100)
print("Number of samples test against", len(jaccards))
print("Maximum jaccards similarity index", max(jaccards))
countmax = jaccards.count(max(jaccards))
print("Number of samples with max similarity ", countmax)
print("Minimum jaccards similarity index", min(jaccards))
countmin = jaccards.count(min(jaccards))
print("Number of samples with min similarity", countmin)
print("Overall jaccards similarity index of test set", sum(jaccards)/len(jaccards))

# Referred to put all the tested data and resuts to a file
# https://www.geeksforgeeks.org/create-a-pandas-dataframe-from-lists/
df = pd.DataFrame(list(zip(final_test, final_y, final_jacc)),columns =['Document', 'Generated Tags', 'Jaccards Similarity Index']) 
df.to_csv(path_or_buf=r'tags_gen_dataset.csv',index = None, header=True)

jacc_count_dict = OrderedDict()
for i in range(len(jaccards)):
	if jaccards[i] in jacc_count_dict:
		jacc_count_dict[jaccards[i]] += 1
	else:
		jacc_count_dict[jaccards[i]] = 1

# Plotting jaccards similarity index on x axis and count of samples on y axis
# To plot a dictionary 
# https://stackoverflow.com/questions/37266341/plotting-a-python-dict-in-order-of-key-values/37266356
lists = sorted(jacc_count_dict.items()) # sorted by key, return a list of tuples

x, y = zip(*lists) 

plt.plot(x, y)
plt.show()
