Download the dataset:
https://www.kaggle.com/datasnaek/youtube-new/data

Instructions to run program for Question 1:

There are two files MergeQ1.py and Q1.py

1.Download the dataset.
2.Open MergeQ1.py 
3.In the places where the path exists as r'C:\Users\umamg\Downloads\youtube-new/, please add your own path where you downloaded the dataset and ensure that the '/' at the end of the line is there.
4.This has to be done is five places, each of which has been highlighted by a comment.
5.Once this is done, you will have a new file called dataset in your path, which will have the new merged dataset.
6.Next, run Q1.py
7.With the python file, you will also have to input the path (like this: r'C:\Users\umamg\Downloads\youtube-new) to your dataset (if it is not already in the same location).
8. In case this does not work, you can put your path name in the code directly, where it has been indicated.

Probelm 2

Steps to run the code:
1) First extract the dataset folder and place the three files mergeDatasets.py, trendingScoreCalculation.py and modelTrainer.py inside the youtube-new folder
2) Run the mergeDatasets.py file first. This should generate the dataset.csv file inside the youtube-new folder
3) Next, run the trendingScoreCalculation.py file. This should generate two files - binnedDataset.csv and WoEDataset.csv
4) Finally, run the modelTrainer.py file. This should generate the testScore.csv file and print the accuracy in the console. 

Note: Please ensure all the generated csv files are closed before running the code.

Problem 3

Extract the file in the dataset
install langdetect
(We are using this Language detection library to detect languages in the dataset)
> pip install langdetect

Language detection part of code takes time to run
So we have two sets of dataset
The smaller version is composed of videos from the countries Canada and US
The larger dataset is composed of enteries from all countries
Results from both the datasets on evaluation give the same Jaccards Similarity index

Run the create_dataset.py file to create dataset for this problem statement
For a smaller dataset:
> python create_dataset.py small_dataset path-to-the-folder-of-extracted-dataset
For larger dataset:
> python create_dataset.py large_dataset path-to-the-folder-of-extracted-dataset

The dataset will be stored in data.csv file

For preprocessing data
Run the prepdata.py file
python prepdata.py


Run model.py to test, generate tags and evaluate the tags
python model.py
The text description, tags and Jaccards Similarity index are stored in tags_gen_dataset.csv file
