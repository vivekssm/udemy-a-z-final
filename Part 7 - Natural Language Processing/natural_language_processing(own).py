#here we have data of reviews in a restaurent..
#we have to make a model so that whenever a new review is added it tell wheather it is positive
#or negative.

#first collect the important words which tell wheather review is positive or negative.. and remove
#unnecessary ones..

#also stem the words e.g. love and loved and loving are same
#capital and small are same



# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)  #tsv file.. it shows
                                          #two reviews are seperated by "\t"(tab)
                                          #quoting=3 meand " "(quotation) is useless

# Cleaning the texts
import re
import nltk   #it will remove irrelevent words like this,that,those,it,preposition,conjunction etc.
nltk.download('stopwords')  #stopwords contain all the irrelevant words..
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) #we don't want to remove a-z and A-Z
                                  #and " " from dataset[i]
    review = review.lower()   #make all in lower case
    review = review.split()   #split the sentence then review will bocome list of words contained 
                              #in this sentence..
    ps = PorterStemmer()      #it will stem the word i.e withdraw love from loving.
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] 
                                  #this for loop will go word to word in the list
    
    review = ' '.join(review)   #here different words in the review will join together to
                               #change back to words from list.
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)  #max-features is added so that it only select word which
                                #occur frequently..   
                                #so only most frequently used 1500 words will be kept.. rest will be ignored..

X = cv.fit_transform(corpus).toarray()  #spars matrix is created..
                                 #here, we classify which word in review come how many times.
                               #and reduce the words which rarely occur  

y = dataset.iloc[:, 1].values  #all rows and 1st column of dataset is selected..
                  #1 in y means positive review.
                  #o mean negative review



# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)       #12+42=54 incorrect predictions