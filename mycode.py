import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

#from nltk.stem import PorterStemmer
#from nltk.stem import WordNetLemmatizer
# ML Libraries
from sklearn.metrics import accuracy_score
#from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Global Parameters
stop_words = set(stopwords.words('english'))

def load_dataset(filename, cols):
    dataset = pd.read_csv(filename, encoding='latin-1')
    dataset.columns = cols
    return dataset

def preprocess_tweet_text(tweet):
    tweet.lower()
    # Remove urls
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    # Remove user @ references and '#' from tweet
    tweet = re.sub(r'\@\w+|\#', '', tweet)
    # Remove punctuations
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords
    tweet_tokens = word_tokenize(tweet)
    filtered_words = [w for w in tweet_tokens if not w in stop_words]

    # ps = PorterStemmer()
    # stemmed_words = [ps.stem(w) for w in filtered_words]
    # lemmatizer = WordNetLemmatizer()
    # lemma_words = [lemmatizer.lemmatize(w, pos='a') for w in stemmed_words]

    return " ".join(filtered_words)


def int_to_string(sentiment):
    if sentiment == 0:
        return "Negative"
    elif sentiment == 2:
        return "Neutral"
    else:
        return "Positive"



def main():
    test_file_name = 'tweets.csv'
    test_ds = load_dataset(test_file_name, ["text"])
# Creating text feature
    preprocess_tweet_text(test_ds["text"])
    test_feature = tf_vector.transform(np.array(test_ds.iloc[:, 1]).ravel())

# Using Logistic Regression model for prediction
    test_prediction_lr = LR_model.predict(test_feature)

# Averaging out the hashtags result
    test_result_ds = pd.DataFrame({'hashtag': test_ds.hashtag, 'prediction':test_prediction_lr})
    test_result = test_result_ds.groupby(['hashtag']).max().reset_index()
    test_result.columns = ['hashtag', 'predictions']
    test_result.predictions = test_result['predictions'].apply(int_to_string)

    print(test_result)
