import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib
import pickle
import numpy as np
from csv import writer


def append_list_as_row(file_name, list_of_elem):
    print('Function called')
    # Open file in append mode
    try:
        with open(file_name, 'a+', newline='') as write_obj:
            csv_writer = writer(write_obj)
            csv_writer.writerow(list_of_elem)
    except Exception as e:
        print('Error occured while appending new rows in the CSV file: ', e.__class__)


def load_dataset(filepath):
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        print('Error occured while reading the data file: ', e.__class__)


def preprocessing(df):
    try:
        feature_data_tokenize = []  # This list contains tokenized data
        for line in df['User Story']:
            line = word_tokenize(line)
            feature_data_tokenize.append(line)

        stop_words = set(stopwords.words('english') + list(punctuation))
        # This list contains tokenized data without stop words and punctuation marks
        feature_data_rmStopWords = []
        for line in feature_data_tokenize:
            sub_arr = []
            for l in line:
                if l not in stop_words:
                    sub_arr.append(l)
            feature_data_rmStopWords.append(sub_arr)

        features = []  # This list contains features dataset with all preprocessing steps performed
        for line in feature_data_rmStopWords:
            full_line = str()
            for word in line:
                full_line = full_line + word + ' '
            full_line = full_line[:-1]
            features.append(full_line)
        # This list contains the corresponding labels for the features
        labels = df['Rating']
        dataFrame = pd.DataFrame(
            {
                'Input': features,
                'Output': labels
            })
        return dataFrame
    except Exception as e:
        print('Error occured while pre-processing: ', e)


def test_train_split(df):
    try:
        train, test = train_test_split(df, test_size=0.2)
        return train, test
    except Exception as e:
        print('Error occured while train test split: ', e)


def feature_extraction(train, test):
    try:
        grams = [1, 3]  # This means (uni-gram, bi-gram, tri-gram)

        vectorizer = CountVectorizer(ngram_range=grams)

        train_X = vectorizer.fit_transform(train['Input'])
        train_X = train_X.toarray()

        train_Y = train['Output']

        test_X = vectorizer.transform(test['Input'])
        test_X = test_X.toarray()
        test_Y = test['Output']

        return train_X, train_Y, test_X, test_Y, vectorizer
    except Exception as e:
        print('Error occured while feature extraction: ', e)


def model_training(train_X, train_Y):
    try:
        naive_bayes_model = MultinomialNB()
        naive_bayes_model.fit(train_X, train_Y)
        return naive_bayes_model
    except Exception as e:
        print('Error occured while model training: ', e)


def model_testing(test_X, test_Y, model):
    try:
        model_pred_Y = model.predict(test_X)
        accuracy = accuracy_score(test_Y, model_pred_Y)
        accuracy = round(accuracy, 2)

        print('The accuracy of the algorithm is : ', accuracy * 100, '%')
        return accuracy
    except Exception as e:
        print('Error occured while model testing:  ', e)


def save_model(model):
    try:
        filename = 'saved_models/model.pkl'
        joblib.dump(model, filename, compress=9)
    except Exception as e:
        print('Error occured while saving the model: ', e)


def load_model(filepath):
    try:
        return joblib.load(filepath)
    except:
        print('Error occured while loading the model:  ', e)


def predict(data, model):
    try:
        result = model.predict(data)
        return result
    except Exception as e:
        print('Error occured while making prediction: ', e)
