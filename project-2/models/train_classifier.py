"""
Train Classifier

This Python script is a machine learning pipeline that loads data from the SQLite database then splits the dataset into training and test sets.
Next, it builds a text processing pipeline using NLP and machine learning pipeline using sklearn. The final steps are to train and tune the model using
GridSearchCV before outputting the results on the test set. Once the final model has been processed it is exported as a pickle file.
"""

# import ntlk modules
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# import libraries
import numpy as np
import pandas as pd
import sys
import os
import re
from sqlalchemy import create_engine
from scipy.stats import gmean
import pickle

# import sklearn modules
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator,TransformerMixin

def load_data_from_db(database_filepath):
    """
    Load Data from the database function
    
    Arguments:
        database_filepath: path to the SQLite database
    Output:
        X: dataframe column containing features
        y: dataframe columns containing labels
        category_names: a list of category names
    """

    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('DisasterResponse_table', engine)
    
    # assigning x and y variables
    X = df['message']
    y = df.iloc[:,4:]
    
    # used for visualization
    category_names = y.columns 
    
    return X, y, category_names

def tokenize(text, url_place_holder_string = "urlplaceholder"):
    """
    Tokenize text messages function
    
    Arguments:
        text: text message to be tokenized
    Output:
        clean_tokens: a list containing tokens extracted from text input
    """
    
    # use regex to replace each url with a placeholder string
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # use regex to extract all the urls from input text
    detected_urls = re.findall(url_regex, text)
    
    # replace urls by looping through detected_urls
    for detected_url in detected_urls:
        text = text.replace(detected_url, url_place_holder_string)

    # extract word tokens from text input
    tokens = nltk.word_tokenize(text)
    
    # use lemmatization to obtain the root forms of inflected/derived words
    lemmatizer = nltk.WordNetLemmatizer()

    # creating a list of clean tokens
    clean_tokens = [lemmatizer.lemmatize(w).lower().strip() for w in tokens]
    
    return clean_tokens

# Build a custom transformer which will extract the starting verb of a sentence
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Starting Verb Extractor Class
    
    This class extract the starting verb of a sentence,
    creating a new feature for the ML classifier
    """

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    # Given it is a tranformer we can return the self 
    def fit(self, X, y = None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def build_pipeline():
    """
    Build Pipeline Function
    
    Output:
        Scikit ML Pipeline to process text based messages and apply a classifier.
        
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('count_vectorizer', CountVectorizer(tokenizer=tokenize)),
                ('tfidf_transformer', TfidfTransformer())
            ])),

            ('starting_verb_transformer', StartingVerbExtractor())
        ])),

        ('classifier', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    parameters = {'classifier__estimator__learning_rate': [0.01, 0.02, 0.05],
              'classifier__estimator__n_estimators': [10, 20, 40]}

    cv = GridSearchCV(pipeline, param_grid = parameters, 
                      scoring = 'f1_micro', n_jobs = -1, verbose = 2)

    #cv.fit(X_train, y_train)
    
    return cv

def multioutput_fscore(y_true,y_pred,beta=1):
    """
    MultiOutput F-score Function
    <CHANGE THIS WHOLE SECTION>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    A performance metric creating a geometric mean of the fbeta_score, it is computed on each label.
    - Compatible with multi-label and multi-class problems.
    - Features some peculiarities such as geometric mean, 100% removal, etc. These are used to exclude
    trivial solutions by deliberatly under-estimating the f-beta_score average.
    The aim is to avoid issues in dealing with multi-class/multi-label imbalanced cases.
    
    Used as a scorer for GridSearchCV:
        scorer = make_scorer(multioutput_fscore,beta=1)
        
    Arguments:
        y_true: List of labels
        y_prod: List of predictions
        beta: Beta value to be used to calculate fscore metric
    
    Output:
        f1-score: Calculation geometric mean of f-score
    """
    
    # If provided y predictions is a dataframe then extract the values from that
    if isinstance(y_pred, pd.DataFrame) == True:
        y_pred = y_pred.values
    
    # If provided y actuals is a dataframe then extract the values from that
    if isinstance(y_true, pd.DataFrame) == True:
        y_true = y_true.values
    
    f1score_list = []
    for column in range(0,y_true.shape[1]):
        score = fbeta_score(y_true[:,column],y_pred[:,column],beta,average='weighted')
        f1score_list.append(score)
        
    f1score = np.asarray(f1score_list)
    f1score = f1score[f1score<1]
    
    # Get the geometric mean of f1score
    f1score = gmean(f1score)
    return f1score

def evaluate_pipeline(pipeline, X_test, y_test, category_names):
    """
    Evaluate Model Function
    
    This function applies a ML pipeline to a test set then prints out the model performance with the accuracy and f1-score
    
    Arguments:
        pipeline: A valid scikit ML Pipeline
        X_test: Test features
        y_test: Test labels
        category_names: label names (multi-output)
    """
    y_pred = pipeline.predict(X_test)
    
    multi_f1 = multioutput_fscore(y_test,y_pred, beta = 1)
    overall_accuracy = (y_pred == y_test).mean().mean()

    print('Average overall accuracy {0:.2f}%'.format(overall_accuracy*100))
    print('F1 score (custom definition) {0:.2f}%'.format(multi_f1*100))

    # Print the whole classification report.
    y_pred = pd.DataFrame(y_pred, columns = y_test.columns)
    
    for column in y_test.columns:
        print('Model Performance with Category: {}'.format(column))
        print(classification_report(y_test[column],y_pred[column]))


def save_model_as_pickle(pipeline, pickle_filepath):
    """
    Save Pipeline Function
    
    This function saves the trained model as a Pickle (.pkl) file, which will be loaded later.
    
    Arguments:
        pipeline: GridSearchCV or Scikit Pipelin object
        pickle_filepath: destination path to save .pkl file
    
    """
    pickle.dump(pipeline, open(pickle_filepath, 'wb'))
    
    #pkl_filename = '{}'.format(pickle_filepath)
    #with open(pkl_filename, 'wb') as file:
    #    pickle.dump(pipline, file)

def main():
    """
    Train Classifier Main Function
    
    This function applies the Machine Learning Pipeline:
        1) Extract data from SQLite db
        2) Train ML model on training set
        3) Estimate model performance on test set
        4) Save trained model as Pickle
    
    """
    if len(sys.argv) == 3:
        database_filepath, pickle_filepath = sys.argv[1:]
        print('Loading data from {} ...'.format(database_filepath))
        X, y, category_names = load_data_from_db(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building the pipeline ...')
        pipeline = build_pipeline()
        
        print('Training the pipeline ...')
        pipeline.fit(X_train, y_train)
        
        print('Best parameters set:')
        print(pipeline.best_params_)
        pipeline = pipeline.best_estimator_
        
        print('Evaluating model...')
        evaluate_pipeline(pipeline, X_test, y_test, category_names)

        print('Saving pipeline to {} ...'.format(pickle_filepath))
        save_model_as_pickle(pipeline, pickle_filepath)

        print('Trained model saved!')

    else:
         print("Please provide the arguments correctly: \nSample Script Execution:\n\
                > python train_classifier.py ../data/disaster_response_db classifier.pkl \n\
                Arguments Description: \n\
                1) Path to SQLite destination database (e.g. DisasterResponse.db)\n\
                2) Path to pickle file name where ML model needs to be saved (e.g. classifier.pkl")

if __name__ == '__main__':
    main()
