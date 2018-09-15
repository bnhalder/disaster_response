# import libraries
import sys
import sqlite3
import pandas as pd
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet'])
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib


def load_data(database_filepath):
    """ Read data from sqlite database
        arguments: database file name
        returns: returns X, y and labels
    """

    # create sqlalchemy engine and read data from database
    conn = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('disaster_messages_table', conn)
    
    # extract input strings for ML model
    X = df.message.values

    # extract label data for ML model
    labels = df.columns[4:]
    y = df[labels].values

    # returns x, y and labels
    return X, y, labels


def tokenize(text):
    """ String tokenizer
        arguments: string text
        returns: list of tokens
    """
 
    # detect url if any in input string and removes them using string 'urlplaceholder'
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    # tokenize the text
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    # clean tokens
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    # return list of clean tokens
    return clean_tokens   


def build_model():
    """ Builds Machine learning pipeline
        arguments: None
        returns: Machine learning model
    """

    # build the pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier())
    ])

    # builds the parameter list for grid search
    parameters = {
        #'vect__ngram_range': ((1, 1), (1, 2)),
        #'vect__max_df': (0.5, 0.75, 1.0),
        #'vect__max_features': (None, 5000, 10000),
        #'tfidf__use_idf': (True, False),
        'clf__n_estimators': [50, 100, 200],
        'clf__min_samples_split': [2, 3, 4]
    }
    
    # create grid object
    grid_obj = GridSearchCV(pipeline, param_grid=parameters)
    
    #return grid object
    return grid_obj


def evaluate_model(model, X_test, Y_test, category_names):
    """ Evaluate model on test set
        arguments: model, test set, test labels, category names
        retunrs: none
    """

    # get predictions
    Y_pred = model.predict(X_test)

    # evaluate and get summary
    print(classification_report(Y_test, Y_pred, target_names=category_names))


def save_model(model, model_filepath):
    """ Save the trained model
        arguments: model, filename
        returns: None
    """

    # save model using joblib library
    model = joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
