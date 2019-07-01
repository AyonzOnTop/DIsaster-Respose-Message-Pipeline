import sys
from sqlalchemy import create_engine
import pandas as pd
import numpy as np

import nltk
nltk.download(['punkt', 'wordnet'])
nltk.download('stopwords')
import re

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score


def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath )
    df = pd.read_sql(database_filepath,engine)
    X = df.message.values
    Y = df.iloc[:, 5:]
    category_names = list(df.columns[4:])
    return X, Y, category_names
    


def tokenize(text):
    '''
    Tokenize message test
    input:
        Message text
    
    output: 
        lemmed tokenize and lemmatized text
        
    '''
    #Normalize text
    
    text = re.sub(r"[^a-zA-Z0-9]", ' ', text.lower())
    
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    detected_urls = re.findall(url_regex, text)  # find urls
    
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')  # replace urls
    
    #Tokenize
    words = word_tokenize(text)
    
    #Remove stopwords
    words = [w for w in words if w not in stopwords.words('english')]
    
    #Lemmatize
    lemmatizer = WordNetLemmatizer()
    lemmed = [lemmatizer.lemmatize(w, pos = 'n').strip() for w in words]
    lemmed = [lemmatizer.lemmatize(w, pos = 'v').strip() for w in lemmed]
    
    return lemmed
    


def build_model():
    '''
    input: 
    output: catgorization of result
    '''
    
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {'clf__estimator__n_estimators': [50, 100]
                  #'clf__estimator__min_samples_split': [2, 3, 4]
                  #'clf__estimator__criterion': ['entropy', 'gini']
                 }
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, y_test, category_names):
    '''
    Evaluate model performance using test data
    Input: 
        model: Model to be evaluated
        X_test: Test data (features)
        y_test: True lables for Test data
        category_names: Labels for 36 categories
    Output:
        Print accuracy and classfication report for each category
    '''
    y_pred = model.predict(X_test)
    for category in range(len(category_names)):
        print("Category:", category_names[category],"\n", classification_report(y_test.iloc[:, category], y_pred[:, category]))
        print('Accuracy of %25s: %.2f' %(category_names[category], accuracy_score(y_test.iloc[:, category].values, y_pred[:,category])))


def save_model(model, model_filepath):
    '''
    Saves the model to disk
    INPUT 
        model: The model to be saved
        model_filepath: Filepath for where the model is to be saved
    OUTPUT
        While there is no specific item that is returned to its calling method, this method will save the model as a pickle file.
    '''    
    temp_pickle = open(model_filepath, 'wb')
    pickle.dump(model, temp_pickle)
    temp_pickle.close()

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

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