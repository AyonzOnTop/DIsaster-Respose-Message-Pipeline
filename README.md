# Disaster Response Pipeline Project

### Project Motivation
With the frequency of disasters around the world increasing daily, the ability to repsond promptly to this cases is very important.
This project is on creating a machine learning pipeline to categorize disaster events so that we can promptly send the messages to an appropriate disaster relief agency. We will train on data set containing real messages that were sent during disaster events from Figure Eight.


### Description

Project consists from three parts:

1. ETL Pipeline
Data cleaning pipeline contained in data/process_data.py:
- Loads the messages and categories datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database

2. ML Pipeline
Machine learning pipeline contained in model/train_classifier.py:
- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

3. Flask Web App
Web app where an emergency worker can input a new message and get classification results in several categories. The web app also displays visualizations of the data.

### Work Files Descriptions
1. data
- Disaster_categories.csv: dataset including all the categories    
- Disaster_messages.csv: dataset including all the messages
- Process_data.py: ETL pipeline scripts to read, clean, and save data into a database
- DisasterResponse.db: output of the ETL pipeline, i.e. SQLite database containing messages and categories data

2. models
- Train_classifier.py: machine learning pipeline scripts to train a RandomForest classification algorithm and export the classifier for the web app
- Classifier.pkl: output of the machine learning pipeline, i.e. a trained classifer

3. app
- run.py: Flask file to run the web application
- templates contains html file for the web application

### Result
- An ETL pipleline was built to read data from two csv files, clean data, and save data into a SQLite database.
- A machine learning pipepline was developed to train a classifier to performs multi-output classification on the 36 categories in the dataset.
- A Flask app was created to show data visualization and classify the message that user enters on the web page



### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
          `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
