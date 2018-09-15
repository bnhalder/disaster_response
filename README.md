# Disaster Response Pipeline Project

### Description
In this project, we have a data set containing real messages that were sent during disaster events and we are creating an ETL pipeline to process that data which will eventually be fed to a machine learning pipeline to categorize disaster events so that we can send the messages to an appropriate disaster relief agency.

This project also includes a web app where an emergency worker can input a new message and get classification results on several categories. The web app will also display visualizations of the data.


### Project Components:
There are three components in this project.

1. ETL Pipeline
    A Python script - process_data.py, contains the data cleaning pipeline that:
        Loads the messages and categories datasets
        Merges the two datasets
        Cleans the data
        Stores it in a SQLite database

2. ML Pipeline
    A Python script - train_classifier.py, contains a machine learning pipeline that:
	Loads data from the SQLite database
	Splits the dataset into training and test sets
	Builds a text processing and machine learning pipeline
	Trains and tunes a model using GridSearchCV
	Outputs results on the test set
	Exports the final model as a pickle file

3. Flask Web App


### File Structure:
Below is the file and folder structure of the repository

1. ./app - contains the flask app
   ./app/templates - contains the html templates

2. ./data/disaster_categories.csv - categories data in csv format
   ./data/disaster_messages.csv - messages data in csv format
   ./data/process_data.py - python script for ETL pipeline
   ./data/DisasterResponse.db - file to contain sqlite3 database

3. ./model/train_classifier.py - python script for ML pipeline
   ./model/classifier.pkl - saved machine learning model

4. ./notebooks - notebooks for own experimentation, not needed for the app to run


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


