# Disaster Response Pipeline Project

### File Structure:
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


