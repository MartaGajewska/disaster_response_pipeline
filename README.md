# Disaster Response Pipeline Project

## About the project
This project is based on a data set containing real messages that were sent during disaster events, which was used to create a machine learning pipeline categorizing these events so that they can be distributed across appropriate disaster relief agencies. 

## Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Files in this repository
- data folder
    - disaster_messages.csv - messages sent during disaster events
    - disaster_categories.csv - categories assigned to messages
    - process_data.py - code processing two csv files mentioned above
    - DisasterRespnse.db - db in which results of the data cleaning stage are stored
- models folder
    - train_classifier.py - code building a classification model 
    - classifier.pkl - saved model
- app
    - template
    - master.html - main page of web app
    - go.html - classification result page of web app
    - run.py - Flask file that runs app
- README.md

## Acknowledgements
Data was provided by Figure Eight. Web app template and code scheleton was provided by Udacity.