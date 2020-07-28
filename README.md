# Disaster Response Pipeline Project

## Purpose

This repository contains all the nessary files to run a webapp that can classify requests for disaster relief in natural language to one of several categories. In categorizing messages, disaster response efforts can be more quickly organized and carried out and so relief can be delivered more quickly and to more people in need.

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - Run the ETL pipeline that cleans training data and stores in database
        `python scripts/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - Run the ML pipeline that pulls data from the database created in the ETL phase and trains a classifier and saves
        `python scripts/train_classifier.py data/DisasterResponse.db RandomForestClassifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. From a terminal run 'env|grep WORK' and you should see output similar to

![TerminalScreenshot](/images/TerminalScreenshot.png)

4. Access the app in a web browser by going to https://SPACEID-3001.SPACEDOMAIN
note - if you do not have any returned output of the 'env|grep WORK' command access the webapp at http://0.0.0.0:3001/

5. Once on the site, type in a message in the text box and click 'classify message' to see how that message would be routed in a disaster scenario
