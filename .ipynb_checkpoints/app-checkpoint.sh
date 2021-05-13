#!/usr/bin/env bash
 
# This is a bash script for Domino's App publishing feature
# Learn more at http://support.dominodatalab.com/hc/en-us/articles/209150326
 
 
## First install necessary packages not already in environment
pip install plotly

## Flask example
## This is an example of the code you would need in this bash script for a Python/Flask app
#export LC_ALL=C.UTF-8
#export LANG=C.UTF-8
#export FLASK_APP=app-flask.py
 
#set FLASK_APP=run.py
export FLASK_APP=run.py
export FLASK_DEBUG=1
 
#python -m flask run --host=0.0.0.0 --port=8888
python -m flask run --host=0.0.0.0 --port=8888