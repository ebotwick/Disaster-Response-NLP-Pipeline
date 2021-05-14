from flask import render_template
from flask_app import app

@app.route('/')
def index():
    return render_template('master.html')


@app.route('/go')
def index():
    return render_template('go.html')