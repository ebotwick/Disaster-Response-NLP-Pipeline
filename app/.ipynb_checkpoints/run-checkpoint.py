import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings('ignore')



app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        if tok not in stopwords.words("english"):
            cleaned = lemmatizer.lemmatize(tok).strip()
            clean_tokens.append(cleaned)

    return clean_tokens

# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('messagesCategorizedCommandLine', engine)

# load model
model = joblib.load("/mnt/artifacts/clf.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    category_counts = list(df.iloc[:,4:].sum().sort_values().values)
    category_keys = list(df.iloc[:,4:].sum().sort_values().keys())


    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {

            'data': [
                Bar(
                    x=category_keys,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Frequency of Request Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Request Category",
                     'tickangle':25
                },
                'margin':{
                'l':0,
                'r':-20,
                't':-25,
                'b':-20
                }
            }
        }


    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
