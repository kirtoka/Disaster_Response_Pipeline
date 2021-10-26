import json
import plotly
import pandas as pd
import re

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
#from sklearn.externals import joblib
import sqlalchemy as db

# Add tokenize function and TextLengthExtractor class to make joblib
# working as expected
import sys
import os
sys.path.append(
    os.path.join(os.path.dirname(__file__), "..", "models"))
from functions import (tokenize, TextLengthExtractor)


app = Flask(__name__,static_url_path='/static')

# load data
engine = db.create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_data', engine)

# load model
model = joblib.load("../models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # Create statistics for visualizations
    graphs = []

    # 
    # graph 1: Count of genres
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    graph_one = Bar(x = genre_names, y = genre_counts)
    layout_one = dict(title = 'Distribution of Message Genres',
                    xaxis = dict(title = 'Count'),
                    yaxis = dict(title = 'Genre'),
                    )

    graphs.append(dict({'data': [graph_one], 'layout': layout_one}))

    # 
    # graph 2: Count of messages per category
    df_cat = df.drop(columns=['index', 'id', 'message', 'original', 'genre'])
    cat_counts = df_cat.sum().sort_values(ascending=False)
    cat_names = cat_counts.index
    
    graph_two = Bar(x = cat_names, y = cat_counts)
    layout_two = dict(title = 'Distribution of Categories',
                    xaxis = dict(title = 'Count'),
                    yaxis = dict(title = 'Category'),
                    )

    graphs.append(dict({'data': [graph_two], 'layout': layout_two}))
    
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
    first_category = [i for i, x in enumerate(df.columns) if x == 'related'][0]
    categories = df.columns[first_category:]
    classification_prediction = model.predict([query])[0]
    classification_results = dict(zip(categories, classification_prediction))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=False)


if __name__ == '__main__':
    main()