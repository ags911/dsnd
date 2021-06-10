# import libraries
import json
import plotly
import pandas as pd
from sklearn.base import BaseEstimator,TransformerMixin
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

# download library for all-nltk
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag, word_tokenize

app = Flask(__name__)

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Starting Verb Extractor class
    
    This class is used to extract the starting verb of each sentence, this will be used to create new features for the machine learning classifier.
    """

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    # return the self from transformer
    def fit(self, X, y = None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)
    
def tokenize(text):
    """
    Tokenize text messages function
    
    Arguments:
        text: text message to be tokenized
    Output:
        clean_tokens: a list containing tokens extracted from text input
    """

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens

# load data
engine = create_engine('sqlite:///{}'.format('data/DisasterResponse.db'))
df = pd.read_sql_table('DisasterResponse_table', engine)

# load model
model = joblib.load('models/classifier.pkl')

# index webpage displays visuals and receives user input text for model
@app.route('/')
@app.route('/index')

def index():
    """
    Index function
    
    This function is used to init Plotly and show visualizations to users on the web app by inputting data.
    """
    
    # use data to display charts
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    related_counts = df.groupby('related').count()['message']
    related_names = list(related_counts.index)
    
    # create charts
    graphs = [
            # GRAPH 1 - Genre Graph
        {
            'data': [
                Bar(
                    x = genre_names,
                    y = genre_counts
                )
            ],
            
            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
            # GRAPH 2 - Category Graph    
        {
            'data': [
                Bar(
                    x = related_names,
                    y = related_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Messages Related to Disaster',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Related",
                }
            }
        }
    ]
    
    # encode plotly graphs using JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

# web page that handles user query and displays model results
@app.route('/go')

def go():
    """
    Go function
    
    This function is used to save inputs as queries and refer to model for classification. The go.html file will also be rendered as a web page for viewing.
    """
    # save user input in query
    query = request.args.get('query', '') 
    
    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))
    
    # this will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query = query,
        classification_result = classification_results
    )

def main():
    app.run(host = '0.0.0.0', port = 3001, debug = True)
if __name__ == '__main__':
    main()
