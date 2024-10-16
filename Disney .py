#!/usr/bin/env python
# coding: utf-8

# In[19]:


pip install git+https://github.com/tweepy/tweepy.git


# In[20]:


get_ipython().system('pip install dash')


# In[ ]:


get_ipython().system('pip install tweepy')


# In[ ]:


get_ipython().system('pip install textblob')


# In[17]:


get_ipython().system('pip install twikit')


# In[ ]:





# In[ ]:





# In[24]:


consumer_key = os.environ.get('VDCvfPJkSmuzo7pYCiH7jXDS7')
consumer_secret = os.environ.get('GtdTHkeMhVWwokOCINCBjWYIaX4KnBR04Dc4DRRBkzwB3UPfy6')
access_token = os.environ.get('1843760341819219968-f5T3OsBo8FMESgETbUZM70HcuI6vim')
access_token_secret = os.environ.get('OAsly5gCb6Hs3kOM0G0O5NQTjT4qMg7MlUAfwCnmZB6Ng')


# In[1]:


import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import tweepy
from textblob import TextBlob
import re
from datetime import datetime, timedelta
import json
import os

# Function to load Twitter API credentials from config file
def load_config():
    config_path = 'twitter_config.json'
    if not os.path.exists(config_path):
        # Create a template config file if it doesn't exist
        template_config = {
            "consumer_key": "VDCvfPJkSmuzo7pYCiH7jXDS7",
            "consumer_secret": "GtdTHkeMhVWwokOCINCBjWYIaX4KnBR04Dc4DRRBkzwB3UPfy6",
            "access_token": "1843760341819219968-f5T3OsBo8FMESgETbUZM70HcuI6vim",
            "access_token_secret": "OAsly5gCb6Hs3kOM0G0O5NQTjT4qMg7MlUAfwCnmZB6Ng"
        }
        with open(config_path, 'w') as f:
            json.dump(template_config, f, indent=4)
        print(f"Please fill in your Twitter API credentials in {config_path}")
        exit(1)
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Check if all credentials are present and not placeholder values
    for key, value in config.items():
        if value.startswith("YOUR_"):
            print(f"Please replace the placeholder value for {key} in {config_path}")
            exit(1)
    
    return config

# Load Twitter API credentials
config = load_config()

# Authenticate with Twitter
try:
    auth = tweepy.OAuthHandler(config['consumer_key'], config['consumer_secret'])
    auth.set_access_token(config['access_token'], config['access_token_secret'])
    api = tweepy.API(auth)
except Exception as e:
    print(f"Error authenticating with Twitter API: {str(e)}")
    exit(1)

# Initialize data storage
sentiment_data = pd.DataFrame(columns=['timestamp', 'sentiment', 'topic'])

# Function to clean tweets
def clean_tweet(tweet):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

# Function to get sentiment
def get_tweet_sentiment(tweet):
    analysis = TextBlob(clean_tweet(tweet))
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity == 0:
        return 'neutral'
    else:
        return 'negative'

# Function to fetch tweets and analyze sentiment
def fetch_tweets(topic):
    global sentiment_data
    try:
        tweets = tweepy.Cursor(api.search_tweets, q=topic, lang="en").items(100)
        for tweet in tweets:
            sentiment = get_tweet_sentiment(tweet.text)
            sentiment_data = sentiment_data.append({
                'timestamp': tweet.created_at,
                'sentiment': sentiment,
                'topic': topic
            }, ignore_index=True)
        
        # Keep only the last 24 hours of data
        sentiment_data = sentiment_data[sentiment_data['timestamp'] > datetime.now() - timedelta(days=1)]
    except Exception as e:
        print(f"Error fetching tweets: {str(e)}")

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1("Disney Real-time Sentiment Dashboard"),
    dcc.Dropdown(
        id='topic-dropdown',
        options=[
            {'label': 'Disney+', 'value': 'Disney+'},
            {'label': 'Marvel', 'value': 'Marvel'},
            {'label': 'Star Wars', 'value': 'StarWars'},
            {'label': 'Disney Parks', 'value': 'DisneyParks'}
        ],
        value='Disney+',
        style={'width': '50%'}
    ),
    dcc.Graph(id='sentiment-graph'),
    dcc.Interval(
        id='interval-component',
        interval=60*1000,  # in milliseconds, update every 1 minute
        n_intervals=0
    )
])

# Callback to update the graph
@app.callback(Output('sentiment-graph', 'figure'),
              [Input('interval-component', 'n_intervals'),
               Input('topic-dropdown', 'value')])
def update_graph(n, topic):
    fetch_tweets(topic)
    df = sentiment_data[sentiment_data['topic'] == topic]
    
    positive = df[df['sentiment'] == 'positive'].groupby('timestamp').size()
    negative = df[df['sentiment'] == 'negative'].groupby('timestamp').size()
    neutral = df[df['sentiment'] == 'neutral'].groupby('timestamp').size()
    
    return {
        'data': [
            go.Scatter(x=positive.index, y=positive.values, name='Positive'),
            go.Scatter(x=negative.index, y=negative.values, name='Negative'),
            go.Scatter(x=neutral.index, y=neutral.values, name='Neutral')
        ],
        'layout': go.Layout(
            title=f'Sentiment Analysis for {topic}',
            xaxis={'title': 'Time'},
            yaxis={'title': 'Number of Tweets'}
        )
    }

if __name__ == '__main__':
    app.run_server(debug=True)


# In[ ]:





# In[2]:


import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import random
from datetime import datetime, timedelta

# Initialize data storage
sentiment_data = pd.DataFrame(columns=['timestamp', 'sentiment', 'topic'])

# Function to generate simulated sentiment data
def generate_sentiment_data(topic, num_points=100):
    global sentiment_data
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=24)
    
    new_data = []
    for _ in range(num_points):
        timestamp = start_time + (end_time - start_time) * random.random()
        sentiment = random.choice(['positive', 'negative', 'neutral'])
        new_data.append({
            'timestamp': timestamp,
            'sentiment': sentiment,
            'topic': topic
        })
    
    # Use concat instead of append
    sentiment_data = pd.concat([sentiment_data, pd.DataFrame(new_data)], ignore_index=True)
    
    # Keep only the last 24 hours of data
    sentiment_data = sentiment_data[sentiment_data['timestamp'] > datetime.now() - timedelta(days=1)]

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1("Disney Simulated Sentiment Dashboard"),
    dcc.Dropdown(
        id='topic-dropdown',
        options=[
            {'label': 'Disney+', 'value': 'Disney+'},
            {'label': 'Marvel', 'value': 'Marvel'},
            {'label': 'Star Wars', 'value': 'StarWars'},
            {'label': 'Disney Parks', 'value': 'DisneyParks'}
        ],
        value='Disney+',
        style={'width': '50%'}
    ),
    dcc.Graph(id='sentiment-graph'),
    dcc.Interval(
        id='interval-component',
        interval=60*1000,  # in milliseconds, update every 1 minute
        n_intervals=0
    )
])

# Callback to update the graph
@app.callback(Output('sentiment-graph', 'figure'),
              [Input('interval-component', 'n_intervals'),
               Input('topic-dropdown', 'value')])
def update_graph(n, topic):
    generate_sentiment_data(topic)
    df = sentiment_data[sentiment_data['topic'] == topic]
    
    positive = df[df['sentiment'] == 'positive'].groupby('timestamp').size()
    negative = df[df['sentiment'] == 'negative'].groupby('timestamp').size()
    neutral = df[df['sentiment'] == 'neutral'].groupby('timestamp').size()
    
    return {
        'data': [
            go.Scatter(x=positive.index, y=positive.values, name='Positive'),
            go.Scatter(x=negative.index, y=negative.values, name='Negative'),
            go.Scatter(x=neutral.index, y=neutral.values, name='Neutral')
        ],
        'layout': go.Layout(
            title=f'Simulated Sentiment Analysis for {topic}',
            xaxis={'title': 'Time'},
            yaxis={'title': 'Number of Simulated Data Points'}
        )
    }

if __name__ == '__main__':
    app.run_server(debug=True)


# In[3]:


import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from collections import Counter

# Initialize data storage
sentiment_data = pd.DataFrame(columns=['timestamp', 'sentiment', 'topic', 'engagement', 'source', 'keywords'])

# Function to generate simulated sentiment data
def generate_sentiment_data(topic, num_points=100):
    global sentiment_data
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=24)
    
    sources = ['Twitter', 'Facebook', 'Instagram', 'News Articles']
    keywords = ['movie', 'show', 'character', 'theme park', 'streaming', 'franchise', 'entertainment', 'family']
    
    new_data = []
    for _ in range(num_points):
        timestamp = start_time + (end_time - start_time) * random.random()
        sentiment = random.choice(['positive', 'negative', 'neutral'])
        engagement = random.randint(1, 1000)
        source = random.choice(sources)
        tweet_keywords = random.sample(keywords, random.randint(1, 3))
        
        new_data.append({
            'timestamp': timestamp,
            'sentiment': sentiment,
            'topic': topic,
            'engagement': engagement,
            'source': source,
            'keywords': ', '.join(tweet_keywords)
        })
    
    sentiment_data = pd.concat([sentiment_data, pd.DataFrame(new_data)], ignore_index=True)
    
    # Keep only the last 24 hours of data
    sentiment_data = sentiment_data[sentiment_data['timestamp'] > datetime.now() - timedelta(days=1)]

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1("Enhanced Disney Simulated Sentiment Dashboard"),
    dcc.Dropdown(
        id='topic-dropdown',
        options=[
            {'label': 'Disney+', 'value': 'Disney+'},
            {'label': 'Marvel', 'value': 'Marvel'},
            {'label': 'Star Wars', 'value': 'StarWars'},
            {'label': 'Disney Parks', 'value': 'DisneyParks'}
        ],
        value='Disney+',
        style={'width': '50%'}
    ),
    dcc.Tabs([
        dcc.Tab(label='Sentiment Over Time', children=[
            dcc.Graph(id='sentiment-graph')
        ]),
        dcc.Tab(label='Sentiment Distribution', children=[
            dcc.Graph(id='sentiment-pie')
        ]),
        dcc.Tab(label='Engagement Analysis', children=[
            dcc.Graph(id='engagement-graph')
        ]),
        dcc.Tab(label='Source Distribution', children=[
            dcc.Graph(id='source-bar')
        ]),
        dcc.Tab(label='Keyword Analysis', children=[
            dcc.Graph(id='keyword-treemap')
        ])
    ]),
    dcc.Interval(
        id='interval-component',
        interval=60*1000,  # in milliseconds, update every 1 minute
        n_intervals=0
    )
])

@app.callback(
    [Output('sentiment-graph', 'figure'),
     Output('sentiment-pie', 'figure'),
     Output('engagement-graph', 'figure'),
     Output('source-bar', 'figure'),
     Output('keyword-treemap', 'figure')],
    [Input('interval-component', 'n_intervals'),
     Input('topic-dropdown', 'value')]
)
def update_graphs(n, topic):
    generate_sentiment_data(topic)
    df = sentiment_data[sentiment_data['topic'] == topic]
    
    # Sentiment Over Time
    positive = df[df['sentiment'] == 'positive'].groupby('timestamp').size()
    negative = df[df['sentiment'] == 'negative'].groupby('timestamp').size()
    neutral = df[df['sentiment'] == 'neutral'].groupby('timestamp').size()
    
    sentiment_time = {
        'data': [
            go.Scatter(x=positive.index, y=positive.values, name='Positive'),
            go.Scatter(x=negative.index, y=negative.values, name='Negative'),
            go.Scatter(x=neutral.index, y=neutral.values, name='Neutral')
        ],
        'layout': go.Layout(
            title=f'Sentiment Over Time for {topic}',
            xaxis={'title': 'Time'},
            yaxis={'title': 'Number of Mentions'}
        )
    }
    
    # Sentiment Distribution
    sentiment_counts = df['sentiment'].value_counts()
    sentiment_pie = px.pie(
        values=sentiment_counts.values, 
        names=sentiment_counts.index, 
        title=f'Sentiment Distribution for {topic}'
    )
    
    # Engagement Analysis
    engagement_scatter = px.scatter(
        df, 
        x='timestamp', 
        y='engagement', 
        color='sentiment', 
        title=f'Engagement Analysis for {topic}',
        labels={'timestamp': 'Time', 'engagement': 'Engagement Level'}
    )
    
    # Source Distribution
    source_counts = df['source'].value_counts()
    source_bar = px.bar(
        x=source_counts.index, 
        y=source_counts.values, 
        title=f'Source Distribution for {topic}',
        labels={'x': 'Source', 'y': 'Number of Mentions'}
    )
    
    # Keyword Analysis
    all_keywords = [keyword for keywords in df['keywords'].str.split(', ') for keyword in keywords]
    keyword_counts = Counter(all_keywords)
    keyword_df = pd.DataFrame(list(keyword_counts.items()), columns=['keyword', 'count'])
    keyword_treemap = px.treemap(
        keyword_df, 
        path=['keyword'], 
        values='count',
        title=f'Keyword Analysis for {topic}'
    )
    
    return sentiment_time, sentiment_pie, engagement_scatter, source_bar, keyword_treemap

if __name__ == '__main__':
    app.run_server(debug=True)
    


# In[4]:


import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import Counter
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Initialize data storage
sentiment_data = pd.DataFrame(columns=['timestamp', 'text', 'sentiment', 'topic', 'engagement', 'source'])

# Function to scrape news articles
def scrape_news(topic, num_articles=10):
    url = f"https://news.google.com/rss/search?q={topic}&hl=en-US&gl=US&ceid=US:en"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, features="xml")
    articles = soup.findAll('item')
    
    news_data = []
    for a in articles[:num_articles]:
        title = a.title.text
        pub_date = datetime.strptime(a.pubDate.text, "%a, %d %b %Y %H:%M:%S %Z")
        news_data.append({
            'timestamp': pub_date,
            'text': title,
            'topic': topic,
            'source': 'News'
        })
    return news_data

# Function to perform sentiment analysis
def get_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity == 0:
        return 'neutral'
    else:
        return 'negative'

# Function to generate engagement
def generate_engagement(sentiment):
    if sentiment == 'positive':
        return np.random.normal(500, 100)
    elif sentiment == 'neutral':
        return np.random.normal(300, 50)
    else:
        return np.random.normal(400, 150)

# Function to fetch and process data
def fetch_data(topic):
    global sentiment_data
    new_data = scrape_news(topic)
    
    for item in new_data:
        item['sentiment'] = get_sentiment(item['text'])
        item['engagement'] = generate_engagement(item['sentiment'])
    
    sentiment_data = pd.concat([sentiment_data, pd.DataFrame(new_data)], ignore_index=True)
    
    # Keep only the last 7 days of data
    sentiment_data = sentiment_data[sentiment_data['timestamp'] > datetime.now() - timedelta(days=7)]

# Train sentiment analysis model
def train_model():
    # Assuming we have a labeled dataset 'labeled_data.csv'
    labeled_data = pd.read_csv('labeled_data.csv')
    X_train, X_test, y_train, y_test = train_test_split(labeled_data['text'], labeled_data['sentiment'], test_size=0.2)
    
    vectorizer = TfidfVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    
    model = MultinomialNB()
    model.fit(X_train_vectorized, y_train)
    
    joblib.dump(vectorizer, 'vectorizer.joblib')
    joblib.dump(model, 'sentiment_model.joblib')
    
    y_pred = model.predict(X_test_vectorized)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    return accuracy, conf_matrix, class_report

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1("Advanced Disney Sentiment Dashboard"),
    dcc.Dropdown(
        id='topic-dropdown',
        options=[
            {'label': 'Disney+', 'value': 'Disney+'},
            {'label': 'Marvel', 'value': 'Marvel'},
            {'label': 'Star Wars', 'value': 'StarWars'},
            {'label': 'Disney Parks', 'value': 'DisneyParks'}
        ],
        value='Disney+',
        style={'width': '50%'}
    ),
    dcc.Tabs([
        dcc.Tab(label='Sentiment Over Time', children=[
            dcc.Graph(id='sentiment-graph')
        ]),
        dcc.Tab(label='Sentiment Distribution', children=[
            dcc.Graph(id='sentiment-pie')
        ]),
        dcc.Tab(label='Engagement Analysis', children=[
            dcc.Graph(id='engagement-graph')
        ]),
        dcc.Tab(label='Source Distribution', children=[
            dcc.Graph(id='source-bar')
        ]),
        dcc.Tab(label='Keyword Analysis', children=[
            dcc.Graph(id='keyword-treemap')
        ]),
        dcc.Tab(label='Model Accuracy', children=[
            html.Div(id='accuracy-stats')
        ])
    ]),
    dcc.Interval(
        id='interval-component',
        interval=300*1000,  # in milliseconds, update every 5 minutes
        n_intervals=0
    ),
    html.Button('Train Model', id='train-model-button'),
    html.Div(id='model-training-output')
])

@app.callback(
    [Output('sentiment-graph', 'figure'),
     Output('sentiment-pie', 'figure'),
     Output('engagement-graph', 'figure'),
     Output('source-bar', 'figure'),
     Output('keyword-treemap', 'figure'),
     Output('accuracy-stats', 'children')],
    [Input('interval-component', 'n_intervals'),
     Input('topic-dropdown', 'value')]
)
def update_graphs(n, topic):
    fetch_data(topic)
    df = sentiment_data[sentiment_data['topic'] == topic]
    
    # Sentiment Over Time
    sentiment_time = px.line(
        df.groupby(['timestamp', 'sentiment']).size().unstack(),
        title=f'Sentiment Over Time for {topic}'
    )
    
    # Sentiment Distribution
    sentiment_pie = px.pie(
        df, 
        names='sentiment', 
        title=f'Sentiment Distribution for {topic}'
    )
    
    # Engagement Analysis
    engagement_scatter = px.scatter(
        df, 
        x='timestamp', 
        y='engagement', 
        color='sentiment', 
        title=f'Engagement Analysis for {topic}',
        labels={'timestamp': 'Time', 'engagement': 'Engagement Level'}
    )
    
    # Source Distribution
    source_bar = px.bar(
        df['source'].value_counts().reset_index(),
        x='index',
        y='source',
        title=f'Source Distribution for {topic}',
        labels={'index': 'Source', 'source': 'Number of Mentions'}
    )
    
    # Keyword Analysis
    all_keywords = df['text'].str.split().explode()
    keyword_counts = all_keywords.value_counts().head(20)
    keyword_treemap = px.treemap(
        names=keyword_counts.index,
        parents=[''] * len(keyword_counts),
        values=keyword_counts.values,
        title=f'Top 20 Keywords for {topic}'
    )
    
    # Model Accuracy
    try:
        model = joblib.load('sentiment_model.joblib')
        vectorizer = joblib.load('vectorizer.joblib')
        X_test = df['text']
        y_true = df['sentiment']
        X_test_vectorized = vectorizer.transform(X_test)
        y_pred = model.predict(X_test_vectorized)
        accuracy = accuracy_score(y_true, y_pred)
        conf_matrix = confusion_matrix(y_true, y_pred)
        class_report = classification_report(y_true, y_pred)
        
        accuracy_stats = [
            html.H3('Model Accuracy Statistics'),
            html.P(f'Accuracy: {accuracy:.2f}'),
            html.P('Confusion Matrix:'),
            html.Pre(str(conf_matrix)),
            html.P('Classification Report:'),
            html.Pre(class_report)
        ]
    except:
        accuracy_stats = html.P('Model not trained yet. Click "Train Model" to train the model.')
    
    return sentiment_time, sentiment_pie, engagement_scatter, source_bar, keyword_treemap, accuracy_stats

@app.callback(
    Output('model-training-output', 'children'),
    [Input('train-model-button', 'n_clicks')]
)
def train_model_callback(n_clicks):
    if n_clicks is None:
        return ''
    accuracy, conf_matrix, class_report = train_model()
    return [
        html.H3('Model Training Results'),
        html.P(f'Accuracy: {accuracy:.2f}'),
        html.P('Confusion Matrix:'),
        html.Pre(str(conf_matrix)),
        html.P('Classification Report:'),
        html.Pre(class_report)
    ]

if __name__ == '__main__':
    app.run_server(debug=True)


# In[ ]:




