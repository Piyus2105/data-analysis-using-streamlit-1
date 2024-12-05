import streamlit as st
import pandas as pd
import re
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob

# Load the data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/skathirmani/datasets/refs/heads/main/narendramodi_tweets.csv"
    return pd.read_csv(url)

# Load tweets
tweets = load_data()

# Preprocess data
tweets['created_at'] = pd.to_datetime(tweets['created_at'], errors='coerce')
tweets['year'] = tweets['created_at'].dt.year
tweets['month'] = tweets['created_at'].dt.month_name()
tweets['hour'] = tweets['created_at'].dt.hour
tweets['text_length'] = tweets['text'].apply(len)
tweets['sentiment'] = tweets['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
tweets['mention'] = tweets['text'].apply(lambda x: re.findall(r'@(\w+)', x))

# Sidebar filters
year_filter = st.sidebar.selectbox("Select Year:", sorted(tweets['year'].dropna().unique())) 
source_filter = st.sidebar.selectbox("Select Source:", sorted(tweets['source'].unique()))  
hour_filter = st.sidebar.slider(
    "Select Hour Range (Start Hour to End Hour):", 
    min_value=0, 
    max_value=23, 
    value=(0, 23), 
    step=1
)

# Filter data
filtered_tweets = tweets[ 
    (tweets['year'] == year_filter) & 
    (tweets['source'] == source_filter) & 
    (tweets['hour'].between(*hour_filter))
]

# Page title
st.title("Narendra Modi - Twitter Analysis")
st.subheader("by Piyush Kumar Agrawal")

# Metrics
num_tweets = filtered_tweets.shape[0]
avg_retweets = filtered_tweets['retweets_count'].mean() if num_tweets > 0 else 0
avg_likes = filtered_tweets['favorite_count'].mean() if num_tweets > 0 else 0
top_hashtag = (
    filtered_tweets['text'].str.findall(r'#\w+').explode().value_counts().idxmax()
    if num_tweets > 0 else "N/A"
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("No. of Tweets", num_tweets)
col2.metric("Average Retweets", round(avg_retweets, 2))
col3.metric("Average Likes", round(avg_likes, 2))
col4.metric("Top Hashtag", top_hashtag)

# Visualization 1: Month-wise Total Number of Tweets
month_data = filtered_tweets['month'].value_counts().reindex(
    ['January', 'February', 'March', 'April', 'May', 'June', 
     'July', 'August', 'September', 'October', 'November', 'December'], fill_value=0
)
st.subheader("Month-wise Total Number of Tweets")
st.line_chart(month_data)

# Visualization 2: Hour-wise Distribution of Tweets
hourly_data = filtered_tweets['hour'].value_counts().sort_index()
st.subheader("Hour-wise Tweet Activity")
st.bar_chart(hourly_data)

# Visualization 3: Top 10 Hashtags
st.subheader("Top 10 Hashtags Used")
all_hashtags = filtered_tweets['text'].str.findall(r'#\w+').explode().value_counts()
st.bar_chart(all_hashtags.head(10))

# Visualization 4: Retweets vs Likes Scatter Plot
st.subheader("Retweets vs Likes Scatter Plot")
if num_tweets > 0:
    scatter_fig = px.scatter(
        data_frame=filtered_tweets,
        x='retweets_count',
        y='favorite_count',
        size='favorite_count',
        hover_data=['text'],
        labels={'retweets_count': 'Retweets', 'favorite_count': 'Likes'},
        title="Retweets vs Likes",
        color_discrete_sequence=px.colors.sequential.Sunset
    )
    st.plotly_chart(scatter_fig)
else:
    st.write("No data available for Retweets vs Likes plot.")

# Table: Top 10 Tweets by Likes
st.subheader("Top 10 Tweets by Likes")
top_tweets = filtered_tweets.nlargest(10, 'favorite_count')[['text', 'favorite_count', 'retweets_count', 'created_at']]
st.dataframe(top_tweets)

# Pie chart for tweet sources
st.write("### Tweet Source Distribution")
source_counts = tweets['source'].value_counts().reset_index()
source_counts.columns = ['source', 'count']

# Pie chart for tweet sources
fig4 = px.pie(source_counts, values='count', names='source', title='Tweet Source Distribution', hole=0.3)
st.plotly_chart(fig4, use_container_width=True)

# 1. Sentiment Analysis of Tweets
st.subheader("Sentiment Analysis of Tweets")
sentiment_counts = filtered_tweets['sentiment'].apply(lambda x: 'Positive' if x > 0 else 'Negative' if x < 0 else 'Neutral').value_counts()
st.bar_chart(sentiment_counts)

# 2. Word Cloud of Most Frequent Words
st.subheader("Word Cloud of Most Frequent Words")
wordcloud = WordCloud(width=800, height=400, max_words=100).generate(' '.join(filtered_tweets['text']))
st.image(wordcloud.to_array(), use_column_width=True)

# 3. Tweet Length Distribution
st.subheader("Tweet Length Distribution")
st.bar_chart(filtered_tweets['text_length'].value_counts().sort_index())

# 4. Most Frequent Mentions
st.subheader("Most Frequent Mentions")
mentions = filtered_tweets['mention'].explode().value_counts().head(10)
st.bar_chart(mentions)

# 6. Time of Day Analysis (Heatmap of tweet activity by hour and day)
st.subheader("Time of Day Analysis (Heatmap)")
time_of_day_data = filtered_tweets.groupby(['hour', 'month']).size().unstack(fill_value=0)

# Create the heatmap using seaborn
plt.figure(figsize=(10, 6))
sns.heatmap(time_of_day_data, annot=True, cmap='YlGnBu', fmt='d', linewidths=0.5)

# Display the heatmap in Streamlit
st.pyplot(plt)

# 7. Comparison Between Retweets and Likes Over Time
st.subheader("Retweets vs Likes Over Time")
time_series = filtered_tweets.groupby(['created_at']).agg({'retweets_count': 'sum', 'favorite_count': 'sum'}).reset_index()
st.line_chart(time_series.set_index('created_at'))

# 8. Word Frequency Trends Over Time
st.subheader("Word Frequency Trends Over Time")
filtered_tweets['date'] = filtered_tweets['created_at'].dt.date
word_freq_over_time = filtered_tweets.groupby(['date']).apply(lambda x: x['text'].str.findall(r'\w+').explode().value_counts().head(10))
st.write(word_freq_over_time)

# 9. Top 10 Most Engaged Tweets (Based on Retweets and Likes)
st.subheader("Top 10 Most Engaged Tweets")
top_engaged_tweets = filtered_tweets.nlargest(10, ['retweets_count', 'favorite_count'])[["text", "retweets_count", "favorite_count", "created_at"]]
st.dataframe(top_engaged_tweets)

# 10. Interactive Tweet Timeline
st.subheader("Interactive Tweet Timeline")
tweet_timeline = px.timeline(filtered_tweets, x_start="created_at", x_end="created_at", y="text", title="Tweet Timeline")
st.plotly_chart(tweet_timeline)
