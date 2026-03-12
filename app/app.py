import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv('data/processed/tweets_clean.csv')
topic_df = pd.read_csv('data/processed/bertopic_output.csv')

# Engineer features
df['tweet_length'] = df['text'].str.len()
df['exclamation_count'] = df['text'].str.count('!')
df['has_url'] = df['text'].str.contains('http').astype(int)
df['mention_count'] = df['text'].str.count(r'@\w+')

df['score'] = 0
df['score'] += (df['tweet_length'] > 100).astype(int)
df['score'] += (df['exclamation_count'] >= 2).astype(int)
df['score'] += (df['has_url'] == 1).astype(int)
df['score'] += (df['mention_count'] > 1).astype(int)
df['score'] += (df['negativereason'].isin(['Lost Luggage', 'Cancelled Flight', 'Customer Service Issue'])).astype(int)

df['churn'] = np.where(
    (df['airline_sentiment'] == 'negative') & (df['score'] >= 3), 1, 0
)

# Title
st.title('Airline Tweet Sentiment Dashboard')
st.write(f'Total tweets analysed: {len(df):,}')

# Section 1 - Sentiment by Airline
st.header('Sentiment by Airline')
airline_sentiment = df.groupby(['airline', 'airline_sentiment']).size().unstack()
airline_pct = airline_sentiment.div(airline_sentiment.sum(axis=1), axis=0) * 100

fig1, ax1 = plt.subplots(figsize=(10, 5))
airline_pct[['negative', 'neutral', 'positive']].plot(
    kind='bar', stacked=True, ax=ax1,
    color=['#E63946', '#A8DADC', '#2E86AB']
)
ax1.set_title('Sentiment Breakdown by Airline (%)')
ax1.set_xlabel('Airline')
ax1.set_ylabel('Percentage')
plt.xticks(rotation=30)
st.pyplot(fig1)

# Section 2 - Churn Risk
st.header('Churn Risk by Airline')
churn_by_airline = df.groupby('airline')['churn'].sum().sort_values(ascending=False)

fig2, ax2 = plt.subplots(figsize=(10, 5))
churn_by_airline.plot(kind='bar', ax=ax2, color='#E63946')
ax2.set_title('High Churn Risk Customers by Airline')
ax2.set_xlabel('Airline')
ax2.set_ylabel('Number of High Risk Customers')
plt.xticks(rotation=30)
st.pyplot(fig2)

# Section 3 - BERTopic
st.header('Top Complaint Topics')
topic_counts = topic_df[topic_df['topic'] != -1]['topic'].value_counts().head(8)

topic_names = {
    0: 'Baggage Issues',
    1: 'Flight Cancellations',
    2: 'Waiting/Gate Issues',
    3: 'Phone Hold/Service',
    4: 'Delays',
    5: 'Rebooking',
    6: 'Lost Luggage',
    7: 'Staff Issues'
}
topic_counts.index = [topic_names.get(i, f'Topic {i}') for i in topic_counts.index]

fig3, ax3 = plt.subplots(figsize=(10, 5))
topic_counts.plot(kind='bar', ax=ax3, color='#E63946')
ax3.set_title('Most Common Complaint Topics')
ax3.set_xlabel('Topic')
ax3.set_ylabel('Number of Tweets')
plt.xticks(rotation=30)
st.pyplot(fig3)