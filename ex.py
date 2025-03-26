from visualisation import plot_sentiment_bar, plot_emotion_radar

test_sentiments = {"Positive": 10, "Neutral": 5, "Negative": 8}
test_emotions = {"Joy": 5, "Anger": 3, "Sadness": 7}

plot_sentiment_bar(test_sentiments)
plot_emotion_radar(test_emotions)