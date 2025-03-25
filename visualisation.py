import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Function to generate bar graph for sentiment distribution
def plot_sentiment_bar(sentiment_counts, save_path="static/sentiment_bar.png"):
    plt.figure(figsize=(6, 4))
    sns.barplot(x=sentiment_counts.keys(), y=sentiment_counts.values(), palette="coolwarm")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.title("Sentiment Distribution")
    plt.savefig(save_path)
    plt.close()  # Close the plot to free memory

# Function to generate radar chart for emotion analysis
def plot_emotion_radar(emotion_scores, save_path="static/emotion_radar.png"):
    labels = list(emotion_scores.keys())
    values = list(emotion_scores.values())

    # Close the radar shape by repeating the first value at the end
    values += values[:1]
    labels += labels[:1]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
    ax.fill(angles, values, color="blue", alpha=0.3)
    ax.plot(angles, values, color="blue", linewidth=2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    plt.title("Emotion Breakdown")
    plt.savefig(save_path)
    plt.close()
