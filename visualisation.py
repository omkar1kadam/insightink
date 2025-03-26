import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
    
# Function to generate bar graph for sentiment distribution
def plot_sentiment_bar(sentiment_counts, save_path="static/sentiment_bar.png"):
    if os.path.exists("static/sentiment_bar.png"):
     os.remove("static/sentiment_bar.png")

    plt.figure(figsize=(6, 4))
    sns.barplot(x=sentiment_counts.keys(), y=sentiment_counts.values(), palette="coolwarm")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.title("Sentiment Distribution")
    plt.savefig(save_path)
    plt.close()  # Close the plot to free memory

# Function to generate radar chart for emotion analysis
def plot_emotion_radar(emotion_scores, save_path="static/emotion_radar.png"):
    if os.path.exists("static/emotion_radar.png"):
     os.remove("static/emotion_radar.png")
    print("Generating Emotion Radar Chart...")  # Debugging
    print("Data:", emotion_scores)  # Debugging

    labels = list(emotion_scores.keys())
    values = list(emotion_scores.values())

    # Close the radar shape
    values.append(values[0])  # Append first value again
    labels.append(labels[0])  # Append first label again (Fixing the mismatch)

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
    ax.fill(angles, values, color="blue", alpha=0.3)
    ax.plot(angles, values, color="blue", linewidth=2)

    ax.set_xticks(angles)  # Ensure tick count matches labels
    ax.set_xticklabels(labels)  # Match labels with angles

    plt.title("Emotion Breakdown")
    plt.savefig(save_path)
    print(f"Saved Emotion Graph at {save_path}")  # Debugging
    plt.close()



