from flask import Flask, request, jsonify , render_template , url_for
from flask_cors import CORS
import pandas as pd
import google.generativeai as genai
from sentimental import analyze_sentiment_nlp, translateHitoEn
from visualisation import plot_sentiment_bar, plot_emotion_radar
from langdetect import detect
from datetime import datetime
from reddit_api import RedditAPI 
import re  # Import regex module

app = Flask(__name__)
CORS(app)

GEMINI_API_KEY = "AIzaSyB2yxeiFcUsEp2dZoekeRka5hX4FHI4C2U"
genai.configure(api_key=GEMINI_API_KEY)

def main(post_url):
    reddit_api = RedditAPI()

    post_url = post_url
    post_id = re.search(r"/comments/([a-zA-Z0-9]+)/", post_url).group(1)

    comments_list = reddit_api.fetch_comments(post_id, max_comments=5)
    print(comments_list)
    return comments_list


@app.route("/", methods=["GET", "POST"])
def index():
    top_comments = []  # Default empty list

    if request.method == "POST":
        data = request.json
        post_url = data.get("post_url", "").strip()
        print("Received JSON:", data)

        if not post_url:
            return jsonify({"error": "No URL provided"}), 400

        # Dummy Sentiment Analysis Results (Replace with NLP model results)
        sentiment_counts = {"Positive": 20, "Neutral": 15, "Negative": 10}
        emotion_scores = {"Joy": 0.8, "Anger": 0.3, "Sadness": 0.5, "Fear": 0.2, "Surprise": 0.6}

        # Generate graphs
        plot_sentiment_bar(sentiment_counts)
        plot_emotion_radar(emotion_scores)

        # Fetch top comments from main() function
        top_comments = main(post_url)  
        print(top_comments)

    return render_template(
        "home.html",
        bar_graph="static/sentiment_bar.png" if top_comments else None,
        radar_chart="static/emotion_radar.png" if top_comments else None,
        top_comments=top_comments
    )


@app.route('/services')
def services():
    return render_template("services.html")

@app.route('/flow')
def flow():
    return render_template("flow.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")



# Sentimental analysis Integration starts : Parth 


@app.route('/fetch_comments', methods=['POST'])
def fetch_comments():
    try:
        data = request.json
        post_url = data.get("post_url", "").strip()
        print("Received JSON:", data)

        if not post_url:
            return jsonify({"error": "No URL provided"}), 400
        
        print("Fetching comments for:", post_url)  # Debugging

        comments = main(post_url)
        print("Fetched comments:", comments)  # Debugging

        analyzed_comments = []
        negative, neutral, positive = 0, 0, 0
        emotions_dict = {}

        for comment in comments:
            try:
                lang = detect(comment)
                if lang == 'unknown':
                    comment = translateHitoEn(comment)

                sentiment, scores, emotion = analyze_sentiment_nlp(comment)  # Ensure this works!
                
                print(f"Analyzing Comment: {comment}")  # Debugging
                print(f"Sentiment: {sentiment}, Scores: {scores}, Emotion: {emotion}")  # Debugging

                negative += scores["Negative"]
                neutral += scores["Neutral"]
                positive += scores["Positive"]

                emotion_label = emotion[0][0]['label']
                emotions_dict[emotion_label] = emotions_dict.get(emotion_label, 0) + 1

                analyzed_comments.append({
                    "comment": comment,
                    "sentiment": sentiment,
                    "scores": scores,
                    "emotion": emotion
                })
            except Exception as e:
                print("Error processing comment:", e)  # Debugging

        total_comments = len(comments)
        percent_negative = (negative / total_comments) * 100 if total_comments > 0 else 0
        percent_neutral = (neutral / total_comments) * 100 if total_comments > 0 else 0
        percent_positive = (positive / total_comments) * 100 if total_comments > 0 else 0

        return jsonify({
            "comments": analyzed_comments,
            "percent_negative": percent_negative,
            "percent_neutral": percent_neutral,
            "percent_positive": percent_positive,
            "emotions": emotions_dict
        })
    
    except Exception as e:
        print("Critical Error:", e)  # Debugging
        return jsonify({"error": str(e)}), 500


# Sentimental analysis Integration Ends : Parth



# omkar`s code starting here 


def format_text(text):
    """Converts '*' into a new line and '**text**' into bold uppercase with an extra new line before it"""
    text = re.sub(r"\*\*(.*?)\*\*", lambda match: f"<br><br><b>{match.group(1).upper()}</b>", text)
    text = text.replace("*", "<br>")
    return text

# Load the Gemini Model only once (outside the function)
genai_model = genai.GenerativeModel("gemini-1.5-flash")

def analyze_sentiment(comment):
    """Sends a comment to Gemini API for sentiment analysis."""
    if not comment.strip():
        return "No comment provided."

    prompt = f"You are an expert in sentiment analysis. Explain your reasoning.\nComment: {comment}"

    try:
        response = genai_model.generate_content(prompt)  # Reuse the loaded model
        formatted_response = format_text(response.text) if response else "Error analyzing sentiment."
        return formatted_response
    except Exception as e:
        return f"API Error: {str(e)}"

    
@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    comment = data.get("comment", "").strip()

    if not comment:
        return jsonify({"error": "No comment provided."}), 400

    sentiment = analyze_sentiment(comment)
    return jsonify({"analysis": sentiment})  

# # omkar`s code ending here 

if __name__ == "__main__":  
    test_sentiment = {"Positive": 20, "Neutral": 15, "Negative": 10}
    test_emotions = {"Joy": 0.8, "Anger": 0.3, "Sadness": 0.5, "Fear": 0.2, "Surprise": 0.6}

    print("Generating test graphs...")
    plot_sentiment_bar(test_sentiment)
    plot_emotion_radar(test_emotions)
    print("Graphs generated! Check the 'static' folder.")
    app.run(port=5001)

