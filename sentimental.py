# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from reddit_api import main
# from langdetect import detect, detect_langs
# from transformers import pipeline
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from scipy.special import softmax
# import torch
# import numpy as np
# import matplotlib.pyplot as plt

# classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)
# translator_tokenizer = AutoTokenizer.from_pretrained("rudrashah/RLM-hinglish-translator")
# translator_model = AutoModelForCausalLM.from_pretrained("rudrashah/RLM-hinglish-translator")
# def translateHitoEn(hinglish_text):
#     template = "Hinglish:\n{hi_en}\n\nEnglish:\n{en}"
#     input_text = translator_tokenizer(template.format(hi_en=hinglish_text, en=""), return_tensors="pt")

#     output = translator_model.generate(**input_text)
#     english_text = translator_tokenizer.decode(output[0], skip_special_tokens=True)

#     if "English:" in english_text:
#         english_text = english_text.split("English:")[-1].strip()

#     print(f"Hinglish Input     : {hinglish_text}")
#     print(f"Translated English : {english_text}")

#     return english_text

# MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
# labels = ['Negative', 'Neutral', 'Positive']
# def analyze_sentiment(text):
#     encoded_input = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    
#     with torch.no_grad():
#         output = model(**encoded_input)
    
#     scores = softmax(output.logits.numpy()[0])
    
#     sentiment_score = {label: float(score) for label, score in zip(labels, scores)}
#     predicted_sentiment = labels[scores.argmax()]
#     emotion=classifier(text)
#     return predicted_sentiment, sentiment_score, emotion

# comms=main()
# ng=[]
# nu=[]
# po=[]
# negative,neutral,positive=0,0,0
# emotions_dict={}
# for i in comms:
#     lang = detect(i)
#     if lang=='unknown':
#         i=translateHitoEn(i)
#     sentiment, scores ,emotion= analyze_sentiment(i)
#     print(f"Predicted Sentiment: {sentiment}")
#     print("Scores:", scores)
#     print(f"Emotion: {emotion}")
#     ng.append(scores["Negative"])
#     nu.append(scores["Neutral"])
#     po.append(scores["Positive"])
    
#     negative+=scores["Negative"]
#     neutral+=scores["Neutral"]
#     positive+=scores["Positive"]
#     emotion_label = emotion[0][0]['label']
#     emotions_dict[emotion_label] = emotions_dict.get(emotion_label, 0) + 1
    
# percent_negative = (negative / len(comms)) * 100
# percent_neutral = (neutral / len(comms)) * 100
# percent_positive = (positive / len(comms)) * 100



# # Ensure all required emotions are present
# all_emotions = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
# emotions_dict = {emotion: emotions_dict.get(emotion, 0) for emotion in all_emotions}

# emotion_labels = list(emotions_dict.keys())
# emotion_values = list(emotions_dict.values())

# # Create angles for the heptagon
# angles = np.linspace(0, 2 * np.pi, 7, endpoint=False).tolist()
# angles += angles[:1]  # Closing the shape
# emotion_values += emotion_values[:1]  # Closing the shape

# # Plot radar chart
# fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})
# ax.set_ylim(0, max(emotion_values) + 1)
# ax.set_xticks(angles[:-1])
# ax.set_xticklabels(emotion_labels)
# ax.set_yticklabels([])
# ax.set_title("Emotion Distribution (Heptagon)")

# # Draw Heptagon
# ax.fill(angles, emotion_values, color='#ED3EF7', alpha=0.25)
# ax.plot(angles, emotion_values, color='#ED3EF7', linewidth=2, linestyle='solid')

# plt.show()

# # Plot Bar Chart for Sentiment
# sentiments = ['Negative', 'Neutral', 'Positive']
# sentiment_counts = [negative, neutral, positive]
# plt.figure(figsize=(6, 4))
# plt.bar(sentiments, sentiment_counts, color=['#7A1CAC','#BF2EF0', '#ED3EF7'])
# plt.xlabel("Sentiment")
# plt.ylabel("Count")
# plt.title("Sentiment Analysis Distribution")
# plt.show()


import numpy as np
import torch
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, AutoModelForCausalLM

# Load sentiment analysis model
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
labels = ['Negative', 'Neutral', 'Positive']

# Load emotion classifier
classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)

# Load translator model for Hinglish-to-English
translator_tokenizer = AutoTokenizer.from_pretrained("rudrashah/RLM-hinglish-translator", use_fast=False)
translator_model = AutoModelForCausalLM.from_pretrained("rudrashah/RLM-hinglish-translator")

def translateHitoEn(hinglish_text):
    template = "Hinglish:\n{hi_en}\n\nEnglish:\n{en}"
    input_text = translator_tokenizer(template.format(hi_en=hinglish_text, en=""), return_tensors="pt")
    output = translator_model.generate(**input_text)
    english_text = translator_tokenizer.decode(output[0], skip_special_tokens=True)
    
    if "English:" in english_text:
        english_text = english_text.split("English:")[-1].strip()

    return english_text

def analyze_sentiment_nlp(text):
    """Analyzes the sentiment and emotion of the given text."""
    encoded_input = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    
    with torch.no_grad():
        output = model(**encoded_input)
    
    scores = softmax(output.logits.numpy()[0])
    
    sentiment_score = {label: float(score) for label, score in zip(labels, scores)}
    predicted_sentiment = labels[scores.argmax()]
    emotion = classifier(text)

    return predicted_sentiment, sentiment_score, emotion