from transformers import pipeline

# Load sentiment-analysis pipeline from Hugging Face (default is DistilBERT)
sentiment_pipeline = pipeline("sentiment-analysis")

def predict_sentiment(text):
    result = sentiment_pipeline(text)[0]
    label = result['label']
    
    if label == "POSITIVE":
        return "Positive 😊"
    elif label == "NEGATIVE":
        return "Negative 😞"
    else:
        return label
