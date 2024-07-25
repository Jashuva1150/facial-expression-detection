from transformers import pipeline

# Initialize sentiment analysis pipeline
sentiment_pipeline = pipeline('sentiment-analysis')

def analyze_sentiment(text):
    result = sentiment_pipeline(text)
    return result[0]['label'], result[0]['score']

# Example usage
if __name__ == "__main__":
    text = "I am very happy today!"
    sentiment, score = analyze_sentiment(text)
    print(f"Sentiment: {sentiment}, Score: {score}")
