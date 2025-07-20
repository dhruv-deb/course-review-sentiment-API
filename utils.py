import re
import logging
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from wordcloud import STOPWORDS as WORDCLOUD_STOPWORDS
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from collections import Counter
import nltk

try:
    stopwords.words('english')
    nltk.data.find('corpora/wordnet.zip')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')

logger = logging.getLogger(__name__)

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

nltk_stopwords = set(stopwords.words('english'))
combined_stopwords = nltk_stopwords.union(WORDCLOUD_STOPWORDS)

CLEAN_REGEX = re.compile(r'[^a-zA-Z\s]')

def clean_and_stem(comment):
    """
    Cleans and STEMS text to match the preprocessing from the training notebook.
    This is used ONLY for the sentiment prediction model.
    """
    if not comment or not isinstance(comment, str):
        return ""
    
    review = re.sub('[^a-zA-Z]', ' ', comment)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if not word in combined_stopwords]
    return ' '.join(review)

def clean_and_lemmatize(comment):
    """
    Cleans and LEMMATIZES text. This is used for keyword extraction to produce
    readable, complete words.
    """
    if not comment or not isinstance(comment, str):
        return ""
    
    review = re.sub('[^a-zA-Z]', ' ', comment)
    review = review.lower().split()
    review = [lemmatizer.lemmatize(word) for word in review if not word in combined_stopwords and len(word) > 1]
    return ' '.join(review)


def extract_keywords(text, top_k=5):
    """Extract keywords using TF-IDF. Uses lemmatization for readable output."""
    try:
        # Use the lemmatizing function for cleaner keywords
        cleaned = clean_and_lemmatize(text)
        if not cleaned:
            return []
        
        tfidf = TfidfVectorizer(
            stop_words='english',
            max_features=100,
            ngram_range=(1, 2),
            min_df=1,
            max_df=1.0
        )
        
        tfidf_matrix = tfidf.fit_transform([cleaned])
        feature_names = tfidf.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [feature_names[i] for i in top_indices if scores[i] > 0]
    
    except Exception as e:
        logger.error(f"Error extracting keywords: {e}")
        return []

def extract_top_keywords(text_list, top_k_per_text=5):
    """Extract keywords for each text in a list."""
    return [extract_keywords(text, top_k=top_k_per_text) for text in text_list]

def predict_sentiment(comment, model, cv, scaler):
    """
    Predicts sentiment. Uses stemming to match the model's training process.
    """
    try:
        # Use the stemming function for prediction
        cleaned_comment = clean_and_stem(comment)
        if not cleaned_comment:
            return "Neutral", 0
        
        comment_vector = cv.transform([cleaned_comment])
        comment_scaled = scaler.transform(comment_vector)
        
        prediction = model.predict(comment_scaled)
        
        sentiment = "Positive" if prediction[0] == 1 else "Negative"
        return sentiment, float(prediction[0])
    
    except Exception as e:
        logger.error(f"Error predicting sentiment: {e}")
        return "Error", 0

def get_aggregates(results):
    """Calculate aggregated statistics from a list of results."""
    if not results:
        return {"total": 0, "count": {}, "percent": {}, "most_common": None}
    
    label_count = Counter([r["label"] for r in results if r["label"] != "Error"])
    total = len(results)
    percent = {k: f"{(v / total) * 100:.1f}%" for k, v in label_count.items()}
    
    return {
        "total": total,
        "count": dict(label_count),
        "percent": percent,
        "most_common": label_count.most_common(1)[0][0] if label_count else None
    }

def separate_keywords_by_sentiment(results, top_k=50):
    """Separates keywords into positive and negative lists based on sentiment."""
    positive_keywords = []
    negative_keywords = []

    for result in results:
        if 'keywords' in result and isinstance(result['keywords'], list):
            if result.get('label') == 'Positive':
                positive_keywords.extend(result['keywords'])
            elif result.get('label') == 'Negative':
                negative_keywords.extend(result['keywords'])

    top_positive = [word for word, _ in Counter(positive_keywords).most_common(top_k)]
    top_negative = [word for word, _ in Counter(negative_keywords).most_common(top_k)]

    return top_positive, top_negative
