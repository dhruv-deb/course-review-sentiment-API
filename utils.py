import re
import logging
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from wordcloud import STOPWORDS as WORDCLOUD_STOPWORDS
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from collections import Counter
import nltk

nltk.data.path.append('./nltk_data')
logger = logging.getLogger(__name__)

lemmatizer = WordNetLemmatizer()
nltk_stopwords = set(stopwords.words('english'))
combined_stopwords = nltk_stopwords.union(WORDCLOUD_STOPWORDS)

CLEAN_REGEX = re.compile(r'[^a-zA-Z\s]')

def clean_and_lemmatize(comment):
    """Clean and lemmatize text"""
    if not comment or not isinstance(comment, str):
        return ""
    
    comment = CLEAN_REGEX.sub(' ', comment)
    comment = comment.lower().split()
    
    comment = [lemmatizer.lemmatize(word) for word in comment 
               if word not in combined_stopwords and len(word) > 1]
    
    return ' '.join(comment)

def extract_keywords(text, top_k=5):
    """Extract keywords using TF-IDF for a single text"""
    try:
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
    """Extract keywords for each text"""
    top_keywords_per_text = []
    for text in text_list:
        kws = extract_keywords(text, top_k=top_k_per_text)
        top_keywords_per_text.append(kws)
    return top_keywords_per_text

def predict_sentiment(comment, model, cv, scaler):
    """Predict sentiment with error handling"""
    try:
        cleaned_comment = clean_and_lemmatize(comment)
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
    """Calculate aggregated statistics"""
    if not results:
        return {"total": 0, "count": {}, "percent": {}, "most_common": None}
    
    label_count = Counter([r["label"] for r in results])
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
        # Ensure 'keywords' key exists and is a list
        if 'keywords' in result and isinstance(result['keywords'], list):
            if result.get('label') == 'Positive':
                positive_keywords.extend(result['keywords'])
            elif result.get('label') == 'Negative':
                negative_keywords.extend(result['keywords'])

    top_positive = [word for word, _ in Counter(positive_keywords).most_common(top_k)]
    top_negative = [word for word, _ in Counter(negative_keywords).most_common(top_k)]

    return top_positive, top_negative
