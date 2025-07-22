import nltk
import os
import pickle
import logging
import pandas as pd
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
from functools import wraps  # Import 'wraps'
from utils import (
    extract_top_keywords,
    predict_sentiment,
    get_aggregates,
    separate_keywords_by_sentiment
)

# --- NLTK Setup ---
NLTK_DATA_DIR = os.path.join(os.path.dirname(__file__), "nltk_data")
nltk.data.path.append(NLTK_DATA_DIR)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# --- Load Models ---
try:
    model = pickle.load(open('model/model_destree.pkl', 'rb'))
    cv = pickle.load(open('model/countVectorizer.pkl', 'rb'))
    scaler = pickle.load(open('model/scaler.pkl', 'rb'))
    logger.info("Models loaded successfully.")
except Exception as e:
    logger.error(f"Fatal Error: Could not load models. {e}")
    raise


API_KEY = os.environ.get("API_KEY", "changeme")

def require_api_key(f):
    """Decorator to protect routes with an API key."""
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get("x-api-key")
        if not key or key != API_KEY:
            return jsonify({"error": "Unauthorized", "message": "A valid API key must be provided in the 'x-api-key' header."}), 401
        return f(*args, **kwargs)
    return decorated

@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    aggregate = {}
    error_message = None
    top_positive_keywords = []
    top_negative_keywords = []

    if request.method == "POST":
        try:
            texts = []
            if 'file' in request.files and request.files['file'].filename:
                file = request.files['file']
                filename = secure_filename(file.filename)
                ext = os.path.splitext(filename)[1].lower()
                if ext == ".csv":
                    df = pd.read_csv(file, on_bad_lines='skip')
                    if 'text' not in df.columns:
                        error_message = "CSV file must contain a 'text' column."
                    else:
                        texts = df['text'].dropna().astype(str).tolist()
                elif ext == ".txt":
                    content = file.read().decode('utf-8', errors='ignore')
                    texts = [line.strip() for line in content.splitlines() if line.strip()]
                else:
                    error_message = "Unsupported file format. Please use CSV or TXT."
            elif 'text' in request.form and request.form['text'].strip():
                texts = [request.form['text'].strip()]
            else:
                error_message = "Please provide text input or upload a file."

            if not error_message and texts:
                top_keywords_per_text = extract_top_keywords(texts)
                for i, text in enumerate(texts[:1000]):
                    sentiment, _ = predict_sentiment(text, model, cv, scaler)
                    display_text = text[:200] + '...' if len(text) > 200 else text
                    results.append({
                        "text": display_text,
                        "keywords": top_keywords_per_text[i],
                        "label": sentiment
                    })
                
                if results:
                    aggregate = get_aggregates(results)
                    top_positive_keywords, top_negative_keywords = separate_keywords_by_sentiment(results)

        except Exception as e:
            logger.error(f"Error in index(): {e}")
            error_message = "An error occurred while processing your request"

    return render_template("index.html",
                           results=results,
                           aggregate=aggregate,
                           error_message=error_message,
                           top_positive_keywords=top_positive_keywords,
                           top_negative_keywords=top_negative_keywords)

# --- API Routes ---

@app.route("/api/analyze-text", methods=["POST"])
@require_api_key  # Secure this endpoint
def api_analyze_text():
    """Analyzes a single text string from a JSON payload."""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' field"}), 400

        text = data['text']
        keywords = extract_top_keywords([text])[0] 
        sentiment, label = predict_sentiment(text, model, cv, scaler)

        positive_kws = keywords if sentiment == 'Positive' else []
        negative_kws = keywords if sentiment == 'Negative' else []

        return jsonify({
            "text": text,
            "sentiment": sentiment,
            "confidence_label": label,
            "keywords": {
                "positive": positive_kws,
                "negative": negative_kws
            }
        })

    except Exception as e:
        logger.error(f"API error in /api/analyze-text: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/api/analyze-file", methods=["POST"])
@require_api_key 
def api_analyze_file():
    """Analyzes a full file (CSV or TXT)."""
    if 'file' not in request.files:
        return jsonify({"error": "No 'file' part in the request."}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading."}), 400

    filename = secure_filename(file.filename)
    ext = os.path.splitext(filename)[1].lower()

    try:
        texts = []
        if ext == ".csv":
            df = pd.read_csv(file, on_bad_lines='skip')
            if 'text' not in df.columns:
                return jsonify({"error": "CSV file must contain a 'text' column."}), 400
            texts = df['text'].dropna().astype(str).tolist()
        elif ext == ".txt":
            content = file.read().decode('utf-8', errors='ignore')
            texts = [line.strip() for line in content.splitlines() if line.strip()]
        else:
            return jsonify({"error": "Unsupported file format. Please use CSV or TXT."}), 415

        if not texts:
            return jsonify({"error": "No valid text to process in the file."}), 400

        results = []
        top_keywords_per_text = extract_top_keywords(texts)
        for i, text in enumerate(texts):
            sentiment, _ = predict_sentiment(text, model, cv, scaler)
            results.append({
                "text": text[:200] + '...' if len(text) > 200 else text,
                "keywords": top_keywords_per_text[i],
                "label": sentiment
            })

        top_positive_keywords, top_negative_keywords = separate_keywords_by_sentiment(results)

        return jsonify({
            "filename": filename,
            "sentiment_results": results,
            "aggregate_sentiment": get_aggregates(results),
            "keywords_analysis": {
                "top_positive_keywords": top_positive_keywords,
                "top_negative_keywords": top_negative_keywords
            }
        })

    except Exception as e:
        logger.error(f"API error in /api/analyze-file: {e}")
        return jsonify({"error": "An error occurred while processing the file."}), 500


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(port=port, host='0.0.0.0')