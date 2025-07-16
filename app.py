# # app.py - Optimized version
# from flask import Flask, render_template, request, jsonify
# import pandas as pd
# import pickle
# import os
# import logging
# from werkzeug.utils import secure_filename
# from utils import extract_keywords, predict_sentiment, get_aggregates

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app = Flask(__name__)
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# # Load models once at startup
# def load_models():
#     try:
#         model = pickle.load(open('model/model_destree.pkl', 'rb'))
#         cv = pickle.load(open('model/countVectorizer.pkl', 'rb'))
#         scaler = pickle.load(open('model/scaler.pkl', 'rb'))
#         logger.info("Models loaded successfully")
#         return model, cv, scaler
#     except Exception as e:
#         logger.error(f"Error loading models: {e}")
#         raise

# model, cv, scaler = load_models()

# @app.route("/", methods=["GET", "POST"])
# def index():
#     results = []
#     aggregate = {}
#     error_message = None

#     if request.method == "POST":
#         try:
#             if 'text' in request.form and request.form['text'].strip():
#                 text = request.form['text'].strip()
#                 keywords = extract_keywords(text)
#                 sentiment, label = predict_sentiment(text, model, cv, scaler)
#                 results = [{"text": text, "keywords": keywords, "label": sentiment}]
#                 aggregate = get_aggregates(results)

#             elif 'file' in request.files and request.files['file'].filename:
#                 file = request.files['file']
#                 filename = secure_filename(file.filename)
                
#                 if not filename:
#                     error_message = "Invalid file name"
#                 else:
#                     ext = filename.split('.')[-1].lower()
                    
#                     if ext == "csv":
#                         try:
#                             df = pd.read_csv(file)
#                             if 'text' not in df.columns:
#                                 error_message = "CSV file must contain a 'text' column"
#                             else:
#                                 results = process_dataframe(df)
#                         except Exception as e:
#                             error_message = f"Error reading CSV: {str(e)}"
                    
#                     elif ext == "txt":
#                         try:
#                             content = file.read().decode('utf-8')
#                             lines = [line.strip() for line in content.splitlines() if line.strip()]
#                             df = pd.DataFrame({"text": lines})
#                             results = process_dataframe(df)
#                         except Exception as e:
#                             error_message = f"Error reading text file: {str(e)}"
                    
#                     else:
#                         error_message = "Unsupported file format. Please use CSV or TXT files."

#                 if results:
#                     aggregate = get_aggregates(results)
            
#             else:
#                 error_message = "Please provide text input or upload a file"

#         except Exception as e:
#             logger.error(f"Error processing request: {e}")
#             error_message = "An error occurred while processing your request"

#     return render_template("index.html", 
#                          results=results, 
#                          aggregate=aggregate, 
#                          error_message=error_message)

# def process_dataframe(df):
#     """Process dataframe and return results"""
#     results = []
#     max_rows = 1000  # Limit processing to avoid timeouts
    
#     for i, txt in enumerate(df['text']):
#         if i >= max_rows:
#             break
            
#         if pd.isna(txt) or not str(txt).strip():
#             continue
            
#         try:
#             keywords = extract_keywords(str(txt))
#             sentiment, label = predict_sentiment(str(txt), model, cv, scaler)
#             results.append({
#                 "text": str(txt)[:100] + "..." if len(str(txt)) > 100 else str(txt),
#                 "keywords": keywords, 
#                 "label": sentiment
#             })
#         except Exception as e:
#             logger.warning(f"Error processing text at row {i}: {e}")
#             continue
    
#     return results

# @app.route("/api/analyze", methods=["POST"])
# def api_analyze():
#     """API endpoint for programmatic access"""
#     try:
#         data = request.get_json()
#         if not data or 'text' not in data:
#             return jsonify({"error": "Missing 'text' field"}), 400
        
#         text = data['text']
#         keywords = extract_keywords(text)
#         sentiment, label = predict_sentiment(text, model, cv, scaler)
        
#         return jsonify({
#             "text": text,
#             "keywords": keywords,
#             "sentiment": sentiment,
#             "confidence": float(label)
#         })
    
#     except Exception as e:
#         logger.error(f"API error: {e}")
#         return jsonify({"error": "Internal server error"}), 500

# @app.errorhandler(413)
# def too_large(e):
#     return render_template("index.html", error_message="File too large. Maximum size is 16MB"), 413

# if __name__ == "__main__":
#     port = int(os.environ.get('PORT', 5000))
#     debug = os.environ.get('FLASK_ENV') == 'development'
    
#     print(f"\n🌐 Open your browser and visit: http://127.0.0.1:{port}\n")
#     app.run(debug=debug, port=port, host='0.0.0.0')
import nltk
import os

NLTK_DATA_DIR = os.path.join(os.path.dirname(__file__), "nltk_data")
nltk.data.path.append(NLTK_DATA_DIR)

required_packages = ['wordnet', 'omw-1.4', 'stopwords', 'punkt']
for pkg in required_packages:
    found = False
    for folder in ['corpora', 'tokenizers']:
        try:
            nltk.data.find(f'{folder}/{pkg}')
            print(f"{pkg} found in {folder}")
            found = True
            break
        except LookupError:
            continue
    if not found:
        raise RuntimeError(f"❌ NLTK corpus '{pkg}' not found in {NLTK_DATA_DIR}. Please pre-download before deploying.")


from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import os
import logging
from werkzeug.utils import secure_filename
from flask_cors import CORS
from utils import (
    extract_keywords,
    extract_top_keywords,
    predict_sentiment,
    get_aggregates,
    get_keywords_json
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Load models once at startup
def load_models():
    try:
        model = pickle.load(open('model/model_destree.pkl', 'rb'))
        cv = pickle.load(open('model/countVectorizer.pkl', 'rb'))
        scaler = pickle.load(open('model/scaler.pkl', 'rb'))
        logger.info("Models loaded successfully")
        return model, cv, scaler
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise

model, cv, scaler = load_models()

# @app.route("/", methods=["GET", "POST"])
# def index():
#     results = []
#     aggregate = {}
#     error_message = None

#     if request.method == "POST":
#         try:
#             if 'text' in request.form and request.form['text'].strip():
#                 text = request.form['text'].strip()
#                 keywords = extract_keywords(text)
#                 sentiment, label = predict_sentiment(text, model, cv, scaler)
#                 results = [{"text": text, "keywords": keywords, "label": sentiment}]
#                 aggregate = get_aggregates(results)

#             elif 'file' in request.files and request.files['file'].filename:
#                 file = request.files['file']
#                 filename = secure_filename(file.filename)

#                 if not filename:
#                     error_message = "Invalid file name"
#                 else:
#                     ext = filename.split('.')[-1].lower()
#                     if ext == "csv":
#                         try:
#                             df = pd.read_csv(file)
#                             if 'text' not in df.columns:
#                                 error_message = "CSV file must contain a 'text' column"
#                             else:
#                                 results = process_dataframe(df)
#                         except Exception as e:
#                             error_message = f"Error reading CSV: {str(e)}"
#                     elif ext == "txt":
#                         try:
#                             content = file.read().decode('utf-8')
#                             lines = [line.strip() for line in content.splitlines() if line.strip()]
#                             df = pd.DataFrame({"text": lines})
#                             results = process_dataframe(df)
#                         except Exception as e:
#                             error_message = f"Error reading text file: {str(e)}"
#                     else:
#                         error_message = "Unsupported file format. Please use CSV or TXT files."

#                 if results:
#                     aggregate = get_aggregates(results)
#         except Exception as e:
#             logger.error(f"Error processing request: {e}")
#             error_message = "An error occurred while processing your request"

#     return render_template("index.html", results=results, aggregate=aggregate, error_message=error_message, global_keywords=global_top_keywords)

@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    aggregate = {}
    error_message = None
    global_keywords = []

    if request.method == "POST":
        try:
            if 'text' in request.form and request.form['text'].strip():
                text = request.form['text'].strip()
                keywords = extract_keywords(text)
                sentiment, label = predict_sentiment(text, model, cv, scaler)
                results = [{"text": text, "keywords": keywords, "label": sentiment}]
                aggregate = get_aggregates(results)
                global_keywords = keywords  # from single input

            elif 'file' in request.files and request.files['file'].filename:
                file = request.files['file']
                filename = secure_filename(file.filename)

                if not filename:
                    error_message = "Invalid file name"
                else:
                    ext = filename.split('.')[-1].lower()

                    if ext == "csv":
                        try:
                            df = pd.read_csv(file, on_bad_lines='skip')
                            if 'text' not in df.columns:
                                error_message = "CSV file must contain a 'text' column"
                            else:
                                texts = df['text'].dropna().astype(str).tolist()
                        except Exception as e:
                            error_message = f"Error reading CSV: {str(e)}"

                    elif ext == "txt":
                        try:
                            content = file.read().decode('utf-8', errors='ignore')
                            lines = [line.strip() for line in content.splitlines() if line.strip()]
                            texts = lines
                        except Exception as e:
                            error_message = f"Error reading text file: {str(e)}"
                    else:
                        error_message = "Unsupported file format. Please use CSV or TXT."

                    if not error_message and texts:
                        top_keywords_per_text, global_keywords = extract_top_keywords(texts)
                        results = []
                        for i, text in enumerate(texts[:1000]):
                            sentiment, label = predict_sentiment(text, model, cv, scaler)
                            display_text = text[:200] + '...' if len(text) > 200 else text
                            results.append({
                                "text": display_text,
                                "keywords": top_keywords_per_text[i],
                                "label": sentiment
                            })
                        aggregate = get_aggregates(results)

        except Exception as e:
            logger.error(f"Error in index(): {e}")
            error_message = "An error occurred while processing your request"

    return render_template("index.html",
                           results=results,
                           aggregate=aggregate,
                           error_message=error_message,
                           global_keywords=global_keywords)


def process_dataframe(df):
    results = []
    max_rows = 1000
    
    for i, txt in enumerate(df['text']):
        if i >= max_rows:
            break

        if pd.isna(txt) or not str(txt).strip():
            continue

        try:
            keywords = extract_keywords(str(txt))
            sentiment, label = predict_sentiment(str(txt), model, cv, scaler)
            results.append({
                "text": str(txt)[:100] + "..." if len(str(txt)) > 100 else str(txt),
                "keywords": keywords,
                "label": sentiment
            })
        except Exception as e:
            logger.warning(f"Error processing text at row {i}: {e}")
            continue

    return results

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' field"}), 400

        text = data['text']
        keywords = extract_keywords(text)
        sentiment, label = predict_sentiment(text, model, cv, scaler)

        return jsonify({
            "text": text,
            "keywords": keywords,
            "sentiment": sentiment,
            "confidence": float(label)
        })

    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/api/analyze-file", methods=["POST"])
def api_analyze_file():
    if 'file' not in request.files:
        return jsonify({"error": "No 'file' part in the request."}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading."}), 400

    filename = secure_filename(file.filename)
    ext = os.path.splitext(filename)[1].lower()

    try:
        if ext == ".csv":
            df = pd.read_csv(file, on_bad_lines='skip')
            if 'text' not in df.columns:
                return jsonify({"error": "CSV file must contain a 'text' column."}), 400
        elif ext == ".txt":
            content = file.read().decode('utf-8', errors='ignore')
            lines = [line.strip() for line in content.splitlines() if line.strip()]
            df = pd.DataFrame({"text": lines})
        else:
            return jsonify({"error": "Unsupported file format. Please use CSV or TXT."}), 415

        texts = df['text'].dropna().astype(str).tolist()
        if not texts:
            return jsonify({"error": "No valid text to process in the file."}), 400

        top_keywords_per_text, global_top_keywords = extract_top_keywords(texts)

        results = []
        for i, text in enumerate(texts):
            sentiment, label = predict_sentiment(text, model, cv, scaler)
            display_text = text[:200] + '...' if len(text) > 200 else text
            results.append({
                "text": display_text,
                "keywords": top_keywords_per_text[i],
                "label": sentiment
            })

        aggregate_data = get_aggregates(results)
        keywords_data = get_keywords_json(results, global_top_keywords)

        return jsonify({
            "filename": filename,
            "sentiment_results": results,
            "aggregate_sentiment": aggregate_data,
            "keywords_analysis": keywords_data
        })

    except Exception as e:
        logger.error(f"API file processing error: {e}")
        return jsonify({"error": "An error occurred while processing the file."}), 500

@app.route("/api/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok"}), 200

@app.errorhandler(413)
def too_large(e):
    return render_template("index.html", error_message="File too large. Maximum size is 16MB"), 413

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    # debug = os.environ.get('FLASK_ENV') == 'development'

    # print(f"\n🌐 Open your browser and visit: http://127.0.0.1:{port}\n")
    app.run( port=port, host='0.0.0.0')
    # app.run(debug=debug, port=port, host='0.0.0.0')
