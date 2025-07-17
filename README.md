# Course Review Sentiment Analysis API

<!-- It's highly recommended to add a screenshot of your app here! -->

A powerful Flask-based API that performs sentiment analysis on course reviews. This tool uses a machine learning model trained on 1.4 million reviews to predict sentiment (Positive/Negative), extract meaningful keywords, and provide detailed analytics. It features a sleek, modern frontend for easy testing and robust API endpoints for integration with other applications.

**Live Demo:** <https://course-review-sentiment-api.onrender.com>

## ✨ Key Features

* **High-Accuracy Predictions:** Utilizes a Decision Tree model trained on a massive dataset, achieving a **98.7% test accuracy**.
* **Advanced NLP Processing:** Employs Natural Language Processing (NLP) for intelligent text parsing, cleaning, and lemmatization.
* **Intelligent Keyword Extraction:** Identifies the most relevant keywords for each review and provides an aggregated list of top positive and negative keywords.
* **In-Depth Analytics:** Delivers a comprehensive dashboard showing total reviews, sentiment distribution, and the dominant sentiment.
* **Dual-Mode Interface:**
    * **Interactive UI:** A simple, modern frontend for direct text input or file uploads (`.csv`, `.txt`).
    * **Developer API:** Clean, accessible API endpoints for seamless integration into other projects.
* **Glassmorphic Design:** A stunning dark theme with a grainy texture and orangish-red accents for a modern user experience.

## 🛠️ Technology Stack

* **Backend:** Python, Flask
* **Machine Learning:** Scikit-learn
* **NLP:** NLTK (Natural Language Toolkit)
* **Data Handling:** Pandas, NumPy
* **Frontend:** HTML5, CSS3, JavaScript
* **Deployment:** Render

## 🚀 Getting Started

Follow these instructions to get a local copy up and running for development and testing purposes.

### Prerequisites

* Python 3.8+
* pip (Python package installer)

### Installation

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/dhruv-deb/course-review-sentiment-API.git](https://github.com/dhruv-deb/course-review-sentiment-API.git)
    cd course-review-sentiment-API
    ```

2.  **Create and activate a virtual environment:**
    * **Windows:**
        ```sh
        python -m venv venv
        .\venv\Scripts\activate
        ```
    * **macOS / Linux:**
        ```sh
        python3 -m venv venv
        source venv/bin/activate
        ```

3.  **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

4.  **Download NLTK Data:**
    The application is configured to download the necessary NLTK packages on the first run.

5.  **Run the Flask application:**
    ```sh
    python app.py
    ```
    The application will be available at `http://127.0.0.1:5000`.

## 🔌 API Endpoints

The API provides two main endpoints for programmatic access.

### 1. Analyze Single Text

* **Endpoint:** `/api/analyze-text`
* **Method:** `POST`
* **Body:** Raw JSON

**Example Request (cURL):**
```sh
curl -X POST -H "Content-Type: application/json" \
-d '{"text": "This course was absolutely fantastic!"}' \
[http://127.0.0.1:5000/api/analyze-text](http://127.0.0.1:5000/api/analyze-text)
```

**Example Response:**
```json
{
  "text": "This course was absolutely fantastic!",
  "sentiment": "Positive",
  "confidence_label": 1.0,
  "keywords": {
    "positive": ["fantastic", "course"],
    "negative": []
  }
}
```

### 2. Analyze a File

* **Endpoint:** `/api/analyze-file`
* **Method:** `POST`
* **Body:** `multipart/form-data`

**Example Request (cURL):**
```sh
curl -X POST -F "file=@/path/to/your/reviews.csv" [http://127.0.0.1:5000/api/analyze-file](http://127.0.0.1:5000/api/analyze-file)
```

**Example Response:**
```json
{
  "filename": "reviews.csv",
  "sentiment_results": [
    {
      "text": "Great content and clear explanations.",
      "keywords": ["great", "content", "clear"],
      "label": "Positive"
    },
    {
      "text": "The instructor was boring and hard to follow.",
      "keywords": ["boring", "hard"],
      "label": "Negative"
    }
  ],
  "aggregate_sentiment": {
    "total": 2,
    "count": { "Positive": 1, "Negative": 1 },
    "percent": { "Positive": "50.0%", "Negative": "50.0%" },
    "most_common": "Positive"
  },
  "keywords_analysis": {
    "top_positive_keywords": ["great", "content", "clear"],
    "top_negative_keywords": ["boring", "hard"]
  }
}
```

## 📂 Project Structure

```
course-review-sentiment-API/
├── model/                  # Contains the serialized ML model and vectorizers
│   ├── model_destree.pkl
│   ├── countVectorizer.pkl
│   └── scaler.pkl
├── static/                 # CSS and other static assets
│   └── style.css
├── templates/              # HTML templates for the frontend
│   └── index.html
├── app.py                  # Main Flask application file
├── utils.py                # NLP processing and helper functions
├── requirements.txt        # Project dependencies
└── README.md               # This file
```

## ✍️ Author

* **Dhruv Deb** - [dhruv-deb](https://github.com/dhruv-deb)
