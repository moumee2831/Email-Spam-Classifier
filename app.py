import os
import json
import logging
import joblib
import pickle
from flask import Flask, render_template, request, jsonify

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Base directory for reliable relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Global variables for loaded model and metadata
model = None
model_name = "Unknown Model"
test_accuracy = None
model_load_error = None
metadata_load_error = None

def load_metadata():
    """Load model information from model_info.json once at startup."""
    global model_name, test_accuracy, metadata_load_error
    metadata_path = os.path.join(BASE_DIR, "model_info.json")
    try:
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file '{metadata_path}' not found.")
        
        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            model_name = data.get("model_name", "Trained ML Model")
            test_accuracy = data.get("test_accuracy")
            logger.info("Metadata loaded successfully: Model=%s, Accuracy=%s%%", model_name, test_accuracy)
    except Exception as e:
        metadata_load_error = str(e)
        logger.error("Failed to load metadata: %s", e)

def load_model():
    """Load the machine learning model once at startup."""
    global model, model_load_error
    # Check for primary filename and fallback if saved as (1)
    candidate_filenames = [
        "email_spam_model.pkl",
        "email_spam_model (1).pkl"
    ]
    
    model_path = None
    for fname in candidate_filenames:
        candidate_path = os.path.join(BASE_DIR, fname)
        if os.path.exists(candidate_path):
            model_path = candidate_path
            break

    if not model_path:
        model_load_error = (
            f"Model file not found in '{BASE_DIR}'. Expected 'email_spam_model.pkl' "
            f"or 'email_spam_model (1).pkl'."
        )
        logger.error(model_load_error)
        return

    logger.info("Found model file at: %s", model_path)

    # Attempt to load using joblib first (standard for scikit-learn models), then pickle
    try:
        model = joblib.load(model_path)
        logger.info("Model loaded successfully with joblib from %s", model_path)
    except Exception as joblib_err:
        logger.warning("Joblib loading failed (%s). Retrying with pickle...", joblib_err)
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            logger.info("Model loaded successfully with pickle from %s", model_path)
        except Exception as pickle_err:
            model_load_error = (
                f"Failed to load model file. Joblib error: {joblib_err} | "
                f"Pickle error: {pickle_err}"
            )
            logger.error(model_load_error)

# Load resources once when application initializes
load_metadata()
load_model()

def make_prediction(text):
    """
    Predict whether text is SPAM or HAM using the loaded pipeline.
    Returns a dictionary with prediction details or raises an Exception.
    """
    if model is None:
        raise RuntimeError(model_load_error or "Model is not loaded.")

    # Model inference
    raw_pred = model.predict([text])[0]
    
    # Normalize class label (e.g., 'ham' -> 'HAM', 'spam' -> 'SPAM', 0 -> 'HAM', 1 -> 'SPAM')
    pred_str = str(raw_pred).strip().lower()
    if pred_str in ("spam", "1", "true"):
        label = "SPAM"
    else:
        label = "HAM"

    spam_prob = None
    ham_prob = None
    confidence = None

    # Calculate probabilities if model supports predict_proba
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba([text])[0]
            classes = list(getattr(model, "classes_", []))
            # Map classes to probabilities
            classes_lower = [str(c).lower() for c in classes]
            if "spam" in classes_lower and "ham" in classes_lower:
                spam_idx = classes_lower.index("spam")
                ham_idx = classes_lower.index("ham")
                spam_prob = round(proba[spam_idx] * 100, 2)
                ham_prob = round(proba[ham_idx] * 100, 2)
            elif len(proba) == 2:
                # Assuming index 0 is HAM, index 1 is SPAM
                ham_prob = round(proba[0] * 100, 2)
                spam_prob = round(proba[1] * 100, 2)

            if label == "SPAM" and spam_prob is not None:
                confidence = spam_prob
            elif label == "HAM" and ham_prob is not None:
                confidence = ham_prob
        except Exception as prob_err:
            logger.warning("Could not calculate probabilities: %s", prob_err)

    return {
        "label": label,
        "is_spam": (label == "SPAM"),
        "confidence": confidence,
        "spam_probability": spam_prob,
        "ham_probability": ham_prob
    }

@app.route("/", methods=["GET", "POST"])
def index():
    subject = ""
    message = ""
    result = None
    error = None

    if request.method == "POST":
        subject = request.form.get("subject", "").strip()
        message = request.form.get("message", "").strip()

        # Input validation
        if not subject and not message:
            error = "Please enter an Email Subject or Message body before checking."
        elif model_load_error:
            error = f"Cannot analyze email because the model failed to load: {model_load_error}"
        else:
            # Combine subject and message
            combined_text = f"{subject} {message}".strip()
            try:
                result = make_prediction(combined_text)
            except Exception as e:
                logger.error("Prediction error: %s", e)
                error = f"An error occurred while making the prediction: {str(e)}"

    return render_template(
        "index.html",
        subject=subject,
        message=message,
        result=result,
        error=error,
        model_name=model_name,
        test_accuracy=test_accuracy,
        model_loaded=(model is not None),
        model_load_error=model_load_error,
        metadata_load_error=metadata_load_error
    )

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """API endpoint for asynchronous checks via AJAX."""
    data = request.get_json(silent=True) or request.form
    subject = data.get("subject", "").strip()
    message = data.get("message", "").strip()

    if not subject and not message:
        return jsonify({
            "success": False,
            "error": "Please enter an Email Subject or Message body before checking."
        }), 400

    if model is None:
        return jsonify({
            "success": False,
            "error": f"Model is currently unavailable: {model_load_error}"
        }), 503

    combined_text = f"{subject} {message}".strip()
    try:
        prediction_data = make_prediction(combined_text)
        return jsonify({
            "success": True,
            "result": prediction_data
        })
    except Exception as e:
        logger.error("API Prediction error: %s", e)
        return jsonify({
            "success": False,
            "error": f"Failed to predict: {str(e)}"
        }), 500

if __name__ == "__main__":
    # Run development server on port 5000
    app.run(host="127.0.0.1", port=5000, debug=True)
