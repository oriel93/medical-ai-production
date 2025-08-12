from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import logging
import os
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration - set these in Render environment variables
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_MODEL_URL = os.getenv("HF_API_URL", "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium")


def query_huggingface_api(payload):
    """Query Hugging Face API with proper error handling"""
    headers = {"Content-Type": "application/json"}

    # Fixed f-string syntax - this was the original error
    if HF_API_TOKEN:
        headers["Authorization"] = f"Bearer {HF_API_TOKEN}"

    try:
        response = requests.post(HF_MODEL_URL, headers=headers, json=payload, timeout=30)

        if response.status_code == 200:
            return response.json(), 200
        elif response.status_code == 503:
            logger.warning("Model is loading, please wait...")
            return {"error": "Model is warming up, try again in a moment"}, 503
        else:
            logger.warning(f"AI API failed: status {response.status_code}")
            return {"error": f"API error: {response.status_code}"}, response.status_code

    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {str(e)}")
        return {"error": "Failed to connect to AI service"}, 500


def analyze_symptoms_with_ai(symptoms_text):
    """Analyze symptoms using AI with medical-focused prompting"""
    if not symptoms_text.strip():
        return "Please provide symptoms to analyze."

    # Medical-focused prompt
    prompt = (
        f"As a medical AI assistant, analyze these symptoms and provide general health guidance. "
        f"Include possible conditions (not diagnoses), severity assessment, and when to seek care. "
        f"Always emphasize consulting healthcare professionals. Symptoms: {symptoms_text}"
    )

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 300,
            "temperature": 0.7,
            "do_sample": True,
            "return_full_text": False
        },
        "options": {
            "wait_for_model": True
        }
    }

    result, status_code = query_huggingface_api(payload)

    if status_code == 200 and isinstance(result, list) and len(result) > 0:
        generated_text = result[0].get("generated_text", "").strip()
        if generated_text:
            return generated_text

    # Fallback response
    return ("I'm unable to analyze your symptoms right now. Please consult with a healthcare "
            "professional for proper medical evaluation and advice.")


# Routes
@app.route('/', methods=['GET'])
def home():
    """Home endpoint with service information"""
    return jsonify({
        'message': 'Welcome to Medical AI Assistant Backend!',
        'status': 'online',
        'endpoints': {
            '/health': 'Check API status',
            '/test': 'Test basic connectivity',
            '/analyze-symptoms': 'POST for symptom analysis'
        },
        'version': '2.0.0',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'Medical AI Assistant',
        'ai_configured': bool(HF_API_TOKEN)
    }), 200


@app.route('/test', methods=['GET'])
def test_endpoint():
    """Test endpoint for connectivity"""
    return jsonify({
        'message': 'API is working correctly',
        'timestamp': datetime.now().isoformat(),
        'test': 'passed'
    }), 200


@app.route('/analyze-symptoms', methods=['POST'])
def analyze_symptoms():
    """Main symptom analysis endpoint"""
    try:
        # Validate JSON request
        if not request.is_json:
            return jsonify({
                'error': 'Content-Type must be application/json',
                'message': 'Please send symptoms in JSON format'
            }), 400

        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'No JSON data provided',
                'message': 'Request body cannot be empty'
            }), 400

        # Extract symptoms from request
        symptoms = data.get('symptoms', '').strip()
        if not symptoms:
            return jsonify({
                'error': 'No symptoms provided',
                'message': 'Please provide symptoms to analyze'
            }), 400

        logger.info(f"Analyzing symptoms: {symptoms[:100]}...")

        # Get AI analysis
        analysis = analyze_symptoms_with_ai(symptoms)

        # Return response
        response_data = {
            'analysis': analysis,
            'disclaimer': 'This is for informational purposes only. Always consult healthcare professionals for medical advice.',
            'timestamp': datetime.now().isoformat(),
            'symptoms_received': symptoms[:100] + ('...' if len(symptoms) > 100 else '')
        }

        logger.info("Analysis completed successfully")
        return jsonify(response_data), 200

    except Exception as e:
        logger.error(f"Error in analyze_symptoms: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': 'An error occurred while processing your request'
        }), 500


# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Not Found',
        'message': 'The requested endpoint does not exist',
        'available_endpoints': ['/', '/health', '/test', '/analyze-symptoms']
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        'error': 'Method Not Allowed',
        'message': 'The requested method is not allowed for this endpoint'
    }), 405


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal Server Error',
        'message': 'An internal server error occurred'
    }), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
