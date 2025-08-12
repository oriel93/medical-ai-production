from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create Flask app
app = Flask(__name__)
CORS(app)

# Configuration
HF_API_TOKEN = os.getenv('HF_API_TOKEN')
HF_MODEL = os.getenv('HF_MODEL', 'microsoft/BioGPT-Large')
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

# Headers for API requests
headers = {}
if HF_API_TOKEN:
    headers["Authorization"] = f"Bearer {HF_API_TOKEN"


def determine_urgency_level(symptoms):
    """Determine urgency level based on symptom keywords"""
    symptoms_lower = symptoms.lower()

    # High urgency indicators
    high_urgency_keywords = [
        'chest pain', 'difficulty breathing', 'severe bleeding',
        'unconscious', 'stroke', 'heart attack', 'severe headache',
        'cannot breathe', 'severe burn', 'poisoning'
    ]

    # Medium urgency indicators
    medium_urgency_keywords = [
        'fever over 101', 'persistent vomiting', 'severe pain',
        'high fever', 'dehydration', 'severe cough', 'blood in stool'
    ]

    # Check for high urgency
    for keyword in high_urgency_keywords:
        if keyword in symptoms_lower:
            return 'High - Seek immediate medical attention!'

    # Check for medium urgency
    for keyword in medium_urgency_keywords:
        if keyword in symptoms_lower:
            return 'Medium - Consider seeing a doctor soon'

    return 'Low - Monitor symptoms and consider self-care'


def get_enhanced_fallback_analysis(symptoms):
    """Provide condition-specific advice when AI is unavailable"""
    symptoms_lower = symptoms.lower()

    # Nosebleed specific advice
    if 'nosebleed' in symptoms_lower or 'nose bleed' in symptoms_lower:
        return """For nosebleeds:
1. Sit upright and lean slightly forward
2. Pinch the soft part of your nose firmly for 10-15 minutes
3. Apply ice to the bridge of your nose
4. Breathe through your mouth and avoid swallowing blood
5. After bleeding stops, avoid blowing your nose for several hours

Seek immediate medical care if:
- Bleeding continues after 20 minutes of direct pressure
- You feel dizzy, weak, or faint
- The nosebleed follows a head injury
- You have frequent nosebleeds (more than once per week)"""

    # Headache specific advice
    elif 'headache' in symptoms_lower:
        return """For headaches:
1. Rest in a quiet, dark room
2. Apply cold compress to forehead or warm compress to neck
3. Stay hydrated with water
4. Practice gentle neck stretches
5. Consider over-the-counter pain relievers as directed

Seek medical care if:
- Sudden, severe headache unlike any you've had before
- Headache with fever, stiff neck, confusion, or vision changes
- Headache after a head injury
- Headaches that worsen progressively over days or weeks"""

    # Default comprehensive advice
    return """Based on your symptoms, here are general recommendations:

Immediate Steps:
1. Monitor your symptoms closely and note any changes
2. Stay well-hydrated with water or clear fluids
3. Get adequate rest in a comfortable environment
4. Take your temperature if you feel unwell

Self-Care Measures:
- Maintain good nutrition with light, easily digestible foods
- Avoid alcohol, smoking, and excessive caffeine
- Practice stress-reduction techniques if applicable

When to Seek Medical Care:
- Symptoms worsen significantly or don't improve within 24-48 hours
- You develop new concerning symptoms
- You feel severely unwell or have difficulty with daily activities
- You have any doubts about your condition

For serious symptoms like severe pain, difficulty breathing, chest pain, or signs of emergency, seek immediate medical attention."""


@app.route('/', methods=['GET'])
def home():
    """Root endpoint to prevent 404 errors"""
    return jsonify({
        "message": "Medical AI Assistant Backend is Running",
        "status": "healthy",
        "endpoints": {
            "health_check": "/health",
            "test": "/test",
            "analyze_symptoms": "POST /analyze-symptoms"
        },
        "version": "2.0.0"
    })


@app.route('/analyze-symptoms', methods=['POST'])
def analyze_symptoms():
    try:
        # Get request data
        data = request.get_json()
        if not data or 'symptoms' not in data:
            return jsonify({'error': 'Please provide symptoms to analyze'}), 400

        symptoms = data['symptoms'].strip()
        if not symptoms or len(symptoms) < 5:
            return jsonify({'error': 'Please provide more detailed symptoms'}), 400

        # Determine urgency level
        urgency_level = determine_urgency_level(symptoms)

        # Enhanced medical prompt for better AI responses
        medical_prompt = f"""As a medical information assistant, analyze these symptoms: "{symptoms}"

Provide a professional assessment including:
1. Most likely condition(s) based on the symptoms described
2. Specific self-care recommendations that are safe and appropriate
3. Clear guidance on when professional medical care should be sought
4. Any immediate safety considerations

Keep the response under 250 words, be cautious and professional, and emphasize this is not medical advice."""

        # Try to get AI analysis
        try:
            payload = {
                "inputs": medical_prompt,
                "parameters": {
                    "max_new_tokens": 300,
                    "temperature": 0.4,
                    "do_sample": True,
                    "return_full_text": False
                },
                "options": {
                    "wait_for_model": True,
                    "use_cache": True
                }
            }

            response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=45)

            if response.status_code == 200:
                ai_response = response.json()
                if isinstance(ai_response, list) and len(ai_response) > 0:
                    analysis_text = ai_response[0].get('generated_text', '').strip()
                else:
                    analysis_text = ai_response.get('generated_text', '').strip()

                if not analysis_text:
                    raise Exception("Empty response from AI")

            else:
                raise Exception(f"API returned status {response.status_code}")

        except Exception as e:
            print(f'AI API failed: {e}')
            analysis_text = get_enhanced_fallback_analysis(symptoms)

        # Prepare response
        response_data = {
            'symptoms': symptoms,
            'analysis': analysis_text,
            'urgency_level': urgency_level,
            'disclaimer': 'This is NOT medical advice. Always consult a qualified healthcare professional for medical concerns.',
            'timestamp': datetime.now().isoformat(),
            'emergency_note': 'If this is a life-threatening emergency, call 911 immediately!' if urgency_level.startswith(
                'High') else None
        }

        return jsonify(response_data)

    except Exception as e:
        print(f'Error in analyze_symptoms: {e}')
        return jsonify({
            'error': 'Sorry, the analysis failed. Please try again.',
            'details': str(e) if app.debug else None
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'version': '2.0.0',
        'ai_configured': bool(HF_API_TOKEN),
        'timestamp': datetime.now().isoformat()
    })


@app.route('/test', methods=['GET'])
def test_endpoint():
    """Test endpoint to verify server is working"""
    return jsonify({
        'message': 'Your Medical AI server is working perfectly!',
        'status': 'success',
        'timestamp': datetime.now().isoformat()
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f'Medical AI server starting on port {port}')
    print('Ready for production deployment!')

    app.run(host='0.0.0.0', port=port, debug=False)
@app.route(`'/`', methods=[`'GET`'])`nef home():`n    `'`'`'Simple home page to confirm server is running`'`'`'`n    return jsonify({`n        `'message`': `'Welcome to Medical AI Assistant Backend!`',`n        `'status`': `'online`',`n        `'endpoints`': {`n            `'/health`': `'Check API status`',`n            `'/test`': `'Test basic connectivity`',`n            `'/analyze-symptoms`': `'POST for symptom analysis`'`n        },`n        `'version`': `'2.0.0`',`n        `'timestamp`': datetime.now().isoformat()`n    })`n
