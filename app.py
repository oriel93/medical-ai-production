from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
import sqlite3
import uuid
from datetime import datetime, timedelta
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    headers["Authorization"] = f"Bearer {HF_API_TOKEN}"

# Enhanced medical knowledge base
MEDICAL_CONDITIONS = {
    'emergency_keywords': [
        'chest pain', 'difficulty breathing', 'severe bleeding', 'unconscious',
        'stroke symptoms', 'heart attack', 'cannot breathe', 'severe burn',
        'poisoning', 'severe allergic reaction', 'seizure'
    ],
    'urgent_keywords': [
        'high fever', 'persistent vomiting', 'severe pain', 'blood in stool',
        'severe headache', 'dehydration', 'severe cough with blood'
    ],
    'condition_specific_advice': {
        'nosebleed': """
For nosebleeds:
1. Sit upright and lean slightly forward
2. Pinch the soft part of your nose firmly for 10-15 minutes
3. Apply ice to the bridge of your nose
4. Breathe through your mouth and avoid swallowing blood
5. After bleeding stops, avoid blowing your nose for several hours

Seek immediate medical care if:
- Bleeding continues after 20 minutes of direct pressure
- You feel dizzy, weak, or faint
- The nosebleed follows a head injury
- You have frequent nosebleeds (more than once per week)
        """,
        'headache': """
For headaches:
1. Rest in a quiet, dark room
2. Apply cold compress to forehead or warm compress to neck
3. Stay hydrated with water
4. Practice gentle neck stretches
5. Consider over-the-counter pain relievers as directed

Seek medical care if:
- Sudden, severe headache unlike any you've had before
- Headache with fever, stiff neck, confusion, or vision changes
- Headache after a head injury
- Headaches that worsen progressively over days or weeks
        """,
        'fever': """
For fever management:
1. Stay well-hydrated with water, clear broths, or electrolyte solutions
2. Rest and avoid strenuous activity
3. Use lightweight clothing and blankets
4. Consider fever-reducing medications as appropriate
5. Monitor temperature regularly

Seek medical care if:
- Temperature over 103°F (39.4°C)
- Fever persists more than 3 days
- Severe symptoms like difficulty breathing, persistent vomiting
- Signs of dehydration or confusion
        """
    }
}


# Initialize simple database for user tracking
def init_database():
    conn = sqlite3.connect('medical_ai.db')
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS assessments (
            id TEXT PRIMARY KEY,
            symptoms TEXT NOT NULL,
            analysis TEXT NOT NULL,
            urgency_level TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS rate_limits (
            ip_address TEXT PRIMARY KEY,
            requests_count INTEGER DEFAULT 0,
            window_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    conn.close()


def determine_urgency_level(symptoms):
    """Enhanced urgency determination with specific medical knowledge"""
    symptoms_lower = symptoms.lower()

    # Check for emergency indicators
    for keyword in MEDICAL_CONDITIONS['emergency_keywords']:
        if keyword in symptoms_lower:
            return 'High - Seek immediate medical attention!'

    # Check for urgent indicators
    for keyword in MEDICAL_CONDITIONS['urgent_keywords']:
        if keyword in symptoms_lower:
            return 'Medium - Consider seeing a doctor soon'

    return 'Low - Monitor symptoms and consider self-care'


def get_enhanced_fallback_analysis(symptoms):
    """Provide condition-specific advice when AI is unavailable"""
    symptoms_lower = symptoms.lower()

    # Check for specific conditions
    for condition, advice in MEDICAL_CONDITIONS['condition_specific_advice'].items():
        if condition in symptoms_lower:
            return advice

    # Default comprehensive advice
    return """
Based on your symptoms, here are general recommendations:

Immediate Steps:
1. Monitor your symptoms closely and note any changes
2. Stay well-hydrated with water or clear fluids
3. Get adequate rest in a comfortable environment
4. Take your temperature if you feel unwell

Self-Care Measures:
- Maintain good nutrition with light, easily digestible foods
- Avoid alcohol, smoking, and excessive caffeine
- Practice stress-reduction techniques if applicable
- Keep a symptom diary noting times, severity, and triggers

When to Seek Medical Care:
- Symptoms worsen significantly or don't improve within 24-48 hours
- You develop new concerning symptoms
- You feel severely unwell or have difficulty with daily activities
- You have any doubts about your condition

For serious symptoms like severe pain, difficulty breathing, chest pain, or signs of emergency, seek immediate medical attention.
    """


def check_rate_limit(ip_address, max_requests=20):
    """Simple rate limiting: 20 requests per hour"""
    conn = sqlite3.connect('medical_ai.db')
    cursor = conn.cursor()

    cursor.execute(
        'SELECT requests_count, window_start FROM rate_limits WHERE ip_address = ?',
        (ip_address,)
    )
    result = cursor.fetchone()

    now = datetime.now()

    if result:
        requests_count, window_start = result
        window_start = datetime.fromisoformat(window_start)

        # Reset if window expired (1 hour)
        if now - window_start > timedelta(hours=1):
            cursor.execute(
                'UPDATE rate_limits SET requests_count = 1, window_start = ? WHERE ip_address = ?',
                (now.isoformat(), ip_address)
            )
        else:
            if requests_count >= max_requests:
                conn.close()
                return False

            cursor.execute(
                'UPDATE rate_limits SET requests_count = requests_count + 1 WHERE ip_address = ?',
                (ip_address,)
            )
    else:
        cursor.execute(
            'INSERT INTO rate_limits (ip_address, requests_count, window_start) VALUES (?, 1, ?)',
            (ip_address, now.isoformat())
        )

    conn.commit()
    conn.close()
    return True


@app.route('/analyze-symptoms', methods=['POST'])
def analyze_symptoms():
    try:
        # Rate limiting
        client_ip = request.environ.get('HTTP_X_REAL_IP', request.remote_addr)
        if not check_rate_limit(client_ip):
            return jsonify({
                'error': 'Rate limit exceeded. Please try again in an hour.',
                'rate_limited': True
            }), 429

        # Get request data
        data = request.get_json()
        if not data or 'symptoms' not in data:
            return jsonify({'error': 'Please provide symptoms to analyze'}), 400

        symptoms = data['symptoms'].strip()
        if not symptoms or len(symptoms) < 10:
            return jsonify({'error': 'Please provide more detailed symptoms (at least 10 characters)'}), 400

        # Determine urgency level
        urgency_level = determine_urgency_level(symptoms)

        # Enhanced medical prompt for better AI responses
        medical_prompt = f"""
        As a medical information assistant, analyze these symptoms: "{symptoms}"

        Provide a professional assessment including:
        1. Most likely condition(s) based on the symptoms described
        2. Specific self-care recommendations that are safe and appropriate
        3. Clear guidance on when professional medical care should be sought
        4. Any immediate safety considerations

        Keep the response under 250 words, be cautious and professional, and emphasize this is not medical advice.
        """

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
            logger.warning(f'AI API failed: {e}')
            analysis_text = get_enhanced_fallback_analysis(symptoms)

        # Save assessment to database
        assessment_id = str(uuid.uuid4())
        conn = sqlite3.connect('medical_ai.db')
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO assessments (id, symptoms, analysis, urgency_level) VALUES (?, ?, ?, ?)',
            (assessment_id, symptoms, analysis_text, urgency_level)
        )
        conn.commit()
        conn.close()

        # Prepare response
        response_data = {
            'assessment_id': assessment_id,
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
        logger.error(f'Error in analyze_symptoms: {e}')
        return jsonify({
            'error': 'Sorry, the analysis failed. Please try again.',
            'technical_details': str(e) if app.debug else None
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'version': '2.0.0',
        'ai_configured': bool(HF_API_TOKEN),
        'timestamp': datetime.now().isoformat()
    })


@app.route('/stats', methods=['GET'])
def get_stats():
    try:
        conn = sqlite3.connect('medical_ai.db')
        cursor = conn.cursor()

        cursor.execute('SELECT COUNT(*) FROM assessments')
        total_assessments = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM assessments WHERE created_at > datetime("now", "-24 hours")')
        daily_assessments = cursor.fetchone()[0]

        conn.close()

        return jsonify({
            'total_assessments': total_assessments,
            'daily_assessments': daily_assessments,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({'error': 'Stats unavailable'}), 500


# Initialize database on startup
init_database()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f'Enhanced Medical AI server starting on port {port}')
    print(f'Health check: http://localhost:{port}/health')
    print('Ready for production deployment!')

    app.run(host='0.0.0.0', port=port, debug=False)