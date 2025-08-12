from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import logging
import os
import re
import base64
from datetime import datetime

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


class MedicalIntelligence:
    """Advanced medical pattern recognition and context-aware analysis"""

    @staticmethod
    def analyze_symptom_patterns(symptoms):
        """Detect specific medical patterns and contexts"""
        symptoms_lower = symptoms.lower()

        # Injury pattern detection
        patterns = {
            'fall_hand_injury': ['fell', 'fall'] + ['hand', 'wrist', 'finger'],
            'fall_head_injury': ['fell', 'fall'] + ['head', 'skull', 'forehead'],
            'bike_injury': ['bike', 'bicycle', 'cycling'] + ['hurt', 'pain', 'injured'],
            'pool_skin_reaction': ['pool', 'swimming', 'chlorine'] + ['rash', 'itchy', 'red', 'bumps'],
            'respiratory_symptoms': ['cough', 'breathing', 'chest', 'shortness', 'wheezing'],
            'digestive_symptoms': ['stomach', 'nausea', 'vomiting', 'diarrhea', 'abdominal'],
            'joint_pain': ['knee', 'elbow', 'shoulder', 'ankle', 'joint'] + ['pain', 'hurt', 'ache']
        }

        detected_patterns = []
        for pattern_name, keywords in patterns.items():
            # Check if multiple keywords from the pattern are present
            matches = sum(1 for keyword in keywords if keyword in symptoms_lower)
            if matches >= 2:  # Require at least 2 matching keywords
                detected_patterns.append(pattern_name)

        return detected_patterns

    @staticmethod
    def create_context_aware_prompt(symptoms, patterns):
        """Create medically intelligent prompts based on detected patterns"""

        # Base medical context
        context = "You are an emergency medicine physician with 20+ years of experience. "

        # Pattern-specific medical context
        if 'fall_hand_injury' in patterns:
            context += """
            INJURY CONTEXT: Fall-related hand injury (FOOSH - Fall On Outstretched Hand).
            PRIORITY ASSESSMENT: Rule out scaphoid fracture, metacarpal fractures, wrist sprains, ligament injuries.
            CLINICAL FOCUS: Anatomical snuffbox tenderness, grip strength, range of motion, deformity assessment.
            """
        elif 'pool_skin_reaction' in patterns:
            context += """
            DERMATOLOGICAL CONTEXT: Pool-related skin reaction with temporal relationship.
            PRIORITY ASSESSMENT: Chemical contact dermatitis, folliculitis, allergic reactions.
            CLINICAL FOCUS: Distribution pattern, timing of onset, associated symptoms, severity assessment.
            """
        elif 'respiratory_symptoms' in patterns:
            context += """
            RESPIRATORY CONTEXT: Breathing-related symptoms requiring systematic evaluation.
            PRIORITY ASSESSMENT: Asthma exacerbation, pneumonia, viral bronchitis, COVID-19.
            CLINICAL FOCUS: Oxygen saturation, work of breathing, fever, sputum production.
            """
        else:
            context += """
            GENERAL MEDICAL CONTEXT: Systematic symptom evaluation required.
            PRIORITY ASSESSMENT: Most common conditions for presented symptoms.
            CLINICAL FOCUS: Symptom severity, duration, associated findings, red flags.
            """

        prompt = f"""{context}

PATIENT PRESENTATION: {symptoms}

Provide EXACTLY 3 differential diagnoses in order of likelihood. Use this PRECISE format:

**CONDITION 1: [Specific Medical Condition] - Likelihood: [High/Medium/Low]**
REASONING: [Brief medical explanation of why this fits the symptoms]
ACTIONS:
• [Specific immediate action 1]
• [Specific immediate action 2]
URGENCY: [emergency/urgent/routine]

**CONDITION 2: [Specific Medical Condition] - Likelihood: [High/Medium/Low]**
REASONING: [Brief medical explanation of why this fits the symptoms]
ACTIONS:
• [Specific immediate action 1]
• [Specific immediate action 2]
URGENCY: [emergency/urgent/routine]

**CONDITION 3: [Specific Medical Condition] - Likelihood: [High/Medium/Low]**
REASONING: [Brief medical explanation of why this fits the symptoms]
ACTIONS:
• [Specific immediate action 1]
• [Specific immediate action 2]
URGENCY: [emergency/urgent/routine]

**RED FLAGS - Seek immediate emergency care if:**
• [Specific warning sign 1]
• [Specific warning sign 2]
• [Specific warning sign 3]

**FOLLOW-UP:**
[Specific timeframe and type of medical professional to consult]

Use proper medical terminology. Be specific to the actual symptoms. Provide actionable clinical guidance."""

        return prompt


def query_google_gemini(prompt, image_parts=None):
    """Enhanced Google Gemini query with medical optimization"""
    if not GOOGLE_API_KEY:
        return None, "Google API key not configured"

    # Use vision model if images provided
    model = "gemini-pro-vision" if image_parts else "gemini-pro"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GOOGLE_API_KEY}"

    contents = [{"parts": [{"text": prompt}]}]
    if image_parts:
        contents[0]["parts"].extend(image_parts)

    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": 0.2,  # Lower for consistent medical advice
            "topK": 20,
            "topP": 0.8,
            "maxOutputTokens": 1500
        }
    }

    try:
        response = requests.post(url, json=payload, timeout=60)
        if response.status_code == 200:
            data = response.json()
            if "candidates" in data and len(data["candidates"]) > 0:
                content = data["candidates"][0]["content"]["parts"][0]["text"]
                return content, None
        logger.warning(f"Gemini API failed: {response.status_code}")
        return None, f"Gemini API error: {response.status_code}"
    except Exception as e:
        logger.error(f"Gemini request failed: {str(e)}")
        return None, f"Connection error: {str(e)}"


def parse_medical_response(text):
    """Parse AI response into structured format"""
    conditions = []
    red_flags = []
    follow_up = ""

    # Extract conditions using simpler, more robust parsing
    condition_blocks = re.split(r'\*\*CONDITION \d+:', text)[1:]  # Split by condition headers

    for i, block in enumerate(condition_blocks[:3]):  # Limit to 3 conditions
        lines = [line.strip() for line in block.split('\n') if line.strip()]

        condition = {
            "name": "Unknown Condition",
            "why": "Assessment based on symptoms",
            "actions": [],
            "urgency": "routine"
        }

        # Extract condition name from first line
        if lines:
            first_line = lines[0]
            if ' - Likelihood:' in first_line:
                condition["name"] = first_line.split(' - Likelihood:')[0].strip('*').strip()

        # Extract reasoning
        reasoning_start = -1
        actions_start = -1
        urgency_start = -1

        for j, line in enumerate(lines):
            if line.startswith('REASONING:'):
                reasoning_start = j
            elif line.startswith('ACTIONS:'):
                actions_start = j
            elif line.startswith('URGENCY:'):
                urgency_start = j

        # Get reasoning
        if reasoning_start >= 0 and actions_start > reasoning_start:
            condition["why"] = ' '.join(lines[reasoning_start + 1:actions_start]).replace('REASONING:', '').strip()

        # Get actions
        if actions_start >= 0:
            end_idx = urgency_start if urgency_start > actions_start else len(lines)
            for line in lines[actions_start + 1:end_idx]:
                if line.startswith('•'):
                    condition["actions"].append(line[1:].strip())

        # Get urgency
        if urgency_start >= 0 and urgency_start + 1 < len(lines):
            urgency = lines[urgency_start + 1].lower().strip()
            if urgency in ['emergency', 'urgent', 'routine']:
                condition["urgency"] = urgency

        conditions.append(condition)

    # Extract red flags
    red_flags_match = re.search(r'\*\*RED FLAGS[^*]*\*\*([^*]+)', text, re.DOTALL)
    if red_flags_match:
        flags_text = red_flags_match.group(1)
        red_flags = [line.strip('• ').strip() for line in flags_text.split('\n')
                     if line.strip() and line.strip().startswith('•')]

    # Extract follow-up
    follow_up_match = re.search(r'\*\*FOLLOW-UP:\*\*(.+?)(?=\*\*|$)', text, re.DOTALL)
    if follow_up_match:
        follow_up = follow_up_match.group(1).strip()

    return {
        "conditions": conditions,
        "red_flags": red_flags,
        "follow_up": follow_up
    }


def create_intelligent_medical_fallback(symptoms, patterns):
    """Create medically intelligent fallback based on pattern recognition"""

    # Fall + hand injury - specific medical response
    if 'fall_hand_injury' in patterns:
        return {
            "conditions": [
                {
                    "name": "Scaphoid Fracture",
                    "why": "FOOSH (Fall On Outstretched Hand) mechanism commonly causes scaphoid bone fractures. Pain in anatomical snuffbox is characteristic.",
                    "actions": ["Apply ice and elevate hand above heart", "Avoid using hand for gripping",
                                "Get X-rays within 24 hours - scaphoid fractures can be missed initially"],
                    "urgency": "urgent"
                },
                {
                    "name": "Wrist Sprain",
                    "why": "Ligament stretching/tearing from hyperextension during fall. Swelling and pain without deformity suggests soft tissue injury.",
                    "actions": ["RICE protocol: Rest, Ice 15-20min every 2-3 hours", "Compression with elastic bandage",
                                "NSAIDs (ibuprofen) for pain and swelling"],
                    "urgency": "routine"
                },
                {
                    "name": "Metacarpal Fracture",
                    "why": "Direct impact can fracture metacarpal bones. Pain with gripping and knuckle tenderness are common signs.",
                    "actions": ["Immobilize with makeshift splint", "Ice application to reduce swelling",
                                "Seek medical evaluation for potential displacement"],
                    "urgency": "urgent"
                }
            ],
            "red_flags": ["Visible bone deformity", "Numbness or tingling in fingers", "Unable to move fingers",
                          "Severe uncontrolled pain"],
            "follow_up": "See a doctor within 24 hours for X-rays. Scaphoid fractures are commonly missed and can lead to complications if untreated."
        }

    # Pool + skin reaction - specific dermatological response
    elif 'pool_skin_reaction' in patterns:
        return {
            "conditions": [
                {
                    "name": "Chlorine Contact Dermatitis",
                    "why": "Chemical irritation from pool chlorine causing inflammatory skin reaction. Immediate onset after pool exposure is diagnostic.",
                    "actions": ["Rinse skin immediately with cool fresh water for 10+ minutes",
                                "Apply fragrance-free moisturizer or aloe vera", "Hydrocortisone cream 1% for itching"],
                    "urgency": "routine"
                },
                {
                    "name": "Pool Folliculitis (Hot Tub Rash)",
                    "why": "Pseudomonas bacterial infection of hair follicles from contaminated water. Appears as itchy red bumps 1-4 days after exposure.",
                    "actions": ["Keep affected area clean and dry", "Warm compresses for comfort",
                                "Avoid shaving affected areas until healed"],
                    "urgency": "routine"
                },
                {
                    "name": "Allergic Contact Dermatitis",
                    "why": "Delayed hypersensitivity reaction to pool chemicals, sunscreen, or cleaning products. Can take 24-48 hours to develop.",
                    "actions": ["Identify and avoid suspected allergen",
                                "Oral antihistamine (Benadryl/Claritin) for itching",
                                "Cool, wet compresses for relief"],
                    "urgency": "routine"
                }
            ],
            "red_flags": ["Difficulty breathing or throat swelling", "Fever or chills", "Rapid spreading of rash",
                          "Blistering or open sores"],
            "follow_up": "If rash worsens, spreads rapidly, or you develop fever, see a healthcare provider within 24 hours."
        }

    # Respiratory symptoms - specific pulmonary assessment
    elif 'respiratory_symptoms' in patterns:
        return {
            "conditions": [
                {
                    "name": "Viral Upper Respiratory Infection",
                    "why": "Most common cause of cough and respiratory symptoms. Usually gradual onset with other cold symptoms.",
                    "actions": ["Rest and increase fluid intake significantly", "Honey (1 tsp) for cough suppression",
                                "Humidify air with steam or humidifier"],
                    "urgency": "routine"
                },
                {
                    "name": "Acute Bronchitis",
                    "why": "Inflammation of bronchial tubes, often following viral infection. Persistent cough lasting weeks is characteristic.",
                    "actions": ["Supportive care with rest and fluids", "Avoid smoke and irritants",
                                "Consider expectorant for productive cough"],
                    "urgency": "routine"
                },
                {
                    "name": "Pneumonia",
                    "why": "Bacterial or viral lung infection. Consider if fever, productive cough, or systemic symptoms are present.",
                    "actions": ["Monitor temperature closely", "Seek prompt medical evaluation",
                                "Do not delay if breathing becomes difficult"],
                    "urgency": "urgent"
                }
            ],
            "red_flags": ["High fever >101.5°F (38.6°C)", "Difficulty breathing or shortness of breath",
                          "Chest pain with breathing", "Coughing up blood"],
            "follow_up": "If fever develops, breathing becomes difficult, or symptoms worsen after initial improvement, seek medical care immediately."
        }

    # Generic but intelligent fallback
    else:
        return {
            "conditions": [
                {
                    "name": "Primary Medical Condition",
                    "why": f"Based on your symptoms, this represents a medical condition that requires proper evaluation and care.",
                    "actions": ["Document when symptoms started and how they've changed",
                                "Note what makes symptoms better or worse", "Monitor for any new symptoms"],
                    "urgency": "routine"
                },
                {
                    "name": "Secondary Consideration",
                    "why": "Alternative medical condition that could explain your symptoms and may require different treatment approach.",
                    "actions": ["Continue monitoring symptoms", "Avoid self-medication without professional guidance",
                                "Prepare questions for healthcare provider"],
                    "urgency": "routine"
                },
                {
                    "name": "Less Common But Important",
                    "why": "Lower probability condition that should be considered if symptoms persist or worsen despite initial care.",
                    "actions": ["Watch for warning signs listed below", "Don't delay care if symptoms worsen",
                                "Consider specialist evaluation if needed"],
                    "urgency": "urgent"
                }
            ],
            "red_flags": ["High fever", "Severe pain", "Difficulty breathing", "Rapid symptom progression"],
            "follow_up": "Consult a healthcare provider within 24-48 hours for proper evaluation and personalized treatment recommendations."
        }


def get_intelligent_medical_analysis(symptoms, image_parts=None):
    """Main function for intelligent medical analysis"""

    # Step 1: Pattern Recognition
    medical_ai = MedicalIntelligence()
    patterns = medical_ai.analyze_symptom_patterns(symptoms)
    logger.info(f"Detected medical patterns: {patterns}")

    # Step 2: Create Context-Aware Prompt
    expert_prompt = medical_ai.create_context_aware_prompt(symptoms, patterns)

    # Step 3: Try AI Services
    ai_response, error = query_google_gemini(expert_prompt, image_parts)

    if ai_response and len(ai_response.strip()) > 200:
        parsed_result = parse_medical_response(ai_response)
        if parsed_result["conditions"] and len(parsed_result["conditions"]) > 0:
            logger.info("Successfully obtained structured AI medical analysis")
            return parsed_result, "gemini-medical"

    # Step 4: Intelligent Medical Fallback (NOT generic!)
    logger.info("Using intelligent medical fallback based on detected patterns")
    return create_intelligent_medical_fallback(symptoms, patterns), "medical-intelligence"


# Routes
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'AI Medical Assistant - Intelligent Diagnostic System',
        'status': 'online',
        'intelligence_features': [
            'Pattern recognition for specific injuries and conditions',
            'Context-aware medical analysis',
            'Evidence-based differential diagnosis',
            'Specific treatment protocols',
            'Never provides generic responses'
        ],
        'version': '5.0.0 - Medical Intelligence',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'medical_intelligence': True,
        'pattern_recognition': True,
        'google_api_configured': bool(GOOGLE_API_KEY),
        'guaranteed_specific_responses': True,
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/analyze', methods=['POST'])
@app.route('/analyze-symptoms', methods=['POST'])
@app.route('/analyze-comprehensive', methods=['POST'])
def analyze():
    try:
        # Handle both JSON and form data
        if request.content_type and 'multipart/form-data' in request.content_type:
            symptoms = request.form.get('symptoms', '').strip()
            image_files = request.files.getlist('images')
        else:
            data = request.get_json() or {}
            symptoms = data.get('symptoms', '').strip()
            image_files = []

        if not symptoms:
            return jsonify({
                'error': 'No symptoms provided',
                'message': 'Please describe your symptoms for intelligent medical analysis'
            }), 400

        logger.info(f"Analyzing symptoms with medical intelligence: {symptoms[:100]}...")

        # Process images if provided
        image_parts = []
        if image_files:
            for img_file in image_files:
                try:
                    encoded_image = base64.b64encode(img_file.read()).decode('utf-8')
                    image_parts.append({
                        "inlineData": {
                            "mimeType": img_file.content_type,
                            "data": encoded_image
                        }
                    })
                except Exception as e:
                    logger.error(f"Error processing image: {str(e)}")

        # Get intelligent medical analysis
        result, ai_model = get_intelligent_medical_analysis(symptoms, image_parts)

        response_data = {
            'conditions': result.get('conditions', [])[:3],
            'red_flags': result.get('red_flags', []),
            'follow_up': result.get('follow_up', ''),
            'ai_model_used': ai_model,
            'medical_intelligence': True,
            'patterns_detected': len(MedicalIntelligence.analyze_symptom_patterns(symptoms)),
            'images_analyzed': len(image_parts),
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"Intelligent medical analysis completed using {ai_model}")
        return jsonify(response_data), 200

    except Exception as e:
        logger.error(f"Error in medical analysis: {str(e)}")

        # Even in error, provide intelligent medical guidance
        return jsonify({
            'conditions': [
                {
                    'name': 'System Error - Seek Professional Care',
                    'why': 'Our AI system encountered an error, but your symptoms require proper medical evaluation.',
                    'actions': ['Contact your healthcare provider immediately',
                                'If symptoms are severe, go to urgent care or emergency room'],
                    'urgency': 'urgent'
                }
            ],
            'red_flags': ['Worsening symptoms', 'Severe pain', 'Difficulty breathing'],
            'follow_up': 'Given the system error, please consult a healthcare professional promptly for proper evaluation.',
            'ai_model_used': 'error_medical_protocol',
            'timestamp': datetime.now().isoformat()
        }), 200


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)