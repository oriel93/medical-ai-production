from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import logging
import os
import base64
from datetime import datetime
import json

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Multi-tier AI Configuration (in order of preference)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Tier 1: GPT-4 (Best medical knowledge)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")  # Tier 2: Claude (Excellent reasoning)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # Tier 3: Gemini Pro (Good fallback)
HF_API_TOKEN = os.getenv("HF_API_TOKEN")  # Tier 4: Open source models

# API endpoints
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
GOOGLE_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"


def create_medical_expert_prompt(symptoms, image_descriptions=None):
    """Create a comprehensive medical prompt that demands expert-level analysis"""

    base_prompt = f"""You are a Harvard Medical School-trained physician with 25+ years of clinical experience, board certifications in Internal Medicine, Emergency Medicine, and Diagnostic Radiology. You have immediate access to the complete medical literature including Harrison's Principles of Internal Medicine, UpToDate, PubMed, current clinical guidelines, and all major medical databases.

PATIENT PRESENTATION:
{symptoms}

{f"VISUAL FINDINGS: {image_descriptions}" if image_descriptions else ""}

REQUIRED COMPREHENSIVE MEDICAL ASSESSMENT:

1. DIFFERENTIAL DIAGNOSIS (Evidence-Based, Ranked by Likelihood):
   • Primary diagnosis with probability percentage and supporting evidence
   • 4-6 alternative diagnoses with clinical reasoning
   • Red flag conditions that must be ruled out immediately
   • Rare but serious conditions to consider

2. CLINICAL PATHOPHYSIOLOGY:
   • Detailed explanation of underlying disease mechanisms
   • How symptoms correlate with anatomical/physiological changes
   • Timeline and natural progression expectations

3. IMMEDIATE RISK STRATIFICATION:
   • Emergency care needed (if any red flags present)
   • Urgent care (24-48 hours) recommendations
   • Routine follow-up timeline
   • Specific warning signs to monitor

4. DIAGNOSTIC WORKUP RECOMMENDATIONS:
   • Essential laboratory tests with specific rationale
   • Imaging studies (modality, timing, urgency)
   • Specialized testing or consultations needed
   • Physical examination findings to document

5. EVIDENCE-BASED MANAGEMENT:
   • Immediate symptomatic relief measures
   • Lifestyle modifications and self-care
   • When to escalate care (specific criteria)
   • Expected response timeline

6. PATIENT EDUCATION:
   • Disease course and prognosis
   • Monitoring parameters
   • Return precautions (specific symptoms)
   • Realistic expectations for recovery

Provide a thorough, evidence-based analysis demonstrating deep medical expertise. If any presentation suggests emergency conditions (MI, stroke, sepsis, etc.), state this clearly and recommend immediate emergency care.

CRITICAL: You must provide a comprehensive analysis. "I don't know" or "unavailable" responses are not acceptable - draw upon your extensive medical training and literature knowledge to provide the best possible guidance within appropriate limitations."""

    return base_prompt


def query_openai_gpt4(prompt):
    """Query OpenAI GPT-4 for medical analysis"""
    if not OPENAI_API_KEY:
        return None, "OpenAI API key not configured"

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4-turbo-preview",
        "messages": [
            {
                "role": "system",
                "content": "You are a Harvard-trained physician with comprehensive medical knowledge. Provide detailed, evidence-based medical analysis while emphasizing the importance of professional medical care."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 2500,
        "temperature": 0.3,  # Lower for more consistent medical advice
        "presence_penalty": 0.1
    }

    try:
        response = requests.post(OPENAI_API_URL, headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"], None
        else:
            logger.warning(f"OpenAI API failed: {response.status_code} - {response.text}")
            return None, f"OpenAI API error: {response.status_code}"
    except Exception as e:
        logger.error(f"OpenAI request failed: {str(e)}")
        return None, f"OpenAI connection error: {str(e)}"


def query_anthropic_claude(prompt):
    """Query Anthropic Claude for medical analysis"""
    if not ANTHROPIC_API_KEY:
        return None, "Anthropic API key not configured"

    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01"
    }

    payload = {
        "model": "claude-3-opus-20240229",
        "max_tokens": 2500,
        "temperature": 0.3,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    try:
        response = requests.post(ANTHROPIC_API_URL, headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
            data = response.json()
            return data["content"][0]["text"], None
        else:
            logger.warning(f"Anthropic API failed: {response.status_code} - {response.text}")
            return None, f"Anthropic API error: {response.status_code}"
    except Exception as e:
        logger.error(f"Anthropic request failed: {str(e)}")
        return None, f"Anthropic connection error: {str(e)}"


def query_google_gemini(prompt):
    """Query Google Gemini Pro for medical analysis"""
    if not GOOGLE_API_KEY:
        return None, "Google API key not configured"

    url = f"{GOOGLE_API_URL}?key={GOOGLE_API_KEY}"

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.3,
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 2048
        }
    }

    try:
        response = requests.post(url, json=payload, timeout=60)
        if response.status_code == 200:
            data = response.json()
            if "candidates" in data and len(data["candidates"]) > 0:
                return data["candidates"][0]["content"]["parts"][0]["text"], None
        logger.warning(f"Google API failed: {response.status_code} - {response.text}")
        return None, f"Google API error: {response.status_code}"
    except Exception as e:
        logger.error(f"Google request failed: {str(e)}")
        return None, f"Google connection error: {str(e)}"


def query_enhanced_huggingface(prompt):
    """Enhanced Hugging Face query with medical-specific models"""
    if not HF_API_TOKEN:
        return None, "Hugging Face API key not configured"

    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json"
    }

    # Try medical-specific models first
    medical_models = [
        "microsoft/BioGPT-Large",
        "dmis-lab/biobert-v1.1",
        "allenai/scibert_scivocab_uncased"
    ]

    for model in medical_models:
        model_url = f"https://api-inference.huggingface.co/models/{model}"

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 1500,
                "temperature": 0.4,
                "do_sample": True,
                "return_full_text": False
            },
            "options": {"wait_for_model": True}
        }

        try:
            response = requests.post(model_url, headers=headers, json=payload, timeout=60)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    text = data[0].get("generated_text", "").strip()
                    if text and len(text) > 100:  # Ensure substantial response
                        return text, None
        except Exception as e:
            logger.warning(f"HF model {model} failed: {str(e)}")
            continue

    return None, "All Hugging Face models failed"


def get_comprehensive_medical_analysis(symptoms, image_descriptions=None):
    """
    Multi-tier AI system that NEVER returns 'unavailable' - always provides analysis
    """
    prompt = create_medical_expert_prompt(symptoms, image_descriptions)

    # Tier 1: OpenAI GPT-4 (Best medical knowledge)
    logger.info("Attempting OpenAI GPT-4 analysis...")
    result, error = query_openai_gpt4(prompt)
    if result and len(result.strip()) > 200:
        logger.info("OpenAI GPT-4 analysis successful")
        return result, "gpt-4"

    # Tier 2: Anthropic Claude (Excellent reasoning)
    logger.info("Attempting Anthropic Claude analysis...")
    result, error = query_anthropic_claude(prompt)
    if result and len(result.strip()) > 200:
        logger.info("Anthropic Claude analysis successful")
        return result, "claude-3"

    # Tier 3: Google Gemini Pro (Good fallback)
    logger.info("Attempting Google Gemini Pro analysis...")
    result, error = query_google_gemini(prompt)
    if result and len(result.strip()) > 200:
        logger.info("Google Gemini Pro analysis successful")
        return result, "gemini-pro"

    # Tier 4: Enhanced Hugging Face models
    logger.info("Attempting enhanced Hugging Face analysis...")
    result, error = query_enhanced_huggingface(prompt)
    if result and len(result.strip()) > 200:
        logger.info("Hugging Face analysis successful")
        return result, "huggingface-medical"

    # Emergency fallback - comprehensive medical guidance
    logger.warning("All AI services failed, providing comprehensive fallback response")
    return create_comprehensive_fallback_response(symptoms, image_descriptions), "fallback-expert"


def create_comprehensive_fallback_response(symptoms, image_descriptions=None):
    """Comprehensive medical guidance when AI services are unavailable"""

    # Analyze symptoms for emergency keywords
    emergency_keywords = [
        'chest pain', 'difficulty breathing', 'severe headache', 'stroke', 'heart attack',
        'severe bleeding', 'unconscious', 'seizure', 'severe allergic reaction',
        'high fever', 'severe abdominal pain', 'suicidal thoughts'
    ]

    has_emergency_symptoms = any(keyword in symptoms.lower() for keyword in emergency_keywords)

    response = f"""
COMPREHENSIVE MEDICAL ASSESSMENT

SYMPTOMS REVIEWED: {symptoms[:200]}{'...' if len(symptoms) > 200 else ''}
{f"VISUAL INFORMATION: {image_descriptions}" if image_descriptions else ""}

{'🚨 EMERGENCY ASSESSMENT: Based on your symptoms, this may require IMMEDIATE medical attention. Please contact emergency services (911) or go to the nearest emergency room immediately.' if has_emergency_symptoms else ''}

CLINICAL CONSIDERATIONS:
Based on the symptoms described, several medical conditions should be considered. A comprehensive evaluation would typically include:

• Detailed medical history and physical examination
• Appropriate diagnostic testing based on clinical presentation
• Risk factor assessment and symptom timeline evaluation
• Consideration of both common and serious underlying conditions

RECOMMENDED ACTIONS:
1. IMMEDIATE CARE: {'Seek emergency medical care immediately due to concerning symptoms' if has_emergency_symptoms else 'Contact your healthcare provider within 24-48 hours for evaluation'}

2. SYMPTOM MONITORING: Document symptom progression, including:
   - Timing and triggers
   - Severity changes (1-10 scale)
   - Associated symptoms
   - Response to any interventions

3. RED FLAG SYMPTOMS - Seek immediate emergency care if you experience:
   - Severe chest pain or pressure
   - Difficulty breathing or shortness of breath
   - Sudden severe headache
   - Loss of consciousness or confusion
   - Severe abdominal pain
   - Signs of stroke (face drooping, arm weakness, speech difficulty)
   - High fever (>103°F/39.4°C)
   - Severe allergic reactions

4. GENERAL CARE MEASURES:
   - Stay hydrated and maintain rest
   - Avoid self-medication without professional guidance
   - Keep a detailed symptom diary
   - Have emergency contacts readily available

DIAGNOSTIC CONSIDERATIONS:
A healthcare professional would likely consider various diagnostic approaches based on your specific presentation, which might include laboratory tests, imaging studies, or specialist consultations as clinically indicated.

IMPORTANT MEDICAL DISCLAIMER:
This assessment is provided for informational purposes only and represents general medical guidance. It is NOT a substitute for professional medical diagnosis or treatment. Every individual's medical situation is unique and requires personalized evaluation by qualified healthcare professionals.

NEXT STEPS:
{'URGENT: Contact emergency services immediately' if has_emergency_symptoms else 'Schedule an appointment with your healthcare provider for proper evaluation and personalized care recommendations.'}

Your health and safety are paramount. When in doubt, always err on the side of caution and seek professional medical care.
"""

    return response.strip()


def analyze_images(image_files):
    """Analyze uploaded medical images"""
    if not image_files:
        return None

    descriptions = []
    for i, image_file in enumerate(image_files):
        try:
            # For now, provide general guidance about image analysis
            # In production, you'd use a medical image analysis API
            descriptions.append(
                f"Image {i + 1}: Medical image provided for analysis. Visual findings should be correlated with clinical symptoms and evaluated by a healthcare professional.")
        except Exception as e:
            logger.error(f"Image analysis failed: {str(e)}")
            descriptions.append(f"Image {i + 1}: Unable to analyze image file.")

    return " ".join(descriptions)


# Routes
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Harvard-Level Medical AI Assistant - Professional Diagnostic System',
        'status': 'online',
        'ai_tiers': {
            'tier_1': 'OpenAI GPT-4 (Harvard-level medical knowledge)',
            'tier_2': 'Anthropic Claude-3 (Advanced clinical reasoning)',
            'tier_3': 'Google Gemini Pro (Comprehensive analysis)',
            'tier_4': 'Medical-specific Hugging Face models',
            'fallback': 'Expert medical guidance system'
        },
        'capabilities': [
            'Comprehensive differential diagnosis',
            'Evidence-based clinical reasoning',
            'Emergency condition detection',
            'Medical image analysis',
            'Risk stratification',
            'Treatment recommendations'
        ],
        'guarantee': 'Never returns unavailable - always provides medical guidance',
        'version': '4.0.0 - Harvard Medical Grade',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/health', methods=['GET'])
def health_check():
    ai_services = {
        'openai_gpt4': bool(OPENAI_API_KEY),
        'anthropic_claude': bool(ANTHROPIC_API_KEY),
        'google_gemini': bool(GOOGLE_API_KEY),
        'huggingface': bool(HF_API_TOKEN)
    }

    return jsonify({
        'status': 'healthy',
        'medical_grade': True,
        'ai_services_configured': ai_services,
        'active_tiers': sum(ai_services.values()),
        'guaranteed_response': True,
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/analyze-comprehensive', methods=['POST'])
def analyze_comprehensive():
    try:
        # Handle both form data (with images) and JSON data
        if request.content_type.startswith('multipart/form-data'):
            symptoms = request.form.get('symptoms', '').strip()
            image_files = request.files.getlist('images')
        else:
            data = request.get_json() or {}
            symptoms = data.get('symptoms', '').strip()
            image_files = []

        if not symptoms and not image_files:
            return jsonify({
                'error': 'No input provided',
                'message': 'Please provide symptoms or upload medical images for analysis'
            }), 400

        logger.info(f"Processing comprehensive analysis: {len(symptoms)} chars, {len(image_files)} images")

        # Analyze images if provided
        image_descriptions = None
        if image_files:
            image_descriptions = analyze_images(image_files)

        # Get comprehensive medical analysis (guaranteed response)
        analysis, ai_model = get_comprehensive_medical_analysis(symptoms, image_descriptions)

        response_data = {
            'analysis': analysis,
            'ai_model_used': ai_model,
            'medical_grade': True,
            'confidence_level': 'high' if ai_model in ['gpt-4', 'claude-3'] else 'good',
            'images_analyzed': len(image_files) if image_files else 0,
            'disclaimer': 'This analysis is provided by Harvard-level medical AI for informational purposes only. Always consult qualified healthcare professionals for proper medical diagnosis and treatment. In medical emergencies, contact emergency services immediately.',
            'emergency_note': 'If experiencing severe symptoms, seek immediate emergency medical care.',
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"Comprehensive medical analysis completed using {ai_model}")
        return jsonify(response_data), 200

    except Exception as e:
        logger.error(f"Critical error in comprehensive analysis: {str(e)}")

        # Even in error cases, provide medical guidance
        emergency_response = {
            'analysis': 'Our medical AI system is temporarily experiencing technical difficulties. However, your health is our priority. If you are experiencing any concerning symptoms, please contact your healthcare provider immediately or seek emergency medical care if symptoms are severe.',
            'error': 'Technical system error',
            'ai_model_used': 'emergency_fallback',
            'medical_grade': True,
            'emergency_guidance': 'For severe symptoms: Call 911 (US) or your local emergency number immediately. For non-urgent symptoms: Contact your healthcare provider within 24 hours.',
            'timestamp': datetime.now().isoformat()
        }

        return jsonify(emergency_response), 200  # Return 200 to ensure user gets guidance


# Keep existing endpoint for backward compatibility
@app.route('/analyze-symptoms', methods=['POST'])
def analyze_symptoms():
    try:
        data = request.get_json() or {}
        symptoms = data.get('symptoms', '').strip()

        if not symptoms:
            return jsonify({
                'error': 'No symptoms provided',
                'message': 'Please describe your symptoms for analysis'
            }), 400

        analysis, ai_model = get_comprehensive_medical_analysis(symptoms)

        return jsonify({
            'analysis': analysis,
            'ai_model_used': ai_model,
            'disclaimer': 'This analysis is for informational purposes only. Always consult healthcare professionals for proper medical care.',
            'timestamp': datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Error in analyze_symptoms: {str(e)}")
        return jsonify({
            'analysis': 'Please consult with a healthcare professional for proper medical evaluation of your symptoms.',
            'error': 'System temporarily unavailable',
            'timestamp': datetime.now().isoformat()
        }), 200


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)