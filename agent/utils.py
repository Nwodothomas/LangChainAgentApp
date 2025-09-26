"""Utility functions for the medical agent"""

import re
import time
from datetime import datetime

def sanitize_filename(filename):
    """Sanitize filename for safe storage"""
    return re.sub(r'[^\w\.-]', '_', filename)

def format_timestamp():
    """Get formatted timestamp"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def simulate_typing(text, speed=0.01):
    """Simulate typing effect for streaming responses"""
    for char in text:
        yield char
        time.sleep(speed)

def validate_medical_query(query):
    """Validate if query is medically relevant"""
    medical_keywords = [
        'cardio', 'heart', 'blood', 'pressure', 'hypertension', 
        'cholesterol', 'diabetes', 'risk', 'treatment', 'symptom',
        'diagnosis', 'prevention', 'biomarker', 'clinical', 'patient'
    ]
    
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in medical_keywords)