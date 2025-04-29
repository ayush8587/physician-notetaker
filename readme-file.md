# Physician Notetaker

A comprehensive AI system for medical transcription, NLP-based summarization, and sentiment analysis.

## Overview

This project implements an NLP pipeline for:
1. Extracting key medical details from physician-patient conversations
2. Analyzing patient sentiment and intent
3. Generating structured SOAP notes from transcribed conversations

## Features

### 1. Medical NLP Summarization
- Named Entity Recognition (NER) for extracting medical entities
- Identification of symptoms, treatments, diagnoses, and prognoses
- Text summarization to convert transcripts into structured medical reports
- Keyword extraction for identifying important medical phrases

### 2. Sentiment & Intent Analysis
- Classification of patient sentiment (Anxious, Neutral, Reassured)
- Detection of patient intent (Seeking reassurance, Reporting symptoms, etc.)
- Analysis of patient dialogue to identify concerns and reassurance needs

### 3. SOAP Note Generation
- Automated generation of SOAP (Subjective, Objective, Assessment, Plan) notes
- Logical mapping of conversation content to SOAP sections
- Structured output suitable for clinical documentation

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/physician-notetaker.git
cd physician-notetaker
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Download required models:
```bash
python -m spacy download en_core_web_md
```

## Usage

### Basic Usage

```python
from physician_notetaker import PhysicianNotetakerPipeline

# Initialize the pipeline
pipeline = PhysicianNotetakerPipeline()

# Process a transcript
with open('transcript.txt', 'r') as file:
    transcript = file.read()

results = pipeline.process_transcript(transcript)
print(results)
```

### Medical NLP Summarization

```python
from medical_nlp_pipeline import MedicalNLPPipeline

# Initialize the pipeline
medical_nlp = MedicalNLPPipeline()

# Process a transcript
summary = medical_nlp.analyze_transcript(transcript)
print(summary)
```

### Sentiment & Intent Analysis

```python
from sentiment_intent_analysis import PatientSentimentAnalyzer

# Initialize the analyzer
analyzer = PatientSentimentAnalyzer()

# Analyze patient dialogue
sentiment_intent = analyzer.analyze_patient_dialogue(transcript)
print(sentiment_intent)
```

### SOAP Note Generation

```python
from soap_note_generator import SOAPNoteGenerator

# Initialize the generator
generator = SOAPNoteGenerator()

# Generate SOAP note
soap_note = generator.generate_soap_note(transcript)
print(soap_note)
```

## Project Structure

```
physician-notetaker/
├── medical_nlp_pipeline.py     # NER and medical entity extraction
├── sentiment_intent_analysis.py # Patient sentiment and intent analysis
├── soap_note_generator.py      # SOAP note generation
├── physician_notetaker.py      # Main pipeline integration
├── requirements.txt            # Required dependencies
└── README.md                   # Project documentation
```

## Requirements

- Python 3.8+
- SpaCy 3.0+
- Transformers 4.0+
- PyTorch 1.8+
- KeyBERT

## Future Improvements

- Fine-tune transformer models specifically for medical text
- Improve entity recognition with domain-specific medical datasets
- Implement more advanced sentiment analysis using healthcare-specific data
- Enhance SOAP note generation with more sophisticated mapping techniques
- Add support for voice-to-text transcription

## Handling Ambiguous or Missing Medical Data

The system employs several strategies for handling ambiguous or missing data:

1. Rule-based fallbacks when model confidence is low
2. Contextual analysis to infer missing information
3. Hierarchical classification to narrow down medical entities
4. Clear indication of uncertainty in outputs when information is ambiguous

## Notes on Fine-tuning for Medical Domain

For optimal performance, the models should be fine-tuned on medical datasets such as:

- MIMIC-III clinical notes
- MT-Clinical BERT
- MedNLI for medical natural language inference
- PubMed abstracts for medical terminology

## License

MIT License
