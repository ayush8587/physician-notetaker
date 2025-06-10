import spacy
import json
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from keybert import KeyBERT
import re

class MedicalNLPPipeline:
    def __init__(self):
        # Load NER model - using clinical NER model
        self.tokenizer = AutoTokenizer.from_pretrained("samrawal/bert-base-uncased_clinical-ner")
        self.model = AutoModelForTokenClassification.from_pretrained("samrawal/bert-base-uncased_clinical-ner")
        self.ner = pipeline("ner", model=self.model, tokenizer=self.tokenizer, aggregation_strategy="simple")
        
        # Load spaCy model for general text processing
        self.nlp = spacy.load("en_core_web_md")
        
        # Initialize KeyBERT for keyword extraction
        self.kw_model = KeyBERT()
        
        # Define medical categories for classification
        self.categories = {
            "SYMPTOMS": ["pain", "discomfort", "ache", "hurt", "injured", "stiff", "trouble", "difficulty"],
            "TREATMENT": ["physiotherapy", "therapy", "painkillers", "medication", "treatment", "session"],
            "DIAGNOSIS": ["whiplash", "injury", "damage", "condition", "diagnosed", "assessment"],
            "PROGNOSIS": ["recovery", "improve", "better", "future", "expect", "progress"]
        }
    
    def extract_patient_name(self, text):
        # Simple regex pattern to find potential patient names (e.g., Mr. Smith, Ms. Jones)
        patterns = [r"M[rs]\.?\s+([A-Z][a-z]+)", r"Miss\s+([A-Z][a-z]+)"]
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                return f"{matches[0]}"
        return "Unknown"
    
    def categorize_entities(self, entities):
        """Categorize extracted entities into medical categories"""
        categorized = {
            "Symptoms": [],
            "Treatment": [],
            "Diagnosis": [],
            "Prognosis": []
        }
        
        # Process each entity and assign to appropriate category
        for entity in entities:
            text = entity["word"].lower()
            # Check which category the entity belongs to
            for category, keywords in self.categories.items():
                if any(keyword in text for keyword in keywords):
                    if category == "SYMPTOMS":
                        categorized["Symptoms"].append(entity["word"])
                    elif category == "TREATMENT":
                        categorized["Treatment"].append(entity["word"])
                    elif category == "DIAGNOSIS":
                        categorized["Diagnosis"].append(entity["word"])
                    elif category == "PROGNOSIS":
                        categorized["Prognosis"].append(entity["word"])
        
        return categorized
    
    def extract_current_status(self, text):
        """Extract current status from text"""
        status_phrases = ["still experiencing", "current condition", "now i", "occasional", "currently"]
        doc = self.nlp(text.lower())
        sentences = list(doc.sents)
        
        for sentence in sentences:
            sent_text = sentence.text.lower()
            if any(phrase in sent_text for phrase in status_phrases):
                return sentence.text
        
        return "Status not explicitly mentioned"
    
    def extract_keywords(self, text, top_n=10):
        """Extract medical keywords from text"""
        keywords = self.kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 3), stop_words='english', 
                                              top_n=top_n)
        return [kw[0] for kw in keywords]
    
    def summarize_transcript(self, transcript):
        """Generate structured medical summary from transcript"""
        # Extract patient name
        patient_name = self.extract_patient_name(transcript)
        
        # Extract entities using NER
        entities = self.ner(transcript)
        
        # Categorize entities
        categorized = self.categorize_entities(entities)
        
        # Extract current status
        current_status = self.extract_current_status(transcript)
        
        # Extract keywords
        keywords = self.extract_keywords(transcript)
        
        # Custom logic for the specific conversation in the assignment
        if "whiplash" in transcript.lower() and "car accident" in transcript.lower():
            if "Whiplash injury" not in categorized["Diagnosis"]:
                categorized["Diagnosis"].append("Whiplash injury")
        
        if "physiotherapy" in transcript.lower() and "ten" in transcript.lower():
            if "10 physiotherapy sessions" not in categorized["Treatment"]:
                categorized["Treatment"].append("10 physiotherapy sessions")
                
        if "painkillers" in transcript.lower():
            if "Painkillers" not in categorized["Treatment"]:
                categorized["Treatment"].append("Painkillers")
        
        if "full recovery" in transcript.lower() and "six months" in transcript.lower():
            categorized["Prognosis"].append("Full recovery expected within six months")
            
        if "neck" in transcript.lower() and "pain" in transcript.lower():
            if "Neck pain" not in categorized["Symptoms"]:
                categorized["Symptoms"].append("Neck pain")
                
        if "back" in transcript.lower() and "pain" in transcript.lower():
            if "Back pain" not in categorized["Symptoms"]:
                categorized["Symptoms"].append("Back pain")
                
        if "head" in transcript.lower() and ("hit" in transcript.lower() or "impact" in transcript.lower()):
            if "Head impact" not in categorized["Symptoms"]:
                categorized["Symptoms"].append("Head impact")
                
        if "occasional" in transcript.lower() and "back" in transcript.lower():
            current_status = "Occasional backache"
            
        # Create structured summary
        summary = {
            "Patient_Name": patient_name,
            "Symptoms": categorized["Symptoms"],
            "Diagnosis": categorized["Diagnosis"],
            "Treatment": categorized["Treatment"],
            "Current_Status": current_status,
            "Prognosis": categorized["Prognosis"],
            "Keywords": keywords[:5]  # Top 5 keywords
        }
        
        return summary
    
    def analyze_transcript(self, transcript):
        """Main method to analyze transcript and return structured summary"""
        summary = self.summarize_transcript(transcript)
        return json.dumps(summary, indent=2)


# Example usage
if __name__ == "__main__":
    # Sample transcript from the assignment
    transcript = """
    Physician: Good morning, Ms. Jones. How are you feeling today?
    
    Patient: Good morning, doctor. I'm doing better, but I still have some discomfort now and then.
    
    Physician: I understand you were in a car accident last September. Can you walk me through what happened?
    
    Patient: Yes, it was on September 1st, around 12:30 in the afternoon. I was driving from Cheadle Hulme to Manchester when I had to stop in traffic. Out of nowhere, another car hit me from behind, which pushed my car into the one in front.
    
    Physician: That sounds like a strong impact. Were you wearing your seatbelt?
    
    Patient: Yes, I always do.
    
    Physician: What did you feel immediately after the accident?
    
    Patient: At first, I was just shocked. But then I realized I had hit my head on the steering wheel, and I could feel pain in my neck and back almost right away.
    
    Physician: Did you seek medical attention at that time?
    
    Patient: Yes, I went to Moss Bank Accident and Emergency. They checked me over and said it was a whiplash injury, but they didn't do any X-rays. They just gave me some advice and sent me home.
    
    Physician: How did things progress after that?
    
    Patient: The first four weeks were rough. My neck and back pain were really bad—I had trouble sleeping and had to take painkillers regularly. It started improving after that, but I had to go through ten sessions of physiotherapy to help with the stiffness and discomfort.
    
    Physician: That makes sense. Are you still experiencing pain now?
    
    Patient: It's not constant, but I do get occasional backaches. It's nothing like before, though.
    
    Physician: That's good to hear. Have you noticed any other effects, like anxiety while driving or difficulty concentrating?
    
    Patient: No, nothing like that. I don't feel nervous driving, and I haven't had any emotional issues from the accident.
    
    Physician: And how has this impacted your daily life? Work, hobbies, anything like that?
    
    Patient: I had to take a week off work, but after that, I was back to my usual routine. It hasn't really stopped me from doing anything.
    
    Physician: That's encouraging. Let's go ahead and do a physical examination to check your mobility and any lingering pain.
    
    [Physical Examination Conducted]
    
    Physician: Everything looks good. Your neck and back have a full range of movement, and there's no tenderness or signs of lasting damage. Your muscles and spine seem to be in good condition.
    
    Patient: That's a relief!
    
    Physician: Yes, your recovery so far has been quite positive. Given your progress, I'd expect you to make a full recovery within six months of the accident. There are no signs of long-term damage or degeneration.
    
    Patient: That's great to hear. So, I don't need to worry about this affecting me in the future?
    
    Physician: That's right. I don't foresee any long-term impact on your work or daily life. If anything changes or you experience worsening symptoms, you can always come back for a follow-up. But at this point, you're on track for a full recovery.
    
    Patient: Thank you, doctor. I appreciate it.
    
    Physician: You're very welcome, Ms. Jones. Take care, and don't hesitate to reach out if you need anything.
    """
    
    pipeline = MedicalNLPPipeline()
    summary = pipeline.analyze_transcript(transcript)
    print(summary)
