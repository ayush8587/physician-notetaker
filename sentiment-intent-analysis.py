import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F
import json
import re

class PatientSentimentAnalyzer:
    def __init__(self):
        # Load pre-trained model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # In a real implementation, we would fine-tune BERT for medical sentiment
        # This is a placeholder for the fine-tuned model
        self.sentiment_model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased', 
            num_labels=3  # Anxious, Neutral, Reassured
        )
        
        # Intent detection model
        self.intent_model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=4  # Different intent categories
        )
        
        # Define sentiment classes
        self.sentiment_classes = ['Anxious', 'Neutral', 'Reassured']
        
        # Define intent classes
        self.intent_classes = [
            'Seeking reassurance', 
            'Reporting symptoms', 
            'Expressing concern',
            'Sharing information'
        ]
        
        # Keywords for rule-based sentiment analysis (as a fallback)
        self.sentiment_keywords = {
            'Anxious': ['worried', 'concern', 'anxious', 'nervous', 'scared', 'afraid', 'fear'],
            'Reassured': ['relief', 'better', 'good', 'great', 'happy', 'reassured', 'confident'],
            'Neutral': ['okay', 'fine', 'alright', 'understand']
        }
        
        # Keywords for rule-based intent analysis (as a fallback)
        self.intent_keywords = {
            'Seeking reassurance': ['will i', 'should i', 'worry', 'concern', '?', 'right?', 'okay?'],
            'Reporting symptoms': ['pain', 'hurt', 'feel', 'symptom', 'discomfort', 'ache'],
            'Expressing concern': ['worried', 'afraid', 'scared', 'concerned', 'anxious'],
            'Sharing information': ['happened', 'i was', 'i had', 'i did', 'i went']
        }
    
    def extract_patient_dialogue(self, transcript):
        """Extract only the patient's dialogue from the transcript"""
        lines = transcript.split('\n')
        patient_dialogue = []
        
        for line in lines:
            if line.strip().startswith('Patient:'):
                dialogue = line.replace('Patient:', '').strip()
                patient_dialogue.append(dialogue)
        
        return ' '.join(patient_dialogue)
    
    def rule_based_sentiment(self, text):
        """Rule-based sentiment analysis as fallback"""
        text = text.lower()
        scores = {sentiment: 0 for sentiment in self.sentiment_classes}
        
        for sentiment, keywords in self.sentiment_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    scores[sentiment] += 1
        
        # Find sentiment with highest score
        max_score = max(scores.values())
        if max_score == 0:
            return 'Neutral'  # Default to neutral if no keywords found
        
        # Return sentiment with highest score
        for sentiment, score in scores.items():
            if score == max_score:
                return sentiment
    
    def rule_based_intent(self, text):
        """Rule-based intent analysis as fallback"""
        text = text.lower()
        scores = {intent: 0 for intent in self.intent_classes}
        
        for intent, keywords in self.intent_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    scores[intent] += 1
        
        # Find intent with highest score
        max_score = max(scores.values())
        if max_score == 0:
            return 'Sharing information'  # Default intent
        
        # Return intent with highest score
        for intent, score in scores.items():
            if score == max_score:
                return intent
    
    def analyze_sentiment(self, text):
        """Analyze sentiment using transformer model or rule-based fallback
        
        In a real implementation, this would use the fine-tuned model.
        For this example, we'll use the rule-based approach.
        """
        try:
            # Placeholder for transformer-based sentiment analysis
            # In a real implementation, we would use the fine-tuned model
            # inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            # outputs = self.sentiment_model(**inputs)
            # predictions = F.softmax(outputs.logits, dim=-1)
            # sentiment_id = torch.argmax(predictions, dim=-1).item()
            # sentiment = self.sentiment_classes[sentiment_id]
            
            # For now, use rule-based approach
            sentiment = self.rule_based_sentiment(text)
            return sentiment
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            return self.rule_based_sentiment(text)
    
    def analyze_intent(self, text):
        """Analyze intent using transformer model or rule-based fallback
        
        In a real implementation, this would use the fine-tuned model.
        For this example, we'll use the rule-based approach.
        """
        try:
            # Placeholder for transformer-based intent analysis
            # In a real implementation, we would use the fine-tuned model
            # inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            # outputs = self.intent_model(**inputs)
            # predictions = F.softmax(outputs.logits, dim=-1)
            # intent_id = torch.argmax(predictions, dim=-1).item()
            # intent = self.intent_classes[intent_id]
            
            # For now, use rule-based approach
            intent = self.rule_based_intent(text)
            return intent
        except Exception as e:
            print(f"Error in intent analysis: {e}")
            return self.rule_based_intent(text)
    
    def analyze_patient_dialogue(self, transcript):
        """Analyze patient dialogue from transcript"""
        patient_dialogue = self.extract_patient_dialogue(transcript)
        
        # Analyze sentiment
        sentiment = self.analyze_sentiment(patient_dialogue)
        
        # Analyze intent
        intent = self.analyze_intent(patient_dialogue)
        
        # Return results
        results = {
            "Sentiment": sentiment,
            "Intent": intent
        }
        
        return json.dumps(results, indent=2)


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
    
    analyzer = PatientSentimentAnalyzer()
    results = analyzer.analyze_patient_dialogue(transcript)
    print(results)
    
    # Test with sample input from assignment
    sample_input = "I'm a bit worried about my back pain, but I hope it gets better soon."
    sample_results = analyzer.analyze_sentiment(sample_input), analyzer.analyze_intent(sample_input)
    print(json.dumps({"Sentiment": sample_results[0], "Intent": sample_results[1]}, indent=2))
