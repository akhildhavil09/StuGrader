# app/ml_model/model.py
from typing import List, Dict
from transformers import AutoTokenizer, AutoModel
import torch
import re

class AssignmentAnalyzer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModel.from_pretrained('bert-base-uncased')

    def analyze(self, rubric_text: str, assignment_text: str, materials_text: str) -> Dict:
        """Analyze assignment based on rubric requirements."""
        # Extract requirements from rubric
        requirements = self._extract_requirements(rubric_text)
        
        # Analyze each requirement
        feedback = []
        total_score = 0
        available_points = 0
        
        for req in requirements:
            requirement_met = self._check_requirement(req['text'], assignment_text)
            points_earned = req['points'] if requirement_met else 0
            
            feedback.append({
                'requirement': req['text'],
                'met': requirement_met,
                'points_earned': points_earned,
                'points_possible': req['points'],
                'feedback': self._generate_feedback(req, requirement_met, assignment_text)
            })
            
            total_score += points_earned
            available_points += req['points']

        # Calculate overall score
        percentage_score = (total_score / available_points * 100) if available_points > 0 else 0

        return {
            'estimatedScore': round(percentage_score, 1),
            'feedback': feedback
        }

    def _extract_requirements(self, rubric_text: str) -> List[Dict]:
        """Extract requirements and their point values from rubric."""
        requirements = []
        # Split into sections based on common rubric formatting
        sections = rubric_text.split('\n\n')
        
        for section in sections:
            # Extract points (if specified)
            points_match = re.search(r'(\d+)\s*points?', section, re.IGNORECASE)
            points = int(points_match.group(1)) if points_match else 5  # Default points
            
            # Look for requirement statements
            req_patterns = [
                r'must\s+(.*?)(?=\.|$)',
                r'should\s+(.*?)(?=\.|$)',
                r'needs to\s+(.*?)(?=\.|$)',
                r'demonstrate\s+(.*?)(?=\.|$)',
                r'include\s+(.*?)(?=\.|$)'
            ]
            
            for pattern in req_patterns:
                matches = re.finditer(pattern, section, re.IGNORECASE)
                for match in matches:
                    req_text = match.group(1).strip()
                    if req_text:
                        requirements.append({
                            'text': req_text,
                            'points': points
                        })
        
        return requirements

    def _check_requirement(self, requirement: str, assignment_text: str) -> bool:
        """Check if a specific requirement is met in the assignment."""
        # Generate embeddings for requirement and assignment
        req_embedding = self._get_embeddings(requirement)
        assign_embedding = self._get_embeddings(assignment_text)
        
        # Calculate similarity
        similarity = torch.cosine_similarity(req_embedding, assign_embedding)
        return similarity.item() > 0.7  # Threshold for requirement being met

    def _get_embeddings(self, text: str) -> torch.Tensor:
        """Generate BERT embeddings for text."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)

    def _generate_feedback(self, requirement: Dict, met: bool, assignment_text: str) -> str:
        """Generate specific feedback for a requirement."""
        if not met:
            # Check for partial matches
            req_keywords = set(requirement['text'].lower().split())
            assign_words = set(assignment_text.lower().split())
            
            # If some keywords are found but requirement not fully met
            if req_keywords.intersection(assign_words):
                return f"Partially addressed: {requirement['text']}. Expand your discussion to fully meet this requirement."
            else:
                return f"Missing requirement: {requirement['text']}. Add this to your assignment."
        
        return f"Successfully addressed: {requirement['text']}"