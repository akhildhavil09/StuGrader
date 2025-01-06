# app/ml_model/grading_model.py
from transformers import AutoTokenizer, AutoModel
import torch
from typing import List, Dict
import re
import numpy as np

class AIGrader:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModel.from_pretrained('bert-base-uncased')

    def analyze_rubric_and_assignment(self, rubric_text: str, assignment_text: str) -> Dict:
        """Main method to analyze assignment against rubric."""
        # Extract criteria from rubric
        criteria = self.extract_rubric_criteria(rubric_text)
        
        # Grade assignment
        return self.grade_assignment(criteria, assignment_text)

    def extract_rubric_criteria(self, rubric_text: str) -> List[Dict]:
        """
        Extracts grading criteria and their weights from rubric text.
        Uses NLP to identify requirements, expectations, and point values.
        """
        # Split rubric into sections
        sections = [s.strip() for s in rubric_text.split('\n') if s.strip()]
        criteria = []
        
        for section in sections:
            # Extract point values if present
            points = self._extract_points(section)
            
            # Extract requirements using BERT embeddings
            requirements = self._extract_requirements(section)
            
            for req in requirements:
                criteria.append({
                    'requirement': req,
                    'points': points,
                    'section': self._classify_requirement_type(req),
                    'keywords': self._extract_key_concepts(req)
                })
        
        return criteria

    def grade_assignment(self, criteria: List[Dict], assignment_text: str) -> Dict:
        """
        Grades assignment based on extracted criteria using semantic similarity.
        """
        results = []
        total_points = sum(c['points'] for c in criteria)
        earned_points = 0
        
        for criterion in criteria:
            # Get embeddings for requirement and assignment
            req_embedding = self._get_text_embedding(criterion['requirement'])
            assign_embedding = self._get_text_embedding(assignment_text)
            
            # Calculate similarity score
            similarity = self._calculate_similarity(req_embedding, assign_embedding)
            
            # Check for specific requirements
            requirement_met = self._check_requirement_fulfillment(
                criterion,
                assignment_text,
                similarity
            )
            
            # Calculate points
            points_earned = criterion['points'] * requirement_met['score']
            earned_points += points_earned
            
            results.append({
                'requirement': criterion['requirement'],
                'points_possible': criterion['points'],
                'points_earned': round(points_earned),
                'fulfillment_level': requirement_met['level'],
                'feedback': requirement_met['feedback'],
                'improvement_suggestions': requirement_met['suggestions']
            })
        
        return {
            'score': round((earned_points / total_points) * 100, 1) if total_points > 0 else 0,
            'points_earned': round(earned_points, 2),  # Round points earned to 2 decimal places
            'total_points': round(total_points, 2),   # Round total points to 2 decimal places
            'detailed_feedback': results,
            'overall_feedback': self._generate_overall_feedback(results)
        }


    def _get_text_embedding(self, text: str) -> torch.Tensor:
        """Generates BERT embeddings for text comparison."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)

    def _calculate_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """Calculates semantic similarity between embeddings."""
        similarity = torch.nn.functional.cosine_similarity(emb1, emb2)
        return similarity.item()

    def _extract_points(self, section: str) -> int:
        """Extract point values from rubric text."""
        point_patterns = [
            r'(\d+)\s*points?',
            r'(\d+)\s*marks?',
            r'worth\s*(\d+)',
            r'value:\s*(\d+)',
            r'points:\s*(\d+)'
        ]
        
        for pattern in point_patterns:
            match = re.search(pattern, section, re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        return 10  # Default points if no explicit value found

    def _classify_requirement_type(self, requirement: str) -> str:
        """Classify the type of requirement."""
        requirement_lower = requirement.lower()
        
        patterns = {
            'analysis': ['analyze', 'examine', 'evaluate', 'assess'],
            'implementation': ['implement', 'create', 'develop', 'build'],
            'understanding': ['understand', 'explain', 'describe', 'discuss'],
            'demonstration': ['demonstrate', 'show', 'display', 'present']
        }
        
        for req_type, keywords in patterns.items():
            if any(keyword in requirement_lower for keyword in keywords):
                return req_type
                
        return 'general'

    def _extract_key_concepts(self, requirement: str) -> List[str]:
        """Extract key concepts and terms from requirement text."""
        words = requirement.lower().split()
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of'}
        words = [word for word in words if word not in stop_words]
        return [word for word in words if len(word) > 3]

    def _extract_requirements(self, section: str) -> List[str]:
        """Extract individual requirements from a section."""
        requirements = []
        
        indicators = [
            r'must\s+(.*?)[.]',
            r'should\s+(.*?)[.]',
            r'needs? to\s+(.*?)[.]',
            r'required to\s+(.*?)[.]',
            r'demonstrate\s+(.*?)[.]',
            r'explain\s+(.*?)[.]',
            r'analyze\s+(.*?)[.]',
            r'discuss\s+(.*?)[.]'
        ]
        
        for pattern in indicators:
            matches = re.finditer(pattern, section, re.IGNORECASE)
            for match in matches:
                requirement = match.group(1).strip()
                if requirement:
                    requirements.append(requirement)
        
        if not requirements and len(section.strip()) > 0:
            requirements.append(section.strip())
            
        return requirements

    def _check_requirement_fulfillment(self, criterion: Dict, text: str, similarity: float) -> Dict:
        """
        Checks how well a specific requirement is fulfilled in the assignment.
        """
        if similarity > 0.85:
            score = 1.0
            level = "Met"
        elif similarity > 0.65:
            score = 0.5
            level = "Partially Met"
        else:
            score = 0.0
            level = "Not Met"

        keyword_presence = sum(1 for k in criterion['keywords'] 
                             if k.lower() in text.lower()) / len(criterion['keywords'])
        
        final_score = (similarity + keyword_presence) / 2
        
        feedback, suggestions = self._generate_criterion_feedback(
            criterion, level, similarity, keyword_presence
        )

        return {
            'score': final_score,
            'level': level,
            'feedback': feedback,
            'suggestions': suggestions
        }

    def _generate_criterion_feedback(self, criterion: Dict, level: str, 
                                   similarity: float, keyword_presence: float) -> tuple:
        """Generates detailed feedback and suggestions for improvement."""
        if level == "Met":
            feedback = f"Excellent demonstration of {criterion['section']} requirements."
            suggestions = ["Consider adding more examples to strengthen your argument."]
        elif level == "Partially Met":
            feedback = f"Basic understanding shown, but needs more depth."
            suggestions = [
                "Expand your discussion of key concepts.",
                "Add more specific examples.",
                "Link your ideas more clearly to the requirements."
            ]
        else:
            feedback = f"Requirement not adequately addressed."
            suggestions = [
                "Review the requirement carefully.",
                "Include specific discussion of required topics.",
                "Add supporting evidence and examples."
            ]

        return feedback, suggestions

    def _generate_overall_feedback(self, results: List[Dict]) -> Dict:
        """Generates comprehensive feedback summary."""
        strengths = [r['requirement'] for r in results if r['fulfillment_level'] == "Met"]
        improvements = [r['requirement'] for r in results if r['fulfillment_level'] != "Met"]
        
        return {
            'strengths': strengths,
            'areas_for_improvement': improvements,
            'summary': self._generate_summary_text(results)
        }

    def _generate_summary_text(self, results: List[Dict]) -> str:
        """Generates a text summary of the overall assessment."""
        met_count = sum(1 for r in results if r['fulfillment_level'] == "Met")
        partial_count = sum(1 for r in results if r['fulfillment_level'] == "Partially Met")
        not_met_count = sum(1 for r in results if r['fulfillment_level'] == "Not Met")
        
        total_requirements = len(results)
        
        return f"Out of {total_requirements} requirements, {met_count} were fully met, " \
               f"{partial_count} were partially met, and {not_met_count} need improvement."