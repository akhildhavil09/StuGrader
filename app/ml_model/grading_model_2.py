from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import re

class AIGrader:
    def __init__(self):
        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

    def analyze_rubric_and_assignment(self, rubric_text: str, assignment_text: str) -> dict:
        prompt = f"""
        Act as an experienced professor grading this assignment.
        
        Rubric Requirements:
        {rubric_text}

        Student's Work:
        {assignment_text}

        Provide detailed feedback in this format:
        1. For each requirement:
           - What was asked for
           - What the student provided
           - What's missing
           - Specific suggestions to improve
           - Points deducted for gaps
        2. Overall score with justification
        3. Key actions needed to improve the grade
        """

        input_ids = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).input_ids
        outputs = self.model.generate(
            input_ids, 
            max_length=512,
            temperature=0.7,
            num_return_sequences=1,
            no_repeat_ngram_size=2
        )
        analysis = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Structure the personalized feedback
        feedback_items = self._parse_feedback(analysis)
        score = self._calculate_score(feedback_items)

        return {
            "score": score,
            "detailed_feedback": feedback_items,
            "overall_feedback": {
                "strengths": self._extract_strengths(analysis),
                "improvement_actions": self._extract_improvement_actions(analysis),
                "score_justification": self._extract_score_justification(analysis)
            }
        }
    def _calculate_score(self, feedback_items):
        total_points = 100
        deducted_points = 0
        
        for item in feedback_items:
            if 'points_impact' in item:
                points_match = re.search(r'-(\d+)', item['points_impact'])
                if points_match:
                    deducted_points += int(points_match.group(1))
        
        final_score = max(0, total_points - deducted_points)
        return round(final_score, 1)

    def _parse_feedback(self, analysis):
        feedback_items = []
        current_requirement = ""
        
        for line in analysis.split('\n'):
            if "Required:" in line:
                current_requirement = line.replace("Required:", "").strip()
                continue
                
            if "Provided:" in line:
                provided = line.replace("Provided:", "").strip()
                missing = next((l.replace("Missing:", "").strip() for l in analysis.split('\n') if "Missing:" in l), "")
                suggestions = next((l.replace("Improve:", "").strip() for l in analysis.split('\n') if "Improve:" in l), "")
                
                feedback_items.append({
                    "requirement": current_requirement,
                    "provided": provided,
                    "missing": missing,
                    "improvement": suggestions,
                    "points_impact": self._extract_points_impact(missing)
                })
                
        return feedback_items

    def _extract_points_impact(self, missing_text):
        if not missing_text or missing_text.lower() == "none":
            return "No points deducted"
        severity = len(missing_text.split())
        return f"-{min(severity * 2, 10)} points for this gap"

    def _extract_improvement_actions(self, analysis):
        actions = []
        lines = analysis.split('\n')
        for line in lines:
            if "To improve:" in line or "Action needed:" in line:
                action = line.split(':', 1)[1].strip()
                points_match = re.search(r'add (\d+) points', action.lower())
                if points_match:
                    points = points_match.group(1)
                    actions.append(f"Complete this to gain {points} points: {action}")
                else:
                    actions.append(action)
        return actions