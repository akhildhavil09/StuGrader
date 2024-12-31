# app/ml_model/text_processor.py
import PyPDF2
from docx import Document
import re
from typing import List, Dict

class TextProcessor:
    def __init__(self):
        pass

    def extract_requirements(self, rubric_text: str) -> List[Dict]:
        """Extract specific requirements from rubric text."""
        requirements = []
        
        # Split text into sections
        sections = rubric_text.split('\n\n')
        
        for section in sections:
            # Look for point values and requirements
            points_match = re.search(r'(\d+)\s*points?', section, re.IGNORECASE)
            points = int(points_match.group(1)) if points_match else 0
            
            # Look for specific requirements using common patterns
            requirement_patterns = [
                (r'must\s+(.*?)(?=\.|$)', 'required'),
                (r'should\s+(.*?)(?=\.|$)', 'required'),
                (r'needs?\s+to\s+(.*?)(?=\.|$)', 'required'),
                (r'demonstrate\s+(.*?)(?=\.|$)', 'skill'),
                (r'include\s+(.*?)(?=\.|$)', 'content'),
                (r'analyze\s+(.*?)(?=\.|$)', 'analysis'),
                (r'explain\s+(.*?)(?=\.|$)', 'explanation')
            ]
            
            for pattern, req_type in requirement_patterns:
                matches = re.finditer(pattern, section, re.IGNORECASE)
                for match in matches:
                    requirement_text = match.group(1).strip()
                    if requirement_text:
                        requirements.append({
                            'text': requirement_text,
                            'type': req_type,
                            'points': points,
                            'required': req_type == 'required'
                        })
        
        return requirements

    def extract_text(self, file_path: str) -> str:
        """Extract text from various file formats."""
        try:
            if file_path.endswith('.pdf'):
                return self._extract_from_pdf(file_path)
            elif file_path.endswith(('.docx', '.doc')):
                return self._extract_from_docx(file_path)
            else:
                with open(file_path, 'r', encoding='utf-8') as file:
                    return file.read()
        except Exception as e:
            print(f"Error extracting text: {str(e)}")
            return ""

    def _extract_from_pdf(self, file_path: str) -> str:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            return ' '.join(page.extract_text() for page in reader.pages)

    def _extract_from_docx(self, file_path: str) -> str:
        doc = Document(file_path)
        return ' '.join(paragraph.text for paragraph in doc.paragraphs)
