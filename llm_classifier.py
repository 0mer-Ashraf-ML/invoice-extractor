"""
LLM-based document classification for PDFs.
Extracted from process_email.py
"""

import json
from typing import List, Dict, Any
from google.generativeai import GenerativeModel
from src.core.config import settings
from src.core.exceptions import AIServiceError

class LLMClassifier:
    """Classifies PDF documents using LLM."""
    
    def __init__(self):
        self.model = GenerativeModel(model_name="models/gemini-2.5-flash-preview-05-20")
    
    def classify_pdfs_with_llm(self, pdf_filenames: List[str]) -> Dict[str, Any]:
        """
        Classify PDF files to identify trade license and tables of benefits.
        
        Args:
            pdf_filenames: List of PDF filenames to classify
            
        Returns:
            Dictionary with classification results
        """
        if not pdf_filenames:
            return {
                "trade_license": None,
                "tables_of_benefits": []
            }
        
        instruction = """
You are helping identify insurance-related documents based on their filenames.

From the following filenames, identify:
- The file most likely to be a Trade License (government-issued document that allows a company to operate legally).
- ALL files that are likely to be Tables of Benefits (TOB) (documents listing insurance plan benefits, terms, perks, coverages, limits, and conditions). There can be multiple TOB files for different categories like TOB-A, TOB-B, TOB-C, or similar variations.

**Rules**:
1. Return the best match for Trade License and ALL matches for Tables of Benefits.
2. For trade_license, return a single filename or null.
3. For tables_of_benefits, return an array of filenames (can be empty if no matches).
4. Look for patterns like "TOB", "Table of Benefits", "Benefits", category indicators (A, B, C), etc.
5. Return the exact filenames, without any modifications.

### PDF Filenames:
        """
        
        prompt_text = f"{instruction}\n" + "\n".join([f"- {name}" for name in pdf_filenames]) + """

### JSON Format (Respond ONLY with this JSON):
{
  "trade_license": "filename.pdf" or null,
  "tables_of_benefits": ["filename1.pdf", "filename2.pdf", ...] or []
}
        """
        
        try:
            response = self.model.generate_content([
                {"role": "user", "parts": [{"text": prompt_text}]}
            ])
            response_text = response.text.strip().replace("```", "").replace("json", "")
            
            try:
                result = json.loads(response_text)
                return {
                    "trade_license": result.get("trade_license"),
                    "tables_of_benefits": result.get("tables_of_benefits", [])
                }
            except json.JSONDecodeError as e:
                print(f"❌ Error parsing LLM response: {e}")
                return {
                    "trade_license": None,
                    "tables_of_benefits": []
                }
        except Exception as e:
            print(f"❌ Error in LLM classification: {e}")
            return {
                "trade_license": None,
                "tables_of_benefits": []
            }

# Global classifier instance
llm_classifier = LLMClassifier()

def classify_pdfs_with_llm(pdf_filenames: List[str]) -> Dict[str, Any]:
    """Convenience function to classify PDFs."""
    return llm_classifier.classify_pdfs_with_llm(pdf_filenames)