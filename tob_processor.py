"""
Table of Benefits (TOB) processing using AI.
Handles both single and multi-category TOB extraction.
"""

import time
import json
from typing import List, Dict, Any
import google.generativeai as genai
from src.core.config import settings
from src.core.constants import STANDARDIZE_PRICES_FIELDS,DATA_FOR_TOB,BENEFIT_DETAILS_DATA
from src.core.exceptions import AIServiceError, DocumentProcessingError

class TOBProcessor:
    """Processes Table of Benefits documents using AI."""
    
    def __init__(self):
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(model_name="models/gemini-2.5-pro-preview-03-25")
    
    def extract_markdown_from_pdf(self, pdf_path: str) -> str:
        """
        Extract markdown content from TOB PDF.
        
        Args:
            pdf_path: Path to the TOB PDF file
            
        Returns:
            Markdown representation of the TOB content
        """
        try:
            uploaded_file = genai.upload_file(pdf_path)
            
            while uploaded_file.state.name == "PROCESSING":
                time.sleep(1)
                uploaded_file = genai.get_file(uploaded_file.name)
                print(f"Waiting for file to become active. Current state: {uploaded_file.state.name}")
            
            if uploaded_file.state.name != "ACTIVE":
                raise DocumentProcessingError(
                    f"File processing failed with state: {uploaded_file.state.name}"
                )
            
            print(f"File is now active: {uploaded_file.name}")
            
            prompt = [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": """This is the benefits table. Carefully read the content of the table and return me a markdown for it. Do not miss anything. Include all the details. Do not miss the complete company name which could be written in the text or in the logo. Pay special attention to any categories, classes, or plans mentioned (like Category A, Category B, Class A, Class B, etc.). Do not include any introductory or explanatory text. The markdown should be the only output."""
                        },
                        {
                            "file_data": {
                                "file_uri": uploaded_file.uri, 
                                "mime_type": "application/pdf"
                            }
                        }
                    ]
                }
            ]
            
            response = self.model.generate_content(prompt)
            markdown_content = response.text.strip().replace("```", "").replace("markdown", "")
            
            # Clean up the file
            genai.delete_file(uploaded_file.name)
            
            return markdown_content
            
        except Exception as e:
            if isinstance(e, DocumentProcessingError):
                raise
            raise DocumentProcessingError(f"Failed to extract markdown from PDF: {e}")
    
    def format_price(self, value: Any, prefix: str = "", comma: bool = False) -> str:
        """Format price values according to specified rules."""
        try:
            num = int(str(value).replace(",", "").replace("AED", "").strip())
            formatted = f"{num:,}" if comma else str(num)
            return f"{prefix}{formatted}".strip()
        except:
            return value  # Leave as-is for "Upto AML", "Not Covered", etc.
    
    def extract_structured_data_from_tob(self, tob_markdown: str) -> List[Dict[str, Any]]:
        """
        Extract structured data from TOB markdown.
        Returns a list of dictionaries for multi-category support.
        """
        
        try:
            prompt = [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": f"""
                    {DATA_FOR_TOB}
                    
                    Now process the TOB below and return only the populated JSON LIST with detailed reasoning for each field. Do not return any text or explanations outside the JSON.
                    
                    **CRITICAL REQUIREMENTS**: 
                    1. First, carefully analyze if the TOB contains multiple categories/classes/plans
                    2. If multiple categories exist, return a list with multiple dictionaries (one for each category)
                    3. If only one category exists, still return a list with one dictionary
                    4. Each field must follow the enhanced structure: {{"value": "", "changed": True/False, "explanation": ""}}
                    5. Make sure to populate the "category_name" field for each dictionary with the appropriate category identifier
                    6. For monetary fields, return clean integers in the "value" field (no commas, no currency symbols)
                    7. Provide detailed explanations for every field, especially when values are adjusted
                    8. Be specific about what original values were found and how they were processed
                    
                    **EXPLANATION GUIDELINES:**
                    - If adjusting a value: "You requested for '[original_value]' for [field_description] but we have selected '[selected_value]' which aligns more closely with our business rules"
                    - If exact match: "Found exact match '[selected_value]' for [field_description]"
                    - If using default: "No specific value found for [field_description], using default '[selected_value]'"
                    - If semantic matching: "Identified '[original_text]' as referring to [field_description], selected '[selected_value]'"
                    
                    ### TOB (Markdown):
                    {tob_markdown}
                    
                    ### JSON Template (for each category):
                    {BENEFIT_DETAILS_DATA}
                    
                    **RETURN FORMAT**: [{{...}}, {{...}}, ...] - Always a list of dictionaries with enhanced field structure
                    """
                        }
                    ]
                }
            ]
            
            response = self.model.generate_content(prompt)
            response_text = response.text.strip().replace("```", "").replace("json", "")
            
            try:
                raw_json_list = json.loads(response_text)
                
                # Ensure we have a list
                if not isinstance(raw_json_list, list):
                    raw_json_list = [raw_json_list]
                
                # Apply price formatting to monetary fields in each category
                for category_data in raw_json_list:
                    for field, rules in STANDARDIZE_PRICES_FIELDS.items():
                        if field in category_data and isinstance(category_data[field], dict):
                            value = category_data[field].get("value", "")
                            if isinstance(value, (int, float)) or (isinstance(value, str) and value.replace(",", "").isdigit()):
                                formatted_value = self.format_price(str(value), prefix=rules["prefix"], comma=rules["comma"])
                                category_data[field]["value"] = formatted_value
                
                return raw_json_list
                
            except json.JSONDecodeError as e:
                raise AIServiceError(f"Error parsing JSON response: {e}")
            
        except Exception as e:
            if isinstance(e, AIServiceError):
                raise
            raise AIServiceError(f"Failed to extract structured data from TOB: {e}")
    
    def process_tob(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Process a TOB PDF file and return structured data.
        
        Args:
            pdf_path: Path to the TOB PDF file
            
        Returns:
            List of dictionaries containing benefit details for each category
        """
        try:
            # Extract markdown from PDF
            markdown_text = self.extract_markdown_from_pdf(pdf_path)
            
            # Extract structured data from markdown
            tob_data = self.extract_structured_data_from_tob(markdown_text)
            
            return tob_data
            
        except Exception as e:
            if isinstance(e, (DocumentProcessingError, AIServiceError)):
                raise
            raise DocumentProcessingError(f"Failed to process TOB: {e}")

# Global TOB processor instance
tob_processor = TOBProcessor()

def extract_markdown_from_pdf(pdf_path: str) -> str:
    """Convenience function to extract markdown from PDF."""
    return tob_processor.extract_markdown_from_pdf(pdf_path)

def extract_structured_data_from_tob(tob_markdown: str) -> List[Dict[str, Any]]:
    """Convenience function to extract structured data from TOB markdown."""
    return tob_processor.extract_structured_data_from_tob(tob_markdown)

def process_tob(pdf_path: str) -> List[Dict[str, Any]]:
    """Convenience function to process TOB PDF."""
    return tob_processor.process_tob(pdf_path)