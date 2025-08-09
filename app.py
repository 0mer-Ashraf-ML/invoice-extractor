import streamlit as st
import pandas as pd
import json
import argparse
import sys
import os
import io
from datetime import datetime
from pathlib import Path
import base64
from typing import Dict, List, Any, Optional
import requests

# Multi-model support imports
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from mistralai import Mistral
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

class MultiModelProcessor:
    def __init__(self, model_provider: str, specific_model: str = None):
        self.model_provider = model_provider.lower()
        self.specific_model = specific_model
        
        # Get API key from environment variables
        self.api_key = self._get_api_key()
        
        if not self.api_key:
            raise ValueError(f"API key for {model_provider} not found in environment variables")
        
        # Initialize the appropriate client
        if self.model_provider == "gemini" and GEMINI_AVAILABLE:
            self.client = genai.Client(api_key=self.api_key)
            self.model_name = specific_model or "gemini-2.5-pro"
        elif self.model_provider == "openai" and OPENAI_AVAILABLE:
            self.client = OpenAI(api_key=self.api_key)
            self.model_name = specific_model or "gpt-4o-mini"
        elif self.model_provider == "mistral" and MISTRAL_AVAILABLE:
            self.client = Mistral(api_key=self.api_key)
            self.model_name = specific_model or "mistral-large-latest"
        else:
            raise ValueError(f"Model provider '{model_provider}' not available or not installed")
    
    def _get_api_key(self) -> str:
        """Get API key from environment variables"""
        if self.model_provider == "gemini":
            return os.getenv("GEMINI_API_KEY")
        elif self.model_provider == "openai":
            return os.getenv("OPENAI_API_KEY")
        elif self.model_provider == "mistral":
            return os.getenv("MISTRAL_API_KEY")
        return None
    
    def encode_image_base64(self, filepath: Path) -> str:
        """Convert image to base64 for OpenAI/Mistral"""
        import base64
        with open(filepath, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def get_mime_type(self, filepath: Path) -> str:
        """Get MIME type for the file"""
        if filepath.suffix.lower() == '.pdf':
            return 'application/pdf'
        elif filepath.suffix.lower() in ['.jpg', '.jpeg']:
            return 'image/jpeg'
        elif filepath.suffix.lower() == '.png':
            return 'image/png'
        else:
            return "unknown"
    
    def generate_content(self, prompt: str, filepath: Path = None) -> str:
        """Unified content generation across all models"""
        
        if self.model_provider == "gemini":
            return self._generate_gemini(prompt, filepath)
        elif self.model_provider == "openai":
            return self._generate_openai(prompt, filepath)
        elif self.model_provider == "mistral":
            return self._generate_mistral(prompt, filepath)
        else:
            raise ValueError(f"Unsupported model provider: {self.model_provider}")
    
    def _generate_gemini(self, prompt: str, filepath: Path = None) -> str:
        """Generate content using Gemini"""
        try:
            if filepath:
                mime_type = self.get_mime_type(filepath)
                if mime_type == "unknown":
                    raise ValueError("Unsupported file type for Gemini")
                
                contents = [
                    types.Part.from_bytes(
                        data=filepath.read_bytes(),
                        mime_type=mime_type,
                    ),
                    prompt
                ]
            else:
                contents = [prompt]
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents
            )
            
            # Check if response has text attribute and it's not None
            if hasattr(response, 'text') and response.text:
                return response.text
            else:
                print(f"Gemini returned empty response. Response object: {response}")
                return ""
                
        except Exception as e:
            print(f"Error in Gemini generation: {e}")
            return ""
    
    def _generate_openai(self, prompt: str, filepath: Path = None) -> str:
        """Generate content using OpenAI"""
        try:
            messages = []
            
            if filepath:
                if filepath.suffix.lower() == '.pdf':
                    # For PDF, we need to extract text first or use a different approach
                    raise ValueError("PDF support for OpenAI requires additional processing")
                
                # For images
                base64_image = self.encode_image_base64(filepath)
                mime_type = self.get_mime_type(filepath)
                
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            }
                        }
                    ]
                })
            else:
                messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages
            )
            
            # Check if response has the expected structure
            if response.choices and len(response.choices) > 0 and response.choices[0].message.content:
                return response.choices[0].message.content
            else:
                print(f"OpenAI returned empty response. Response object: {response}")
                return ""
                
        except Exception as e:
            print(f"Error in OpenAI generation: {e}")
            return ""
    
    def _generate_mistral(self, prompt: str, filepath: Path = None) -> str:
        """Generate content using Mistral"""
        try:
            messages = []
            
            if filepath:
                if filepath.suffix.lower() == '.pdf':
                    raise ValueError("PDF support for Mistral requires additional processing")
                
                # For images with Mistral vision models - using your sample format
                base64_image = self.encode_image_base64(filepath)
                
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"}
                        ]
                    }
                ]
            else:
                messages = [{"role": "user", "content": prompt}]
            
            # Using your sample format: client.chat.complete()
            chat_response = self.client.chat.complete(
                model=self.model_name,
                messages=messages
            )
            
            # Check if response has the expected structure
            if chat_response.choices and len(chat_response.choices) > 0 and chat_response.choices[0].message.content:
                return chat_response.choices[0].message.content
            else:
                print(f"Mistral returned empty response. Response object: {chat_response}")
                return ""
                
        except Exception as e:
            print(f"Error in Mistral generation: {e}")
            return ""

class WeFlyProcessor:
    def __init__(self, model_provider: str, specific_model: str = None):
        self.processor = MultiModelProcessor(model_provider, specific_model)
        
    def detect_document_type(self, filepath: Path) -> str:
        mime_type = self.processor.get_mime_type(filepath)
        if mime_type == "unknown":
            return "unknown"
        
        detection_prompt = """
        Analyze this document and determine its type. Look for these indicators:
        
        FLIGHT LOG indicators:
        - Pilot names/signatures
        - Flight routes (airport codes like EDFM, EDNY)
        - Flight duration/time columns
        - Passenger/crew counts
        - German headers like "Datum", "Pilot", "Betriebszeit"
        - Handwritten entries in logbook format
        - Flight-related data
        
        FUEL INVOICE indicators:
        - Aircraft registrations (D-XXXX format)
        - Fuel types (AvGas, Jet A1)
        - Quantities in liters
        - Prices in EUR
        - Invoice numbers
        - Fuel supplier information
        - Transaction timestamps
        - "Betankung" or fuel-related terms
        
        OTHER DOCUMENT indicators:
        - If it doesn't clearly match flight log or fuel invoice patterns
        - Random documents, receipts, forms not related to aviation
        
        Return ONLY one word: "flight_log", "fuel_invoice", or "unknown"
        """
        
        try:
            print(f"Detecting document type for: {filepath.name}")
            response = self.processor.generate_content(detection_prompt, filepath)
            
            if response is None:
                print(f"AI model returned None for document type detection: {filepath.name}")
                return "unknown"
            
            if not isinstance(response, str):
                print(f"AI model returned unexpected type {type(response)} for document type detection: {filepath.name}")
                return "unknown"
            
            detected_type = response.strip().lower()
            print(f"Document type detected for {filepath.name}: {detected_type}")
            
            if "flight" in detected_type:
                return "flight_log"
            elif "fuel" in detected_type or "invoice" in detected_type:
                return "fuel_invoice"
            else:
                return "unknown"
                
        except Exception as e:
            print(f"Error in document type detection for {filepath.name}: {e}")
            return "unknown"
    
    def process_document(self, filepath: Path) -> tuple[str, str]:
        doc_type = self.detect_document_type(filepath)
        
        if doc_type == "unknown":
            return "", doc_type
        
        if doc_type == "flight_log":
            extraction_prompt = """
            Extract this flight log as detailed markdown. Include:
            - All pilot names and signatures
            - Flight dates and times
            - Routes (departure/arrival airports)
            - Flight durations
            - Passenger/crew counts
            - Aircraft IDs if visible
            - Any technical remarks
            Preserve all structure and formatting.
            """
        else:  # fuel_invoice
            extraction_prompt = """
            Extract this fuel invoice as detailed markdown. Include:
            - All transaction dates and times
            - Aircraft registrations (D-XXXX)
            - Fuel types and quantities
            - Prices and totals
            - Transaction IDs
            - Supplier information
            Preserve all structure and formatting.
            """
        
        try:
            print(f"Processing document: {filepath.name} as {doc_type}")
            response = self.processor.generate_content(extraction_prompt, filepath)
            
            if response is None:
                print(f"AI model returned None for document: {filepath.name}")
                return "", "unknown"
            
            if not isinstance(response, str):
                print(f"AI model returned unexpected type {type(response)} for document: {filepath.name}")
                return "", "unknown"
                
            if not response.strip():
                print(f"AI model returned empty response for document: {filepath.name}")
                return "", "unknown"
            
            print(f"Successfully extracted {len(response)} characters from {filepath.name}")
            return response, doc_type
            
        except Exception as e:
            print(f"Error processing document {filepath.name}: {e}")
            return "", "unknown"
    
    def markdown_to_json_with_confidence(self, markdown_text: str, detected_type: str) -> List[Dict]:
        """Enhanced method that returns data with confidence scores and alternatives"""
        
        if detected_type == "flight_log":
            fields = "date, pilot_name, aircraft_id, departure_airport, arrival_airport, flight_duration_hours, flight_duration_minutes, passengers, crew_members, technical_issues"
            context = "flight log with pilot entries"
        else:  # fuel_invoice
            fields = "date, time, aircraft_id, fuel_type, quantity_gallons, price_per_liter, total_amount, transaction_id, supplier, location"
            context = "fuel invoice with transaction entries"
        
        # Check if using OpenAI to adjust prompt for better performance
        model_provider = self.processor.model_provider
        
        if model_provider == "openai":
            # Simplified prompt for OpenAI to avoid overly long responses
            prompt = f"""
            Extract structured data from this {context} markdown and return as a concise JSON array with confidence scoring.
            
            For each record, provide this structure (keep it concise):
            {{
              "record_data": {{
                "field_name": {{
                  "extracted_value": "value",
                  "confidence_score": 0.95,
                  "confidence_reasoning": "Brief reason",
                  "source_text": "original text", 
                  "alternatives": [{{"value": "alt1", "confidence": 0.1, "reasoning": "brief reason"}}]
                }}
              }},
              "overall_record_confidence": 0.87
            }}

            Required fields: {fields}
            
            Instructions:
            - Keep confidence_reasoning under 20 words
            - Provide max 2 alternatives for low confidence fields only
            - Use concise reasoning
            - confidence_score: 0.0-1.0
            - High confidence (0.8-1.0): Clear text
            - Medium confidence (0.5-0.79): Readable but ambiguous
            - Low confidence (0.0-0.49): Unclear/damaged text
            
            Document: {markdown_text}
            
            Return only JSON array, no other text.
            """
        else:
            # Original detailed prompt for Gemini/Mistral
            prompt = f"""
            Extract structured data from this {context} markdown and return as JSON array with confidence scoring.
            
            For each record, provide the following structure:
            {{
              "record_data": {{
                "field_name": {{
                  "extracted_value": "the extracted value",
                  "confidence_score": 0.95,
                  "confidence_reasoning": "Clear text, easily readable",
                  "source_text": "original text from document", 
                  "alternatives": [
                    {{"value": "alternative1", "confidence": 0.15, "reasoning": "possible but less likely"}},
                    {{"value": "alternative2", "confidence": 0.10, "reasoning": "third possibility"}}
                  ]
                }}
              }},
              "overall_record_confidence": 0.87
            }}

            Required fields: {fields}
            
            Instructions for confidence scoring:
            - confidence_score: 0.0-1.0 (1.0 = completely certain, 0.0 = complete guess)
            - Consider text clarity, handwriting quality, context clues, format consistency
            - High confidence (0.8-1.0): Clear printed text, standard formats, obvious context
            - Medium confidence (0.5-0.79): Readable handwriting, minor ambiguity
            - Low confidence (0.0-0.49): Unclear handwriting, damaged text, unusual formats
            - Provide 2-3 alternatives for fields with confidence < 0.8
            - Use null for completely missing fields
            - Be very accurate with numbers and dates
            
            For fuel invoices: 
            - total_amount should be the final price paid for fuel
            - price_per_liter should be the unit price (Einzelpreis)  
            - quantity_gallons should be the fuel amount in gallons (Menge)
            
            Document text:
            {markdown_text}
            
            Return only the JSON array, no other text.
            """
        
        try:
            response = self.processor.generate_content(prompt)
            
            # Check if response is None or empty
            if response is None:
                print("Error converting to JSON with confidence: AI model returned None response")
                return []
            
            if not isinstance(response, str):
                print(f"Error converting to JSON with confidence: Expected string, got {type(response)}")
                return []
            
            json_text = response.strip()
            if not json_text:
                print("Error converting to JSON with confidence: Empty response after stripping")
                return []
                
            if json_text.startswith("```json"):
                json_text = json_text[7:-3]
            elif json_text.startswith("```"):
                json_text = json_text[3:-3]
            
            # Additional check for empty json_text after cleaning
            json_text = json_text.strip()
            if not json_text:
                print("Error converting to JSON with confidence: Empty JSON content after cleaning")
                return []
            
            # Check if JSON appears to be truncated (doesn't end with ] or })
            if not (json_text.endswith(']') or json_text.endswith('}') or json_text.endswith('}')):
                print("Warning: JSON appears to be truncated, attempting to fix...")
                # Try to fix common truncation issues
                if json_text.count('[') > json_text.count(']'):
                    json_text += ']'
                elif json_text.count('{') > json_text.count('}'):
                    json_text += '}'
            
            return json.loads(json_text)
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            if 'response' in locals():
                print(f"Response length: {len(response)} characters")
                print(f"Response content preview (first 500 chars): {response[:500]}...")
                print(f"Response content end (last 200 chars): ...{response[-200:]}")
                
                # Try to extract partial JSON if possible
                try:
                    # Find the start of JSON array
                    json_start = response.find('[')
                    if json_start != -1:
                        # Find the last complete record
                        partial_json = response[json_start:]
                        # Try to close incomplete JSON
                        if partial_json.count('{') > partial_json.count('}'):
                            # Find last complete record
                            last_complete = partial_json.rfind('},{')
                            if last_complete != -1:
                                partial_json = partial_json[:last_complete+1] + ']'
                                print("Attempting to parse partial JSON...")
                                return json.loads(partial_json)
                except:
                    pass
                    
            return []
        except Exception as e:
            print(f"Error converting to JSON with confidence: {e}")
            if 'response' in locals():
                print(f"Response type: {type(response)}")
                print(f"Response length: {len(response) if isinstance(response, str) else 'N/A'}")
            else:
                print("Response variable not defined - AI model failed to generate content")
            return []
    
    def extract_simple_values(self, confidence_records: List[Dict]) -> tuple[List[Dict], List[Dict]]:
        """Extract simple values from confidence-enhanced records for compatibility"""
        flight_records = []
        fuel_records = []
        
        for record in confidence_records:
            if "record_data" in record:
                simple_record = {}
                confidence_metadata = {}
                
                for field_name, field_data in record["record_data"].items():
                    if isinstance(field_data, dict) and "extracted_value" in field_data:
                        simple_record[field_name] = field_data["extracted_value"]
                        confidence_metadata[field_name] = {
                            "confidence": field_data.get("confidence_score", 0.0),
                            "reasoning": field_data.get("confidence_reasoning", ""),
                            "alternatives": field_data.get("alternatives", [])
                        }
                    else:
                        simple_record[field_name] = field_data
                
                simple_record["_confidence_metadata"] = confidence_metadata
                simple_record["_overall_confidence"] = record.get("overall_record_confidence", 0.0)
                
                # Determine if it's flight log or fuel invoice based on fields
                if "pilot_name" in simple_record or "flight_duration_hours" in simple_record:
                    flight_records.append(simple_record)
                else:
                    fuel_records.append(simple_record)
        
        return flight_records, fuel_records
    
    def format_date(self, date_str) -> str:
        if not date_str:
            return date_str
        
        date_str = str(date_str).strip()
        
        # Handle YYYY-MM-DD format (fuel invoices)
        if '-' in date_str and len(date_str) >= 8:
            try:
                parts = date_str.split('-')
                year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
                months = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                return f"{day} {months[month]} {year % 100:02d}"
            except:
                pass
        
        # Handle DD.MM.YYYY format (flight logs)
        if '.' in date_str:
            try:
                parts = date_str.split('.')
                day, month = int(parts[0]), int(parts[1])
                year = int(parts[2]) if len(parts) > 2 and parts[2] else 2025
                months = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                return f"{day} {months[month]} {year % 100:02d}"
            except:
                pass
        
        # Handle DD/MM format
        if '/' in date_str:
            try:
                parts = date_str.split('/')
                day, month = int(parts[0]), int(parts[1])
                year = 2025  # Default year
                months = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                return f"{day} {months[month]} {year % 100:02d}"
            except:
                pass
        
        return date_str
    
    def normalize_records_with_confidence(self, flight_records: List[Dict], fuel_records: List[Dict]) -> tuple[List[Dict], List[Dict]]:
        """Normalize records while preserving confidence metadata"""
        records = []
        confidence_data = []
        
        for record in flight_records:
            # Convert to integers safely
            try:
                hours = int(record.get("flight_duration_hours", 0) or 0)
            except (ValueError, TypeError):
                hours = 0
            
            try:
                minutes = int(record.get("flight_duration_minutes", 0) or 0)
            except (ValueError, TypeError):
                minutes = 0
                
            duration = f"{hours}:{minutes:02d}" if hours or minutes else None
            
            normalized_record = {
                "Date": self.format_date(record.get("date")),
                "Airplane ID": record.get("aircraft_id"),
                "Pilot": record.get("pilot_name"),
                "Crew members": record.get("crew_members"),
                "Route from": record.get("departure_airport"),
                "Route to": record.get("arrival_airport"),
                "Flight duration": duration,
                "paid for fuel": "No"
            }
            
            confidence_record = {
                "record_type": "flight_log",
                "overall_confidence": record.get("_overall_confidence", 0.0),
                "field_confidence": record.get("_confidence_metadata", {}),
                "record_index": len(records)
            }
            
            records.append(normalized_record)
            confidence_data.append(confidence_record)
        
        for record in fuel_records:
            normalized_record = {
                "Date": self.format_date(record.get("date")),
                "Airplane ID": record.get("aircraft_id"),
                "Fuel Flat Rate (per liter)": record.get("price_per_liter"),
                "Fuel filled up (gal)": record.get("quantity_gallons"),
                "Fuel cost": record.get("total_amount"),
                "Fuel cost (per liter)": record.get("price_per_liter"),
                "paid for fuel": "Yes"
            }
            
            confidence_record = {
                "record_type": "fuel_invoice",
                "overall_confidence": record.get("_overall_confidence", 0.0),
                "field_confidence": record.get("_confidence_metadata", {}),
                "record_index": len(records)
            }
            
            records.append(normalized_record)
            confidence_data.append(confidence_record)
        
        return records, confidence_data
    
    def create_wefly_dataframe_with_confidence(self, normalized_data: List[Dict]) -> pd.DataFrame:
        """Create WeFly dataframe with confidence indicators"""
        
        wefly_columns = [
            "Date", "Airplane ID", "Pilot", "Crew members", "Invoice Address",
            "Route from", "Route to", "Fuel Start (gal)", "Fuel End (gal)",
            "Fuel Flat Rate (per liter)", "Fuel filled up (gal)", "paid for fuel",
            "Flight duration", "Fuel cost", "Fuel cost (per liter)",
            "GAT fee", "Take-off and landing fees", "Parking fees", "DFS",
            "Eurocontrol", "Austrocontrol", "Shuttle", "Emission", "Towing",
            "Parking fee", "Night handling"
        ]
        
        df = pd.DataFrame(normalized_data)
        wefly_df = df.reindex(columns=wefly_columns)
        
        numeric_cols = ["Crew members", "Fuel Start (gal)", "Fuel End (gal)", 
                        "Fuel Flat Rate (per liter)", "Fuel filled up (gal)",
                        "Fuel cost", "Fuel cost (per liter)", "GAT fee",
                        "Take-off and landing fees", "Parking fees", "DFS",
                        "Eurocontrol", "Austrocontrol", "Shuttle", "Emission",
                        "Towing", "Parking fee", "Night handling"]
        
        for col in numeric_cols:
            wefly_df[col] = pd.to_numeric(wefly_df[col], errors='coerce')
        
        return wefly_df

def get_available_models():
    """Get list of available AI models based on installed libraries and API keys"""
    models = {}
    
    # Check Gemini
    if GEMINI_AVAILABLE and os.getenv("GEMINI_API_KEY"):
        models["Gemini"] = {
            "provider": "gemini",
            "models": ["gemini-2.5-pro", "gemini-1.5-pro"]
        }
    
    # Check OpenAI
    if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
        models["OpenAI"] = {
            "provider": "openai", 
            "models": ["gpt-4o", "gpt-4o-mini"]
        }
    
    # Check Mistral
    if MISTRAL_AVAILABLE and os.getenv("MISTRAL_API_KEY"):
        models["Mistral"] = {
            "provider": "mistral",
            "models": ["mistral-large-latest", "mistral-medium-2505", "pixtral-12b-2409"]
        }
    
    return models

def process_uploaded_files_with_confidence(processor: WeFlyProcessor, uploaded_files, progress_bar, status_text, log_container) -> tuple[pd.DataFrame, List[Dict], List[str], List[str]]:
    """Enhanced processing function that returns confidence data"""
    
    all_confidence_records = []
    warnings = []
    processing_log = []
    
    total_files = len(uploaded_files)
    
    for i, uploaded_file in enumerate(uploaded_files):
        progress = (i / total_files)
        progress_bar.progress(progress)
        status_text.text(f"🔍 Processing file {i+1}/{total_files}: {uploaded_file.name}")
        
        current_log = processing_log.copy()
        current_log.append(f"📂 **Processing:** {uploaded_file.name}")
        
        with log_container.container():
            st.markdown("### 📋 Processing Log")
            for log_entry in current_log[-10:]:  
                st.write(log_entry)
        
        temp_path = Path(f"temp_{uploaded_file.name}")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            status_text.text(f"🔍 Detecting document type: {uploaded_file.name}")
            doc_type = processor.detect_document_type(temp_path)
            
            processing_log.append(f"   🎯 **Type detected:** {doc_type}")
            
            if doc_type == "unknown":
                warnings.append(f"⚠️ {uploaded_file.name}: Not a flight log or fuel invoice")
                processing_log.append(f"   ❌ **Result:** Skipped (unknown document type)")
                continue
            
            status_text.text(f"📝 Extracting text from: {uploaded_file.name}")
            markdown, _ = processor.process_document(temp_path)
            
            processing_log.append(f"   📝 **Text extracted:** {len(markdown)} characters")
            
            status_text.text(f"🔄 Converting to structured data with confidence: {uploaded_file.name}")
            confidence_records = processor.markdown_to_json_with_confidence(markdown, doc_type)
            
            if not confidence_records:
                warnings.append(f"⚠️ {uploaded_file.name}: No data could be extracted")
                processing_log.append(f"   ❌ **Result:** No data extracted")
                continue
            
            all_confidence_records.extend(confidence_records)
            processing_log.append(f"   ✅ **Success:** {len(confidence_records)} records extracted with confidence data")
            
        except Exception as e:
            warnings.append(f"⚠️ {uploaded_file.name}: Error processing - {str(e)}")
            processing_log.append(f"   ❌ **Error:** {str(e)}")
        
        finally:
            if temp_path.exists():
                temp_path.unlink()
    
    status_text.text("🔄 Processing confidence data...")
    progress_bar.progress(0.9)
    
    flight_records, fuel_records = processor.extract_simple_values(all_confidence_records)
    normalized_data, confidence_metadata = processor.normalize_records_with_confidence(flight_records, fuel_records)
    wefly_df = processor.create_wefly_dataframe_with_confidence(normalized_data)
    
    progress_bar.progress(1.0)
    status_text.text("✅ Processing complete!")
    
    processing_log.append(f"✅ **Complete:** {len(wefly_df)} total records ready with confidence data")
    
    return wefly_df, confidence_metadata, warnings, processing_log

def create_confidence_report(confidence_metadata: List[Dict]) -> pd.DataFrame:
    """Create a detailed confidence report"""
    report_data = []
    
    for i, record_meta in enumerate(confidence_metadata):
        record_type = record_meta.get("record_type", "unknown")
        overall_confidence = record_meta.get("overall_confidence", 0.0)
        
        for field_name, field_confidence in record_meta.get("field_confidence", {}).items():
            alternatives = field_confidence.get("alternatives", [])
            alt_text = "; ".join([f"{alt['value']} ({alt['confidence']:.2f})" for alt in alternatives[:2]])
            
            report_data.append({
                "Record #": i + 1,
                "Record Type": record_type,
                "Field Name": field_name,
                "Confidence Score": field_confidence.get("confidence", 0.0),
                "Confidence Reasoning": field_confidence.get("reasoning", ""),
                "Alternatives": alt_text if alt_text else "None",
                "Overall Record Confidence": overall_confidence
            })
    
    return pd.DataFrame(report_data)

def main_streamlit():
    st.set_page_config(
        page_title="WeFly Data Processor - Multi-Model",
        page_icon="✈️",
        layout="wide"
    )
    
    st.title("✈️ WeFly Flight Data Processor - Multi-Model AI Support")
    st.markdown("Upload flight logs and fuel invoices to extract data using **OpenAI**, **Mistral**, or **Gemini** AI models")
    
    # Get available models (only those with valid API keys)
    available_models = get_available_models()
    
    if not available_models:
        st.error("❌ No AI models available!")
        st.info("""
        **Setup Instructions:**
        1. Install required packages: `pip install google-generativeai openai mistralai python-dotenv`
        2. Create a `.env` file with your API keys:
        ```
        GEMINI_API_KEY=your_gemini_key_here
        OPENAI_API_KEY=your_openai_key_here
        MISTRAL_API_KEY=your_mistral_key_here
        ```
        3. Restart the application
        """)
        st.stop()
    
    # Simple Model Selection - Only two dropdowns
    st.subheader("🤖 AI Model Configuration")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        provider_names = list(available_models.keys())
        selected_provider = st.selectbox(
            "Choose AI Provider",
            options=provider_names,
            help="Select your preferred AI model provider"
        )
    
    with col2:
        if selected_provider:
            provider_info = available_models[selected_provider]
            selected_model = st.selectbox(
                "Choose Specific Model",
                options=provider_info["models"],
                help=f"Select the specific {selected_provider} model to use"
            )
        else:
            selected_model = None
    
    # Only proceed if both provider and model are selected
    if not selected_provider or not selected_model:
        st.warning("⚠️ Please select both AI Provider and Model to continue")
        st.stop()
    
    # Initialize processor
    try:
        processor = WeFlyProcessor(available_models[selected_provider]["provider"], selected_model)
        st.success(f"✅ {selected_provider} ({selected_model}) ready to use!")
    except Exception as e:
        st.error(f"❌ Failed to initialize {selected_provider}: {str(e)}")
        st.stop()
    
    # File upload section
    st.subheader("📁 Document Upload")
    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=['png', 'jpg', 'jpeg', 'pdf'],
        accept_multiple_files=True,
        help="Upload flight logs, fuel invoices, or mixed documents. Supports PNG, JPG, and PDF files."
    )
    
    if uploaded_files:
        st.success(f"📁 {len(uploaded_files)} files uploaded")
        
        with st.expander("📋 Uploaded Files"):
            for file in uploaded_files:
                st.write(f"• {file.name} ({file.size/1024:.1f} KB)")
        
        if st.button(f"🚀 Process Documents with {selected_provider}", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            log_container = st.empty()
            
            wefly_df, confidence_metadata, warnings, processing_log = process_uploaded_files_with_confidence(
                processor, uploaded_files, progress_bar, status_text, log_container
            )
            
            if warnings:
                st.warning("⚠️ **Processing Warnings:**")
                for warning in warnings:
                    st.write(warning)
            
            if len(wefly_df) > 0:
                st.success(f"✅ Successfully processed {len(wefly_df)} records using {selected_provider} ({selected_model})!")
                
                # Calculate average confidence
                avg_confidence = sum(meta.get("overall_confidence", 0) for meta in confidence_metadata) / len(confidence_metadata) if confidence_metadata else 0
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Records", len(wefly_df))
                with col2:
                    st.metric("Average Confidence", f"{avg_confidence:.2%}")
                with col3:
                    high_confidence = sum(1 for meta in confidence_metadata if meta.get("overall_confidence", 0) >= 0.8)
                    st.metric("High Confidence Records", f"{high_confidence}/{len(confidence_metadata)}")
                with col4:
                    total_fuel_cost = wefly_df['Fuel cost'].sum()
                    st.metric("Total Fuel Cost", f"€{total_fuel_cost:.2f}")
                
                # Tabs for different views
                tab1, tab2, tab3 = st.tabs(["📊 Extracted Data", "🎯 Confidence Analysis", "📋 Processing Log"])
                
                with tab1:
                    st.dataframe(wefly_df, use_container_width=True)
                
                with tab2:
                    st.subheader("🎯 Confidence Analysis Report")
                    confidence_df = create_confidence_report(confidence_metadata)
                    
                    if not confidence_df.empty:
                        # Filter options
                        record_type_filter = st.selectbox("Record Type", ["All", "flight_log", "fuel_invoice"])
                        
                        # Apply filters
                        # filtered_df = confidence_df[confidence_df["Confidence Score"]]
                        
                        filtered_df = confidence_df.copy()
                        if record_type_filter != "All":
                            filtered_df = filtered_df[filtered_df["Record Type"] == record_type_filter]
                        
                        # Color coding for confidence scores
                        def color_confidence(val):
                            if val >= 0.8:
                                return 'background-color: #00008B'
                            elif val >= 0.6:
                                return 'background-color: #008000' 
                            else:
                                return 'background-color: #800000'
                        
                        styled_df = filtered_df.style.map(color_confidence, subset=['Confidence Score'])
                        st.dataframe(styled_df, use_container_width=True)
                        
                        # Summary statistics
                        st.subheader("📈 Confidence Statistics")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            high_conf = len(filtered_df[filtered_df["Confidence Score"] >= 0.8])
                            st.metric("High Confidence (≥80%)", high_conf)
                        with col2:
                            med_conf = len(filtered_df[(filtered_df["Confidence Score"] >= 0.6) & (filtered_df["Confidence Score"] < 0.8)])
                            st.metric("Medium Confidence (60-79%)", med_conf)
                        with col3:
                            low_conf = len(filtered_df[filtered_df["Confidence Score"] < 0.6])
                            st.metric("Low Confidence (<60%)", low_conf)
                
                with tab3:
                    st.markdown("### 📋 Complete Processing Log")
                    for log_entry in processing_log:
                        st.write(log_entry)
                
                st.subheader("📥 Download Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    csv_string = wefly_df.to_csv(index=False)
                    st.download_button(
                        label="📄 Download WeFly CSV",
                        data=csv_string,
                        file_name=f"wefly_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        type="secondary"
                    )
                
                with col2:
                    if confidence_metadata:
                        confidence_json = json.dumps(confidence_metadata, indent=2)
                        st.download_button(
                            label="🎯 Download Confidence Data",
                            data=confidence_json,
                            file_name=f"confidence_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                            type="secondary"
                        )
                
                with col3:
                    if not confidence_df.empty:
                        confidence_csv = confidence_df.to_csv(index=False)
                        st.download_button(
                            label="📊 Download Confidence Report",
                            data=confidence_csv,
                            file_name=f"confidence_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            type="primary"
                        )
            else:
                st.error("❌ No valid data could be extracted from the uploaded files")
                st.info("Please ensure you're uploading flight logs or fuel invoices")

if __name__ == "__main__":
    main_streamlit()