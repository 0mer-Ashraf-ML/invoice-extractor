import streamlit as st
import pandas as pd
import json
import argparse
import sys
import io
from datetime import datetime
from pathlib import Path
import base64
from typing import Dict, List, Any, Optional

from google import genai
from google.genai import types
import pathlib

class WeFlyProcessor:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        
    def detect_document_type(self, filepath: Path) -> str:
        
        # Determine type
        if filepath.suffix.lower() == '.pdf':
            mime_type = 'application/pdf'
        elif filepath.suffix.lower() in ['.jpg', '.jpeg']:
            mime_type = 'image/jpeg'
        elif filepath.suffix.lower() == '.png':
            mime_type = 'image/png'
        else:
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
            response = self.client.models.generate_content(
                model="gemini-2.5-pro",
                contents=[
                    types.Part.from_bytes(
                        data=filepath.read_bytes(),
                        mime_type=mime_type,
                    ),
                    detection_prompt
                ]
            )
            
            detected_type = response.text.strip().lower()
            
            if "flight" in detected_type:
                return "flight_log"
            elif "fuel" in detected_type or "invoice" in detected_type:
                return "fuel_invoice"
            else:
                return "unknown"
                
        except Exception as e:
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
        
        if filepath.suffix.lower() == '.pdf':
            mime_type = 'application/pdf'
        else:
            mime_type = 'image/jpeg' if filepath.suffix.lower() in ['.jpg', '.jpeg'] else 'image/png'
        
        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-pro",
                contents=[
                    types.Part.from_bytes(
                        data=filepath.read_bytes(),
                        mime_type=mime_type,
                    ),
                    extraction_prompt
                ]
            )
            
            return response.text, doc_type
            
        except Exception as e:
            print(f"Error processing document: {e}")
            return "", "unknown"
    
    def markdown_to_json(self, markdown_text: str, detected_type: str) -> List[Dict]:
        
        if detected_type == "flight_log":
            fields = "date, pilot_name, aircraft_id, departure_airport, arrival_airport, flight_duration_hours, flight_duration_minutes, passengers, crew_members, technical_issues"
            context = "flight log with pilot entries"
        else:  # fuel_invoice
            fields = "date, time, aircraft_id, fuel_type, quantity_gallons, price_per_liter, total_amount, transaction_id, supplier, location"
            context = "fuel invoice with transaction entries"
        
        prompt = f"""
        Extract structured data from this {context} markdown and return as JSON array.
        Required fields: {fields}
        
        Instructions:
        - Each record = one JSON object
        - Use null for missing fields
        - Return only JSON array
        - Be very accurate with numbers and dates
        - For fuel invoices: total_amount should be the final price paid for fuel
        - For fuel invoices: price_per_liter should be the unit price (Einzelpreis)
        - For fuel invoices: quantity_gallons should be the fuel amount in gallons (Menge)
        
        {markdown_text}
        """
        
        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-pro",
                contents=[prompt]
            )
            
            json_text = response.text.strip()
            if json_text.startswith("```json"):
                json_text = json_text[7:-3]
            elif json_text.startswith("```"):
                json_text = json_text[3:-3]
            
            return json.loads(json_text)
            
        except Exception as e:
            print(f"Error converting to JSON: {e}")
            return []
    
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
    
    def liters_to_gallons(self, liters) -> float:
        try:
            if liters is None:
                return None
            return round(float(liters) * 0.264172, 2)
        except:
            return None
    
    def normalize_records(self, flight_records: List[Dict], fuel_records: List[Dict]) -> List[Dict]:
        records = []
        
        for record in flight_records:
            hours = record.get("flight_duration_hours", 0) or 0
            minutes = record.get("flight_duration_minutes", 0) or 0
            duration = f"{hours}:{minutes:02d}" if hours or minutes else None
            
            records.append({
                "Date": self.format_date(record.get("date")),
                "Airplane ID": record.get("aircraft_id"),
                "Pilot": record.get("pilot_name"),
                "Crew members": record.get("crew_members"),
                "Route from": record.get("departure_airport"),
                "Route to": record.get("arrival_airport"),
                "Flight duration": duration,
                "paid for fuel": "No"
            })
        
        for record in fuel_records:
            records.append({
                "Date": self.format_date(record.get("date")),
                "Airplane ID": record.get("aircraft_id"),
                "Fuel Flat Rate (per liter)": record.get("price_per_liter"),
                "Fuel filled up (gal)": record.get("quantity_gallons"),  # Direct mapping - no conversion
                "Fuel cost": record.get("total_amount"),
                "Fuel cost (per liter)": record.get("price_per_liter"),
                "paid for fuel": "Yes"
            })
        
        return records
    
    def create_wefly_dataframe(self, normalized_data: List[Dict]) -> pd.DataFrame:
        
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

def process_uploaded_files(processor: WeFlyProcessor, uploaded_files, progress_bar, status_text, log_container) -> tuple[pd.DataFrame, List[str], List[str]]:
    
    all_flight_records = []
    all_fuel_records = []
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
            
            with log_container.container():
                st.markdown("### 📋 Processing Log")
                for log_entry in processing_log[-10:]:
                    st.write(log_entry)
            
            if doc_type == "unknown":
                warnings.append(f"⚠️ {uploaded_file.name}: Not a flight log or fuel invoice")
                processing_log.append(f"   ❌ **Result:** Skipped (unknown document type)")
                continue
            
            status_text.text(f"📝 Extracting text from: {uploaded_file.name}")
            markdown, _ = processor.process_document(temp_path)
            
            processing_log.append(f"   📝 **Text extracted:** {len(markdown)} characters")
            
            with log_container.container():
                st.markdown("### 📋 Processing Log")
                for log_entry in processing_log[-10:]:
                    st.write(log_entry)
            
            status_text.text(f"🔄 Converting to structured data: {uploaded_file.name}")
            records = processor.markdown_to_json(markdown, doc_type)
            
            if not records:
                warnings.append(f"⚠️ {uploaded_file.name}: No data could be extracted")
                processing_log.append(f"   ❌ **Result:** No data extracted")
                continue
            
            if doc_type == "flight_log":
                all_flight_records.extend(records)
                processing_log.append(f"   ✅ **Success:** {len(records)} flight records extracted")
            else:
                all_fuel_records.extend(records)
                processing_log.append(f"   ✅ **Success:** {len(records)} fuel records extracted")
            
            with log_container.container():
                st.markdown("### 📋 Processing Log")
                for log_entry in processing_log[-10:]:
                    st.write(log_entry)
                
        except Exception as e:
            warnings.append(f"⚠️ {uploaded_file.name}: Error processing - {str(e)}")
            processing_log.append(f"   ❌ **Error:** {str(e)}")
        
        finally:
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()
    
    status_text.text("🔄 Normalizing data and creating WeFly format...")
    progress_bar.progress(0.9)
    
    processing_log.append("---")
    processing_log.append(f"📊 **Summary:** {len(all_flight_records)} flight records, {len(all_fuel_records)} fuel records")
    processing_log.append("🔄 **Creating WeFly format...**")
    
    with log_container.container():
        st.markdown("### 📋 Processing Log")
        for log_entry in processing_log[-10:]:
            st.write(log_entry)
    
    normalized_data = processor.normalize_records(all_flight_records, all_fuel_records)
    wefly_df = processor.create_wefly_dataframe(normalized_data)
    
    progress_bar.progress(1.0)
    status_text.text("✅ Processing complete!")
    
    processing_log.append(f"✅ **Complete:** {len(wefly_df)} total records ready for download")
    
    with log_container.container():
        st.markdown("### 📋 Processing Log")
        for log_entry in processing_log[-15:]:
            st.write(log_entry)
    
    return wefly_df, warnings, processing_log

# def process_uploaded_files(processor: WeFlyProcessor, uploaded_files) -> tuple[pd.DataFrame, List[str], List[str]]:
    
#     all_flight_records = []
#     all_fuel_records = []
#     warnings = []
#     processing_log = []
    
#     for uploaded_file in uploaded_files:
#         temp_path = Path(f"temp_{uploaded_file.name}")
#         with open(temp_path, "wb") as f:
#             f.write(uploaded_file.getbuffer())
        
#         try:
#             markdown, doc_type = processor.process_document(temp_path)
            
#             if doc_type == "unknown":
#                 warnings.append(f"⚠️ {uploaded_file.name}: Not a flight log or fuel invoice")
#                 processing_log.append(f"❌ {uploaded_file.name}: Skipped (unknown document type)")
#                 continue
            
#             records = processor.markdown_to_json(markdown, doc_type)
            
#             if not records:
#                 warnings.append(f"⚠️ {uploaded_file.name}: No data could be extracted")
#                 processing_log.append(f"❌ {uploaded_file.name}: No data extracted")
#                 continue
            
#             if doc_type == "flight_log":
#                 all_flight_records.extend(records)
#                 processing_log.append(f"✅ {uploaded_file.name}: {len(records)} flight records")
#             else:
#                 all_fuel_records.extend(records)
#                 processing_log.append(f"✅ {uploaded_file.name}: {len(records)} fuel records")
                
#         except Exception as e:
#             warnings.append(f"⚠️ {uploaded_file.name}: Error processing - {str(e)}")
#             processing_log.append(f"❌ {uploaded_file.name}: Error - {str(e)}")
        
#         finally:
#             # Clean up temp file
#             if temp_path.exists():
#                 temp_path.unlink()
    
#     normalized_data = processor.normalize_records(all_flight_records, all_fuel_records)
#     wefly_df = processor.create_wefly_dataframe(normalized_data)
    
#     return wefly_df, warnings, processing_log

def create_download_link(df: pd.DataFrame, file_format: str = 'csv') -> str:
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if file_format == 'excel':
        buffer = io.BytesIO()
        df.to_excel(buffer, index=False, sheet_name='Overall')
        buffer.seek(0)
        b64 = base64.b64encode(buffer.read()).decode()
        filename = f"wefly_data_{timestamp}.xlsx"
        mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    else:  # CSV
        csv_string = df.to_csv(index=False)
        b64 = base64.b64encode(csv_string.encode()).decode()
        filename = f"wefly_data_{timestamp}.csv"
        mime_type = "text/csv"
    
    return b64, filename, mime_type

def main_streamlit():
    st.set_page_config(
        page_title="WeFly Data Processor",
        page_icon="✈️",
        layout="wide"
    )
    
    st.title("✈️ WeFly Flight Data Processor")
    st.markdown("Upload flight logs and fuel invoices to extract data in WeFly format")
    
    api_key = st.text_input(
        "Google Gemini API Key", 
        type="password",
        help="Get your API key from: https://makersuite.google.com/app/apikey"
    )
    
    if not api_key:
        st.warning("Please enter your Google Gemini API key to continue")
        st.info("🔑 Get your free API key from: https://makersuite.google.com/app/apikey")
        st.stop()
    
    processor = WeFlyProcessor(api_key)
    
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
        
        if st.button("🚀 Process Documents", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            log_container = st.empty()
            
            wefly_df, warnings, processing_log = process_uploaded_files(
                processor, uploaded_files, progress_bar, status_text, log_container
            )
            
            if warnings:
                st.warning("⚠️ **Processing Warnings:**")
                for warning in warnings:
                    st.write(warning)
            
            with st.expander("📋 Complete Processing Log", expanded=True):
                st.markdown("### 📈 Processing Summary")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Files Processed", len(uploaded_files))
                with col2:
                    successful_files = len([log for log in processing_log if "✅ **Success:**" in log])
                    st.metric("Successful Extractions", successful_files)
                with col3:
                    st.metric("Warnings", len(warnings))
                
                st.markdown("### 📋 Detailed Log")
                for log_entry in processing_log:
                    st.write(log_entry)
            
            if len(wefly_df) > 0:
                st.success(f"✅ Successfully processed {len(wefly_df)} records!")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Records", len(wefly_df))
                with col2:
                    aircraft_count = wefly_df['Airplane ID'].dropna().nunique()
                    st.metric("Aircraft", aircraft_count)
                with col3:
                    pilot_count = wefly_df['Pilot'].dropna().nunique()
                    st.metric("Pilots", pilot_count)
                with col4:
                    total_fuel_cost = wefly_df['Fuel cost'].sum()
                    st.metric("Total Fuel Cost", f"€{total_fuel_cost:.2f}")
                
                st.subheader("📊 Extracted Data")
                st.dataframe(wefly_df, use_container_width=True)
                
                st.subheader("📥 Download Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    csv_b64, csv_filename, csv_mime = create_download_link(wefly_df, 'csv')
                    st.download_button(
                        label="📄 Download CSV",
                        data=base64.b64decode(csv_b64),
                        file_name=csv_filename,
                        mime=csv_mime,
                        type="secondary"
                    )
                
                with col2:
                    excel_b64, excel_filename, excel_mime = create_download_link(wefly_df, 'excel')
                    st.download_button(
                        label="📊 Download Excel",
                        data=base64.b64decode(excel_b64),
                        file_name=excel_filename,
                        mime=excel_mime,
                        type="primary"
                    )
            else:
                st.error("❌ No valid data could be extracted from the uploaded files")
                st.info("Please ensure you're uploading flight logs or fuel invoices")

def main_cli():
    parser = argparse.ArgumentParser(description="WeFly Flight Data Processor CLI")
    parser.add_argument("input_files", nargs="+", help="Path to input files (images or PDFs)")
    parser.add_argument("--output", "-o", required=True, help="Output file path (.csv or .xlsx)")
    parser.add_argument("--api-key", help="Google Gemini API key")
    parser.add_argument("--format", choices=["csv", "excel"], default="csv", help="Output format")
    
    args = parser.parse_args()
    
    api_key = args.api_key or input("Enter Google Gemini API key: ")
    if not api_key:
        print("Error: API key is required")
        sys.exit(1)
    
    processor = WeFlyProcessor(api_key)
    
    all_flight_records = []
    all_fuel_records = []
    
    print(f"Processing {len(args.input_files)} files...")
    
    for file_path in args.input_files:
        filepath = Path(file_path)
        if not filepath.exists():
            print(f"❌ File not found: {file_path}")
            continue
        
        print(f"Processing: {filepath.name}")
        
        try:
            markdown, doc_type = processor.process_document(filepath)
            
            if doc_type == "unknown":
                print(f"⚠️  {filepath.name}: Not a flight log or fuel invoice")
                continue
            
            records = processor.markdown_to_json(markdown, doc_type)
            
            if doc_type == "flight_log":
                all_flight_records.extend(records)
                print(f"✅ {filepath.name}: {len(records)} flight records")
            else:
                all_fuel_records.extend(records)
                print(f"✅ {filepath.name}: {len(records)} fuel records")
                
        except Exception as e:
            print(f"❌ Error processing {filepath.name}: {e}")
    
    normalized_data = processor.normalize_records(all_flight_records, all_fuel_records)
    wefly_df = processor.create_wefly_dataframe(normalized_data)
    
    if len(wefly_df) > 0:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if args.format == "excel" or output_path.suffix.lower() in ['.xlsx', '.xls']:
            wefly_df.to_excel(output_path, index=False, sheet_name='Overall')
        else:
            wefly_df.to_csv(output_path, index=False)
        
        print(f"\n✅ Success!")
        print(f"📄 Output saved: {output_path}")
        print(f"📊 Records processed: {len(wefly_df)}")
        print(f"✈️  Flight records: {len(all_flight_records)}")
        print(f"⛽ Fuel records: {len(all_fuel_records)}")
        
    else:
        print("❌ No valid data could be extracted")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) > 1 and not any(arg.startswith('--') for arg in sys.argv[1:2]):
        main_cli()
    else:
        main_streamlit()