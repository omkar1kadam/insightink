from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from datetime import datetime
import os
import traceback
from flask import Flask,render_template, request, jsonify
import google.generativeai as genai
import re  # Import regex module

# Your Gemini API Key (Replace with your actual API Key)
API_KEY = "AIzaSyB2yxeiFcUsEp2dZoekeRka5hX4FHI4C2U"

# Configure Gemini API
genai.configure(api_key=API_KEY)

app = Flask(__name__)
CORS(app)

# Create a data directory if it doesn't exist
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    print(f"Created data directory at: {DATA_DIR}")

EXCEL_FILE = os.path.join(DATA_DIR, 'contact_submissions.xlsx')


@app.route('/')
def index():
    return render_template("../Frontened/pages/home.html")

@app.route('/api/contact', methods=['POST'])
def handle_contact():
    try:
        # Get and validate the data
        data = request.get_json()
        if not data:
            print("Error: No data received")
            return jsonify({"error": "No data received"}), 400

        required_fields = ['name', 'email', 'subject', 'message']
        for field in required_fields:
            if field not in data:
                print(f"Error: Missing required field: {field}")
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # Add timestamp
        data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Create DataFrame from the form data
        df_new = pd.DataFrame([data])
        print(f"Received data: {data}")

        try:
            # If file exists, append to it
            if os.path.exists(EXCEL_FILE):
                print(f"Appending to existing file: {EXCEL_FILE}")
                df_existing = pd.read_excel(EXCEL_FILE)
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            else:
                print(f"Creating new file: {EXCEL_FILE}")
                df_combined = df_new
            
            # Save to Excel with error handling
            try:
                df_combined.to_excel(EXCEL_FILE, index=False)
                print(f"Successfully saved data to {EXCEL_FILE}")
                print(f"Current entries in Excel: {len(df_combined)}")
                return jsonify({"message": "Success", "entries": len(df_combined)}), 200
            except Exception as save_error:
                print(f"Error saving Excel file: {str(save_error)}")
                print(traceback.format_exc())
                return jsonify({"error": "Failed to save data"}), 500
                
        except Exception as excel_error:
            print(f"Error working with Excel file: {str(excel_error)}")
            print(traceback.format_exc())
            return jsonify({"error": "Failed to process Excel file"}), 500
            
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": "Failed to process request"}), 400
    
    #chatbot code by omkar

stored_data = ""

@app.route('/store', methods=['GET' , 'POST' ])
def store():
    global stored_data
    data = request.json.get("input")  # Get input from frontend
    stored_data = data
    print("Received:", stored_data)  # Print in backend console
    return jsonify({"message": "Data stored successfully", "stored": stored_data})

if __name__ == '__main__':
    app.run(debug=True)

if __name__ == '__main__':
    print(f"Starting server...")
    print(f"Excel file will be saved to: {EXCEL_FILE}")
    if os.path.exists(EXCEL_FILE):
        try:
            df = pd.read_excel(EXCEL_FILE)
            print(f"Existing entries in Excel: {len(df)}")
        except Exception as e:
            print(f"Error reading existing Excel file: {str(e)}")
    app.run(debug=True, port=5000) 



