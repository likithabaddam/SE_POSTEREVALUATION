import os
import cv2
import re
import requests
import numpy as np
from flask_lambda import FlaskLambda
import pytesseract
from spellchecker import SpellChecker
import language_tool_python
from PIL import ImageFont, ImageDraw, Image
import difflib
import werkzeug.utils
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from flask import send_file
import pdfkit

app = Flask(__name__)
pdfkit_config = pdfkit.configuration(wkhtmltopdf='/usr/local/bin/wkhtmltopdf')
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
KNOWN_FONTS = ["Arial", "Times New Roman", "Verdana", "Tahoma", "Calibri"]

def extract_references(text):
    url_pattern = r'\b(?:https?://)?(?:(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+\.[a-zA-Z]{2,6}\b/?)'
    return re.findall(url_pattern, text)

def verify_links(references):
    results = []
    for ref in references:
        if not ref.startswith(('http://', 'https://')):
            ref = 'http://' + ref
        try:
            response = requests.head(ref, allow_redirects=True, timeout=5)
            if response.status_code == 200:
                results.append((ref, 'Verified'))
            else:
                results.append((ref, 'Not Verified'))
        except requests.RequestException:
            results.append((ref, 'Not Verified'))
    return results

def is_valid_file_extension(filename):
    return filename.lower().endswith(tuple(ALLOWED_EXTENSIONS))

def compute_contrast(img):
    if len(img.shape) > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    contrast = np.std(gray)
    return contrast

def closest_font(text_from_image):
    img = Image.new('RGB', (200, 60), color = (255, 255, 255))
    d = ImageDraw.Draw(img)
    best_match = None
    highest_ratio = 0
    for font_name in KNOWN_FONTS:
        try:
            font = ImageFont.truetype(font_name + ".ttf", size=12)
            d.text((10,10), text_from_image, font=font, fill=(0,0,0))
            rendered_text = pytesseract.image_to_string(img)
            match_ratio = difflib.SequenceMatcher(None, text_from_image, rendered_text).ratio()
            if match_ratio > highest_ratio:
                highest_ratio = match_ratio
                best_match = font_name
        except Exception as e:
            print(f"Error processing font {font_name}: {e}")
    return best_match

tool = language_tool_python.LanguageTool('en-US')
spell = SpellChecker()

@app.route('/', methods=['GET', 'POST'])
def index():
    hue_avg = saturation_avg = 0
    if request.method == 'POST':
        file = request.files['file']
        if file and is_valid_file_extension(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            img = cv2.imread(filepath)
            if img is None:
                return "Error loading image", 500

            contrast = compute_contrast(img)

            if len(img.shape) == 3:
                small_img = cv2.resize(img, (100, 100))
                hsv = cv2.cvtColor(small_img, cv2.COLOR_BGR2HSV)
                hue, saturation, _ = cv2.split(hsv)
                hue_avg = np.mean(hue)
                saturation_avg = np.mean(saturation)

            custom_config = r'--oem 3 --psm 11'
            text_from_image = pytesseract.image_to_string(img, config=custom_config)

            matches = tool.check(text_from_image)
            error_details = []
            for match in matches:
                from_line = match.context[0:match.offset].count('\n') + 1
                from_column = match.offset - match.context.rfind('\n', 0, match.offset)
                error_detail = {
                    'from_line': from_line,
                    'from_column': from_column,
                    'message': match.message,
                    'replacements': match.replacements,
                    'length': match.errorLength
                }
                error_details.append(error_detail)

            words = re.findall(r'\b\w+\b', text_from_image.lower())
            misspelled = [word for word in words if spell.unknown([word])]

            references = extract_references(text_from_image)
            verified_references = verify_links(references)

            d = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            sentences = [text for text, conf in zip(d['text'], d['conf']) if int(conf) > 60]
            font_sizes = [(height / img.shape[0]) * 72 for height in d['height'] if height != 0]
            text_data = list(zip(sentences, font_sizes))

            detected_font = closest_font(text_from_image)

            return render_template(
                'results.html', 
                text_data=text_data, 
                hue_avg=hue_avg, 
                saturation_avg=saturation_avg, 
                contrast=contrast,
                file_type=file.content_type, 
                misspelled=misspelled, 
                grammar_errors=matches, 
                detected_font=detected_font, 
                references=verified_references,
                error_details=error_details,
            )
    return render_template('index.html')

@app.route('/suggestions')
def suggestions():
    suggestions_data = request.args.get('suggestions', [])  # Adjust as needed for your implementation
    return render_template('suggestions.html', suggestions=suggestions_data)

@app.route('/generate_pdf', methods=['POST'])
def generate_pdf():
    pdfkit.from_file('templates/results.html', 'results.pdf', configuration=pdfkit_config)
    return send_file('results.pdf', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
