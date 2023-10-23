from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
import pytesseract
from spellchecker import SpellChecker
import language_tool_python
from PIL import ImageFont, ImageDraw, Image
import difflib

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

KNOWN_FONTS = ["Arial", "Times New Roman", "Verdana", "Tahoma", "Calibri"]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def compute_contrast(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = np.max(gray)
    darkness = np.min(gray)
    contrast = brightness - darkness
    return contrast, brightness, darkness

def closest_font(text_from_image):
    best_match = None
    highest_ratio = 0
    for font_name in KNOWN_FONTS:
        try:
            font = ImageFont.truetype(font_name + ".ttf", size=12)
            img = Image.new('RGB', (200, 60), color=(255, 255, 255))
            d = ImageDraw.Draw(img)
            d.text((10, 10), text_from_image, font=font, fill=(0, 0, 0))
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
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            img = cv2.imread(filename)
            contrast, brightness, darkness = compute_contrast(img)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hue, saturation, _ = cv2.split(hsv)
            hue = np.mean(hue)
            saturation = np.mean(saturation)
            height, width, _ = img.shape
            d = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            sentences = [d['text'][i] for i, h in enumerate(d['height']) if h != 0]
            font_sizes = [(h / height) * 72 for h in d['height'] if h != 0]  # Convert to points (approximation)
            text_data = list(zip(sentences, font_sizes))
            text_from_image = " ".join(sentences)
            detected_font = closest_font(text_from_image)
            misspelled = spell.unknown(text_from_image.split())
            grammar_errors = tool.check(text_from_image)
            return render_template('results.html', text_data=text_data, hue=hue, saturation=saturation, contrast=contrast, brightness=brightness, darkness=darkness, file_type=file.content_type, misspelled=misspelled, grammar_errors=grammar_errors, detected_font=detected_font)
    return render_template('index.html')

@app.route('/suggestions')
def suggestions():
    return render_template('suggestions.html')

if __name__ == '__main__':
    app.run(debug=True)
