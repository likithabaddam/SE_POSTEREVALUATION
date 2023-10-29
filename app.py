import os
import cv2
import numpy as np
import pytesseract
from spellchecker import SpellChecker
import language_tool_python
from PIL import ImageFont, ImageDraw, Image
import difflib
import werkzeug.utils
from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
#filename = secure_filename(file.filename)

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
KNOWN_FONTS = ["Arial", "Times New Roman", "Verdana", "Tahoma", "Calibri"]

def is_valid_file_extension(filename):
    return filename.lower().endswith(tuple(ALLOWED_EXTENSIONS))
    
def secure_filename(filename):
    return werkzeug.utils.secure_filename(filename)

def compute_contrast(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = np.max(gray)
    darkness = np.min(gray)
    
    contrast = brightness / (darkness + 1e-5)  

    
    return contrast, brightness, darkness

def closest_font(text_from_image):
    img = Image.new('RGB', (200, 60), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    best_match = None
    highest_ratio = 0
    for font_name in KNOWN_FONTS:
        try:
            font = ImageFont.truetype(font_name + ".ttf", size=12)
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
        file = request.files.get('file')
        if not file:
            return "No file part", 400
        
        if not file or not file.filename:
            return "No file selected or file name is empty", 400

        
        if not is_valid_file_extension(file.filename):
            return "Invalid file type or extension. Please upload a valid image file.", 400
        if not file.content_type.startswith('image'):
            return "Invalid file type, please upload an image.", 400
        
        filename = secure_filename(file.filename)
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(filepath)
        except Exception as e:
            return f"Failed to save file: {str(e)}", 500
        
        # Load the image in grayscale directly as contrast computation doesn't need color information.
        gray_img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        contrast, brightness, darkness = compute_contrast(gray_img)

        # Load a smaller version of the image for color analysis to save resources
        small_img = cv2.resize(cv2.imread(filepath), (100, 100))  # Adjust dimensions as appropriate
        hsv = cv2.cvtColor(small_img, cv2.COLOR_BGR2HSV)
        hue, saturation, _ = cv2.split(hsv)

        
        hue_avg = np.mean(hue)
        saturation_avg = np.mean(saturation)
        
        height, width, _ = gray_img.shape
        d = pytesseract.image_to_data(gray_img, output_type=pytesseract.Output.DICT)
        
        
        def extract_text(data):
            return [text for text, height in zip(data['text'], data['height']) if height != 0]

        sentences = extract_text(d)

        
        
        def calculate_font_size(heights, image_height):
            return [(height / image_height) * 72 for height in heights if height != 0]

        font_sizes = calculate_font_size(d['height'], height)

        
        text_data = list(zip(sentences, font_sizes))
        text_from_image = " ".join(sentences)
        detected_font = closest_font(text_from_image)
        
        def get_misspelled_words(text):
            words = text.lower().split()
            return spell.unknown(words)

        misspelled = get_misspelled_words(text_from_image)

        
        
        def detect_grammar_errors(text):
            return tool.check(text)

        grammar_errors = detect_grammar_errors(text_from_image)

        return render_template('results.html', text_data=text_data, hue=hue_avg, saturation=saturation_avg, contrast=contrast, brightness=brightness, darkness=darkness, file_type=file.content_type, misspelled=misspelled, grammar_errors=grammar_errors, detected_font=detected_font)

    return render_template('index.html')

@app.route('/suggestions')
def suggestions():
    return render_template('suggestions.html')

if __name__ == '__main__':
    app.run(debug=True)
