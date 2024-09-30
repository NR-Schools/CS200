from flask import Flask, render_template, request, redirect, url_for
from app.model import classify_image
from PIL import Image
import base64
import io


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return redirect(request.url)
    
    image = request.files['image']
    if image.filename == '':
        return redirect(request.url)
    
    # Read the image as bytes (to preserve the file pointer for both classification and display)
    image_bytes = image.read()
    
    class_result = classify_image(
        Image.open(
            io.BytesIO(
                image_bytes
            )
        )
    )

    result = {}
    result["class"] = class_result
    result["image"] = base64.b64encode(image_bytes).decode('utf-8')
    return render_template('uploaded.html', result=result)


if __name__ == "__main__":
    app.run(port=8000, debug=True)