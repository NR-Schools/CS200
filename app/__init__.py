from flask import Flask, render_template, request, redirect, url_for
from app.model import classify_image
from PIL import Image
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
    
    # Read and classify image
    result = classify_image(
        Image.open(
            io.BytesIO(
                image.read()
            )
        )
    )

    return render_template('uploaded.html', result=result)


if __name__ == "__main__":
    app.run(port=8000, debug=True)