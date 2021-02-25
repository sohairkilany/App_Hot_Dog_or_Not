import base64
import numpy as np
import io
from PIL import Image

from tensorflow.keras.models import load_model

from tensorflow.keras.preprocessing.image import img_to_array, load_img

from flask import request, render_template
from flask import jsonify
from flask import Flask

app = Flask(__name__)


# __name__ is equal to app.py
app = Flask(__name__)

@app.route('/')
def entry_page():

    return render_template('predict.html')


def preprocess_image(img, target_size):
    # if image.mode != "RGB":
    #     image = image.convert("RGB")

    image = load_img(img, target_size=target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    return image


@app.route("/predict", methods=["POST"])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    encoded = encoded.replace('data:image/jpeg;base64,', '')
    print(encoded)
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))      #.convert('RGB')
    image.save('test.jpeg', 'JPEG')
    processed_image = preprocess_image('test.jpeg', target_size=(150, 150))
    model = load_model('model_transfer.h5')
    prediction = model.predict(processed_image)
    print(prediction[0][0])
    if prediction[0][0]==0:
        result='hot dog'
    else:
        result ='not hot gog'
    response = {
        # 'prediction': {
        #     'hot-dog': prediction[0][0]
        'result': result
    }
    return jsonify(response)



if __name__ == "__main__":
    app.run(debug=True)