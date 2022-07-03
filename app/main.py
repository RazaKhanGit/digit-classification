from flask import Flask, request, jsonify
import app.utils as utils

app = Flask(__name__)

ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files.get('file')
        if file == None or file.filename == '':
            return jsonify({'error': 'file not found'})
        if not allowed_file(file.filename):
            return jsonify({'error': 'bad format'})
        try:
            image_bytes = file.read()
            img = utils.tranform_img(image_bytes)
            prediction = utils.get_prediction(img)
            data = {'prediction': prediction.item(), 'class_name':  str(prediction.item())}
            return jsonify(data)            
        except:
            return jsonify({'error': 'error during inference'})

    # Load img
    #transform img
    #predict
    #return json
    return jsonify({'result': 1})

