import os
import librosa
import pickle
from flask import Flask, flash, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "audio_uploads")
ALLOWED_EXTENSIONS = {'wav', 'mp3'}
SAMPLE_RATE = 22050

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_features_csv_row(signal, sample_rate, num_mfcc=13, n_fft=2048, hop_length=512):

    csv_row = []

    # extract mfcc
    mfcc = librosa.feature.mfcc(signal, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfcc = mfcc.T
    csv_row += sum(mfcc.tolist(), [])
        
    # extract spectral centeroid
    spectral_centroids = librosa.feature.spectral_centroid(signal, sr=sample_rate)[0]
    csv_row += spectral_centroids.tolist()
    
    # extract spectral rolloff
    spectral_rolloffs = librosa.feature.spectral_rolloff(signal+0.01, sr=sample_rate)[0]
    csv_row += spectral_rolloffs.tolist()
    
    # extract spectral bandwidth
    spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(signal+0.01, sr=sample_rate)[0]
    csv_row += spectral_bandwidth_2.tolist()
    spectral_bandwidth_3 = librosa.feature.spectral_bandwidth(signal+0.01, sr=sample_rate, p=3)[0]
    csv_row += spectral_bandwidth_3.tolist()
    spectral_bandwidth_4 = librosa.feature.spectral_bandwidth(signal+0.01, sr=sample_rate, p=4)[0]
    csv_row += spectral_bandwidth_4.tolist()
    
    # extract zero-crossing rate                    
    zero_crossing_rates = librosa.feature.zero_crossing_rate(signal, pad=False)[0]
    csv_row += zero_crossing_rates.tolist()
    
    # extract croma features
    chroma_features = librosa.feature.chroma_stft(signal, sr=sample_rate, hop_length=hop_length)
    chroma_features = chroma_features.T
    csv_row += sum(chroma_features.tolist(), [])

    return csv_row

def is_cough_present(feature_vector):

    mapping = {
        0: False,
        1: True,
    }
    try:
        # load the model from disk
        model_file = os.path.join(BASE_DIR, "detection_model.sav")
        loaded_model = pickle.load(open(model_file, 'rb'))
        prediction = int(loaded_model.predict([feature_vector])[0])
        return mapping.get(prediction)
    except Exception as e:
        raise

def get_cough_type(feature_vector):

    mapping = {
        0: "whooping",
        1: "croup",
        2: "wet",
        3: "dry",
    }
    try:
        # load the model from disk
        model_file = os.path.join(BASE_DIR, "classification_model.sav")
        loaded_model = pickle.load(open(model_file, 'rb'))
        prediction = loaded_model.predict([feature_vector])[0]
        return mapping.get(prediction)
    except Exception as e:
        raise

@app.route('/') 
def home(): 
    return "<h3>POST API:</h3>for cough detection - /detect/<br>for cough type classification - /classify/"

@app.route('/detect', methods=['POST']) 
def detect_cough():
    try:
        if request.method == 'POST':
            # check if the post request has the file part
            if 'file' not in request.files:
                raise Exception("File part not found in POST request")
            file = request.files['file']
            # if user does not select file, browser also submit an empty part without filename
            if file.filename == '':
                raise Exception("File not selected")
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)
        feature_vector = get_features_csv_row(signal, sample_rate)
        prediction = is_cough_present(feature_vector)
        os.remove(file_path)

        res = {
            "status": "success",
            "data": {
                "prediction": prediction
            },
            "message": ""
        }
    except Exception as e:
        res = {
            "status": "error",
            "data": {
                "prediction": None
            },
            "message": str(e)
        }
    return jsonify(res)

@app.route('/classify', methods=['POST']) 
def classify_cough():
    try:
        if request.method == 'POST':
            # check if the post request has the file part
            if 'file' not in request.files:
                raise Exception("File part not found in POST request")
            file = request.files['file']
            # if user does not select file, browser also submit an empty part without filename
            if file.filename == '':
                raise Exception("File not selected")
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)
        feature_vector = get_features_csv_row(signal, sample_rate)
        os.remove(file_path)
        
        if not is_cough_present(feature_vector):
            res = {
                "status": "success",
                "data": {
                    "prediction": None
                },
                "message": "No coughing sound detected."
            }
        else:
            prediction = get_cough_type(feature_vector)
            res = {
                "status": "success",
                "data": {
                    "prediction": prediction
                },
                "message": ""
            }
    except Exception as e:
        res = {
            "status": "error",
            "data": {
                "prediction": None
            },
            "message": str(e)
        }

    return jsonify(res)
  
# main driver function 
if __name__ == '__main__': 
    # Bind to PORT if defined, otherwise default to 5000.
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)