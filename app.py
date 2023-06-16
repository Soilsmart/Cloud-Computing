from flask import Flask, jsonify, request
import json
from datetime import datetime
import pytz
import firebase_admin
from firebase_admin import credentials, firestore, auth
import tensorflow as tf
import numpy as np
import tensorflow.lite as tflite
app = Flask(__name__)

cred = credentials.Certificate("serviceAccount.json")
firebase_admin.initialize_app(cred)
db = firestore.client()
data = {
    'Periode': 0,  # Ganti dengan nilai minimum dari data Anda
    'Luas Panen (Ha)': 0,  # Ganti dengan nilai minimum dari data Anda
    'Produksi Padi (Ton-GKG)': 0  # Ganti dengan nilai minimum dari data Anda
}

# Endpoint untuk mendapatkan semua tasks
@app.route('/api/historyprediction', methods=['GET'])
def get_history():
    try:
        email = request.args.get('email')

        if not email:
            return json.dumps({'error': 'Email must be included in the query parameters!'}), 400

        user = auth.get_user_by_email(email)
        uid = user.uid

        try:
          query = db.collection('histori_panen').where(
              'uid', '==', uid).get()
          data_history = []

          for doc in query:
              data = doc.to_dict()
              data_history.append(data)

          response = {'code': 200, 'data': data_history}
          response_json = json.dumps(response)
          return response_json, 200
        except Exception as e:
          error_message = {'error': str(e)}
          return json.dumps(error_message), 500
        # return get_data_firestore_by_id(uid)

    except auth.InvalidIdTokenError as e:
        error_message = {'error': str(e)}
        return json.dumps(error_message), 401

    except Exception as e:
        error_message = {'error': str(e)}
        return json.dumps(error_message), 500

    # data = []
    # collection_ref = db.collection('history_lahan')  # Ganti dengan nama koleksi Anda

    # docs = collection_ref.get()
    # for doc in docs:
    #     data.append(doc.to_dict())

    # return jsonify({'data': data})

# Endpoint untuk mendapatkan task berdasarkan ID
model = tflite.Interpreter(model_path='model_produksipadi.tflite')
@app.route('/api/predict', methods=['POST'])
def get_predict():
    email = request.form['email']
    periode_tanam = float(request.form['periode_tanam'])
    luas_panen = float(request.form['luas_panen'])


    try: 
        user = auth.get_user_by_email(email)
        uid = user.uid 

        def prepare_input_data(periode_tanam, luas_panen):
            scaled_period = (periode_tanam - data['Periode'].min()) / (data['Periode'].max() - data['Periode'].min())
            scaled_luas_panen = (luas_panen - data['Luas Panen (Ha)'].min()) / (data['Luas Panen (Ha)'].max() - data['Luas Panen (Ha)'].min())
            input_data = np.array([[scaled_period, scaled_luas_panen]])
            return input_data
        input_data = prepare_input_data(float(periode_tanam), float(luas_panen))
        y_pred = model.predict(input_data)
        predicted_output = (y_pred * (data['Produksi Padi (Ton-GKG)'].max() - data['Produksi Padi (Ton-GKG)'].min())) + data['Produksi Padi (Ton-GKG)'].min()
       
        def save_history_to_firestore(uid, periode_tanam, luas_panen, predicted_output):
            try:
                server_time = datetime.now()
                user_timezone = pytz.timezone('Asia/Jakarta')
                user_time = server_time.astimezone(user_timezone)

                data = {
                    'uid': uid,
                    'periode_tanam': periode_tanam,
                    'luas_panen' : luas_panen,
                    'predicted_output': predicted_output,
                    'timestamp': user_time.strftime('%Y-%m-%d %H:%M:%S')
                }

                db.collection('history_lahan').add(data)
                response = {'code': 200, 'message': 'Data sudah disimpan!'}
                response_json = json.dumps(response)
                return response_json, 200
            except Exception as e:
                error_message = {'error': str(e)}
                return json.dumps(error_message), 500
        save_history_to_firestore = save_history_to_firestore(uid, periode_tanam, luas_panen, predicted_output)
        return jsonify({'message': "Prediksi berhasil", 'hasil_prediksi': predicted_output})
    except Exception as e:
        error_message = {'error': str(e)}
        return json.dumps(error_message), 500

if __name__ == '__main__':
    app.run(debug=True)
