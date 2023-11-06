import pickle
import numpy as np

from flask import Flask
from flask import request
from flask import jsonify

model_file = 'model.bin'

with open(model_file, 'rb') as f_in:
   dv, model = pickle.load(f_in)

app = Flask('Stock_price')

@app.route('/model_predict', methods=['POST'])

def predict():
   stock = request.get_json()

   X = dv.transform([stock])
   y_pred = model.predict(X)
   
   results = {
      'price_probablity': float(y_pred)
   }
   
   # result = results.to_list()
   return jsonify(results)

if __name__ == '__main__':
   app.run(debug=True, host='0.0.0.0', port=6969)
