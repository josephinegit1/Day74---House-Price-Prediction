from flask import Flask,render_template,request,jsonify
from joblib import load 
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')


model = load('house_price_model.joblib')

@app.route('/predict', methods=['POST']) 
def predict(): 
    data = request.get_json() # Receive JSON input 
    sqft = np.array([[data['sqft']]]) # Extract input value 
    prediction = model.predict(sqft)[0] # Make prediction 
    return jsonify({'predicted_price': round(prediction, 2)}) # Send response


if __name__ == '__main__':
    app.run()