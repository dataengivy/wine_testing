from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

# Initialize Flask app
app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("F:/Dployment/p3d/wine_quality_model.pkl", "rb"))
scaler = pickle.load(open("F:/Dployment/p3d/wine_quality_scaler.pkl", "rb"))

# Mapping function
def pred(value):
    if value == 0:
        return 'Excellent'
    elif value == 1:
        return 'Good'
    elif value == 2:
        return 'Average'
    else:
        return 'Poor'


# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST','GET'])
def predict():
    try:
        data = request.form

        print("Working fine")
        
        # Convert inputs to float
        flavanoids = float(data['flavanoids'])
        color_intensity = float(data['color_intensity'])
        proline = float(data['proline'])
        ash = float(data['ash'])
        alcohol = float(data['alcohol'])

        # Prepare input for model
        raw_data = np.array([[flavanoids, color_intensity, proline, ash, alcohol]])
        scaled_data = scaler.transform(raw_data)
        prediction = model.predict(scaled_data)

        # Map prediction to label
        final_prediction = pred(prediction[0])
        print(f"Prediction: {final_prediction}")

        return render_template('index.html', prediction_text='Wine Quality is: {}'.format(final_prediction))

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)