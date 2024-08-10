import joblib
from flask import Flask, request, jsonify
import traceback
import sys

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model = None
try:
    model = joblib.load('model.pkl')  # Replace 'model.pkl' with the path to your actual model
    print('Model loaded')
except Exception as e:
    print('Error loading the model:', str(e))

# Define the predict route
@app.route('/predict', methods=['POST'])
def predict():
    if model:
        try:
            # Get JSON data from the request
            json_ = request.json
            print('Received JSON data:', json_)
            
            # Validate the input
            if not isinstance(json_, dict) or 'title' not in json_:
                return jsonify({'error': 'Invalid input format'}), 400
            
            # Make prediction
            title = json_['title']
            prediction = model.predict([title])
            
            # Return the prediction as JSON
            return jsonify({'prediction': str(prediction[0])})
        
        except Exception as e:
            # Return the traceback in case of an error
            return jsonify({'trace': traceback.format_exc()})
    
    else:
        print('Train the model first')
        return 'No model here to use'

# Run the Flask app
if __name__ == '__main__':
    try:
        port = int(sys.argv[1])  # Command-line port input
    except IndexError:
        port = 12345  # Default port if not provided
    
    print(f'Running on port {port}')
    app.run(port=port, debug=True)
