
# Import necessary libraries
from pathlib import Path
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize Flask app with a name
superkart_api = Flask("SuperKart")

# Load the trained model pipeline
MODEL_PATH = Path(__file__).resolve().parent / "superkart_model.joblib"
model = joblib.load(MODEL_PATH)

# Expected input features in training order
FEATURES = [
    'Product_Weight',
    'Product_Sugar_Content',
    'Product_Allocated_Area',
    'Product_MRP',
    'Store_Size',
    'Store_Location_City_Type',
    'Store_Type',
    'Product_Id_char',
    'Store_Age_Years',
    'Product_Type_Category',
]


# Define a route for the home page
@superkart_api.get('/')
def home():
    return "Welcome to the SuperKart System"


# Define an endpoint to predict sales for a single product
@superkart_api.post('/v1/predict')
def predict_sales():
    try:
        # Get JSON data from the request
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({
                "error": "Invalid or missing JSON payload."
            }), 400

        # Check for missing features
        missing_features = [col for col in FEATURES if col not in data]

        if missing_features:
            return jsonify({
                "error": f"Missing required features: {missing_features}"
            }), 400

        # Extract relevant features from the input data
        sample = {col: data[col] for col in FEATURES}

        # Convert the extracted data into a DataFrame
        input_data = pd.DataFrame([sample])

        # Make a prediction using the trained model
        prediction = float(model.predict(input_data)[0])

        # Return the prediction as a JSON response
        return jsonify({'Sales': round(prediction, 2)}), 200

    except Exception:
        superkart_api.logger.exception("Single prediction failed")
        return jsonify({
            "error": "Unable to generate sales prediction."
        }), 500


# Define an endpoint to predict sales for a batch of products
@superkart_api.post('/v1/predictbatch')
def predict_sales_batch():
    try:
        # Get the uploaded CSV file from the request
        if 'file' not in request.files:
            return jsonify({
                "error": "No CSV file provided under key 'file'."
            }), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({
                "error": "Uploaded file has an empty filename."
            }), 400

        if not file.filename.lower().endswith('.csv'):
            return jsonify({
                "error": "Only CSV files are supported."
            }), 400

        # Read the uploaded CSV file into a DataFrame
        input_data = pd.read_csv(file)

        if input_data.empty:
            return jsonify({
                "error": "The uploaded CSV file is empty."
            }), 400

        # Check for missing features
        missing_features = [
            col for col in FEATURES if col not in input_data.columns
        ]

        if missing_features:
            return jsonify({
                "error": f"Missing required columns: {missing_features}"
            }), 400

        # Align input columns with training feature order
        input_data = input_data[FEATURES]

        # Make predictions for the batch data
        predictions = model.predict(input_data).tolist()

        # Create an output dictionary mapping row index to predicted sales
        output_dict = {
            str(i): round(float(pred), 2)
            for i, pred in enumerate(predictions)
        }

        return jsonify(output_dict), 200

    except pd.errors.EmptyDataError:
        return jsonify({
            "error": "The uploaded CSV file contains no data."
        }), 400

    except pd.errors.ParserError:
        return jsonify({
            "error": "Unable to parse the uploaded CSV file."
        }), 400

    except Exception:
        superkart_api.logger.exception("Batch prediction failed")
        return jsonify({
            "error": "Unable to process batch predictions."
        }), 500


# Run the Flask app
if __name__ == '__main__':
    superkart_api.run(
        host='0.0.0.0',
        port=7860,
        debug=False
    )
