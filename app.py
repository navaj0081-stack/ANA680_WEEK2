import numpy as np
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

with open("model.pkl", "rb") as f:
    model_data = pickle.load(f)
    model = model_data["model"]
    scaler = model_data["scaler"]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        raw_features = [float(x) for x in request.form.values()]
        input_array = np.array(raw_features).reshape(1, -1)

    # Scale the input values
        scaled_features = scaler.transform(input_array)

    # Make prediction
        prediction = model.predict(scaled_features)
        output = "Malignant" if prediction[0] == 1 else "Benign"
    

    # Return the prediction
        return render_template("index.html", prediction_text=f"Diagnosis Result: {output}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error processing request: {str(e)}"
                           )

if __name__ == "__main__":
    port = init(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
