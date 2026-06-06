from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load saved model bundle
bundle = joblib.load("Housepricepredictor.pkl")

model = bundle["model"]
scaler = bundle["scaler"]
model_columns = bundle["columns"]

# Same categorical columns used during training
cat_cols = [
    "MSZoning",
    "Utilities",
    "BldgType",
    "Heating",
    "KitchenQual",
    "SaleCondition",
    "LandSlope"
]

# Numerical columns that were scaled during training
important_num_cols = [
    "OverallQual",
    "YearBuilt",
    "YearRemodAdd",
    "TotalBsmtSF",
    "1stFlrSF",
    "GrLivArea",
    "FullBath",
    "TotRmsAbvGrd",
    "GarageCars",
    "GarageArea"
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not isinstance(data, dict):
            return jsonify({
                "error": "Send a valid JSON object"
            }), 400

        # Create dataframe from JSON
        X = pd.DataFrame([data])

        # One-hot encode categorical columns
        X = pd.get_dummies(X, columns=cat_cols)

        # Add missing columns expected by model
        for col in model_columns:
            if col not in X.columns:
                X[col] = 0

        # Keep same column order as training
        X = X.reindex(columns=model_columns, fill_value=0)

        # Scale numerical columns
        cols_to_scale = [
            col for col in important_num_cols
            if col in X.columns
        ]

        X[cols_to_scale] = scaler.transform(X[cols_to_scale])

        # Predict
        prediction = model.predict(X)[0]

        return jsonify({
            "predicted_price": round(float(prediction), 2)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True)