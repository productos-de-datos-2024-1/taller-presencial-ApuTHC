# -----------------------------------------------------------------------------
# API Server
# -----------------------------------------------------------------------------

app = Flask(__name__)
app.config["SECRET_KEY"] = "api-server-secret-key"

# Model features used for prediction
FEATURES = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "condition",
]


@app.route("/", methods=["POST"])
def index():
    """API function"""

    # model input
    args = request.json
    filt_args = {key: [int(args[key])] for key in FEATURES}
    df = pd.DataFrame.from_dict(filt_args)
    logging.info("-" * 40)
    logging.info("User values: %s", filt_args)

    # prediction
    path = os.path.join(config["models_dir"], config["house_prices_model"])
    with open(path, "rb") as file:
        loaded_model = pickle.load(file)
    prediction = loaded_model.predict(df)
    logging.info("Prediction: %s", prediction)

    # result
    return str(prediction[0])


if __name__ == "__main__":
    logging.info("Starting API server")
    app.run(debug=True, port=5000, host="0.0.0.0")
    logging.info("Finishing API server")



