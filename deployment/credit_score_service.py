import os
import pickle
from typing import Any

from flask import Flask
from flask import jsonify
from flask import request


def _get_credit_score_object(destination_name: str) -> Any:
    object_file_name = os.path.join(os.getcwd(), destination_name)

    with open(object_file_name, "rb") as object_binary_file:
        score_object = pickle.load(object_binary_file)

    return score_object


def get_model() -> tuple[Any, Any]:
    model = _get_credit_score_object("model2.bin")
    dv = _get_credit_score_object("dv.bin")

    return model, dv


app = Flask("credit_score")

score_model, score_dv = get_model()


@app.route("/credit_score", methods=["POST"])
def get_credit_score():
    score_request = request.get_json()

    # score_request = {"job": "retired", "duration": 445, "poutcome": "success"}

    features = score_dv.transform(score_request)
    prediction = score_model.predict_proba(features)[0, 1]

    result = {
        "prediction": prediction
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
    #
    # print(f"{get_credit_score()}")
