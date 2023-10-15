import requests


SERVICE_URL = "http://localhost:9696/credit_score"
BODY = {"job": "retired", "duration": 445, "poutcome": "success"}


if __name__ == "__main__":
    credit_score = requests.post(url=SERVICE_URL, json=BODY).json()

    print(f"credit score is: {credit_score}")
