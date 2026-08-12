import joblib
from fastapi.testclient import TestClient
from pathlib import Path

from tech_challenge_fase2.api import create_app
from tech_challenge_fase2.data import load_diagnostic_dataset, split_dataset
from tech_challenge_fase2.llm import DeterministicExplanationProvider
from tech_challenge_fase2.models import build_model


def test_predict_endpoint_with_trained_artifact():
    split = split_dataset(load_diagnostic_dataset())
    model = build_model("logistic_regression")
    model.fit(split.X_train, split.y_train)
    artifact_path = Path("artifacts/models/test_api_model.joblib")
    try:
        joblib.dump(
            {
                "model": model,
                "model_name": "logistic_regression",
                "feature_names": list(split.X_train.columns),
            },
            artifact_path,
        )
        app = create_app(
            artifact_path=artifact_path,
            explanation_provider=DeterministicExplanationProvider(),
        )
        client = TestClient(app)
        row = split.X_test.iloc[0].to_dict()

        assert client.get("/ready").status_code == 200
        response = client.post("/predict", json={"features": row})

        assert response.status_code == 200
        payload = response.json()
        assert payload["predicted_class"] in {"benigno", "maligno"}
        assert payload["explanation_provider"] == "offline-template"
    finally:
        artifact_path.unlink(missing_ok=True)
