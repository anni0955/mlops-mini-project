import os
import mlflow

def promote_model():
    mlflow.set_tracking_uri("https://dagshub.com/anni0955/mlops-mini-project.mlflow")

    username = os.getenv("MLFLOW_TRACKING_USERNAME")
    password = os.getenv("MLFLOW_TRACKING_PASSWORD")

    if not username or not password:
        raise EnvironmentError("MLFLOW credentials not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = password

    client = mlflow.MlflowClient()

    model_name = "LR_model"

    versions = client.search_model_versions(f"name='{model_name}'")
    latest_version = sorted(versions, key=lambda v: int(v.version))[-1].version

    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=latest_version
    )

    print(f"Promoted version {latest_version} → @champion")


if __name__ == "__main__":
    promote_model()