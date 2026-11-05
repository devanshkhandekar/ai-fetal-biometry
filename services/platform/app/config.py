from __future__ import annotations

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://fetal:fetal_local_password@localhost:5432/fetal_biometry"

    s3_internal_endpoint: str = "http://localhost:9000"
    s3_public_endpoint: str = "http://localhost:9000"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin123"
    s3_region: str = "us-east-1"
    s3_originals_bucket: str = "fetal-originals"
    s3_derived_bucket: str = "fetal-derived"
    s3_reports_bucket: str = "fetal-reports"

    sqs_endpoint: str = "http://localhost:9324"
    sqs_region: str = "us-east-1"
    sqs_access_key: str = "x"
    sqs_secret_key: str = "x"
    inference_queue_name: str = "inference-jobs"
    report_queue_name: str = "report-jobs"

    auth_disabled: bool = False
    oidc_issuer: str = "http://localhost:8081/realms/fetal-local"
    oidc_audience: str = "fetal-web"
    oidc_jwks_url: str = "http://localhost:8081/realms/fetal-local/protocol/openid-connect/certs"
    default_hospital_name: str = "Local Demo Hospital"

    cors_origins: str = "http://localhost:5173"
    api_base_url: str = "http://localhost:8000"

    # mock      -> deterministic Phase-1 demo
    # python    -> import PIPELINE_MODULE / PIPELINE_FUNCTION in the worker process
    # subprocess-> recommended Phase-2 mode; the platform worker remains lightweight
    #              and invokes the already-working model Python environment.
    pipeline_mode: str = "mock"
    pipeline_module: str = "pipeline_devansh"
    pipeline_function: str = "run_inference"
    pipeline_model_release: str = "local-phase0-7"
    model_repo_path: str = "/opt/model_repo"

    pipeline_python: str = ""
    pipeline_bridge_script: str = ""
    pipeline_timeout_sec: int = 300
    enable_study_aggregation: bool = False

    # Kept for backwards compatibility with the Phase-1 scaffold. Phase 3 no
    # longer invokes the study aggregator from API/report containers; it is run
    # by the inference worker and persisted as StudyAggregate instead.
    study_aggregator_module: str = ""
    study_aggregator_function: str = ""

    openai_api_key: str = ""
    openai_model: str = ""

    @property
    def cors_origin_list(self) -> list[str]:
        return [x.strip() for x in self.cors_origins.split(",") if x.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
