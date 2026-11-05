from __future__ import annotations
import boto3
from botocore.config import Config
from app.config import settings


def _client(endpoint: str):
    return boto3.client(
        "s3",
        endpoint_url=endpoint or None,
        region_name=settings.s3_region,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


internal_s3 = _client(settings.s3_internal_endpoint)
public_s3 = _client(settings.s3_public_endpoint)


def presign_put(bucket: str, key: str, content_type: str) -> str:
    return public_s3.generate_presigned_url(
        "put_object",
        Params={"Bucket": bucket, "Key": key, "ContentType": content_type},
        ExpiresIn=900,
    )


def presign_get(bucket: str, key: str, expires: int = 3600) -> str:
    return public_s3.generate_presigned_url(
        "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=expires
    )


def upload_file(bucket: str, key: str, local_path: str, content_type: str | None = None):
    extra = {"ContentType": content_type} if content_type else None
    internal_s3.upload_file(local_path, bucket, key, ExtraArgs=extra or {})


def download_file(bucket: str, key: str, local_path: str):
    internal_s3.download_file(bucket, key, local_path)


def head_object(bucket: str, key: str):
    return internal_s3.head_object(Bucket=bucket, Key=key)
