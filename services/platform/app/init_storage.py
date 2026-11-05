from __future__ import annotations

import sys
import time

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from app.config import settings


def make_client():
    return boto3.client(
        "s3",
        endpoint_url=settings.s3_internal_endpoint,
        region_name=settings.s3_region,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
        ),
    )


def wait_until_ready(client, attempts: int = 60, sleep_seconds: int = 2):
    print(f"[storage-init] waiting for {settings.s3_internal_endpoint}", flush=True)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            client.list_buckets()
            print("[storage-init] object storage is ready", flush=True)
            return
        except Exception as exc:  # bootstrap retry: connection/auth may not be ready yet
            last_error = exc
            print(f"[storage-init] attempt {attempt}/{attempts}: {type(exc).__name__}: {exc}", flush=True)
            time.sleep(sleep_seconds)
    print(f"[storage-init] storage never became ready: {last_error}", file=sys.stderr)
    raise SystemExit(1)


def ensure_bucket(client, bucket: str):
    try:
        client.head_bucket(Bucket=bucket)
        print(f"[storage-init] exists: {bucket}", flush=True)
        return
    except Exception:
        pass

    try:
        if settings.s3_region == "us-east-1":
            client.create_bucket(Bucket=bucket)
        else:
            client.create_bucket(
                Bucket=bucket,
                CreateBucketConfiguration={"LocationConstraint": settings.s3_region},
            )
        print(f"[storage-init] created: {bucket}", flush=True)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in {"BucketAlreadyExists", "BucketAlreadyOwnedByYou"}:
            print(f"[storage-init] exists: {bucket}", flush=True)
            return
        raise


def main():
    client = make_client()
    wait_until_ready(client)
    for bucket in (
        settings.s3_originals_bucket,
        settings.s3_derived_bucket,
        settings.s3_reports_bucket,
    ):
        ensure_bucket(client, bucket)
    print("[storage-init] all required buckets are ready", flush=True)


if __name__ == "__main__":
    main()
