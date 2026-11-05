import json
import boto3
from app.config import settings


def sqs_client():
    return boto3.client(
        "sqs",
        endpoint_url=settings.sqs_endpoint or None,
        region_name=settings.sqs_region,
        aws_access_key_id=settings.sqs_access_key,
        aws_secret_access_key=settings.sqs_secret_key,
    )


def queue_url(name: str) -> str:
    return sqs_client().get_queue_url(QueueName=name)["QueueUrl"]


def send(name: str, payload: dict):
    client = sqs_client()
    return client.send_message(QueueUrl=queue_url(name), MessageBody=json.dumps(payload))


def receive(name: str, wait_seconds: int = 10):
    client = sqs_client()
    url = queue_url(name)
    response = client.receive_message(
        QueueUrl=url,
        MaxNumberOfMessages=1,
        WaitTimeSeconds=wait_seconds,
        VisibilityTimeout=180,
    )
    return client, url, (response.get("Messages") or [])
