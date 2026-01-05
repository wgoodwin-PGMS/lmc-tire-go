from dataclasses import dataclass

import os
import logging
import time
import random
import math

from datetime import datetime, timezone

import orjson
import boto3
from boto3.s3.transfer import TransferConfig, S3Transfer
from botocore.config import Config as BotoConfig
from botocore.exceptions import WaiterError, ClientError

logger = logging.getLogger(__name__)

_BOTO_CONFIG = BotoConfig(
    retries={"max_attempts": 10, "mode": "standard"},
    max_pool_connections=20,
    tcp_keepalive=True,
)

_SESSION = boto3.session.Session()
_S3_CLIENT = _SESSION.client("s3", config=_BOTO_CONFIG)
_S3_RESOURCE = _SESSION.resource("s3", config=_BOTO_CONFIG)

# Single-thread for tiny/small files (no executor cost)
_SMALL_CFG = TransferConfig(use_threads=False)

# Modest multipart for truly large ones (>10MB)
_BIG_CFG = TransferConfig(
    max_concurrency=6,  # keep modest to reduce CPU/joins
    multipart_threshold=10 * 1024 * 1024,  # only multipart when >10MB
    multipart_chunksize=8 * 1024 * 1024,
    use_threads=True,
)

# Persistent transfer managers (so if we do use threads, they stay alive)
_SMALL_XFER = S3Transfer(_S3_CLIENT, config=_SMALL_CFG)
_BIG_XFER = S3Transfer(_S3_CLIENT, config=_BIG_CFG)

_BIG_SUFFIXES = (".zip", ".pt", ".bin", ".tar", ".gz", ".xz")


def _download_small(bucket: str, key: str, out_path: str):
    resp = _S3_CLIENT.get_object(Bucket=bucket, Key=key)

    with open(out_path, "wb") as f, resp["Body"] as body:
        while True:
            chunk = body.read(1024 * 1024)  # 1 MB chunks
            if not chunk:
                break
            f.write(chunk)


def _pick_download_method(key: str):
    # hard-map by file name for zero HEADs or heuristics
    if key in ("baselineJson.json", "executionPlan.json"):
        return "large"
    else:
        return "small"


def _pick_xfer_for_key(key: str):
    return _BIG_XFER if key.endswith(_BIG_SUFFIXES) else _SMALL_XFER


@dataclass
class AwsInterface:
    sleep_time: int = 5
    max_wait_time: int = 600
    max_hang_time: int = 30

    def __post_init__(self):
        self.s3_resource = _S3_RESOURCE
        self.s3_client = _S3_CLIENT

    def get_s3_object(
        self, bucket_name: str, local_directory: str, s3_file_name: str
    ) -> str:
        os.makedirs(local_directory, exist_ok=True)
        base = s3_file_name.rsplit("/", 1)[-1]
        out = f"{local_directory}/{base}"

        method = _pick_download_method(base)
        if method == "small":
            _download_small(bucket_name, s3_file_name, out)
        else:
            _BIG_XFER.download_file(
                bucket_name, s3_file_name, out
            )  # threaded multipart

        logger.debug(f"File downloaded from S3 to {out}")

        return out

    def upload_file_to_s3(self, s3_path: str, file_path: str):
        bucket = "lmc-nsga-iii-results"
        logger.debug("Uploading %s to %s/%s", file_path, bucket, s3_path)
        logger.info(f"Uploading {file_path} to s3://{bucket}/{s3_path}")
        # choose based on extension; if you prefer exact size, use os.path.getsize here
        xfer = _pick_xfer_for_key(s3_path)
        xfer.upload_file(file_path, bucket, s3_path)

    def wait_for_object(
        self, bucket: str, s3_path: str, delay: int = None, max_attempts: int = None
    ) -> None:

        delay = self.sleep_time if delay is None else delay
        # Convert max_wait_time to attempts if not specified
        if max_attempts is None:
            max_attempts = max(1, math.ceil(self.max_wait_time / delay))

        logger.info(
            f"    [WAIT] Waiting for sweep completion marker (Delay: {delay} sec, MaxAttempts: {max_attempts})..."
        )

        waiter = self.s3_client.get_waiter("object_exists")
        start_time = time.monotonic()
        try:
            waiter.wait(
                Bucket=bucket,
                Key=s3_path,
                WaiterConfig={"Delay": delay, "MaxAttempts": max_attempts},
            )
        except WaiterError as e:
            logger.error(
                f"[AWS INTERFACE] Timed out after {self.max_wait_time} waitng for {s3_path}"
            )
            raise TimeoutError(
                f"Timed out waiting for {s3_path} (Delay: {delay} sec, MaxAttempts: {max_attempts})"
            ) from e
        total_time = math.floor(time.monotonic() - start_time)
        logger.info(f"    Sweep Completed After {total_time} sec")

    def scan_for_object(self, bucket: str, prefix: str, s3_path: str) -> None:
        paginator = self.s3_client.get_paginator("list_objects_v2")

        start_time = time.monotonic()
        stagnation_time = 0
        sweeps_complete = False
        key_count_prior = 0
        while not sweeps_complete:
            total_time = math.floor(time.monotonic() - start_time)

            if total_time > self.max_wait_time and key_count_prior == 0:
                logger.critical(
                    f"  Exiting, generation took too long | {total_time} sec"
                )
                raise TimeoutError(f"No sims were returned after {total_time}")
            elif stagnation_time > self.max_hang_time:
                logger.critical(
                    f"  Exiting, generation took too long | {total_time} sec"
                )
                raise TimeoutError(f"Sims hung after {total_time}")

            key_count = 0
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                # print(json.dumps(page, indent=2, sort_keys=True, default=str))

                key_count += page["KeyCount"]

                if key_count == 0:
                    logger.info(f"   Waiting on Sweep to Start | {total_time} sec")

                elif key_count != 0:
                    for object in page["Contents"]:
                        if object["Key"] == s3_path:
                            sweeps_complete = True

            if sweeps_complete:
                logger.info(f"   Sweep Completed After {total_time} sec")
                break
            elif not sweeps_complete and key_count != 0:
                logger.info(
                    f"    Sweep Running ({key_count} Returned) | Exiting if no new runs after {self.max_hang_time - stagnation_time} sec"
                )
                if key_count > key_count_prior:
                    stagnation_time = 0
                else:
                    stagnation_time += self.sleep_time

                    key_count_prior = key_count

                    time.sleep(self.sleep_time)

    def wait_on_aws_return(
        self, cache: str, s3_directory: str, download_dir: str
    ) -> dict | str:
        bucket_name = "trd-sim-results"
        prefix = f"lmc/3c088896-1005-465b-a484-bda422d71280/sweep/{s3_directory}"

        target_object = f"{prefix}/{cache}_sweep_complete.json"

        target_csv = f"{prefix}/sweep_results.csv"

        logger.debug(f"Target: {target_object}")

        self.wait_for_object(bucket_name, target_object)
        # self.scan_for_object(bucket_name, prefix, target_object)

        sweep_results = self.get_s3_object(bucket_name, download_dir, target_object)

        with open(f"{sweep_results}", "rb") as f:
            sweep_data = orjson.loads(f.read())

        total = sweep_data["total"]
        completed = sweep_data["completed"]
        failed = sweep_data["failed"]
        lost = sweep_data["lost"]

        logger.info(
            f"        Total: {total} | Completed: {completed} | Failed: {failed} | Lost: {lost}"
        )

        max_retries = 10
        base_delay = 2  # initial delay in seconds

        logger.info(f"    [DOWNLOADING] Downloading {target_csv} from {bucket_name}")

        for attempt in range(1, max_retries + 1):
            try:
                sweep_results_csv = self.get_s3_object(
                    bucket_name, download_dir, target_csv
                )
                logger.info(
                    f"    Successfully downloaded {target_csv} on attempt {attempt}"
                )
                break
            except ClientError as e:
                if attempt < max_retries:
                    # Exponential backoff: 2, 4, 8, 16... seconds
                    # Add jitter (0.5x–1.5x) to avoid synchronized retries
                    delay = base_delay * (2 ** (attempt - 1))
                    delay = delay * random.uniform(0.5, 1.5)
                    logger.warning(
                        f"    Attempt {attempt} failed to download {target_csv} from {bucket_name}: {e}. "
                        f"Retrying in {delay:.1f} seconds..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"All {max_retries} attempts failed. Could not retrieve {target_csv} from {bucket_name}."
                    )
                    raise

        return sweep_data, sweep_results_csv

    def upload_folder_to_s3(self, local_folder: str, s3_prefix: str = '', skip_existing: bool = True):
        """Upload folder to S3, optionally skipping unchanged files."""
        bucket = "lmc-nsga-iii-results"

        s3_objects = {}
        if skip_existing:
            logger.info(f"Checking existing files in s3://{bucket}/{s3_prefix}")
            paginator = self.s3_client.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=bucket, Prefix=s3_prefix):
                for obj in page.get('Contents', []):
                    s3_objects[obj['Key']] = obj['LastModified']
        
        uploaded = 0
        skipped = 0
        
        for root, dirs, files in os.walk(local_folder):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, local_folder)
                s3_key = os.path.join(s3_prefix, relative_path).replace("\\", "/")
                
                should_upload = True
                if skip_existing and s3_key in s3_objects:
                    local_mtime = datetime.fromtimestamp(
                        os.path.getmtime(local_path), tz=timezone.utc
                    )
                    if local_mtime <= s3_objects[s3_key]:
                        logger.debug(f"Skipping {local_path} (unchanged)")
                        skipped += 1
                        should_upload = False
                
                if should_upload:
                    logger.info(f"Uploading {local_path} to s3://{bucket}/{s3_key}")
                    xfer = _pick_xfer_for_key(s3_key)
                    xfer.upload_file(local_path, bucket, s3_key)
                    uploaded += 1
        
        logger.info(f"Upload complete! Uploaded: {uploaded}, Skipped: {skipped}")