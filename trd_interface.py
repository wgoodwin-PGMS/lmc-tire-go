from dataclasses import dataclass
from functools import wraps

import copy
import uuid
import logging
import time
import random

import requests
from requests.exceptions import SSLError, ConnectionError, Timeout, RequestException
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
import orjson

logger = logging.getLogger(__name__)


def retry_with_backoff(
    max_retries=10,
    base_delay=3.0,
    max_delay=120.0,
):
    """Decorator for retrying functions with exponential backoff and full jitter."""
    retryable = (
        SSLError,
        ConnectionError,
        Timeout,
        RequestException,
        orjson.JSONDecodeError,
    )

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except retryable as e:
                    last_exception = e

                    if attempt == max_retries - 1:
                        logger.error(
                            f"{func.__name__} failed after {max_retries} attempts: {e}"
                        )
                        raise

                    exp_delay = base_delay * (2**attempt)
                    jitter = random.uniform(0, base_delay)
                    delay = min(exp_delay + jitter, max_delay)

                    logger.warning(
                        f"{func.__name__} attempt {attempt + 1}/{max_retries} failed: {e}. "
                        f"Retrying in {delay:.2f}s"
                    )
                    time.sleep(delay)

            raise last_exception

        return wrapper

    return decorator


def create_resilient_session():
    """Create a requests session with built-in retry logic."""
    session = requests.Session()

    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS"],
        raise_on_status=False,
    )

    adapter = HTTPAdapter(
        max_retries=retry_strategy, pool_connections=10, pool_maxsize=10
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session


@dataclass
class TrdInterface:
    username: str
    password: str

    def __post_init__(self):
        # self.session = requests.Session()
        self.session = create_resilient_session()
        self.cookie = self.apex_login()

    @retry_with_backoff(max_retries=10, base_delay=3)
    def apex_login(self) -> str:
        url = "https://www.apex-mp.com/api/platform/login"

        payload = {
            "username": self.username,
            "password": self.password,
        }

        headers = {"MediaType": "application/json"}

        response = self.session.post(
            url=url, headers=headers, data=payload, timeout=(15, 30)
        )
        response.raise_for_status()

        status = orjson.loads(response.text)

        if "data" in status.keys():
            token = status["data"]["auth"]["access_token"]
            return token

        raise PermissionError("Error Logging into Apex. Check Credentials")

    @retry_with_backoff(max_retries=10, base_delay=3)
    def place_apex_cache(self, payload: dict, sweep_directory=str) -> str:

        url = "https://apex-setup.app.apex-mp.com/api/execution-plans/cache"

        headers = {
            "Content-Type": "application/json",
            "Cookie": f"apex_platform_token={self.cookie}",
        }

        local_payload = copy.deepcopy(payload)

        self.s3_directory = f"sweep/{sweep_directory}"

        for i, module in enumerate(local_payload["data"]["plan"]["$sequence"]):
            if module["module_name"] == "general-exporter":
                module["module_data"]["s3_output_dir"] = self.s3_directory

                local_payload["data"]["plan"]["$sequence"][i] = module

        # with open("updated_payload.json", "w") as f:
        #     json.dump(local_payload, f)

        body = orjson.dumps(local_payload)
        response = self.session.post(
            url=url, headers=headers, data=body, timeout=(15, 30)
        )
        response.raise_for_status()

        status = orjson.loads(response.text)

        if status.get("status") == "error":
            logger.error(f"[TRD Interface] Error sending sweep to Apex | {status}")
            raise RuntimeError("Error Returned From Apex")

        if status.get("status") == 200:
            return status["data"]["session_id"]

        return status

    @retry_with_backoff(max_retries=10, base_delay=3)
    def submit_sweep(
        self,
        unique_id: str,
        population: pd.DataFrame,
        road_course: bool = False,
    ) -> str:

        url = (
            f"https://apex-setup.app.apex-mp.com/api/execution-plans/{unique_id}/sweeps"
        )

        headers = {
            "Content-Type": "application/json",
            "Cookie": f"apex_platform_token={self.cookie}",
        }

        temp_pop = population.copy()

        logger.info(f"    ----- Sending {len(temp_pop)} runs to Apex Sweep")
        col_list = temp_pop.columns.to_list()
        if road_course:
            # logger.debug("===== ROAD COURSE =====")
            for col in temp_pop:
                if col.endswith("_lf"):
                    new_col = col.replace("_lf", "_rf")
                    if new_col not in col_list:
                        temp_pop[new_col] = temp_pop[col]
                if col.endswith("_lr"):
                    new_col = col.replace("_lr", "_rr")
                    if new_col not in col_list:
                        temp_pop[new_col] = temp_pop[col]
                if ".lf." in col:
                    new_col = col.replace(".lf.", ".rf.")
                    if new_col not in col_list:
                        temp_pop[new_col] = temp_pop[col]
                if ".lr." in col:
                    new_col = col.replace(".lr.", ".rr.")
                    if new_col not in col_list:
                        temp_pop[new_col] = temp_pop[col]

        if (
            "vehicle.setup.unhooked_cross_weight" in col_list
            and "vehicle.setup.cross_weight" not in col_list
        ):
            temp_pop["vehicle.setup.cross_weight"] = temp_pop[
                "vehicle.setup.unhooked_cross_weight"
            ]
        elif (
            "vehicle.setup.cross_weight" in col_list
            and "vehicle.setup.unhooked_cross_weight" not in col_list
        ):
            temp_pop["vehicle.setup.unhooked_cross_weight"] = temp_pop[
                "vehicle.setup.cross_weight"
            ]

        metadata_dict = {
            "sim_engine": "gen7-master",
            "sweep_parameters": temp_pop.to_dict("list"),
        }

        body = orjson.dumps(metadata_dict)
        response = self.session.post(
            url=url, headers=headers, data=body, timeout=(15, 30)
        )
        response.raise_for_status()

        status = orjson.loads(response.text)
        # print(send_payload)

        # with open("sweep_payload.json", "w") as f:
        #     json.dump(metadata_dict, f)

        if status.get("status") != 200:
            logger.critical(status)
            raise RuntimeError("[TRD Interface] Error Sending Sweep to Apex")

        return status

    def batch_to_sweep_convert(self, payload_path: str, metric_dict: dict) -> int | int:
        with open(payload_path, "rb") as f:
            payload = orjson.loads(f.read())

        temp_uuid = "bac" + str(uuid.uuid4())[3:]

        payload["session_id"] = temp_uuid
        payload["store_id"] = temp_uuid
        payload["data"]["session_id"] = temp_uuid
        payload["data"]["corr_cancel_id"] = temp_uuid

        plan = payload["data"]["plan"]

        json_sequence = None
        csv_sequence = None
        temp_sequence = []
        for i, module in enumerate(plan["$sequence"]):
            module_name = module["module_name"]
            module_type = module["module_type"]
            if module_type == "dymola" or module_name == "data-bridge":
                temp_sequence.append(module)
            elif module_name == "general-exporter":
                if not module["module_data"]["filename"].startswith(("edf_", "vdf_")):
                    module["module_data"]["client_download"] = False
                    module["module_data"]["s3_output_dir"] = f"sweep/{temp_uuid}"
                    module["module_data"]["upload_to_s3"] = False
                    temp_sequence.append(module)

                    csv_module = copy.deepcopy(module)
                    csv_module["module_data"]["upload_to_s3"] = False
                    csv_module["module_data"]["filename"] = "sweep_results.csv"
                    csv_module["module_data"]["format"] = "batch-export"
                    csv_module["module_data"]["output_channels"] = metric_dict

                    temp_sequence.append(csv_module)
                    json_sequence = i
                    csv_sequence = i + 1

        plan["$sequence"] = temp_sequence

        for i, module in enumerate(plan["$event_handlers"]["on_error"]):
            if "exporter" in module["module_name"]:
                module["module_data"]["client_download"] = False
                plan["$event_handlers"]["on_error"][i] = module

        payload["data"]["plan"] = plan

        with open(payload_path, "wb") as f:
            f.write(orjson.dumps(payload))

        logger.info(f"Converted {payload_path} to Sweep Execution Plan")

        return json_sequence, csv_sequence
