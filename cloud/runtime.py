from __future__ import annotations

import json
import math
import os
import pathlib
import shutil
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import yaml

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    requests = None


def _strtobool(value: Optional[str]) -> bool:
    if value is None:
        return False
    value = value.strip().lower()
    return value in {"1", "true", "t", "yes", "y", "on"}


def _load_overrides_from_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp.read())
    return data or {}


def _load_overrides_from_string(raw: str) -> Dict[str, Any]:
    raw = raw.strip()
    if not raw:
        return {}
    if raw.startswith("{"):
        return json.loads(raw)

    overrides: Dict[str, Any] = {}
    for chunk in raw.split(","):
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(
                "Override chunks must be key=value pairs when using the compact syntax"
            )
        key, value = chunk.split("=", 1)
        key = key.strip()
        literal = value.strip()
        overrides[key] = yaml.safe_load(literal)
    return overrides


def _deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in overrides.items():
        if (
            isinstance(value, dict)
            and isinstance(base.get(key), dict)
        ):
            base[key] = _deep_merge(dict(base[key]), value)
        else:
            base[key] = value
    return base


@dataclass
class CloudTrainingRequest:
    """Represents intent inferred from environment variables."""

    enabled: bool = False
    provider: str = "generic"
    autostart: bool = False
    overrides: Dict[str, Any] = field(default_factory=dict)
    checkpoint_uri: Optional[str] = None
    resume_checkpoint_uri: Optional[str] = None
    checkpoint_interval_epochs: Optional[int] = None
    termination_notice_seconds: Optional[int] = 30
    poll_interval_seconds: float = 5.0


class StorageClient:
    def upload(self, local_path: str, remote_path: str) -> None:
        raise NotImplementedError

    def download(self, remote_path: str, local_path: str) -> None:
        raise NotImplementedError


class LocalStorageClient(StorageClient):
    def upload(self, local_path: str, remote_path: str) -> None:
        remote = pathlib.Path(remote_path)
        remote.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, remote)

    def download(self, remote_path: str, local_path: str) -> None:
        src = pathlib.Path(remote_path)
        if not src.exists():
            raise FileNotFoundError(remote_path)
        pathlib.Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, local_path)


class GCSStorageClient(StorageClient):
    def __init__(self) -> None:
        try:  # pragma: no cover - exercised in cloud environments
            from google.cloud import storage
        except Exception as exc:  # pragma: no cover - handled at runtime
            raise RuntimeError(
                "google-cloud-storage is required to use gs:// URIs"
            ) from exc
        self._client = storage.Client()

    def _split(self, uri: str) -> tuple[str, str]:
        parsed = urlparse(uri)
        bucket = parsed.netloc
        blob_name = parsed.path.lstrip("/")
        if not bucket or not blob_name:
            raise ValueError(f"Invalid GCS URI: {uri}")
        return bucket, blob_name

    def upload(self, local_path: str, remote_path: str) -> None:
        bucket_name, blob_name = self._split(remote_path)
        bucket = self._client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(local_path)

    def download(self, remote_path: str, local_path: str) -> None:
        bucket_name, blob_name = self._split(remote_path)
        bucket = self._client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        pathlib.Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(local_path)


def _client_for_uri(uri: Optional[str]) -> Optional[StorageClient]:
    if not uri:
        return None
    parsed = urlparse(uri)
    if parsed.scheme in ("", "file"):
        return LocalStorageClient()
    if parsed.scheme == "gs":
        return GCSStorageClient()
    raise ValueError(f"Unsupported storage scheme: {parsed.scheme}")


class CloudCheckpointManager:
    def __init__(self, request: CloudTrainingRequest) -> None:
        self._request = request
        self._remote_template = request.checkpoint_uri
        self._initial_uri = request.resume_checkpoint_uri
        self._storage_client = _client_for_uri(self._remote_template)
        self._steps_per_epoch: Optional[float] = None
        self._next_epoch_trigger: Optional[float] = None
        self._lock = threading.Lock()

    def set_steps_per_epoch(self, steps_per_epoch: float) -> None:
        self._steps_per_epoch = max(steps_per_epoch, 1.0)
        if self._request.checkpoint_interval_epochs:
            self._next_epoch_trigger = float(self._request.checkpoint_interval_epochs)

    def _format_remote_dir(self, metadata: Dict[str, Any]) -> str:
        template = self._remote_template
        if not template:
            raise ValueError("Remote checkpoint URI not configured")
        if "{" in template:
            safe_metadata = {k: ("" if v is None else v) for k, v in metadata.items()}
            try:
                template = template.format(**safe_metadata)
            except KeyError:
                pass
        return template.rstrip("/")

    def build_remote_path(self, metadata: Dict[str, Any]) -> Optional[str]:
        if not self._remote_template:
            return None
        remote_dir = self._format_remote_dir(metadata)
        filename = metadata.get("filename")
        if not filename:
            step = metadata.get("step", 0)
            epoch = metadata.get("epoch")
            reason = metadata.get("reason", "periodic")
            filename = f"checkpoint-step-{step:07d}"
            if epoch is not None:
                filename += f"-epoch-{int(epoch):04d}"
            filename += f"-{reason}.pt"
        return f"{remote_dir}/{filename}"

    def upload(self, local_path: str, metadata: Dict[str, Any]) -> None:
        if not local_path or not self._storage_client:
            return
        remote_path = self.build_remote_path(metadata)
        if remote_path is None:
            return
        with self._lock:
            self._storage_client.upload(local_path, remote_path)

    def download_initial(self, destination_dir: str) -> Optional[str]:
        if not self._initial_uri:
            return None
        client = _client_for_uri(self._initial_uri)
        if client is None:
            return self._initial_uri
        parsed = urlparse(self._initial_uri)
        filename = os.path.basename(parsed.path)
        if not filename:
            filename = "init_checkpoint.pt"
        local_path = os.path.join(destination_dir, filename)
        client.download(self._initial_uri, local_path)
        return local_path

    def should_trigger(self, step: int) -> Optional[int]:
        if (
            self._request.checkpoint_interval_epochs is None
            or self._steps_per_epoch is None
            or self._steps_per_epoch <= 0
        ):
            return None
        if self._next_epoch_trigger is None:
            self._next_epoch_trigger = float(self._request.checkpoint_interval_epochs)

        epochs_completed = max(step, 0) / self._steps_per_epoch
        if epochs_completed + 1e-9 >= self._next_epoch_trigger:
            epoch_to_save = math.ceil(self._next_epoch_trigger)
            self._next_epoch_trigger += self._request.checkpoint_interval_epochs
            return epoch_to_save
        return None

    def infer_epoch(self, step: int) -> Optional[int]:
        if not self._steps_per_epoch:
            return None
        if step <= 0:
            return 0
        return max(1, int(math.ceil(step / self._steps_per_epoch)))


class SpotTerminationWatcher:
    _PREEMPT_URL = "http://metadata.google.internal/computeMetadata/v1/instance/preempted"
    _MAINTENANCE_URL = (
        "http://metadata.google.internal/computeMetadata/v1/instance/maintenance-event"
    )

    def __init__(self, event: threading.Event, poll_interval: float = 5.0) -> None:
        self._event = event
        self._poll_interval = poll_interval
        self._reason: Optional[str] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def reason(self) -> Optional[str]:
        return self._reason

    def start(self) -> None:
        if requests is None:
            return
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0)

    def _run(self) -> None:
        headers = {"Metadata-Flavor": "Google"}
        while not self._stop.is_set():
            try:
                preempt_resp = requests.get(  # type: ignore[arg-type]
                    self._PREEMPT_URL, headers=headers, timeout=1
                )
                if preempt_resp.text.strip().upper() == "TRUE":
                    self._reason = "preemption"
                    self._event.set()
                    return
                maintenance_resp = requests.get(  # type: ignore[arg-type]
                    self._MAINTENANCE_URL, headers=headers, timeout=1
                )
                if maintenance_resp.text.strip() == "TERMINATE_ON_HOST_MAINTENANCE":
                    self._reason = "maintenance"
                    self._event.set()
                    return
            except Exception:
                pass
            time.sleep(self._poll_interval)


class CloudRuntimeContext:
    def __init__(self, request: CloudTrainingRequest) -> None:
        self.request = request
        self.checkpoint_manager = CloudCheckpointManager(request)
        self._termination_event = threading.Event()
        self._termination_reason: Optional[str] = None
        self._watchers: list[SpotTerminationWatcher] = []
        self._metadata: Dict[str, Any] = {}
        self._steps_per_epoch_value: Optional[float] = None

    def bind_config(self, project_name: str, run_name: str) -> None:
        self._metadata["project"] = project_name
        self._metadata["run"] = run_name

    def bind_steps(self, total_steps: int, epochs: int) -> None:
        if epochs <= 0:
            return
        steps_per_epoch = max(total_steps / epochs, 1.0)
        self.checkpoint_manager.set_steps_per_epoch(steps_per_epoch)
        self._steps_per_epoch_value = steps_per_epoch

    def prepare_initial_checkpoint(self, base_dir: str) -> Optional[str]:
        os.makedirs(base_dir, exist_ok=True)
        return self.checkpoint_manager.download_initial(base_dir)

    def should_save_periodic(self, step: int) -> Optional[int]:
        return self.checkpoint_manager.should_trigger(step)

    def handle_checkpoint(self, path: Optional[str], *, step: int, epoch: Optional[int], reason: str) -> None:
        if not path:
            return
        metadata = dict(self._metadata)
        metadata.update({"step": step, "epoch": epoch, "reason": reason})
        metadata.setdefault("filename", os.path.basename(path))
        self.checkpoint_manager.upload(path, metadata)

    def should_stop(self) -> bool:
        return self._termination_event.is_set()

    def termination_reason(self) -> Optional[str]:
        if self._termination_reason:
            return self._termination_reason
        for watcher in self._watchers:
            if watcher.reason:
                return watcher.reason
        return None

    def estimate_epoch(self, step: int) -> Optional[int]:
        return self.checkpoint_manager.infer_epoch(step)

    def start_watchers(self) -> None:
        if self.request.provider.lower() != "gcp":
            return
        watcher = SpotTerminationWatcher(
            self._termination_event,
            poll_interval=self.request.poll_interval_seconds,
        )
        watcher.start()
        self._watchers.append(watcher)

    def stop_watchers(self) -> None:
        for watcher in self._watchers:
            watcher.stop()
        self._watchers.clear()


_ACTIVE_CONTEXT: Optional[CloudRuntimeContext] = None


def set_active_context(context: Optional[CloudRuntimeContext]) -> None:
    global _ACTIVE_CONTEXT
    _ACTIVE_CONTEXT = context


def get_active_context() -> Optional[CloudRuntimeContext]:
    return _ACTIVE_CONTEXT


def _collect_request_from_env() -> Optional[CloudTrainingRequest]:
    env = os.environ
    overrides: Dict[str, Any] = {}
    overrides_path = env.get("CLOUD_TRAINING_OVERRIDES_FILE")
    inline_overrides = env.get("CLOUD_TRAINING_OVERRIDES")

    if overrides_path:
        overrides = _deep_merge(overrides, _load_overrides_from_file(overrides_path))
    if inline_overrides:
        overrides = _deep_merge(overrides, _load_overrides_from_string(inline_overrides))

    checkpoint_uri = env.get("CLOUD_CHECKPOINT_URI")
    resume_uri = env.get("CLOUD_INITIAL_CHECKPOINT_URI")
    interval = env.get("CLOUD_CHECKPOINT_INTERVAL_EPOCHS")

    autostart = _strtobool(env.get("CLOUD_TRAINING_AUTOSTART"))
    provider = env.get("CLOUD_PROVIDER", "generic")
    poll_interval = float(env.get("CLOUD_POLL_INTERVAL_SECONDS", 5.0))
    termination_notice = env.get("CLOUD_TERMINATION_NOTICE_SECONDS")
    termination_notice_seconds = (
        int(termination_notice) if termination_notice else 30
    )

    relevant = any(
        [
            overrides,
            checkpoint_uri,
            resume_uri,
            interval,
            autostart,
        ]
    )
    if not relevant:
        return None

    request = CloudTrainingRequest(
        enabled=True,
        provider=provider,
        autostart=autostart,
        overrides=overrides,
        checkpoint_uri=checkpoint_uri,
        resume_checkpoint_uri=resume_uri,
        poll_interval_seconds=poll_interval,
        termination_notice_seconds=termination_notice_seconds,
    )
    if interval:
        request.checkpoint_interval_epochs = int(interval)
    return request


def build_cloud_context_from_env() -> Optional[CloudRuntimeContext]:
    request = _collect_request_from_env()
    if not request:
        return None
    return CloudRuntimeContext(request)


def apply_cloud_overrides(config: Any, context: Optional[CloudRuntimeContext]) -> Any:
    if context is None or not context.request.overrides:
        return config
    config_dict = config.model_dump()
    merged = _deep_merge(config_dict, context.request.overrides)
    return config.__class__(**merged)
