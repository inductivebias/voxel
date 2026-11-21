import json
import socket
from abc import ABC, abstractmethod
from pathlib import Path

_HOST_IP: str = "localhost"
_HOST_NAME: str = socket.gethostname()


def neuroglancer_compatible_zarr_json(file_path: Path) -> bytes | None:
    """Return an OME-Zarr 0.5-compatible ``zarr.json`` payload for Neuroglancer if ``file_path``
    is a Zarr v3 group with OME-NGFF 0.6 metadata, or ``None`` if no rewrite is needed."""
    if file_path.name != "zarr.json":
        return None
    try:
        document = json.loads(file_path.read_text(encoding="utf-8"))
        ome = document.get("attributes", {}).get("ome", {})
        if document.get("node_type") != "group" or ome.get("version") != "0.6":
            return None
        old_multiscales = []
        for multiscale in ome.get("multiscales", []):
            systems = multiscale.get("coordinateSystems", [])
            axes = systems[0].get("axes", []) if systems else []
            old_datasets = []
            for dataset in multiscale.get("datasets", []):
                sequence = next(
                    (ct for ct in dataset.get("coordinateTransformations", []) if ct.get("type") == "sequence"), {}
                )
                transforms = [
                    t
                    for t in sequence.get("transformations", [])
                    if t.get("type") in ("scale", "translation", "identity")
                ]
                old_datasets.append({"path": dataset["path"], "coordinateTransformations": transforms})
            old_multiscales.append(
                {"version": "0.5", "axes": axes, "datasets": old_datasets, "type": multiscale.get("type", "")}
            )
        ome["multiscales"] = old_multiscales
        ome["version"] = "0.5"
        return json.dumps(document, separators=(",", ":")).encode("utf-8")
    except (OSError, ValueError, KeyError, TypeError):
        return None


def get_host_name() -> str:
    return _HOST_NAME


def get_host_ip() -> str:
    global _HOST_IP  # noqa: PLW0603 - process-local lazy cache

    if _HOST_IP == "localhost":
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            if ip is not None:
                _HOST_IP = ip
        except Exception:
            return _HOST_IP

    return _HOST_IP


class ZarrServer(ABC):
    """
    Abstract base class for servers that serve Zarr datasets to Neuroglancer.

    Both FastAPI and httpd-based implementations should inherit from this class
    and implement the required methods.
    """

    @property
    @abstractmethod
    def port(self) -> int:
        """Server port number."""
        ...

    @abstractmethod
    def start(self) -> None:
        """Start the server."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Stop the server."""
        ...

    @abstractmethod
    def get_url_for_zarr(self, zarr_path: Path, use_localhost: bool = True) -> str:
        """Get HTTP URL for a Zarr directory.

        Args:
            zarr_path: Path to Zarr directory
            use_localhost: If True, use localhost; if False, use host IP

        Returns:
            HTTP URL string
        """
        ...
