"""Check the PVCAM stack in stages without starting a Voxel instrument.

    vxl-pvcam-diagnose                    # environment, device nodes, runtime, enumerated interfaces
    vxl-pvcam-diagnose --open             # also open each head and read its identity
    vxl-pvcam-diagnose --open --serial S  # succeed only if the head with serial S opens
    vxl-pvcam-diagnose --recover          # rescan PCIe and reload pvcam_pcie when no head is in use
    vxl-pvcam-diagnose --recover --open --wait 45  # poll until powered-on heads finish booting

Opening a head reads its identity and closes it; nothing is acquired. A head held by another process,
such as a running Voxel, is reported as in use rather than as a fault.
"""

import argparse
import os
import stat
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from vxl_drivers.cameras import teledyne

_OPEN_HINT = (
    "Check camera power, allow 30 seconds to boot, then run `vxl-pvcam-diagnose --recover --open` "
    "(or verify PCIe cables and link LEDs)."
)
_SYS_PCI_DEVICES = Path("/sys/bus/pci/devices")
_SYS_PCI_RESCAN = Path("/sys/bus/pci/rescan")
_SYS_MODULES = Path("/sys/module")
_UDEVADM_PATHS = (Path("/usr/bin/udevadm"), Path("/bin/udevadm"))
_SETPCI_PATHS = (Path("/usr/bin/setpci"), Path("/bin/setpci"), Path("/sbin/setpci"), Path("/usr/sbin/setpci"))
_PVCAM_MODULE_NAME = "pvcam_pcie"
_KINETIX_VENDOR_ID = "0x1b6b"
_KINETIX_DEVICE_ID = "0x0001"
_PCI_BRIDGE_CLASS = "0x060400"
_PLX_VENDOR_ID = "0x10b5"
_COMMAND_TIMEOUT_S = 15.0
_DEFAULT_RECOVER_WAIT_S = 45.0
_RETRY_POLL_INTERVAL_S = 3.0


class DiagnosisError(Exception):
    """A stage of the PVCAM stack failed; the message names the stage and what to check."""


def _device_nodes() -> list[Path]:
    return sorted(Path("/dev").glob("pvcamPCIE_*"))


def _claimed_interfaces(claim_dir: Path | None = None) -> list[str]:
    """Return PVCAM interface names currently claimed via flock by any process on the host."""
    return teledyne.claimed_pvcam_interfaces(claim_dir)


def _read_sysfs_attr(device_dir: Path, attr: str) -> str:
    try:
        return (device_dir / attr).read_text(encoding="ascii").strip().lower()
    except OSError:
        return ""


def _kinetix_pci_devices(pci_devices_dir: Path = _SYS_PCI_DEVICES) -> list[Path]:
    """Return sysfs directories for PCIe endpoints matching Teledyne Photometrics (1b6b:0001)."""
    if not pci_devices_dir.is_dir():
        return []
    return [
        device_dir
        for device_dir in sorted(pci_devices_dir.iterdir())
        if _read_sysfs_attr(device_dir, "vendor") == _KINETIX_VENDOR_ID
        and _read_sysfs_attr(device_dir, "device") == _KINETIX_DEVICE_ID
    ]


def _parent_pci_bridge(endpoint_dir: Path, pci_devices_dir: Path = _SYS_PCI_DEVICES) -> Path | None:
    """Resolve the upstream PCIe bridge directory in ``pci_devices_dir`` for ``endpoint_dir``."""
    try:
        resolved_parent = endpoint_dir.resolve().parent
    except OSError:
        return None
    candidate = pci_devices_dir / resolved_parent.name
    if candidate != endpoint_dir and candidate.is_dir():
        return candidate
    return None


def _switch_downstream_bridges(endpoints: Sequence[Path], pci_devices_dir: Path = _SYS_PCI_DEVICES) -> list[Path]:
    """Return PCIe bridge ports upstream of Kinetix endpoints or belonging to the PLX/Dolphin switch."""
    bridges: dict[str, Path] = {}
    for endpoint in endpoints:
        parent = _parent_pci_bridge(endpoint, pci_devices_dir)
        if parent is not None:
            bridges[parent.name] = parent
    if not bridges and pci_devices_dir.is_dir():
        for device_dir in sorted(pci_devices_dir.iterdir()):
            if (
                _read_sysfs_attr(device_dir, "vendor") == _PLX_VENDOR_ID
                and _read_sysfs_attr(device_dir, "class") == _PCI_BRIDGE_CLASS
            ):
                bridges[device_dir.name] = device_dir
    return [bridges[name] for name in sorted(bridges)]


def _run_privileged_command(args: Sequence[str], *, input_text: str | None = None) -> None:
    cmd = list(args) if os.geteuid() == 0 else ["sudo", *args]
    try:
        result = subprocess.run(  # noqa: S603 - fixed system utility arguments, no shell.
            cmd,
            input=input_text,
            capture_output=True,
            text=True,
            check=False,
            timeout=_COMMAND_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as error:
        raise DiagnosisError(f"{' '.join(cmd)} timed out after {_COMMAND_TIMEOUT_S:.0f} s") from error
    except OSError as error:
        raise DiagnosisError(f"could not execute {' '.join(cmd)}: {error}") from error
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip() or f"exit status {result.returncode}"
        raise DiagnosisError(f"{' '.join(cmd)} failed: {detail}")


def _write_sysfs_text(path: Path, value: str) -> None:
    if os.geteuid() == 0:
        try:
            path.write_text(f"{value}\n", encoding="ascii")
        except OSError as error:
            raise DiagnosisError(f"cannot write to {path}: {error}") from error
        return
    _run_privileged_command(["tee", str(path)], input_text=f"{value}\n")


def _write_sysfs_one(path: Path) -> None:
    _write_sysfs_text(path, "1")


def _disable_runtime_pm(devices: Sequence[Path], *, output: TextIO) -> None:
    """Keep PCIe switch ports and Kinetix endpoints in D0 (``power/control = on``)."""
    for device_dir in devices:
        control_path = device_dir / "power" / "control"
        if not control_path.is_file():
            continue
        if _read_sysfs_attr(device_dir, "power/control") == "on":
            continue
        with teledyne.suppress(DiagnosisError):
            output.write(f"recover: setting {control_path} to on\n")
            _write_sysfs_text(control_path, "on")


def _retrain_bridge_links(bridges: Sequence[Path], *, output: TextIO, settle_s: float) -> None:
    """Toggle Secondary Bus Reset (Bridge Control 0x3e bit 0x40) on downstream switch bridges."""
    if not bridges or not any(p.is_file() for p in _SETPCI_PATHS):
        return
    for bridge in bridges:
        output.write(f"recover: toggling secondary bus reset on bridge {bridge.name}\n")
        with teledyne.suppress(DiagnosisError):
            _run_privileged_command(["setpci", "-s", bridge.name, "BRIDGE_CONTROL=0x40:0x40"])
            if settle_s > 0:
                time.sleep(min(0.1, settle_s))
            _run_privileged_command(["setpci", "-s", bridge.name, "BRIDGE_CONTROL=0x00:0x40"])
    if settle_s > 0:
        time.sleep(settle_s)


def recover_pvcam_pcie(
    *,
    output: TextIO,
    pci_devices_dir: Path = _SYS_PCI_DEVICES,
    pci_rescan_path: Path = _SYS_PCI_RESCAN,
    modules_dir: Path = _SYS_MODULES,
    settle_s: float = 0.5,
    retrain_bridges: bool = False,
) -> None:
    """Rescan the PCIe bus and reload ``pvcam_pcie`` so newly powered camera heads enumerate.

    Refuses to run while any PVCAM interface is claimed by a running Voxel or other process.
    """

    def say(line: str) -> None:
        output.write(f"{line}\n")

    if sys.platform != "linux":
        raise DiagnosisError("PCIe / pvcam_pcie recovery is only supported on Linux hosts")

    claimed = _claimed_interfaces()
    if claimed:
        raise DiagnosisError(
            f"cannot recover PVCAM while interface(s) are in use: {', '.join(claimed)}; stop Voxel first"
        )

    if teledyne.try_non_interactive_host_recovery(retrain_bridges=retrain_bridges):
        say(f"recover: executed {teledyne.PVCAM_RECOVERY_HELPER}")
        if settle_s > 0:
            time.sleep(settle_s)
        return

    if (modules_dir / _PVCAM_MODULE_NAME).exists():
        say(f"recover: unloading {_PVCAM_MODULE_NAME}")
        _run_privileged_command(["modprobe", "-r", _PVCAM_MODULE_NAME])

    endpoints = _kinetix_pci_devices(pci_devices_dir)
    bridges = _switch_downstream_bridges(endpoints, pci_devices_dir)
    _disable_runtime_pm(bridges, output=output)

    for endpoint in endpoints:
        remove_path = endpoint / "remove"
        if remove_path.exists():
            say(f"recover: removing stale PCIe endpoint {endpoint.name}")
            _write_sysfs_one(remove_path)

    if retrain_bridges or not endpoints:
        _retrain_bridge_links(bridges, output=output, settle_s=settle_s)

    if not pci_rescan_path.exists():
        raise DiagnosisError(f"PCIe rescan path {pci_rescan_path} does not exist")
    say(f"recover: rescanning PCIe bus via {pci_rescan_path}")
    _write_sysfs_one(pci_rescan_path)

    rescanned_endpoints = _kinetix_pci_devices(pci_devices_dir)
    _disable_runtime_pm(rescanned_endpoints, output=output)

    say(f"recover: loading {_PVCAM_MODULE_NAME}")
    _run_privileged_command(["modprobe", _PVCAM_MODULE_NAME])
    if any(p.is_file() for p in _UDEVADM_PATHS):
        _run_privileged_command(["udevadm", "settle", "--timeout=5"])
    if settle_s > 0:
        time.sleep(settle_s)


def _open_and_verify_heads(names: Sequence[str], *, serial: str | None, output: TextIO) -> None:
    serials: list[str] = []
    failures: list[str] = []
    for name in names:
        try:
            device = teledyne.open_pvcam_device(name)
        except teledyne.PvcamClaimedError as error:
            output.write(f"head={name} skipped: {error}\n")
            continue
        except Exception as error:  # report every head before failing
            failures.append(f"{name}: {teledyne.describe_error(error)}")
            continue
        try:
            identity = device.identity
        finally:
            device.close()
        serials.append(identity.serial)
        output.write(f"head={name} serial={identity.serial} product={identity.product} firmware={identity.firmware}\n")
        if serial is not None and identity.serial == serial:
            return

    if serial is not None:
        detail = "; ".join(failures) if failures else f"found serials: {', '.join(serials) or '<none>'}"
        raise DiagnosisError(f"expected head serial {serial!r} was not available ({detail})")
    if failures:
        raise DiagnosisError(f"PVCAM could not open a camera head ({'; '.join(failures)}). {_OPEN_HINT}")


def _diagnose_once(
    *, open_heads: bool, serial: str | None, recover: bool, retrain_bridges: bool, output: TextIO
) -> None:
    def say(line: str) -> None:
        output.write(f"{line}\n")

    if recover:
        recover_pvcam_pcie(output=output, retrain_bridges=retrain_bridges)

    teledyne.ensure_pvcam_environment()
    say(f"PVCAM_UMD_PATH={os.getenv('PVCAM_UMD_PATH', '<unset>')}")
    say(f"PVCAM_VERSION={os.getenv('PVCAM_VERSION', '<unset>')}")

    nodes = _device_nodes()
    for node in nodes:
        details = node.stat()
        access = "rw" if os.access(node, os.R_OK | os.W_OK) else "not-rw"
        say(
            f"device={node} access={access} mode={stat.filemode(details.st_mode)} "
            f"uid={details.st_uid} gid={details.st_gid}"
        )
    if not nodes:
        say("device=<none matching /dev/pvcamPCIE_*>")

    try:
        pvc = teledyne.pvcam_runtime(reinit=recover)
    except Exception as error:
        raise DiagnosisError(f"PVCAM initialization failed: {teledyne.describe_error(error)}") from error
    say(f"runtime={pvc.get_pvcam_version()}")

    try:
        names = teledyne.list_pvcam_camera_names()
    except Exception as error:
        raise DiagnosisError(f"PVCAM enumeration failed: {teledyne.describe_error(error)}") from error
    say(f"interfaces={', '.join(names) if names else '<none>'}")
    if not names:
        raise DiagnosisError(
            "PVCAM enumerated no interfaces; check camera power, run `vxl-pvcam-diagnose --recover`, "
            "or check the kernel module, device node, and permissions"
        )
    if not open_heads:
        say("head open skipped; pass --open to test communication with the camera head")
        return

    _open_and_verify_heads(names, serial=serial, output=output)


def diagnose(
    *,
    open_heads: bool,
    serial: str | None,
    recover: bool = False,
    wait_s: float = 0.0,
    poll_interval_s: float | None = None,
    output: TextIO,
) -> None:
    """Report each stage of the PVCAM stack to ``output``, polling up to ``wait_s`` if requested.

    Raises:
        DiagnosisError: PVCAM failed to initialize or enumerate, a head failed to open, or the
            ``serial`` head was not available.
    """
    interval = _RETRY_POLL_INTERVAL_S if poll_interval_s is None else poll_interval_s
    deadline = time.monotonic() + max(0.0, wait_s)
    attempt = 0
    while True:
        attempt += 1
        try:
            _diagnose_once(
                open_heads=open_heads,
                serial=serial,
                recover=recover,
                retrain_bridges=recover and attempt > 1,
                output=output,
            )
        except DiagnosisError as error:
            remaining = deadline - time.monotonic()
            if (
                remaining <= 0
                or "while interface(s) are in use:" in str(error)
                or "only supported on Linux" in str(error)
            ):
                raise
            sleep_s = min(interval, remaining)
            output.write(f"retry: {error} (waiting {sleep_s:.1f}s for camera head boot...)\n")
            if sleep_s > 0:
                time.sleep(sleep_s)
        else:
            return


def run(argv: Sequence[str], *, output: TextIO, errors: TextIO) -> int:
    """Parse ``argv``, run the diagnosis, and return the process exit status."""
    parser = argparse.ArgumentParser(
        prog="vxl-pvcam-diagnose",
        description="Check the PVCAM stack in stages without starting a Voxel instrument.",
    )
    parser.add_argument(
        "--recover",
        action="store_true",
        help="rescan the PCIe bus and reload pvcam_pcie before diagnosing (requires no claimed head)",
    )
    parser.add_argument(
        "--wait",
        type=float,
        default=None,
        metavar="SECONDS",
        help=(
            f"poll for up to SECONDS until camera heads finish booting "
            f"(defaults to {_DEFAULT_RECOVER_WAIT_S:.0f}s with --recover --open, 0s otherwise)"
        ),
    )
    parser.add_argument(
        "--open", action="store_true", dest="open_heads", help="open each enumerated head and read its identity"
    )
    parser.add_argument("--serial", help="expected head serial; requires --open")
    args = parser.parse_args(argv)
    if args.serial is not None and not args.open_heads:
        parser.error("--serial requires --open")
    if args.wait is not None and args.wait < 0:
        parser.error("--wait must be non-negative")
    wait_s = (
        args.wait if args.wait is not None else (_DEFAULT_RECOVER_WAIT_S if (args.recover and args.open_heads) else 0.0)
    )
    try:
        diagnose(open_heads=args.open_heads, serial=args.serial, recover=args.recover, wait_s=wait_s, output=output)
    except DiagnosisError as error:
        errors.write(f"error: {error}\n")
        return 1
    return 0


def main() -> None:
    raise SystemExit(run(sys.argv[1:], output=sys.stdout, errors=sys.stderr))
