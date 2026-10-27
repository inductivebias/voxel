"""Check the PVCAM stack in stages without starting a Voxel instrument.

    vxl-pvcam-diagnose                    # environment, device nodes, runtime, enumerated interfaces
    vxl-pvcam-diagnose --open             # also open each head and read its identity
    vxl-pvcam-diagnose --open --serial S  # succeed only if the head with serial S opens

Opening a head reads its identity and closes it; nothing is acquired. A head held by another process,
such as a running Voxel, is reported as in use rather than as a fault.
"""

import argparse
import os
import stat
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from vxl_drivers.cameras import teledyne

_OPEN_HINT = "Check camera power, allow 30 seconds to boot, then verify PCIe cables and link LEDs."


class DiagnosisError(Exception):
    """A stage of the PVCAM stack failed; the message names the stage and what to check."""


def _device_nodes() -> list[Path]:
    return sorted(Path("/dev").glob("pvcamPCIE_*"))


def diagnose(*, open_heads: bool, serial: str | None, output: TextIO) -> None:
    """Report each stage of the PVCAM stack to ``output``.

    Raises:
        DiagnosisError: PVCAM failed to initialize or enumerate, a head failed to open, or the
            ``serial`` head was not available.
    """

    def say(line: str) -> None:
        output.write(f"{line}\n")

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
        pvc = teledyne.pvcam_runtime()
    except Exception as error:
        raise DiagnosisError(f"PVCAM initialization failed: {teledyne.describe_error(error)}") from error
    say(f"runtime={pvc.get_pvcam_version()}")

    try:
        names = teledyne.list_pvcam_camera_names()
    except Exception as error:
        raise DiagnosisError(f"PVCAM enumeration failed: {teledyne.describe_error(error)}") from error
    say(f"interfaces={', '.join(names) if names else '<none>'}")
    if not names:
        raise DiagnosisError("PVCAM enumerated no interfaces; check the kernel module, device node, and permissions")
    if not open_heads:
        say("head open skipped; pass --open to test communication with the camera head")
        return

    serials: list[str] = []
    failures: list[str] = []
    for name in names:
        try:
            device = teledyne.open_pvcam_device(name)
        except teledyne.PvcamClaimedError as error:
            say(f"head={name} skipped: {error}")
            continue
        except Exception as error:  # report every head before failing
            failures.append(f"{name}: {teledyne.describe_error(error)}")
            continue
        try:
            identity = device.identity
        finally:
            device.close()
        serials.append(identity.serial)
        say(f"head={name} serial={identity.serial} product={identity.product} firmware={identity.firmware}")
        if serial is not None and identity.serial == serial:
            return

    if serial is not None:
        detail = "; ".join(failures) if failures else f"found serials: {', '.join(serials) or '<none>'}"
        raise DiagnosisError(f"expected head serial {serial!r} was not available ({detail})")
    if failures:
        raise DiagnosisError(f"PVCAM could not open a camera head ({'; '.join(failures)}). {_OPEN_HINT}")


def run(argv: Sequence[str], *, output: TextIO, errors: TextIO) -> int:
    """Parse ``argv``, run the diagnosis, and return the process exit status."""
    parser = argparse.ArgumentParser(
        prog="vxl-pvcam-diagnose",
        description="Check the PVCAM stack in stages without starting a Voxel instrument.",
    )
    parser.add_argument(
        "--open", action="store_true", dest="open_heads", help="open each enumerated head and read its identity"
    )
    parser.add_argument("--serial", help="expected head serial; requires --open")
    args = parser.parse_args(argv)
    if args.serial is not None and not args.open_heads:
        parser.error("--serial requires --open")
    try:
        diagnose(open_heads=args.open_heads, serial=args.serial, output=output)
    except DiagnosisError as error:
        errors.write(f"error: {error}\n")
        return 1
    return 0


def main() -> None:
    raise SystemExit(run(sys.argv[1:], output=sys.stdout, errors=sys.stderr))
