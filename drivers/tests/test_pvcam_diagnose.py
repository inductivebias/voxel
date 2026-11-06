import io
import subprocess
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest
from vxl_drivers.cameras import pvcam_diagnose, teledyne


def install_fake_pvcam(
    monkeypatch: pytest.MonkeyPatch, names: list[str], open_device: Callable[[str], SimpleNamespace]
) -> None:
    pvc = SimpleNamespace(get_pvcam_version=lambda: "3.10.3.6")
    monkeypatch.setattr(teledyne, "ensure_pvcam_environment", lambda: None)
    monkeypatch.setattr(teledyne, "pvcam_runtime", lambda **_kwargs: pvc)
    monkeypatch.setattr(teledyne, "list_pvcam_camera_names", lambda **_kwargs: names)
    monkeypatch.setattr(teledyne, "open_pvcam_device", open_device)
    monkeypatch.setattr(pvcam_diagnose, "_device_nodes", list)


def identity(serial: str) -> teledyne.PvcamIdentity:
    return teledyne.PvcamIdentity(
        serial=serial, product="Kinetix", chip="KINETIX", firmware="1.5", pvcam_version="3.10.3"
    )


def run(*argv: str) -> tuple[int, str, str]:
    output, errors = io.StringIO(), io.StringIO()
    status = pvcam_diagnose.run(argv, output=output, errors=errors)
    return status, output.getvalue(), errors.getvalue()


def test_run_without_open_never_opens_a_head(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_open(name: str) -> SimpleNamespace:
        raise AssertionError(f"passive diagnostics opened {name}")

    install_fake_pvcam(monkeypatch, ["pvcamPCIE_0"], fail_open)

    status, output, _ = run()

    assert status == 0
    assert "runtime=3.10.3.6" in output
    assert "interfaces=pvcamPCIE_0" in output
    assert "head open skipped" in output


def test_run_open_transport_failure_exits_nonzero_with_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_open(name: str) -> SimpleNamespace:
        raise RuntimeError(f"{name}: PL_ERR_DDI_DEVICE_WRITE_FAILED")

    install_fake_pvcam(monkeypatch, ["pvcamPCIE_0"], fail_open)

    status, _, errors = run("--open")

    assert status == 1
    assert "PL_ERR_DDI_DEVICE_WRITE_FAILED" in errors
    assert "allow 30 seconds to boot" in errors


def test_run_open_partial_failure_reports_both_heads_and_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    closed: list[str] = []

    def open_device(name: str) -> SimpleNamespace:
        if name == "pvcamPCIE_1":
            raise RuntimeError(f"{name}: PL_ERR_DDI_DEVICE_WRITE_FAILED")
        return SimpleNamespace(identity=identity("KX00000001"), close=lambda: closed.append(name))

    install_fake_pvcam(monkeypatch, ["pvcamPCIE_0", "pvcamPCIE_1"], open_device)

    status, output, errors = run("--open")

    assert status == 1, "a head that failed to open must not report success"
    assert "head=pvcamPCIE_0 serial=KX00000001" in output
    assert "PL_ERR_DDI_DEVICE_WRITE_FAILED" in errors
    assert closed == ["pvcamPCIE_0"]


def test_run_open_head_held_elsewhere_is_reported_not_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    def open_device(name: str) -> SimpleNamespace:
        if name == "pvcamPCIE_0":
            raise teledyne.PvcamClaimedError(f"PVCAM interface {name} is in use by another Kinetix or process")
        return SimpleNamespace(identity=identity("KX00000002"), close=lambda: None)

    install_fake_pvcam(monkeypatch, ["pvcamPCIE_0", "pvcamPCIE_1"], open_device)

    status, output, _ = run("--open")

    assert status == 0
    assert "head=pvcamPCIE_0 skipped: PVCAM interface pvcamPCIE_0 is in use" in output
    assert "head=pvcamPCIE_1 serial=KX00000002" in output


def test_run_open_serial_succeeds_only_when_that_head_opens(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_pvcam(
        monkeypatch, ["pvcamPCIE_0"], lambda _name: SimpleNamespace(identity=identity("KX00000001"), close=lambda: None)
    )

    assert run("--open", "--serial", "KX00000001")[0] == 0
    status, _, errors = run("--open", "--serial", "KX00000009")
    assert status == 1
    assert "expected head serial 'KX00000009' was not available (found serials: KX00000001)" in errors


def test_run_serial_without_open_is_a_usage_error(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_pvcam(monkeypatch, ["pvcamPCIE_0"], lambda name: pytest.fail(f"opened {name}"))

    with pytest.raises(SystemExit) as exited:
        run("--serial", "KX00000001")

    assert exited.value.code == 2


def test_recover_pvcam_pcie_rescans_kinetix_endpoints_and_reloads_module(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    udevadm = tmp_path / "udevadm"
    udevadm.write_text("")
    setpci = tmp_path / "setpci"
    setpci.write_text("")
    monkeypatch.setattr(pvcam_diagnose.sys, "platform", "linux")
    monkeypatch.setattr(pvcam_diagnose.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(pvcam_diagnose, "_UDEVADM_PATHS", (udevadm,))
    monkeypatch.setattr(pvcam_diagnose, "_SETPCI_PATHS", (setpci,))
    monkeypatch.setattr(teledyne, "PVCAM_CLAIM_DIR", tmp_path / "locks")
    monkeypatch.setattr(teledyne, "try_non_interactive_host_recovery", lambda **_kwargs: False)
    (tmp_path / "locks").mkdir()

    pci_dir = tmp_path / "pci"
    bridge_dir = pci_dir / "0000:12:01.0"
    kinetix_ep = bridge_dir / "0000:13:00.0"
    kinetix_ep.mkdir(parents=True)
    (pci_dir / "0000:13:00.0").symlink_to(kinetix_ep)
    (bridge_dir / "vendor").write_text("0x10b5\n")
    (bridge_dir / "class").write_text("0x060400\n")
    (bridge_dir / "power").mkdir()
    (bridge_dir / "power" / "control").write_text("auto\n")
    (kinetix_ep / "vendor").write_text("0x1b6b\n")
    (kinetix_ep / "device").write_text("0x0001\n")
    (kinetix_ep / "remove").write_text("0\n")
    (kinetix_ep / "power").mkdir()
    (kinetix_ep / "power" / "control").write_text("auto\n")

    other_ep = pci_dir / "0000:c1:00.0"
    other_ep.mkdir(parents=True)
    (other_ep / "vendor").write_text("0x1093\n")
    (other_ep / "device").write_text("0xc4c4\n")
    (other_ep / "remove").write_text("0\n")

    rescan_path = tmp_path / "rescan"
    rescan_path.write_text("0\n")
    modules_dir = tmp_path / "modules"
    (modules_dir / "pvcam_pcie").mkdir(parents=True)

    commands: list[tuple[tuple[str, ...], str | None]] = []

    def fake_run(args: list[str] | tuple[str, ...], **kwargs: object) -> subprocess.CompletedProcess[str]:
        timeout = kwargs["timeout"]
        assert isinstance(timeout, int | float)
        assert timeout > 0
        input_text = kwargs.get("input")
        commands.append((tuple(args), str(input_text) if input_text is not None else None))
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(pvcam_diagnose.subprocess, "run", fake_run)

    output = io.StringIO()
    pvcam_diagnose.recover_pvcam_pcie(
        output=output,
        pci_devices_dir=pci_dir,
        pci_rescan_path=rescan_path,
        modules_dir=modules_dir,
        settle_s=0.0,
        retrain_bridges=True,
    )

    assert commands == [
        (("sudo", "modprobe", "-r", "pvcam_pcie"), None),
        (("sudo", "tee", str(bridge_dir / "power" / "control")), "on\n"),
        (("sudo", "tee", str(pci_dir / "0000:13:00.0" / "remove")), "1\n"),
        (("sudo", "setpci", "-s", "0000:12:01.0", "BRIDGE_CONTROL=0x40:0x40"), None),
        (("sudo", "setpci", "-s", "0000:12:01.0", "BRIDGE_CONTROL=0x00:0x40"), None),
        (("sudo", "tee", str(rescan_path)), "1\n"),
        (("sudo", "tee", str(pci_dir / "0000:13:00.0" / "power" / "control")), "on\n"),
        (("sudo", "modprobe", "pvcam_pcie"), None),
        (("sudo", "udevadm", "settle", "--timeout=5"), None),
    ]
    log = output.getvalue()
    assert "unloading pvcam_pcie" in log
    assert "0000:13:00.0" in log
    assert "0000:12:01.0" in log
    assert "0000:c1:00.0" not in log


def test_recover_pvcam_pcie_refuses_when_interface_is_claimed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(pvcam_diagnose.sys, "platform", "linux")
    monkeypatch.setattr(teledyne, "PVCAM_CLAIM_DIR", tmp_path)
    monkeypatch.setattr(
        pvcam_diagnose.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("must not run privileged commands while a head is claimed"),
    )

    claim = teledyne.PvcamClaim("pvcamPCIE_0")
    try:
        with pytest.raises(pvcam_diagnose.DiagnosisError, match="in use: pvcamPCIE_0"):
            pvcam_diagnose.recover_pvcam_pcie(output=io.StringIO(), settle_s=0.0)
    finally:
        claim.release()


def test_run_recover_invokes_recovery_and_reinitializes_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    recovered: list[bool] = []
    reinit_flags: list[bool] = []
    pvc = SimpleNamespace(get_pvcam_version=lambda: "3.10.3.6")
    monkeypatch.setattr(teledyne, "ensure_pvcam_environment", lambda: None)
    monkeypatch.setattr(teledyne, "pvcam_runtime", lambda *, reinit=False: (reinit_flags.append(reinit), pvc)[1])
    monkeypatch.setattr(teledyne, "list_pvcam_camera_names", lambda **_kwargs: ["pvcamPCIE_0"])
    monkeypatch.setattr(pvcam_diagnose, "_device_nodes", list)
    monkeypatch.setattr(
        pvcam_diagnose,
        "recover_pvcam_pcie",
        lambda *, output, retrain_bridges=False: (
            recovered.append(retrain_bridges),
            output.write("recovered\n"),
        )[0],
    )

    status, output, errors = run("--recover")

    assert status == 0, errors
    assert recovered == [False]
    assert reinit_flags == [True]
    assert "recovered" in output


def test_run_recover_open_polls_with_bridge_retrain_until_head_boots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retrain_calls: list[bool] = []
    attempts = 0
    pvc = SimpleNamespace(get_pvcam_version=lambda: "3.10.3.6")
    monkeypatch.setattr(teledyne, "ensure_pvcam_environment", lambda: None)
    monkeypatch.setattr(teledyne, "pvcam_runtime", lambda **_kwargs: pvc)
    monkeypatch.setattr(teledyne, "list_pvcam_camera_names", lambda **_kwargs: ["pvcamPCIE_0"])
    monkeypatch.setattr(pvcam_diagnose, "_device_nodes", list)
    monkeypatch.setattr(pvcam_diagnose, "_RETRY_POLL_INTERVAL_S", 0.01)
    monkeypatch.setattr(
        pvcam_diagnose,
        "recover_pvcam_pcie",
        lambda *, output, retrain_bridges=False: (
            retrain_calls.append(retrain_bridges),
            output.write("recover\n"),
        )[0],
    )

    def open_device(name: str) -> SimpleNamespace:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError(f"{name}: PL_ERR_DDI_DEVICE_WRITE_FAILED")
        return SimpleNamespace(identity=identity("KX00000001"), close=lambda: None)

    monkeypatch.setattr(teledyne, "open_pvcam_device", open_device)

    status, output, errors = run("--recover", "--open", "--wait", "1")

    assert status == 0, errors
    assert retrain_calls == [False, True, True]
    assert "retry:" in output
    assert "head=pvcamPCIE_0 serial=KX00000001" in output
