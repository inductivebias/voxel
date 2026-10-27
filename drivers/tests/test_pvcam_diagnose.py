import io
from collections.abc import Callable
from types import SimpleNamespace

import pytest
from vxl_drivers.cameras import pvcam_diagnose, teledyne


def install_fake_pvcam(
    monkeypatch: pytest.MonkeyPatch, names: list[str], open_device: Callable[[str], SimpleNamespace]
) -> None:
    pvc = SimpleNamespace(get_pvcam_version=lambda: "3.10.3.6")
    monkeypatch.setattr(teledyne, "ensure_pvcam_environment", lambda: None)
    monkeypatch.setattr(teledyne, "pvcam_runtime", lambda: pvc)
    monkeypatch.setattr(teledyne, "list_pvcam_camera_names", lambda: names)
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
