"""Hardware-free checks for NI terminal allocation."""

import pytest

from vxl.devices.daq.hub_ni.resources import NiDaqModel, _NiResourceManager, capabilities_for_model


def _manager() -> _NiResourceManager:
    return _NiResourceManager(device_name="Dev1", capabilities=capabilities_for_model(NiDaqModel.NI6738))


def test_reserve_input_task_claims_free_counters_and_clock_lines_without_output_routes():
    manager = _manager()
    manager.reserve_ao_task(owner_uid="pacer", ao_terminals=["ao0"], hardware_timed=True, needs_counter=True)

    lease = manager.reserve_input_task(
        owner_uid="recorder",
        counters=3,
        timed_di_lines=["port0/line0", "P0.1"],
        input_pfis=["pfi0", "PFI6", "/Dev1/pfi3"],
    )

    assert lease.counter_paths == ("/Dev1/ctr1", "/Dev1/ctr2", "/Dev1/ctr3")
    assert lease.di_paths == ("/Dev1/port0/line0", "/Dev1/port0/line1")
    assert lease.input_pfi_paths == ("/Dev1/PFI0", "/Dev1/PFI6", "/Dev1/PFI3")
    recorder_pins = {pin for pin, owner in manager.assigned_pins.items() if owner == "recorder"}
    assert recorder_pins == {"ctr1", "ctr2", "ctr3", "port0/line0", "port0/line1"}, "input PFIs stay unclaimed"

    lease.release()
    assert set(manager.assigned_pins.values()) == {"pacer"}


def test_reserve_input_task_input_pfis_are_shared_with_other_readers():
    manager = _manager()
    manager.reserve_ao_task(
        owner_uid="pacer", ao_terminals=["ao0"], hardware_timed=True, needs_counter=True, input_pfis=["pfi6"]
    )
    first = manager.reserve_input_task(owner_uid="a", counters=1, timed_di_lines=[], input_pfis=["pfi6"])
    second = manager.reserve_input_task(owner_uid="b", counters=1, timed_di_lines=[], input_pfis=["pfi6"])
    assert first.input_pfi_paths == second.input_pfi_paths == ("/Dev1/PFI6",)


@pytest.mark.parametrize(
    "claim_output",
    [
        lambda manager: manager.reserve_do_task(owner_uid="gate", do_terminals=["pfi3"], hardware_timed=False),
        lambda manager: manager.reserve_ao_task(
            owner_uid="gate", ao_terminals=["ao0"], hardware_timed=True, needs_counter=True, output_pfi="pfi3"
        ),
    ],
    ids=["do_line", "counter_output"],
)
def test_reserve_input_task_pfi_driven_as_output_is_refused_and_claims_nothing(claim_output):
    manager = _manager()
    claim_output(manager)
    before = manager.assigned_pins

    with pytest.raises(ValueError, match="'pfi3' is driven as an output by 'gate'"):
        manager.reserve_input_task(
            owner_uid="recorder", counters=2, timed_di_lines=["port0/line0"], input_pfis=["pfi3"]
        )

    assert manager.assigned_pins == before


def test_reserve_input_task_counter_shortage_raises_and_claims_nothing():
    manager = _manager()
    manager.reserve_input_task(owner_uid="first", counters=2, timed_di_lines=[], input_pfis=[])
    before = manager.assigned_pins

    with pytest.raises(RuntimeError, match="3 free counters are needed on Dev1; 2 are free"):
        manager.reserve_input_task(owner_uid="second", counters=3, timed_di_lines=["port0/line0"], input_pfis=[])

    assert manager.assigned_pins == before


def test_reserve_input_task_timed_di_needs_the_free_port0_engine():
    manager = _manager()
    manager.reserve_input_task(owner_uid="first", counters=0, timed_di_lines=["port0/line0"], input_pfis=[])

    with pytest.raises(ValueError, match="No hardware DI timing engine"):
        manager.reserve_input_task(owner_uid="second", counters=0, timed_di_lines=["port0/line1"], input_pfis=[])
    with pytest.raises(ValueError, match=r"limited to ports \[0\]"):
        manager.reserve_input_task(owner_uid="second", counters=0, timed_di_lines=["pfi0"], input_pfis=[])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"counters": 0, "timed_di_lines": [], "input_pfis": []}, "at least one counter or digital line"),
        ({"counters": 1, "timed_di_lines": ["port0/line0", "p0.0"], "input_pfis": []}, "duplicate"),
        ({"counters": 1, "timed_di_lines": [], "input_pfis": ["ao0"]}, "PFI terminal"),
    ],
)
def test_reserve_input_task_invalid_request_raises(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _manager().reserve_input_task(owner_uid="recorder", **kwargs)


def test_reserve_ao_task_counter_trigger_lists_its_counter_path():
    lease = _manager().reserve_ao_task(owner_uid="pacer", ao_terminals=["ao0"], hardware_timed=True, needs_counter=True)
    assert lease.counter_paths == (lease.counter_path,) == ("/Dev1/ctr0",)
