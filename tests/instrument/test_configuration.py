import re

import pytest
from pydantic import ValidationError

from vxl.instrument.config import ImagingProtocol, InstrumentConfig


@pytest.mark.parametrize(("section", "kind"), [("channels", "Channel id"), ("profiles", "Profile id")])
def test_channel_and_profile_ids_are_safe_in_dataset_paths(
    instrument_config: InstrumentConfig, section: str, kind: str
) -> None:
    data = instrument_config.default.imaging.model_dump()
    data[section]["../escape"] = next(iter(data[section].values()))
    with pytest.raises(ValidationError, match=re.escape(f"{kind} '../escape' may use only")):
        ImagingProtocol.model_validate(data)


def test_channel_filters_use_discrete_axis_position_validation(instrument_config: InstrumentConfig) -> None:
    imaging = instrument_config.default.imaging
    channel = imaging.channels["gfp"].model_copy(update={"filters": {"fw_emission": "missing"}})
    imaging = imaging.model_copy(update={"channels": {**imaging.channels, "gfp": channel}})
    assert [(v.code, v.loc, v.msg) for v in imaging.hal_violations(instrument_config.hal, loc=("imaging",))] == [
        (
            "discrete_axis.position_missing",
            ("imaging", "channels", "gfp", "filters", "fw_emission"),
            "Position 'missing' is not configured for device 'fw_emission' (available: ['BP525', 'BP600', 'BP700']).",
        )
    ]


def test_channel_filters_must_exactly_cover_detection_filter_wheels(instrument_config: InstrumentConfig) -> None:
    imaging = instrument_config.default.imaging
    channel = imaging.channels["gfp"]

    def violations(filters):
        updated = imaging.model_copy(
            update={"channels": {**imaging.channels, "gfp": channel.model_copy(update={"filters": filters})}}
        )
        return [(v.code, v.loc) for v in updated.hal_violations(instrument_config.hal, loc=("imaging",))]

    assert violations({}) == [
        ("imaging.channel.filter.missing", ("imaging", "channels", "gfp", "filters", "fw_emission"))
    ]
    assert violations({**channel.filters, "illumination_side_selector": "left"}) == [
        ("imaging.channel.filter.unexpected", ("imaging", "channels", "gfp", "filters", "illumination_side_selector"))
    ]


def test_legacy_channel_laser_migrates_without_overriding_explicit_selection() -> None:
    data = {
        "channels": {"camera": {"detection": "camera_1", "illumination": "laser_488"}},
        "profiles": {
            "cw": {"channels": ["camera"], "z_step": 1},
            "explicit": {"channels": ["camera"], "z_step": 1, "lasers": {"laser_561": "software"}},
            "dark": {"channels": ["camera"], "z_step": 1, "lasers": {}},
        },
    }
    protocol = ImagingProtocol.model_validate(data)
    assert protocol.profiles["cw"].lasers == {"laser_488": "software"}
    assert protocol.profiles["explicit"].lasers == {"laser_561": "software"}
    assert protocol.profiles["dark"].lasers == {}
    assert "illumination" not in protocol.channels["camera"].model_dump()
    assert ImagingProtocol.model_validate(protocol.model_dump()) == protocol
    assert data["channels"]["camera"]["illumination"] == "laser_488"
