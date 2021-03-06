# @generated by generate_proto_mypy_stubs.py.  Do not edit!
import sys
from google.protobuf.message import (
    Message as google___protobuf___message___Message,
)

from typing import (
    Optional as typing___Optional,
)

from typing_extensions import (
    Literal as typing_extensions___Literal,
)


class EngineConfigurationProto(google___protobuf___message___Message):
    width = ... # type: int
    height = ... # type: int
    quality_level = ... # type: int
    time_scale = ... # type: float
    target_frame_rate = ... # type: int
    show_monitor = ... # type: bool

    def __init__(self,
        width : typing___Optional[int] = None,
        height : typing___Optional[int] = None,
        quality_level : typing___Optional[int] = None,
        time_scale : typing___Optional[float] = None,
        target_frame_rate : typing___Optional[int] = None,
        show_monitor : typing___Optional[bool] = None,
        ) -> None: ...
    @classmethod
    def FromString(cls, s: bytes) -> EngineConfigurationProto: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    if sys.version_info >= (3,):
        def ClearField(self, field_name: typing_extensions___Literal[u"height",u"quality_level",u"show_monitor",u"target_frame_rate",u"time_scale",u"width"]) -> None: ...
    else:
        def ClearField(self, field_name: typing_extensions___Literal[b"height",b"quality_level",b"show_monitor",b"target_frame_rate",b"time_scale",b"width"]) -> None: ...
