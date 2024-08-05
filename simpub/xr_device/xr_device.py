import json

from ..core.subscriber import Subscriber, logger


class InputData:
    def __init__(self, json_str: str) -> None:
        self.json_str = json_str
        self.data = json.loads(json_str)


class XRDevice:
    type = "XRDevice"

    def __init__(
        self,
        device_name: str = "UnityEditor",
    ) -> None:
        self.device = device_name
        self.log_subscriber = Subscriber(f"{device_name}/Log", self.print_log)

    def print_log(self, log: str):
        logger.info(f"{self.type} Log: {log}")

    def get_input_data(self) -> InputData:
        pass
