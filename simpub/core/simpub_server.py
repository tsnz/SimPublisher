from __future__ import annotations
import abc
from typing import Dict, List

from simpub.simdata import SimScene
from .net_manager import init_net_manager
from .net_manager import Streamer, Service
from .log import logger


class ServerBase(abc.ABC):

    def __init__(self, host: str = "127.0.0.1", broadcast=None):
        self.host: str = host
        self.net_manager = init_net_manager(host, broadcast)
        self.initialize()
        self.net_manager.start()

    def join(self):
        self.net_manager.join()

    @abc.abstractmethod
    def initialize(self):
        raise NotImplementedError


class MsgServer(ServerBase):

    def initialize(self):
        pass


class SimPublisher(ServerBase):

    def __init__(
        self,
        sim_scene: SimScene,
        host: str = "127.0.0.1",
        no_rendered_objects: List[str] = None,
        no_tracked_objects: List[str] = None,
        broadcast=None
    ) -> None:
        self.sim_scene = sim_scene
        self.no_rendered_objects = no_rendered_objects if no_rendered_objects is not None else []
        self.no_tracked_objects = no_tracked_objects if no_tracked_objects is not None else []

        super().__init__(host, broadcast=broadcast)

    def initialize(self):
        self.scene_update_streamer = Streamer("SceneUpdate", self.get_update)
        self.scene_service = Service("Scene", self._on_scene_request, str)
        self.asset_service = Service("Asset", self._on_asset_request, bytes)
        self.scene_string = self.sim_scene.to_string()

    def _on_scene_request(self, req: str) -> str:
        return self.scene_string

    def _on_asset_request(self, tag: str) -> bytes:
        if tag not in self.sim_scene.raw_data:
            logger.warning(f"Received invalid data request {tag}")
            return bytes()
        return self.sim_scene.raw_data[tag]

    @abc.abstractmethod
    def get_update(self) -> Dict:
        raise NotImplementedError
