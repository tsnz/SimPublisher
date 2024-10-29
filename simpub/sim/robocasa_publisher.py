from mujoco import mj_name2id, mjtObj
from typing import List, Dict
import numpy as np
from simpub.parser.mjcf.utils import ros2unity

from robosuite.environments.base import MujocoEnv
from robosuite.utils.binding_utils import MjModel, MjData
from simpub.core.simpub_server import SimPublisher
from simpub.parser.mjcf import MJCFParser
from simpub.simdata import SimObject
import time


class RobocasaPublisher(SimPublisher):

    def __init__(
        self,
        env : MujocoEnv,
        host: str = "127.0.0.1",
        broadcast : str = None,
        no_rendered_objects: List[str] = None,
        no_tracked_objects: List[str] = None,
    ) -> None:
        
        self.env = env
        sim_scene = MJCFParser.parse_from_string(env.sim.model.get_xml(), visual_groups={1})
        self.tracked_obj = list()
        
        super().__init__(
            sim_scene,
            host,
            no_rendered_objects,
            no_tracked_objects,
            broadcast=broadcast
        )
        for child in self.sim_scene.root.children:
            self.set_update_objects(child)

    def set_update_objects(self, obj: SimObject):
        if obj.name in self.no_tracked_objects:
            return
         
        self.tracked_obj.append(obj.name)
        for child in obj.children:
            self.set_update_objects(child)

    def get_update(self) -> Dict[str, List[float]]:
        state = {}
        for name in self.tracked_obj:
            id = self.env.sim.model.body_name2id(name)
            pos, rot = self.env.sim.data.body_xpos[id], self.env.sim.data.body_xquat[id]
            state[name] = [
                -pos[1], pos[2], pos[0], rot[2], -rot[3], -rot[1], rot[0]
            ]
        return state
