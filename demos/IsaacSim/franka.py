# launch Isaac Sim before any other imports
# default first two lines in any standalone application
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})  # we can also run as headless.

# This extension has franka related tasks and controllers as well
from omni.isaac.core import World
from omni.isaac.franka import Franka
from omni.isaac.core.objects import DynamicCuboid, VisualCuboid
from omni.isaac.core.prims import XFormPrim
import numpy as np

from omni.isaac.lab.devices import Se3Gamepad
from omni.isaac.core.articulations import Articulation
from omni.isaac.franka.controllers import RMPFlowController

from omni.isaac.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage

from pxr import Gf

from simpub.sim.isaacsim_publisher import IsaacSimPublisher
from simpub.xr_device.meta_quest3 import MetaQuest3

world = World(device="cuda", set_defaults=False)
physContext = world.get_physics_context()
physContext.enable_fabric(True)
physContext.enable_gpu_dynamics(True)

world.scene.add_default_ground_plane()
world.scene.stage.GetPrimAtPath(
    "/World/defaultGroundPlane/Environment/Geometry"
).GetAttribute("xformOp:translate").Set(Gf.Vec3d(0, 0, 0))
world.scene.stage.GetPrimAtPath(
    "/World/defaultGroundPlane/Environment/Geometry"
).GetAttribute("xformOp:scale").Set(Gf.Vec3d(3, 3, 3))

# Robot specific class that provides extra functionalities
# such as having gripper and end_effector instances.
franka = world.scene.add(
    Franka(
        prim_path="/World/Franka",
        name="Franka",
        position=np.array([0.5, 0, 0]),
        orientation=np.array([0, 0, 0, 1]),
    )
)

fancy_cube = world.scene.add(
    DynamicCuboid(
        prim_path="/World/random_cube",
        name="fancy_cube",
        position=np.array([0.3, 0.3, 0.3]),
        scale=np.array([0.0515, 0.0515, 0.0515]),
        color=np.array([0, 0, 1.0]),
    )
)

world.scene.add(
    XFormPrim(
        prim_path="/World/targetxform",
        orientation=np.array([0, 1, 0, 0]),
    )
)

# add a cube for franka to follow
target = world.scene.add(
    VisualCuboid(
        prim_path="/World/targetxform/target",
        name="target",
        position=np.array([0, 0, 0.3]),
        scale=np.array([0.03, 0.03, 0.03]),
        color=np.array([1.0, 0, 0]),
    )
)

asset_root_path = get_assets_root_path()
teddy = add_reference_to_stage(
    usd_path=asset_root_path + "/Isaac/IsaacLab/Objects/Teddy_Bear/teddy_bear.usd",
    prim_path="/World/teddy",
)
teddy.GetAttribute("xformOp:rotateXYZ").Set(Gf.Vec3f(0, 0, -90))
# teddy.GetAttribute("xformOp:translate").Set(Gf.Vec3f(0.5, 0, 0.1))
# teddy.GetChild("geometry").GetChild("bear").GetProperty(
#     "physxDeformable:enableCCD"
# ).Set(False)
# teddy.GetChild("geometry").GetChild("bear").GetAttribute(
#     "physxDeformable:collisionSimplificationRemeshingResolution"
# ).Set(5)
# teddy.GetChild("geometry").GetChild("bear").GetAttribute(
#     "physxDeformable:simulationHexahedralResolution"
# ).Set(5)

bin = add_reference_to_stage(
    usd_path=asset_root_path + "/Isaac/Props/KLT_Bin/small_KLT.usd",
    prim_path="/World/bin",
)
bin.GetAttribute("xformOp:translate").Set(Gf.Vec3f(0.5, -0.5, 0.075))

world.reset()
articulation = Articulation(franka.prim_path)
articulation.initialize()
rmp_controller = RMPFlowController("FC", articulation)

publisher = IsaacSimPublisher(host="192.168.0.208", stage=world.stage)
meta_quest3 = MetaQuest3("ALRMetaQuest3")

while simulation_app.is_running():
    world.step(render=True)

    if world.is_playing():
        
        input_data = meta_quest3.get_input_data()
        if input_data is not None and input_data["right"]["index_trigger"]:
            target.set_world_pose(input_data["right"]["pos"])
            # rotate controller by 180 deg on z axis so that x axis faces robot
            # this prevents a 180 deg turn of the EEF
            # this also means y and z axis have to be inverted to correctly rotate the target cube
            rot_180_deg_z = Gf.Quatf(0, Gf.Vec3f(0, 0, 1))
            controller_rot = Gf.Quatf(input_data["right"]["rot"][0], input_data["right"]["rot"][1:4])
            target_rot = controller_rot  * rot_180_deg_z        
            target_rot_i = target_rot.GetImaginary()
            target.set_local_pose(None, [target_rot.GetReal(), target_rot_i[0], -target_rot_i[1], -target_rot_i[2]])                        
                    
            current_gripper_position = franka.gripper.get_joint_positions()
            if not input_data["B"]:
                franka.gripper.set_joint_positions(
                    current_gripper_position
                    + (franka.gripper.joint_opened_positions - current_gripper_position)
                    * 0.03
                )
            else:
                franka.gripper.set_joint_positions(
                    current_gripper_position
                    + (franka.gripper.joint_closed_positions - current_gripper_position)
                    * 0.03
                )

        target_position, target_rotation = target.get_world_pose()
        action = rmp_controller.forward(target_position, target_rotation)
        articulation.apply_action(action)

simulation_app.close()
