#launch Isaac Sim before any other imports
#default first two lines in any standalone application
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False}) # we can also run as headless.

# This extension has franka related tasks and controllers as well
from omni.isaac.core import World
from omni.isaac.franka import Franka
from omni.isaac.core.objects import DynamicCuboid, VisualSphere
import numpy as np

from omni.isaac.lab.devices import Se3Gamepad
from omni.isaac.core.articulations import Articulation
from omni.isaac.franka.controllers import RMPFlowController

from omni.isaac.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage

from pxr import Gf

from simpub.sim.isaacsim_publisher import IsaacSimPublisher

world = World(device="cuda", set_defaults=False)
physContext = world.get_physics_context()
# physContext.enable_fabric(True)
physContext.enable_gpu_dynamics(True)
world.scene.add_default_ground_plane()
# Robot specific class that provides extra functionalities
# such as having gripper and end_effector instances.

franka = world.scene.add(Franka(prim_path="/World/Fancy_Franka", name="fancy_franka"))
# add a cube for franka to pick up
fancy_cube = world.scene.add(
    DynamicCuboid(
        prim_path="/World/random_cube",
        name="fancy_cube",
        position=np.array([0.3, 0.3, 0.3]),
        scale=np.array([0.0515, 0.0515, 0.0515]),
        color=np.array([0, 0, 1.0]),
    )
)

target = world.scene.add(
    VisualSphere(
        prim_path="/World/target",
        name="target",
        position=np.array([0.5, 0, 0.5]),
        radius=0.015,
        color=np.array([1.0, 0, 0]),
    )
)

asset_root_path = get_assets_root_path()
teddy = add_reference_to_stage(usd_path=asset_root_path + "/Isaac/IsaacLab/Objects/Teddy_Bear/teddy_bear.usd", prim_path="/World/teddy")
teddy.GetAttribute('xformOp:rotateXYZ').Set(Gf.Vec3f(0, 0, 90))
teddy.GetAttribute('xformOp:translate').Set(Gf.Vec3f(0.5, 0, 0.1))
teddy.GetChild('geometry').GetChild('bear').GetProperty('physxDeformable:enableCCD').Set(False)
teddy.GetChild('geometry').GetChild('bear').GetAttribute('physxDeformable:collisionSimplificationRemeshingResolution').Set(5)
teddy.GetChild('geometry').GetChild('bear').GetAttribute('physxDeformable:simulationHexahedralResolution').Set(5)

sensitivity = 0.004
input_controller = Se3Gamepad(dead_zone=0.2)

world.reset()

articulation = Articulation(franka.prim_path)
articulation.initialize()
rmp_controller = RMPFlowController("FC", articulation)

t_pose = np.asarray([0, 1, 0, 0], dtype=np.float32)

publisher = IsaacSimPublisher(host="192.168.170.22", stage=world.stage)

while simulation_app.is_running():
    world.step(render=True)
    input = input_controller.advance()
    input[0][:] = input[0][:] * sensitivity
    input[0][0] = -input[0][0]
    
    if world.is_playing():
        
        cube_position, _ = fancy_cube.get_world_pose()                
        
        current_gripper_position = franka.gripper.get_joint_positions()                
        if (not input[1]):
            franka.gripper.set_joint_positions(current_gripper_position + (franka.gripper.joint_opened_positions - current_gripper_position) * 0.03)
        else:
            franka.gripper.set_joint_positions(current_gripper_position + (franka.gripper.joint_closed_positions - current_gripper_position) * 0.03)
           
        target_position, _ = target.get_world_pose()
        _, target_rotation = target.get_local_pose()
        target_position = target_position + input[0][[-0, 1, 2]]
        target.set_world_pose(target_position)        
        action = rmp_controller.forward(target_position, t_pose)
        articulation.apply_action(action)
    
simulation_app.close()