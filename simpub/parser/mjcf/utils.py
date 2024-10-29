from typing import List, Dict, Callable
import numpy as np
from scipy.spatial.transform import Rotation
from xml.etree.ElementTree import Element as XMLNode
import re

from simpub.simdata import VisualType

RMap: Dict[str, Callable] = {
    "quat": lambda x,y : quat2quat(x,y),
    "axisangle": lambda x,y: axisangle2quat(x,y),
    "euler": lambda x,y: euler2quat(x,y),
    "xyaxes": lambda x,y: xyaxes2quat(x,y),
    "zaxis": lambda x,y: zaxis2quat(x,y),
}


def get_rot_from_xml(obj_xml: XMLNode, degree=True) -> List[float]:
    result: List[float] = [0, 0, 0, 1]
    for key in RMap.keys():
        if key in obj_xml.attrib:
            result = RMap[key](
                str2list(obj_xml.get(key)),
                degree
            )
            break
    return ros2unity_quat(result)


def str2list(input_str: str, default = [1, 1, 1]) -> np.ndarray:
    if not input_str: return np.array(default, dtype=np.float32)
    return np.fromstring(input_str, dtype=np.float32, sep=" ")

def str2listabs(input_str: str, sep: str = ' ', default = [1, 1, 1]) -> np.ndarray:
    if not input_str: return np.array(default, dtype=np.float32)
    return np.abs(np.fromstring(input_str, dtype=np.float32, sep=sep))


def rotation2unity(rotation: Rotation) -> List[float]:
    return rotation.as_quat()


def quat2quat(quat: List[float], degree : bool) -> List[float]:
    quat = np.asarray(quat, dtype=np.float32)
    assert len(quat) == 4, "Quaternion must have four components."
    # Mujoco use wxyz format and Unity uses xyzw format
    w, *xyz = quat
    return rotation2unity(Rotation.from_quat([*xyz, w]))


def axisangle2quat(
    axisangle: List[float], degree : bool
) -> List[float]:
    assert len(axisangle) == 4, (
        "axisangle must contain four values (x, y, z, a)."
    )
    # Extract the axis (x, y, z) and the angle a
    axis = axisangle[:3]
    angle = axisangle[3]
    axis = axis / np.linalg.norm(axis)
    rotation = Rotation.from_rotvec(angle * axis, degrees=degree)
    return rotation2unity(rotation)


def euler2quat(
    euler: List[float], degree : bool
) -> List[float]:
    assert len(euler) == 3, "euler must contain three values (x, y, z)."
    rotation = Rotation.from_euler("xyz", euler, degrees=degree)
    return rotation2unity(rotation)


def xyaxes2quat(xyaxes: List[float], degree : bool) -> List[float]:
    assert len(xyaxes) == 6, (
        "xyaxes must contain six values (x1, y1, z1, x2, y2, z2)."
    )
    x = np.array(xyaxes[:3])
    y = np.array(xyaxes[3:])
    z = np.cross(x, y)
    rotation_matrix = np.array([x, y, z]).T
    rotation = Rotation.from_matrix(rotation_matrix)
    return rotation2unity(rotation)


def zaxis2quat(zaxis: List[float], degree : bool) -> List[float]:
    assert len(zaxis) == 3, "zaxis must contain three values (x, y, z)."
    # Create the rotation object from the z-axis
    rotation = Rotation.from_rotvec(zaxis, degrees=degree)
    return rotation2unity(rotation)


def ros2unity(pos: List[float]) -> List[float]:
    return [-pos[1], pos[2], pos[0]]


def ros2unity_quat(quat: List[float]) -> List[float]:
    return [quat[1], -quat[2], -quat[0], quat[3]]


TypeMap: Dict[str, VisualType] = {
    "plane": VisualType.PLANE,
    "sphere": VisualType.SPHERE,
    "capsule": VisualType.CAPSULE,
    "ellipsoid": VisualType.CAPSULE,
    "cylinder": VisualType.CYLINDER,
    "box": VisualType.CUBE,
    "mesh": VisualType.MESH
}


def scale2unity(scale: List[float], visual_type: str) -> List[float]:
    if visual_type in ScaleMap:
        return ScaleMap[visual_type](scale)
    else:
        return [1, 1, 1]


def plane2unity_scale(scale: List[float]) -> List[float]:
    return list(map(abs, [scale[0] * 2, 0.001, scale[1] * 2]))


def box2unity_scale(scale: List[float]) -> List[float]:
    # return [abs(scale[1]) * 2, abs(scale[2]) * 2, abs(scale[0]) * 2]
    return [abs(scale[i]) * 2 for i in [1, 2, 0]]


def sphere2unity_scale(scale: List[float]) -> List[float]:
    return [abs(scale[0]) * 2] * 3


def cylinder2unity_scale(scale: List[float]) -> List[float]:
    # if len(scale) == 3:
    #     return list(map(abs, [scale[0], scale[1], scale[0]]))
    # else:
    #     return list(map(abs, [scale[0] * 2, scale[1], scale[0] * 2]))
    if len(scale) == 1:
        return list(map(abs, [scale[0] * 2, scale[0] * 2, scale[0] * 2]))
    else:
        return list(map(abs, [scale[0] * 2, scale[1], scale[0] * 2]))

def capsule2unity_scale(scale: List[float]) -> List[float]:
    # assert len(scale) == 3, "Only support scale with three components."
    # return list(map(abs, [scale[0], scale[1], scale[0]]))
    if len(scale) == 2:
        return list(map(abs, [scale[0], scale[1], scale[0]]))
    elif len(scale) == 1:
        return list(map(abs, [scale[0] * 2, scale[0] * 2, scale[0] * 2]))


ScaleMap: Dict[str, Callable] = {
    "plane": lambda x: plane2unity_scale(x),
    "box": lambda x: box2unity_scale(x),
    "sphere": lambda x: sphere2unity_scale(x),
    "cylinder": lambda x: cylinder2unity_scale(x),
    "capsule": lambda x: capsule2unity_scale(x),
    "ellipsoid": lambda x: capsule2unity_scale(x),
}
