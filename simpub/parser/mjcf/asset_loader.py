from asyncio import futures
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor
from dataclasses import dataclass
from enum import Enum
import io
import math
from pathlib import Path
import time
from typing import Final, List, Optional, Tuple, Union

import cv2
import simpub
from simpub.simdata import SimMesh, SimTexture
import numpy as np
import trimesh
import open3d as o3d
import zlib

from simpub.core.log import logger

def hash_bytes(data: bytes) -> str:
    return "{:08x}".format(zlib.adler32(data) & 0xFFFFFFFF)



class AssetType(Enum):
    """Enumeration of supported asset types."""
    MESH = "mesh"
    TEXTURE = "texture"

@dataclass 
class AssetRequest:
    """Request for loading an asset."""
    type: AssetType
    name: str
    file: Path
    textureType: Optional[str] = None
    tint: Optional[np.ndarray] = None
    builtin: Optional[str] = None
    scale: Optional[np.ndarray] = None

    @classmethod
    def from_mesh(cls, name: str, file: Path, scale: Optional[np.ndarray] = None):
        return cls(AssetType.MESH, name, file, scale=scale)

    @classmethod
    def from_texture(cls, name: str, file: Path, texture_type: str, tint: Optional[np.ndarray] = None, builtin: Optional[str] = None):
        return cls(AssetType.TEXTURE, name, file, texture_type, tint, builtin)

class AssetLoader:

    def __init__(self, futures: List[futures.Future]):
        self.futures = futures

    @classmethod
    def load_assets(cls, requests: List[AssetRequest]) -> Tuple[List[Tuple[SimMesh, memoryview]], List[Tuple[SimTexture, memoryview]]]:

        with ThreadPoolExecutor(max_workers=6) as executor:
            results = list(executor.map(cls._load_asset,requests))

        textures = [result[1:] for result in results if result[0] == AssetType.TEXTURE]
        meshes = [result[1:] for result in results if result[0] == AssetType.MESH]
        return meshes, textures

    @classmethod
    def _load_asset(cls, request: AssetRequest):
        try:
            if request.type == AssetType.TEXTURE and request.builtin is not None:
                return (AssetType.TEXTURE, *TextureLoader.from_builtin(request.name, request.builtin, request.tint))
            elif request.type == AssetType.TEXTURE: 
                return (AssetType.TEXTURE, *TextureLoader.from_file(request.name, request.file, request.textureType, request.tint))
            else:
                return (AssetType.MESH, *MeshLoader.from_file(request.file, request.name, request.scale))
        except Exception as e:
            print(f"Failed to load asset {request.name}: {e.with_traceback()}")
            return None

class TextureBuiltinType(Enum):
    """Enumeration of built-in texture types."""
    CHECKER = "checker"
    GRADIENT = "gradient"
    FLAT = "flat"

class TextureLoader:
    
    # Class constants
    RES_PATH: Final[Path] = Path(simpub.__file__).parent
    DEFAULT_TEXTURE_SIZE: Final[Tuple[int, int]] = (256, 256)
    
    @classmethod
    def from_builtin(
        cls,
        name: str,
        builtin_name: Union[str, TextureBuiltinType],
        tint: Optional[np.ndarray] = None
    ) -> Tuple[SimTexture, bytes]:
        try:
            if isinstance(builtin_name, str):
                builtin_type = TextureBuiltinType(builtin_name)
            else:
                builtin_type = builtin_name
        except ValueError:
            raise ValueError(f"Invalid texture builtin: {builtin_name}")

        if builtin_type == TextureBuiltinType.CHECKER:
            # Create checker pattern using OpenCV
            img = cls._create_checker_pattern(cls.DEFAULT_TEXTURE_SIZE)
        elif builtin_type == TextureBuiltinType.GRADIENT:
            img = cls._create_gradient_pattern(cls.DEFAULT_TEXTURE_SIZE)
        else:  # FLAT
            img = np.full((*cls.DEFAULT_TEXTURE_SIZE, 4), 255, dtype=np.uint8)

        return cls._create_texture(name, img, tint=tint)

    @classmethod
    def from_file(
        cls,
        name: str,
        file: Path,
        texture_type: str,
        tint: Optional[np.ndarray] = None
    ) -> Tuple[SimTexture, bytes]:
        

        content = np.fromfile(file, np.uint8)
        # Decode image
        img = cv2.imdecode(content, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError("Failed to decode image data")


        # Convert BGR to RGBA if necessary
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
        elif img.shape[-1] == 3:  # BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        elif img.shape[-1] == 4:  # BGRA
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        else:
            raise IOError(f"Unsupported number of channels: {img.shape[-1]}")



        return cls._create_texture(name, img, texture_type, tint)
    
    @staticmethod
    def _create_checker_pattern(size: Tuple[int, int]) -> np.ndarray:
        """Create a checker pattern using OpenCV."""
        width, height = size
        square_size = min(width, height) // 8
        
        # Create base pattern
        pattern = np.zeros((square_size * 2, square_size * 2), dtype=np.uint8)
        pattern[:square_size, :square_size] = 255
        pattern[square_size:, square_size:] = 255
        
        # Tile pattern
        pattern = np.tile(pattern, ((height + square_size * 2 - 1) // (square_size * 2),
                                  (width + square_size * 2 - 1) // (square_size * 2)))
        pattern = pattern[:height, :width]
        
        # Convert to RGBA
        rgba = cv2.cvtColor(pattern, cv2.COLOR_GRAY2RGBA)
        return rgba

    @staticmethod
    def _create_gradient_pattern(size: Tuple[int, int]) -> np.ndarray:
        """Create a gradient pattern using OpenCV."""
        width, height = size
        gradient = np.linspace(0, 255, width, dtype=np.uint8)
        gradient = np.tile(gradient, (height, 1))
        return cv2.cvtColor(gradient, cv2.COLOR_GRAY2RGBA)
    
    @staticmethod
    def _apply_tint(img: np.ndarray, tint: np.ndarray) -> np.ndarray:
        assert img.shape[-1] == 4, "Image must be RGBA"
        assert len(tint) == 3, "Tint must be RGB"
        
        # Ensure tint is float32 and properly shaped for broadcasting
        tint = tint.reshape(1, 1, 3)
        
        # Simple multiplication approach - fast and stable
        img[..., :3] = (img[..., :3] * tint).astype(np.uint8)

    @staticmethod
    def _create_texture(
        name: str,
        img: np.ndarray,
        texture_type: str = "2d",
        tint : np.ndarray = None
    ) -> Tuple[SimTexture, bytes]:

        
        if tint is not None and tint.mean() != 1:
            TextureLoader._apply_tint(img, tint)

        height, width = img.shape[:2]
        tex_data = io.BytesIO(img).getvalue()
        texture_hash = hash_bytes(tex_data)

        texture = SimTexture(
            id=name,
            width=width,
            height=height,
            textureType=texture_type,
            dataHash=texture_hash
        )
        return texture, tex_data

class MeshLoader:

    ROTATION_TRANSFORM = trimesh.transformations.euler_matrix(
        -math.pi / 2.0, math.pi / 2.0, 0
    )

    O3D_ROTATION_TRANSFORM = np.array([
        [ np.cos(np.pi/2), 0, np.sin(np.pi/2), 0],
        [-np.sin(np.pi/2) * np.sin(-np.pi/2), np.cos(-np.pi/2), np.cos(np.pi/2) * np.sin(-np.pi/2), 0],
        [-np.sin(np.pi/2) * np.cos(-np.pi/2), -np.sin(-np.pi/2), np.cos(np.pi/2) * np.cos(-np.pi/2), 0],
        [0, 0, 0, 1]
    ])


    @classmethod
    def from_file(
        cls,
        file_path: str,
        name: Optional[str] = None,
        scale: Optional[np.ndarray] = None,
    ) -> Tuple[SimMesh, bytes]:
        return cls.from_open3d(file_path, name, scale)
    
    @classmethod
    def from_trimesh(
        cls,
        path: Path,
        name: str,
        scale: Optional[np.ndarray] = None
    ) -> Tuple[SimMesh, bytes]:
        """Saver loading but way slower"""
        
        mesh = trimesh.load_mesh(path)

        if scale is not None:
            mesh.apply_scale(scale)

        mesh = mesh.apply_transform(cls.ROTATION_TRANSFORM)

        """Process mesh"""

        # Vertices
        verts = mesh.vertices.astype(np.float32)
        verts[:, 2] = -verts[:, 2]
        verts = verts.flatten()

        # Normals
        norms = mesh.vertex_normals.astype(np.float32)
        norms[:, 2] = -norms[:, 2]
        norms = norms.flatten()

        # Indices
        indices = mesh.faces.astype(np.int32)
        indices = np.flip(indices, axis=1)
        indices = indices.flatten() 

        # Texture coords
        uvs = None
        if hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
            uvs = mesh.visual.uv.astype(np.float32)
            uvs[:, 1] = 1 - uvs[:, 1]
            uvs = uvs.flatten()

        return cls.from_loaded_mesh(name, indices, verts, norms, uvs)
    
    @classmethod
    def from_open3d(cls, file_path: str, name: str, scale: Optional[np.ndarray] = None) -> Tuple[SimMesh, bytes]:
        mesh = o3d.io.read_triangle_mesh(file_path)

        R = mesh.get_rotation_matrix_from_xyz((-math.pi / 2.0, 0, math.pi / 2.0))
        mesh = mesh.rotate(R, center=(0, 0 ,0))
        
        vertices = np.asarray(mesh.vertices, dtype=np.float32)

        assert scale.shape ==  (3,), "Scale must be a 3D vector"
        if scale is not None: vertices = vertices * scale

        vertices[:, 2] = -vertices[:, 2]
        vertices = vertices.flatten()

        # Transform normals
        normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
        normals[:, 2] = -normals[:, 2]
        normals = normals.flatten()
        
        # Indices
        indices = raw_indices = np.asarray(mesh.triangles , dtype=np.int32)
        indices = np.flip(indices, axis=1)
        indices = indices.flatten() 

        # Texture coords
        if mesh.has_triangle_uvs():
            # Convert triangle to vertex uvs
            vertex_uvs = np.zeros((len(vertices) // 3, 2), dtype=np.float32)
            vertex_uvs[raw_indices.flatten()] = mesh.triangle_uvs
            vertex_uvs[:, 1] = 1.0 - vertex_uvs[:, 1]
            vertex_uvs = vertex_uvs.flatten()

        return cls.from_loaded_mesh(name, indices, vertices, normals, vertex_uvs if mesh.has_triangle_uvs() else None)

    @classmethod
    def from_loaded_mesh(
        cls,
        name : str,
        indices : np.ndarray,
        vertices : np.ndarray,
        normals : np.ndarray,
        uvs : Optional[np.ndarray] = None
    ) -> Tuple[SimMesh, bytes]:
        
        """Write mesh data to binary buffer"""
        bin_buffer = io.BytesIO()
        bin_buffer.seek(0)

        vertices_layout = bin_buffer.tell(), vertices.shape[0]
        bin_buffer.write(vertices)

        normal_layout = bin_buffer.tell(), normals.shape[0]
        bin_buffer.write(normals) 

        indices_layout = bin_buffer.tell(), indices.shape[0]
        bin_buffer.write(indices)
        
        uv_layout = (0, 0)
        if uvs is not None:
            uv_layout = bin_buffer.tell(), uvs.shape[0]
            bin_buffer.write(uvs)

        bin_data = bin_buffer.getvalue()
        hash = hash_bytes(bin_data)

        mesh = SimMesh(
            id=name,
            indicesLayout=indices_layout,
            verticesLayout=vertices_layout,
            normalsLayout=normal_layout,
            uvLayout=uv_layout,
            dataHash=hash
        )
        return mesh, bin_data
