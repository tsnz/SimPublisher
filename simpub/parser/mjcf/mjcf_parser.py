from __future__ import annotations
from functools import lru_cache
from xml.etree.ElementTree import Element as XMLNode
import xml.etree.ElementTree as ET
import copy
import os
from os.path import join as pjoin
from typing import List, Dict, Optional
from pathlib import Path
import numpy as np
from simpub.parser.mjcf.asset_loader import AssetLoader, AssetRequest, MeshLoader, TextureLoader
from simpub.simdata import SimObject, SimScene
from simpub.simdata import SimMaterial, SimTransform
from simpub.simdata import SimVisual
from .utils import str2list, str2listabs, ros2unity
from .utils import get_rot_from_xml, scale2unity, TypeMap
from ...core.log import logger


class MJCFScene(SimScene):

    def __init__(self) -> None:
        super().__init__()
        self.xml_string: str = None


class MJCFDefault:
    def __init__(
        self,
        xml: XMLNode = None,
        parent: MJCFDefault = None,
    ) -> None:
        self._dict: Dict[str, Dict[str, str]] = dict()
        self.class_name = "main" if parent is None else xml.attrib["class"]
        if xml is not None:
            self.import_xml(xml)

    def import_xml(self, xml: XMLNode, parent: MJCFDefault = None) -> None:
        if parent is not None:
            self._dict = copy.deepcopy(parent._dict)
        for child in xml:
            if child.tag == "default":
                continue
            if child.tag not in self._dict.keys():
                self._dict[child.tag] = copy.deepcopy(child.attrib)
            else:
                self._dict[child.tag].update(child.attrib)

    def update_xml(self, xml: XMLNode) -> None:
        if xml.tag in self._dict:
            attrib = copy.deepcopy(self._dict[xml.tag])
            attrib.update(xml.attrib)
            xml.attrib = attrib


class MJCFParser:
    def __init__(
        self,
        file_path: str,
    ):
        
        self._xml_path = os.path.abspath(file_path) if file_path else None
        self._path = os.path.abspath(os.path.join(self._xml_path, "..")) if file_path else None
        self.visual_groups = set()
        self.no_rendered_objects = set()

    def parse(
        self,
        no_rendered_objects: List[str] = None,
    ) -> MJCFScene:
        
        if no_rendered_objects is not None:
            self.no_rendered_objects = no_rendered_objects

        xml = MJCFParser._get_root_from_xml_file(self._xml_path)
        return self._parse_xml(xml)

  
    @classmethod
    def parse_from_string(cls, xml_string : str, root = "/", visual_groups={0, 1}) -> MJCFScene:
        scene = cls(root)  
        scene.visual_groups = set(visual_groups)

        return scene._parse_xml(ET.fromstring(xml_string))

    def _parse_xml(self, xml: XMLNode) -> MJCFScene:
        self.scene = MJCFScene()

        self._merge_includes(xml)
        self._load_compiler(xml)
        self._load_defaults(xml)
        self._load_asset_info(xml)

        wbodys = list(xml.findall("./worldbody"))
        for body in wbodys[1:]:
            wbodys[0].extend(body)

        self.scene.root = self._load_body(wbodys[0])
        self.scene.xml_string = ET.tostring(xml, encoding="unicode")
        self._load_assets()

        return self.scene

    def _merge_includes(self, root_xml: XMLNode) -> XMLNode:
        for child in root_xml:
            if child.tag != "include":
                self._merge_includes(child)
                continue
            sub_xml_path = os.path.join(self._path, child.attrib["file"])
            if not os.path.exists(sub_xml_path):
                logger.warning(f"Warning: File '{sub_xml_path}' does not exist.")
                continue
            sub_xml_root = MJCFParser._get_root_from_xml_file(sub_xml_path)
            root_xml.extend(sub_xml_root)
        for child in root_xml:
            if child.tag == "include":
                root_xml.remove(child)

    @staticmethod
    def _get_root_from_xml_file(xml_path: str) -> XMLNode:

        assert os.path.exists(xml_path), f"File '{xml_path}' does not exist."

        xml_path = os.path.abspath(xml_path)
        tree_xml = ET.parse(xml_path)
        return tree_xml.getroot()
    

    def _load_compiler(self, xml: XMLNode) -> None:
        compiler = xml.find("./compiler")

        self._use_degree = compiler.get("angle", "degree") == "degree"
        self._eulerseq = compiler.get("eulerseq", "xyz")
        
        self._assetdir = pjoin(self._path, compiler.get("assetdir", ""))

        if "meshdir" in compiler.attrib:
            self._meshdir = pjoin(self._path, compiler.get("meshdir"))
        else:
            self._meshdir = self._assetdir
        if "texturedir" in compiler.attrib:
            self._texturedir = pjoin(self._path, compiler.get("texturedir"))
        else:
                self._texturedir = self._assetdir

    def _load_defaults(self, root_xml: XMLNode) -> None:
        default_dict: Dict[str, MJCFDefault] = dict()
        default_dict["main"] = MJCFDefault()
        # only start _loading default tags under the mujoco tag
        for default_child_xml in root_xml.findall("./default"):
            self._parse_default(default_child_xml, default_dict)
        # replace the class attribute with the default values
        self._import_default(root_xml, default_dict)

    def _parse_default(
        self,
        default_xml: XMLNode,
        default_dict: Dict[str, MJCFDefault],
        parent: MJCFDefault = None,
    ) -> None:
        default = MJCFDefault(default_xml, parent)
        default_dict[default.class_name] = default
        for default_child_xml in default_xml.findall("default"):
            self._parse_default(default_child_xml, default_dict, default)

    def _import_default(
        self,
        xml: XMLNode,
        default_dict: Dict[str, MJCFDefault],
        parent_name: str = "main",
    ) -> None:
        if xml.tag == "default":
            return
        default_name = (
            xml.attrib["class"]
            if "class" in xml.attrib.keys()
            else parent_name
        )
        default = default_dict[default_name]
        default.update_xml(xml)

        parent_name = (
            xml.attrib["childclass"]
            if "childclass" in xml.attrib
            else parent_name
        )
        for child in xml:
            self._import_default(child, default_dict, parent_name)

    def _load_asset_info(self, xml: XMLNode) -> None:
        
        scene_assets = [asset for assets in xml.findall("./asset") for asset in assets]

        self.mesh_info = { asset.get("name") : asset for asset in scene_assets if asset.tag == "mesh" and "name" in asset.attrib }
        self.texture_info = { asset.get("name") : asset for asset in scene_assets if asset.tag == "texture" and "name" in asset.attrib }
        self.material_info = { asset.get("name") : asset for asset in scene_assets if asset.tag == "material" and "name" in asset.attrib }

        self.requested_assets = dict()

    def _request_asset(self, type, asset_key) -> None:

        if type + asset_key in self.requested_assets: return

        if type == "mesh":

            mesh = self.mesh_info[asset_key]

            asset_file = pjoin(self._meshdir, mesh.attrib["file"])
            scale = str2list(mesh.get("scale"))
            name = mesh.get("name")

            self.requested_assets[type + asset_key] = (AssetRequest.from_mesh(name, asset_file, scale))

        elif type == "material":

            material = self.material_info[asset_key]

            name = material.get("name") 
            color = str2list(material.get("rgba"), [1, 1, 1, 1])
            emission = float(material.get("emission", "0.0"))
            emissionColor = color * emission

            mat = SimMaterial(
                id=name,
                color=color,
                emissionColor=emissionColor,
                specular=float(material.get("specular") or 0.5),
                shininess=float(material.get("shininess") or 0.5),
                reflectance=float(material.get("reflectance") or 0.0),
                textureSize=str2list(material.get("texrepeat"), [1, 1]),
                texture=material.get("texture")
            )

            self.scene.materials.append(mat)

            # Load the required texture
            if mat.texture is not None:
                self._request_asset("texture", mat.texture)

        elif type == "texture":    

            texture = self.texture_info[asset_key]

            name = texture.get("name")
            type = texture.get("type")
            builtin = texture.get("builtin", "none")
            tint = texture.get("rgb1")
            
            if tint is not None:
                tint = str2list(tint)

            if texture.get("builtin", "none") != "none":    
                self.requested_assets[type + asset_key] = (AssetRequest.from_texture(name, None, tint, builtin))
            else:                
                asset_file = pjoin(self._texturedir, texture.attrib["file"])
                self.requested_assets[type + asset_key] = (AssetRequest.from_texture(name, asset_file, tint))

    def _load_assets(self):

        meshes, textures = AssetLoader.load_assets(self.requested_assets.values()) 
        print(len(meshes), len(textures))
        for mesh, data in meshes:
            self.scene.meshes.append(mesh)
            self.scene.raw_data[mesh.dataHash] = data

        for texture, data in textures:
            self.scene.textures.append(texture)
            self.scene.raw_data[texture.dataHash] = data


    def _load_visual(self, visual: XMLNode) -> Optional[SimVisual]:
        
        if int(visual.get("group", -1)) not in self.visual_groups:
            return None

        visual_type = visual.get("type", "box")
        pos = ros2unity(str2list(visual.get("pos"), [0, 0, 0]))
        rot = get_rot_from_xml(visual, self._use_degree)
        size = str2listabs(visual.get("size", "0.1 0.1 0.1"))
        scale = scale2unity(size, visual_type)
        trans = SimTransform(pos=pos, rot=rot, scale=scale)

        mesh = visual.get("mesh")
        material = visual.get("material")
        
        if mesh: self._request_asset("mesh", mesh)
        if material: self._request_asset("material", material)

        return SimVisual(
            type=TypeMap[visual_type],
            trans=trans,
            mesh=mesh,
            material=material,
            color=str2list(visual.get("rgba"),  [1, 1, 1, 1]),
        )

    def _load_body(self, body: XMLNode) -> Optional[None]:
        
        name = body.get("name")

        if name in self.no_rendered_objects:
            return None

        trans = SimTransform(
            pos=ros2unity(str2list(body.get("pos"), [0, 0, 0])),
            rot=get_rot_from_xml(body, self._use_degree)
        )

        visuals  = [self._load_visual(geom) for geom in body.findall("geom")]
        children = [self._load_body(child) for child in body.findall("body")]
       
        return SimObject(
            name=name,
            trans=trans,
            visuals =[visual for visual in visuals if visual is not None],
            children=[child for child in children if child is not None],
        )
