"""
This module provides the implementation of various specialized 'Body' classes for the Mujoco physics
simulation engine, designed to handle complex physical entities composed of multiple elements
like geometries (geoms), sites, and joints. Bodies are the fundamental physical units within
the simulation, such as fixed pulleys, sliders, and other mechanical components, used to construct
Entities.

Classes:
    Body: Base class for all bodies in the Mujoco XML, supporting basic operations like addition of
        geometries, sites, and joints.
    FixedPulley: Represents a fixed pulley system, derived from Body.
    MovablePulley: Represents a movable pulley system, extending the Body class with specific
        methods for pulley operations.
    Mass: Specializes the Body class to represent masses with specific configurations.
    TriangularPrismBox: Represents complex geometric body configurations such as triangular prisms
        made from boxes.
    MassPrismPlane: Manages complex body setups involving masses, prisms, and planes to model
        intricate simulation scenarios.

"""

import os
from sim.geometry_utils import Frame

import numpy as np
from sim.objects import *
from typing import List, Tuple, Optional, Dict, Any
import math
import random

st = ipdb.set_trace



class TendonSequence:
    def __init__(self, 
                 elements: Optional[List[Union[Site, Geom]]] = None, 
                 description: str = "", 
                 name: str = "", 
                 children: Optional[List['TendonSequence']] = None):
        self.elements = elements if elements is not None else []
        if isinstance(elements, TendonSequence):
            self.elements = elements.get_elements()
        self.description = description
        self.name = name
        self.children = children if children is not None else []

    def add_child(self, child: 'TendonSequence'):
        self.children.append(child)

    def add_element(self, element: Union[Site, Geom]):
        self.elements.append(element)

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def reverse(self):
        """Reverse the order of sequences and children."""
        self.elements.reverse()
        self.children.reverse()
        for child in self.children:
            child.reverse()

    def get_elements(self) -> List[Union[Site, Geom]]:
        if self.is_leaf() or (self.elements and len(self.elements) > 0):
            return self.elements
        else:
            result = []
            for child in self.children:
                for element in child.get_elements():
                    result.append(element)
            return result

    def get_description(self) -> str:
        description = "description: " + self.description + "\n"
        for child in self.children:
            description += "\t" + child.get_description()
        return description


def reverse_tendon_sequence(tendon_sequence: TendonSequence):
    tendon_sequence.reverse()
    return tendon_sequence


class Body(Object):
    """
    Represents a 'body' element in Mujoco XML, which can contain geometries, sites, and joints.
    """

    def __init__(
            self,
            name: str,
            pos: Tuple[float, float, float] = (
                    0,
                    0,
                    0,
            ),
            # TODO: when massplane or massprismplane call body it will raise no pos error, deal with this later!
            quat: Tuple[float, float, float, float] = (1, 0, 0, 0),
            **kwargs,
    ) -> None:
        super().__init__(name)
        self.body_type = "body"
        self.pos = pos
        self.quat = quat
        self.geoms: List[Geom] = []
        self.sites: List[Site] = []
        self.joints: List[Joint] = []
        self.child_bodies: List['Body'] = []
        self.sensor_site = None
        self.sensor_list = []
        self.constant_force_dict = {}
        if not hasattr(self, "init_velocity_dict"): self.init_velocity_dict = {}
        # elif self.init_velocity_dict != {}: st()
        if not hasattr(self, "springs"): self.springs = []

    def add_geom(self, geom: Geom) -> None:
        """
        Add a geometry to the body.
        """
        self.geoms.append(geom)

    def add_site(self, site: Site) -> None:
        """
        Add a site to the body.
        """
        self.sites.append(site)

    def add_joint(self, joint: Joint) -> None:
        """
        Add a joint to the body.
        """
        self.joints.append(joint)

    def add_child_body(self, body: 'Body') -> None:
        """
        Add a child body to this body.
        """
        self.child_bodies.append(body)

    def set_pose(
            self,
            pos: Tuple[float, float, float] = None,
            quat: Tuple[float, float, float, float] = None,
    ) -> None:
        """
        Set the position and orientation of the body.
        """
        if pos is not None:
            self.pos = pos
        if quat is not None:
            self.quat = quat

    def set_quat_with_angle(self, angle, axis='x'):
        theta = math.radians(angle)
        qw = math.cos(theta / 2.0)

        if axis.lower() == 'x':
            qx = math.sin(theta / 2.0)
            qy = 0.0
            qz = 0.0
        elif axis.lower() == 'y':
            qx = 0.0
            qy = math.sin(theta / 2.0)
            qz = 0.0
        else:
            raise ValueError("Invalid axis. Choose 'x' or 'y'")

        quat = (qw, qx, qy, qz)
        self.set_pose(quat=quat)

    def add_rotation(self, angle, axis='x'):
        """
        Add a rotation around the specified axis to the current quaternion.

        Args:
            angle: Rotation angle (in degrees)
            axis: Rotation axis, supports 'x' or 'y'
        """
        # Convert the angle to radians
        theta = np.deg2rad(angle)
        # Calculate the real part of the rotation quaternion
        qw = np.cos(theta / 2.0)

        if axis.lower() == 'x':
            qx = np.sin(theta / 2.0)
            qy = 0.0
            qz = 0.0
        elif axis.lower() == 'y':
            qx = 0.0
            qy = np.sin(theta / 2.0)
            qz = 0.0
        elif axis.lower() == 'z':
            qx = 0.0
            qy = 0.0
            qz = np.sin(theta / 2.0)
        else:
            raise ValueError("Invalid axis. Choose 'x' or 'y'")

        # Generate the new rotation quaternion
        new_rot = np.array([qw, qx, qy, qz])
        # Apply the rotation: note that the order of multiplication affects the rotation order
        self.quat = Frame.quaternion_multiplication(self.quat, new_rot)

    def move(self, displacement: Tuple[float, float, float]) -> None:
        """
        Move the body by the given displacement.
        """
        self.pos = tuple(map(sum, zip(self.pos, displacement)))

    def get_ready_tendon_sequences(
            self, direction: ConnectingDirection
    ) -> List[TendonSequence]:
        return []

    def get_connecting_tendon_sequences(
            self, direction: ConnectingDirection = ConnectingDirection.DEFAULT, connecting_option: Any = None
    ) -> List[TendonSequence]:
        return []

    def get_sensor_list(self) -> List[Sensor]:
        if self.sensor_site is not None:
            self.sensor_list = self.sensor_site.create_sensor_list()
        return self.sensor_list

    def add_spring(self, spring: "Spring") -> None:
        """
        Add a spring to the body.
        """
        self.springs.append(spring)

    def to_xml(self) -> str:
        """
        Convert the body and its components to an XML string.
        """
        body_xml = f"""<body name="{self.name}" pos="{' '.join(map(str, self.pos))}" quat="{' '.join(map(str, self.quat))}">"""
        for geom in self.geoms:
            body_xml += geom.to_xml() + "\n"
        for site in self.sites:
            body_xml += site.to_xml() + "\n"
        for joint in self.joints:
            body_xml += joint.to_xml() + "\n"
        for child_body in self.child_bodies:
            body_xml += child_body.to_xml() + "\n"
        body_xml += "</body>"
        return body_xml

    def get_constant_forces(self) -> Dict[str, Tuple[float, float, float]]:
        return self.constant_force_dict

    def get_init_velocities(self) -> Dict[str, Tuple[float, float, float]]:
        return self.init_velocity_dict

    def get_springs(self) -> List["Spring"]:
        return self.springs

    def get_masses_quality(self) -> List[dict]:
        """
        Get the quality of each mass used for symbolic regression.
        """
        # TODO: Do we need to consider mass quality after child body is used.
        #  We can also adjust other bodies to use child body instead for better code logic
        list_of_masses = []
        for geom in self.geoms:
            if geom.mass is not None or float(geom.mass) != 0:
                sensor_dict = {}
                if len(self.sensor_list) > 0:
                    for sensor in self.sensor_list:
                        sensor_params = sensor.get_params()
                        sensor_dict[sensor_params[1].value] = sensor_params[0]
                mass_dict = {
                    "name": self.name,
                    "mass": float(geom.mass),
                    "sensor": sensor_dict,
                    "use_tendon_angle": False,
                }
                list_of_masses.append(mass_dict)

        return list_of_masses
    
    def get_description(self, simDSL2nlq = False) -> List[dict]:
        """
        Get the description of each body. It provides the information for LLM so it can match the variables.
        """
        # TODO: Do we need to consider mass quality after child body is used.
        #  We can also adjust other bodies to use child body instead for better code logic
        list_of_masses = []
        for geom in self.geoms:
            if geom.mass is not None and float(geom.mass) != 0:
                mass_dict = {
                    "name": self.name,
                    "mass": float(geom.mass),
                    "body_type": self.body_type,
                    "description": f"The {self.body_type} {self.name} has a mass of {float(geom.mass)} kg.",
                }
                if self.constant_force_dict:
                    mass_dict["description"] += (
                        f" A constant force of {self.constant_force_dict[self.name]} N is applied to {mass_dict['name']}."
                    )
                if self.init_velocity_dict:
                    mass_dict["description"] += (
                        f" {mass_dict['name']} has an initial velocity of {self.init_velocity_dict[self.name]} m/s."
                    )
                list_of_masses.append(mass_dict)

        return list_of_masses

    def get_bodies(self):
        """
        Get the body name for recording purposes. 
        The sub-dictionary will contain the measurements or sensor data, like mass, velocity, etc.
        """
        return {self.name: {}}

def get_all_geoms_in_body(body: Body):
    """
    Get all geoms in the body.
    """
    geoms = list(body.geoms)
    
    # Find all attributes of type Body in body
    for attr in dir(body):
        if isinstance(getattr(body, attr), Body):
            # Recursively get geoms from the nested body
            nested_body = getattr(body, attr)
            nested_geoms = get_all_geoms_in_body(nested_body)
            geoms.extend(nested_geoms)
        elif isinstance(getattr(body, attr), list):
            bodies = [b for b in getattr(body, attr) if isinstance(b, Body)]
            for b in bodies:
                nested_geoms = get_all_geoms_in_body(b)
                geoms.extend(nested_geoms)
    
    return geoms