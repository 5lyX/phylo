"""
This module defines classes for managing elements within the Mujoco physics simulation engine.
These classes provide structured representations for fundamental XML tags used in Mujoco,
such as 'geom', 'site', 'spatial', and 'tendon'. Each class implements methods to serialize
instances to XML format. Objects are the most basic building elements that are used to shape
body structures, while tendons are used to define relationships between these objects.

Classes:
    Object: Base class for all elements with common XML conversion interface.
    Geom: Handles 'geom' elements representing geometric shapes.
    Site: Manages 'site' elements used as reference or attachment points.
    Joint: Deals with 'joint' elements defining movement degrees.
    Spatial: Manages 'spatial' elements that can contain multiple sites or geometries.
    Tendon: Represents a 'tendon' structure containing spatial elements.
    Equality: Defines relationships between tendons.
"""

import ipdb
from sim.constants import *

st = ipdb.set_trace
from typing import List, Tuple, Union
from sim.geometry_utils import Frame
import numpy as np

import ipdb


class Object:
    """
    A base class for all objects, providing a common interface for XML conversion.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def to_xml(self) -> str:
        """
        Convert the object to its corresponding XML representation, serializing all non-None attributes.
        """
        class_name = self.__class__.__name__
        private_attr = f"_{class_name}__"
        attributes = []  # Start with the empty attribute if provided
        if "plugin" not in self.__dict__.keys() or self.__dict__["plugin"] is None:
            for attr, value in self.__dict__.items():
                if attr.startswith(private_attr):  # Skip private attributes
                    continue
                if value is not None:  # Skip the one already included
                    if isinstance(value, tuple) or isinstance(value, list):
                        value = " ".join(map(str, value))
                    attributes.append(f'{attr}="{value}"')
            return f'<{self.__class__.__name__.lower()} {" ".join(attributes)} />'
        else:   # If plugin is present, include it in the XML
            for attr, value in self.__dict__.items():
                if attr.startswith(private_attr) or "plugin" in attr:  # Skip private attributes and plugin
                    continue
                if value is not None:  # Skip the one already included
                    if isinstance(value, tuple) or isinstance(value, list):
                        value = " ".join(map(str, value))
                    attributes.append(f'{attr}="{value}"')
            return f'<{self.__class__.__name__.lower()} {" ".join(attributes)}>\n<plugin instance="{self.plugin}" />\n</{self.__class__.__name__.lower()}>' 


class Geom(Object):
    """
    Represents a 'geom' element, defining a geometric shape within a body.
    Optional attributes will only be serialized if they are explicitly set.
    """

    def __init__(
        self,
        name: str = None,
        geom_type: str = None,
        pos: Tuple[float, float, float] = None,
        size: Tuple[
            float, ...
        ] = None,  # Variable-length tuple because it can be shapes other than box
        material: str = None,
        rgba: Tuple[float, float, float, float] = None,
        mass: float = None,
        quat: Tuple[float, float, float, float] = (0, 0, 0, 1),
        geom: str = None,
        mesh: str = None,
        sidesite: str = None,
        condim: str = None,
        plugin: str = None,
        contype: str = "1",
        conaffinity: str = "1",
        density: str | None = None,
    ) -> None:
        super().__init__(name)
        # if name is None: ipdb.set_trace()
        self.type = geom_type
        self.pos = pos
        self.size = size
        self.material = material
        self.rgba = rgba
        self.mass = mass
        self.quat = quat
        self.geom = geom
        self.mesh = mesh
        self.sidesite = sidesite
        self.condim = condim
        self.plugin = plugin
        self.contype = contype
        self.conaffinity = conaffinity
        self.density = density

    def get_body_name(self) -> str:
        return '_'.join([self.name, self.geom][self.name is None].split('_')[:-1])


"""
<equality>
    <tendon tendon1="tendon_2_1" polycoef="0 1 0 0 0"/>
</equality>
"""


class Equality(Object):
    """
    Represents a 'equality' element, defining the relationship between tendons.
    """

    def __init__(
        self,
        tendon1: str,
        name: str = None,
        tendon2: str = None,
        polycoef: Tuple[float, float, float, float, float] = (0, 1, 0, 0, 0),
    ) -> None:
        super().__init__(name)
        self.tendon1 = tendon1
        self.tendon2 = tendon2
        self.polycoef = polycoef

    def to_xml(self) -> str:
        equality_xml = f"<equality>\n"
        equality_xml += f'<tendon tendon1="{self.tendon1}"'
        if self.tendon2:
            equality_xml += f' tendon2="{self.tendon2}"'
        equality_xml += f' polycoef="{" ".join(map(str, self.polycoef))}"/>\n'
        equality_xml += "</equality>\n"
        return equality_xml


class Site(Object):
    """
    Represents a 'site' element, a reference or attachment point within a body.
    """

    def __init__(
        self,
        name: str = None,
        pos: Tuple[float, float, float] = None,
        quat: Tuple[float, float, float, float] = (1, 0, 0, 0),
        site: str = None,
        body_name: str = None,
    ) -> None:
        super().__init__(name)
        self.site = site
        self.pos = pos
        self.quat = quat
        self.__body_name = body_name

    def create_spatial_site(self) -> "Site":
        """
        Create a spatial site from the site that can be used to create Tendon.
        """
        spatial_site = Site(site=self.name, quat=None, body_name=self.__body_name)
        return spatial_site

    def create_sensor_list(self) -> List["Sensor"]:
        """
        Create a spatial site from the site that can be used to create Tendon.
        """
        sensor_list = []
        sensor_list.append(Sensor(site_name=self.name, sensor_type=SensorType.ACC))
        sensor_list.append(Sensor(site_name=self.name, sensor_type=SensorType.FORCE))

        return sensor_list

    def get_body_name(self) -> str:
        return self.__body_name

    def set_body_name(self, body_name: str) -> None:
        self.__body_name = body_name


class Joint(Object):
    """
    Represents a 'joint' element, defining the degrees of freedom for movement within a body.
    """

    def __init__(
        self, joint_type: str, axis: Tuple[float, float, float], name: str = None, pos: Tuple[float, float, float] = (0, 0, 0)
    ) -> None:
        super().__init__(name)
        self.type = joint_type
        self.axis = axis
        self.pos = pos


class Spatial(Object):
    """
    Represents a 'spatial' element in a tendon, which can contain multiple sites or geometries.
    """

    def __init__(
        self,
        name: str,
        width: float = DEFAULT_ROPE_THICKNESS,
        rgba: Tuple[float, float, float, float] = (0.9686, 0.8392, 0.8784, 1),
        stiffness: float = None,
        springlength: float = None,
        damping: float = None,
    ) -> None:
        super().__init__(name)
        self.width = width
        self.rgba = rgba
        self.stiffness = stiffness
        self.springlength = springlength
        self.damping = damping
        self.elements = []  # This will store either Site or Geom references
        self.sensor = {}

    def add_element(self, element: Union[Site, Geom]) -> None:
        """
        Add a site or geometry to the spatial element.
        """
        self.elements.append(element)

    def extend(self, other: "Spatial") -> None:
        """
        Extend this spatial with elements from another spatial.
        """
        self.elements.extend(other.elements)

    def create_sensor_list(self) -> List["Sensor"]:
        """
        Create a spatial site from the site that can be used to create Tendon.
        """
        return [Sensor(tendon_name=self.name, sensor_type=SensorType.TENDONLIMITFRC)]

    def create_custom_sensor_list(self) -> List["Custom"]:
        """
        Create a spatial site from the site that can be used to create Tendon.
        """
        self.sensor["force"] = f"{self.name}_tension_sensor"
        return [Custom(name=f"{self.name}_tension_sensor", data=self.name)]

    @staticmethod
    def combine(
        spatial1: "Spatial",
        spatial2: "Spatial",
        name: str,
        width: float,
        rgba: Tuple[float, float, float, float],
    ) -> "Spatial":
        """
        Combine two spatials into a new one, including all elements from both.
        """
        combined = Spatial(name, width, rgba)
        combined.extend(spatial1)
        combined.extend(spatial2)
        return combined

    def to_xml(self) -> str:
        """
        Convert the spatial and its components to an XML string.
        """
        attributes = f'name="{self.name}" width="{self.width}" rgba="{" ".join(map(str, self.rgba))}"'
        if self.stiffness is not None:
            attributes += f' stiffness="{self.stiffness}"'
        if self.springlength is not None:
            attributes += f' springlength="{self.springlength}"'
        if self.damping is not None:
            attributes += f' damping="{self.damping}"'
        elements_xml = "\n".join([element.to_xml() for element in self.elements])
        return f'<spatial {attributes}>\n{elements_xml}\n</spatial>'
    
    def get_description(self, simDSL2nlq = False):
        element_str = ", ".join([e.get_body_name() for e in self.elements[:-1]]) + f' and {self.elements[-1].get_body_name()}'
        return [{
            'name': self.name,
            'stiffness': self.stiffness,
            **({'spring_length': self.springlength} if self.springlength is not None else {}),
            'description': f'A light {["spring", "string"][self.stiffness is None]} named {self.name} connects {element_str} in that order.'
        }]

class Tendon(Object):
    """
    Represents a 'tendon' structure that contains one or more spatial elements.
    """

    def __init__(self, name: str, spring = False) -> None:
        super().__init__(name)
        self.spatials: List[Spatial] = []
        self.is_spring = spring

    def add_spatial(self, spatial: Spatial) -> None:
        """
        Add a spatial element to the tendon.
        """
        self.spatials.append(spatial)

    def to_xml(self) -> str:
        """
        Convert the tendon and its spatials to an XML string.
        """
        spatials_xml = "\n".join([spatial.to_xml() for spatial in self.spatials])
        return f"<tendon>\n{spatials_xml}\n</tendon>"

    def get_parameters(self) -> List[dict]:
        assert len(self.spatials) == 1, "Assuming tendon has only one spatial"
        sensor_name = self.spatials[0].sensor
        elements = self.spatials[0].elements
        name = self.name
        connectors = list()
        sites = list()
        for element in elements:
            if isinstance(element, Site):
                connectors.append(element.get_body_name())
                sites.append(element.site)
        return [
            {
                "sensor": sensor_name,
                "name": name,
                "connector_names": connectors,
                "site_names": sites,
            }
        ]

    def generate_equality(self) -> Equality:
        """
        Generate the equality string for the tendon.
        """
        return None if self.is_spring else Equality(tendon1=self.spatials[0].name)

    def get_sensor_list(self) -> List["Sensor"]:
        """
        Get the list of sensors from the tendon.
        """
        sensor_list = []
        for spatial in self.spatials:
            sensor_list.extend(spatial.create_sensor_list())
        return sensor_list

    def get_custom_sensor_list(self) -> List["Custom"]:
        """
        Get the list of sensors from the tendon.
        """
        sensor_list = []
        for spatial in self.spatials:
            sensor_list.extend(spatial.create_custom_sensor_list())
        return sensor_list
    
    def get_description(self, simDSL2nlq = False):
        return sum([spatial.get_description() for spatial in self.spatials], [])


class Sensor(Object):
    def __init__(
        self, sensor_type: SensorType, site_name: str = None, tendon_name: str = None
    ) -> None:
        if (
            sensor_type != SensorType.ACC
            and sensor_type != SensorType.FORCE
            and sensor_type != SensorType.TENDONLIMITFRC
        ):
            raise ValueError(f"Sensor type {sensor_type} is not supported.")
        name = f"{site_name if site_name else tendon_name}_{sensor_type.value}"  # this should be more descriptive
        super().__init__(name)
        self.site_name = site_name
        self.sensor_type = sensor_type
        self.tendon_name = tendon_name

    def get_params(self) -> Tuple[str, SensorType]:
        return (self.name, self.sensor_type)

    def to_xml(self) -> str:
        """
        Convert the sensor to its corresponding XML representation.
        """
        if self.sensor_type == SensorType.ACC:
            return f'<accelerometer name="{self.name}" site="{self.site_name}" />\n'
        elif self.sensor_type == SensorType.FORCE:
            return f'<force  name="{self.name}" site="{self.site_name}" />\n'
        elif self.sensor_type == SensorType.TENDONLIMITFRC:
            return (
                f'<tendonlimitfrc name="{self.name}" tendon="{self.tendon_name}" />\n'
            )
        else:
            raise ValueError(f"Sensor type {self.sensor_type} is not supported.")


class Custom(object):
    def __init__(self, name: str, data: str) -> None:
        self.name = name
        self.data = data

    def to_xml(self) -> str:
        """
        Convert the custom sensor to its corresponding XML representation.
        """
        return f"<text name='{self.name}' data='{self.data}' />"


class Actuator(Object):
    def __init__(
        self,
        name: str,
        actuator_type: str = "general",
        joint: str = None,
        tendon: str = None,
        gainprm: float = 0.0,
        biasprm: float = 0.0,
        ctrllimited: bool = False,
        ctrlrange: Tuple[float, float] = (0.0, 1.0),
        kv: float = None,
        velocity: float = None,
    ) -> None:
        super().__init__(name)
        self.type = actuator_type
        self.joint = joint
        self.tendon = tendon
        self.gainprm = gainprm
        self.biasprm = biasprm
        self.ctrllimited = ctrllimited
        self.ctrlrange = ctrlrange
        self.kv = kv
        self.velocity = velocity

    def to_xml(self) -> str:
        if self.type == "velocity":
            actuator_xml = f'<velocity name="{self.name}" '
            if self.joint:
                actuator_xml += f'joint="{self.joint}" '
            if self.kv is not None:
                actuator_xml += f'kv="{self.kv}" '
            actuator_xml += '/>\n'
        else:
            actuator_xml = f'<{self.type} name="{self.name}" '
            if self.tendon:
                actuator_xml += f'tendon="{self.tendon}" '
            actuator_xml += f'gainprm="{self.gainprm}" biasprm="{self.biasprm}" '
            actuator_xml += 'biastype="affine" '
            actuator_xml += '/>\n'
        return actuator_xml
    
class Inertial(Object):
    def __init__(
            self,
            pos: Tuple[float, float, float] = (0.0, 0.0, 0.0),
            mass: float = 0.0,
            diaginertia: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        ):

        super().__init__(name=None)
        self.pos = pos
        self.mass = mass
        self.diaginertia = diaginertia
