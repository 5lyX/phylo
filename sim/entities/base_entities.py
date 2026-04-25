"""
This module extends the 'Body' class to define 'Entity' and its specialized subclasses for
representing complex simulation units within the Mujoco physics engine. An 'Entity' is
composed of one or more 'Bodies', each equipped with elements like geometries (geoms),
sites, and joints, to form larger and more complex simulation units.

These entities are capable of representing intricate mechanisms such as directed masses that
can move in specified directions, masses combined with fixed or movable pulleys, and other
configurations that require dynamic interaction between multiple bodies.

The provided classes allow for the creation and manipulation of these entities, facilitating
the building of detailed simulation scenarios that can represent a variety of mechanical
and physical systems.

Classes:
    Entity: A base class for complex entities, extending basic 'Body' functionalities.
    DirectedMass: Represents a mass that moves in a specified direction, with mechanisms for
                  attaching pulleys and generating XML representations for simulations.
    MassWithFixedPulley: Configures a mass with a fixed pulley system, allowing for complex
                         simulations involving tension and mechanical advantage.
    MassWithMovablePulley: Similar to MassWithFixedPulley but includes movable pulleys to
                           simulate systems with variable mechanical configurations and dynamics.

Usage:
    The classes are designed to be integrated within larger Mujoco XML setups, such as Scenes.
"""

from sim.bodies import *
from dataclasses import dataclass
from sim.geometry_utils import Frame
import random
import inspect
import yaml
import math
import numpy as np
import ipdb
from sim.utils import convert_list_to_natural_language

st = ipdb.set_trace

ENTITY_CONNECTIONS = {
    "FixedPulleyEntity": {
        ConnectingPoint.DEFAULT: {
            "ConnectingDirection": [
                ConnectingDirection.INNER_TO_OUTER,
                ConnectingDirection.OUTER_TO_INNER,
                ConnectingDirection.LEFT_TO_RIGHT,
                ConnectingDirection.RIGHT_TO_LEFT,
            ],
            "ConnectingPointNum": 1,
        }
    },
    "ConstantForceFixedPulley": {
        ConnectingPoint.DEFAULT: {
            "ConnectingDirection": [
                ConnectingDirection.INNER_TO_OUTER,
                ConnectingDirection.OUTER_TO_INNER,
                ConnectingDirection.LEFT_TO_RIGHT,
                ConnectingDirection.RIGHT_TO_LEFT,
            ],
            "ConnectingPointNum": 1,
        }
    },
    "DirectedMass": {
        ConnectingPoint.SIDE_1: {
            "ConnectingDirection": [
                ConnectingDirection.INNER_TO_OUTER,
                ConnectingDirection.OUTER_TO_INNER,
            ],
            "ConnectingPointNum": 1,
        },
        ConnectingPoint.SIDE_2: {
            "ConnectingDirection": [
                ConnectingDirection.INNER_TO_OUTER,
                ConnectingDirection.OUTER_TO_INNER,
            ],
            "ConnectingPointNum": 1,
        },
    },
    "MassWithFixedPulley": {
        ConnectingPoint.DEFAULT: {
            "ConnectingDirection": [
                ConnectingDirection.INNER_TO_OUTER,
                ConnectingDirection.OUTER_TO_INNER,
            ],
            "ConnectingPointNum": 1,
        }
    },
    "MassWithMovablePulley": {
        ConnectingPoint.DEFAULT: {
            "ConnectingDirection": [
                ConnectingDirection.LEFT_TO_RIGHT,
                ConnectingDirection.RIGHT_TO_LEFT,
            ],
            "ConnectingPointNum": 1,
        },
        ConnectingPoint.RIGHT: {
            "ConnectingDirection": [
                ConnectingDirection.OUTER_TO_INNER,
                ConnectingDirection.INNER_TO_OUTER,
            ],
            "ConnectingPointNum": 1,
        },
    },
    "TwoSideMassPlane": {
        ConnectingPoint.LEFT: {
            "ConnectingDirection": [
                ConnectingDirection.INNER_TO_OUTER,
                ConnectingDirection.OUTER_TO_INNER,
            ],
            "ConnectingPointNum": 1,
        },
        ConnectingPoint.RIGHT: {
            "ConnectingDirection": [
                ConnectingDirection.INNER_TO_OUTER,
                ConnectingDirection.OUTER_TO_INNER,
            ],
            "ConnectingPointNum": 1,
        },
    },
    "MassPrismPlaneEntity": {
        ConnectingPoint.LEFT: {
            "ConnectingDirection": [
                ConnectingDirection.INNER_TO_OUTER,
                ConnectingDirection.OUTER_TO_INNER,
            ],
            "ConnectingPointNum": 1,
        },
        ConnectingPoint.RIGHT: {
            "ConnectingDirection": [
                ConnectingDirection.INNER_TO_OUTER,
                ConnectingDirection.OUTER_TO_INNER,
            ],
            "ConnectingPointNum": 1,
        },
    },
    "ComplexMovablePulley": {
        ConnectingPoint.DEFAULT: {
            "ConnectingDirection": [
                ConnectingDirection.LEFT_TO_RIGHT,
                ConnectingDirection.RIGHT_TO_LEFT,
            ],
            "ConnectingPointNum": 1,
        },
        ConnectingPoint.TOP: {
            "ConnectingDirection": [
                ConnectingDirection.INNER_TO_OUTER,
                ConnectingDirection.OUTER_TO_INNER,
            ],
            "ConnectingPointNum": 1,
        },
        ConnectingPoint.BOTTOM: {
            "ConnectingDirection": [
                ConnectingDirection.INNER_TO_OUTER,
                ConnectingDirection.OUTER_TO_INNER,
            ],
            "ConnectingPointNum": 1,
        },
    },
    "PulleyGroupEntity": {
        ConnectingPoint.DEFAULT: {
            "ConnectingDirection": [
                ConnectingDirection.LEFT_TO_RIGHT,
                ConnectingDirection.RIGHT_TO_LEFT,
            ],
            "ConnectingPointNum": 1,
        }
    },
    "StackedMassPlane": {
        ConnectingPoint.LEFT: {
            "ConnectingDirection": [
                ConnectingDirection.INNER_TO_OUTER,
                ConnectingDirection.OUTER_TO_INNER,
            ],
            "ConnectingPointNum": 5,
        },
        ConnectingPoint.RIGHT: {
            "ConnectingDirection": [
                ConnectingDirection.INNER_TO_OUTER,
                ConnectingDirection.OUTER_TO_INNER,
            ],
            "ConnectingPointNum": 5,
        },
    },
    "MassWithReversedMovablePulley": {
        ConnectingPoint.TOP: {
            "ConnectingDirection": [
                ConnectingDirection.INNER_TO_OUTER,
                ConnectingDirection.OUTER_TO_INNER,
            ],
            "ConnectingPointNum": 1,
        },
        # TODO: add side_1 and side_2 if needed
    },
    "TwoDCollisionPlane": {},  # No connecting points for this entity
    "SliderWithArchPlaneSpheres": {},  # No connecting points for this entity
    "ComplexCollisionPlane": {
        ConnectingPoint.LEFT: {
            "ConnectingDirection": [
                ConnectingDirection.INNER_TO_OUTER,
                ConnectingDirection.OUTER_TO_INNER,
            ],
            "ConnectingPointNum": 1,
        },
        ConnectingPoint.RIGHT: {
            "ConnectingDirection": [
                ConnectingDirection.INNER_TO_OUTER,
                ConnectingDirection.OUTER_TO_INNER,
            ],
            "ConnectingPointNum": 1,
        },
    },  # No connecting points for this entity
    "SpringBlockEntity": {
        ConnectingPoint.DEFAULT: {
            "ConnectingDirection": [
                ConnectingDirection.INNER_TO_OUTER,
                ConnectingDirection.OUTER_TO_INNER,
            ],
            "ConnectingPointNum": 5,
        },
        ConnectingPoint.SURROUNDING: {
            "ConnectingDirection": [
                ConnectingDirection.INNER_TO_OUTER,
                ConnectingDirection.OUTER_TO_INNER,
            ],
            "ConnectingPointNum": 5,
        },
    },
    "RigidRotationEntity": {
    },
    "BarPlaneSupport": {
    },
    "DiskRackWithSphereEntity": {
    },
    "PendulumEntity": {
    },
    "OrbitalMotionEntity": {
    },
    "GeneralCelestialEntity": {
    },
    "RollingPlaneEntity": {
    },
    "ThrowingMotionEntity": {
    },
    "MassBoxPlaneEntity": {
        ConnectingPoint.RIGHT: {
            "ConnectingDirection": [
                ConnectingDirection.INNER_TO_OUTER,
                ConnectingDirection.OUTER_TO_INNER,
            ],
            "ConnectingPointNum": 1,
        },
    },
    "MassPrismPulleyPlane": {
    },
    "ConstantVelocityPuller": {
        ConnectingPoint.DEFAULT: {
            "ConnectingDirection": [
                ConnectingDirection.LEFT_TO_RIGHT,
                ConnectingDirection.RIGHT_TO_LEFT,
            ],
            "ConnectingPointNum": 1,
        },
        ConnectingPoint.RIGHT: {
            "ConnectingDirection": [
                ConnectingDirection.INNER_TO_OUTER,
                ConnectingDirection.OUTER_TO_INNER,
            ],
            "ConnectingPointNum": 1,
        },
    },
    "SpringMassPlaneEntity": {
    },
    "SolarSystemEntity": {
    },
    "RocketEntity": {
    },
    "ThrowingMotionEntity": {
    },
    "MagneticElectricEntity": {
    },
    "ElectroMagneticEntity": {
    },
}


def create_mass_body(
    name: str,
    mass_type: str,
    positions: List[Tuple[float, float, float]],
    mass_values: List[float],
    joint_option: Tuple[str, Tuple[float, float, float]] = ("slide", (0, 0, 1)),
    plane_slope: float = 0,
    prism_left_slope: float = 30,
    prism_right_slope: float = 60,
    prism_mass_value: float = 1,
    use_left_site: DirectionsEnum = DirectionsEnum.USE_LEFT,
    use_prism_left: bool = True,
    padding_z: float = DEFAULT_PULLEY_RADIUS + 2 * DEFAULT_MASS_SIZE,
    condim: str = "1",
    use_bottom_site: bool = False,
    constant_force: Optional[Dict[str, List[Union[List, float]]]] = None,
    init_velocity: Optional[Dict[str, List[Union[List, float]]]] = None,
) -> Body:
    if mass_type == "Mass":
        mass = Mass(
            name=name,
            positions=positions,
            joint_option=joint_option,
            mass_value=mass_values[0],
            padding_z=padding_z,
            use_bottom_site=use_bottom_site,
            constant_force=constant_force,
            init_velocity=init_velocity,
        )
    elif mass_type == "MassPlane":
        mass = MassPlane(
            name=name + "_plane",
            plane_slope=plane_slope,
            mass_values=mass_values,
            use_left_site=use_left_site,
            positions=positions,
            padding_z=padding_z,
            condim=condim,
            constant_force=constant_force,
            init_velocity=init_velocity,
        )
    elif mass_type == "MassPrismPlane":
        mass = MassPrismPlane(
            name=name + "_prism_plane",
            plane_slope=plane_slope,
            prism_left_slope=prism_left_slope,
            prism_right_slope=prism_right_slope,
            block_mass_value=mass_values[0],
            prism_mass_value=prism_mass_value,
            use_left_site=use_left_site,
            use_prism_left=use_prism_left,
            positions=positions,
            padding_z=padding_z,
            condim=condim,
            constant_force=constant_force,
            init_velocity=init_velocity,
        )
    else:
        raise ValueError(f"Unsupported mass_type: {mass_type}")

    return mass


def round_floats(obj):
    """Recursively traverse the object and round all float values to 2 decimal places."""
    if isinstance(obj, float):
        return round(obj, 2)
    elif isinstance(obj, dict):
        return {key: round_floats(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [round_floats(element) for element in obj]
    else:
        return obj


class ConnectPoint:
    def __init__(
        self,
        connecting_point: ConnectingPoint = ConnectingPoint.DEFAULT,
        connecting_point_seq_id: ConnectingPointSeqId = 1,
        direction: ConnectingDirection = ConnectingDirection.DEFAULT,
        entity_type: str = "",
    ):
        self.connecting_point = connecting_point
        self.connecting_point_seq_id = connecting_point_seq_id
        self.direction = direction
        self.entity_type = entity_type

@dataclass
class PulleyParam:
    angle: float
    distance: float
    side: str
    offset: float

class FrictionType(Enum):
    DEFAULT = 0
    ROLLING = 1

class Entity(Body):
    def __init__(
        self,
        name: str,
        pos: Tuple[float, float, float] = (0, 0, 0),
        quat: Tuple[float, float, float, float] = (1, 0, 0, 0),
        entity_type: str = "",
        constant_force: Optional[Dict[str, List[Union[List, float]]]] = None,
        init_randomization_degree: DegreeOfRandomization = None,
        **kwargs,
    ):
        super().__init__(name, pos, quat, **kwargs)
        self.entity_type = entity_type
        if (
            init_randomization_degree
        ):  # TODO: There reinit may cause some performance overhead
            self.randomize_parameters(
                degree_of_randomization=init_randomization_degree,
                reinitialize_instance=True,
            )
        self.constant_force = constant_force
        self.available_connecting_points: List[ConnectPoint] = []
        self.used_connecting_points = []
        self.initialize_connecting_points()
        if not hasattr(self, "resolution_coefficient_list"): self.resolution_coefficient_list = []  # [(collision_body_1, collision_body_2, restitution),...]
        if not hasattr(self, "friction_coefficient_list"): self.friction_coefficient_list = []  # [(collision_body_1, collision_body_2, restitution),...]
        if not hasattr(self, "friction_type"): self.friction_type = FrictionType.DEFAULT
        if not hasattr(self, "trail_bodies"): self.trail_bodies = []  # List of bodies that will be used to create trails in the visualization

    def initialize_connecting_points(self, connection_constraints: dict = {}):
        """
        connection_constraints: {ConnectingPoint: MaxConnectingPointNum}
        """
        self.available_connecting_points.clear()
        if self.entity_type not in ENTITY_CONNECTIONS:
            raise ValueError(f"Unsupported entity type: {self.entity_type}")
        for connecting_point, connecting_point_info in ENTITY_CONNECTIONS[
            self.entity_type
        ].items():
            max_connecting_points = (
                connection_constraints[connecting_point]
                if connecting_point in connection_constraints
                else connecting_point_info["ConnectingPointNum"]
            )  # get the max_connecting_points_num from connection_constraints, otherwise from connecting_point_info
            for i in range(max_connecting_points):
                for conect_point_dir in connecting_point_info["ConnectingDirection"]:
                    connect_point = ConnectPoint(
                        connecting_point=connecting_point,
                        connecting_point_seq_id=i + 1,
                        entity_type=self.entity_type,
                        direction=conect_point_dir
                    )
                    self.available_connecting_points.append(connect_point)

    def get_available_connecting_points_num(
        self, required_directions: List[ConnectingDirection] = None
    ) -> int:
        """
        Description: Get the number of available connecting points in the specified directions.
        Returns the count of matching available connecting points.
        """
        available_connect_point_set = set()

        for connect_point in self.available_connecting_points:
            if (not required_directions) or (connect_point.direction in required_directions):
                available_connect_point_set.add((connect_point.connecting_point, connect_point.connecting_point_seq_id))

        return len(available_connect_point_set)

    def check_connecting_point_availability(
        self, required_directions: List[ConnectingDirection] = None
    ) -> bool:
        """
        Description: Check if there are any available connecting points in the specified directions.
        Returns True if at least one available connecting point is found; otherwise, False.
        """
        # Use `get_available_connecting_points_num` to check availability
        return self.get_available_connecting_points_num(required_directions) > 0

    def get_next_connecting_point(
        self, required_directions: List[ConnectingDirection] = None
    ) -> ConnectPoint:
        """
        Description: Get the next available connecting point in the specified directions
        Note: currently always return the first available connecting point
        """
        if len(self.available_connecting_points) == 0:
            raise ValueError("No available connecting points")
        target_connect_point = None
        for connect_point in self.available_connecting_points:
            if (not required_directions) or (len(required_directions) == 0) or (connect_point.direction in required_directions):
                # find the target connect point
                target_connect_point = connect_point
                break
    
        if target_connect_point is None:
            print("No available connecting points in the specified directions")
            return None
        # delete the target_connect_point in self.available_connecting_points
        self.available_connecting_points = [
            connect_point for connect_point in self.available_connecting_points
            if not (connect_point.connecting_point == target_connect_point.connecting_point and
                    connect_point.connecting_point_seq_id == target_connect_point.connecting_point_seq_id)
        ]
        # add the used connect_point to self.used_connecting_points
        self.used_connecting_points.append(target_connect_point)
        return target_connect_point

    def reinitialize(self):
        # Get the signature of the __init__ method
        init_signature = inspect.signature(self.__init__)

        # Collect the current values of the attributes corresponding to the __init__ parameters
        init_params = {
            param.name: getattr(self, param.name, None)
            for param in init_signature.parameters.values()
            if param.name != "self" and hasattr(self, param.name)
        }

        # Call the __init__ method with the collected parameters
        self.__init__(**init_params)

    def get_connecting_tendon_sequences(
        self, direction: ConnectingDirection = ConnectingDirection.DEFAULT, connecting_option: Any = None
    ) -> List[TendonSequence]:
        raise RuntimeError(
            "'get_connecting_tendon_sequences' is disabled for Entity. Please use 'get_connecting_tendon_sequence' instead."
        )

    def get_connecting_tendon_sequence(
        self,
        direction: ConnectingDirection,
        connecting_point: ConnectingPoint = ConnectingPoint.DEFAULT,
        connecting_point_seq_id: Optional[ConnectingPointSeqId] = None,
        use_sidesite: bool = False,
    ) -> TendonSequence:
        return TendonSequence()

    def get_parameters(self) -> List[dict]:
        """
        Get the parameters of the entity in a list of dictionaries
        eg:
            [
                {
                    'body_name': 'xxx_mass_prism_plane_mass',
                    'geom_name': 'xxx_mass_prism_plane_mass_geom',
                    'mass': 1.0,
                    'prism_slope': 30,
                    'plane_slope': 30
                }
            ]

        """
        list_of_parameters = []
        for attr, value in self.__dict__.items():
            if isinstance(value, Body):
                mass_dict_list = value.get_masses_quality()
                list_of_parameters.extend(mass_dict_list)
            elif isinstance(value, list) and all(
                isinstance(item, Entity) for item in value
            ):
                for entity_item in value:
                    mass_dict_list = entity_item.get_parameters()
                    list_of_parameters.extend(mass_dict_list)
            elif isinstance(value, list) and all(
                isinstance(item, Body) for item in value
            ):
                for body_item in value:
                    mass_dict_list = body_item.get_masses_quality()
                    list_of_parameters.extend(mass_dict_list)

        return list_of_parameters
    
    def get_description(self, simDSL2nlq = False) -> List[dict]:
        """
        Get the description for each body of the entity in a list of dictionaries
        eg:
            [
                {
                    'body_name': 'xxx_mass_prism_plane_mass',
                    'mass': 1.0,
                    'prism_name': 'xxx_mass_prism_plane_prism',
                    'prism_slope': 30,
                    'plane_name': 'xxx_mass_prism_plane_plane',
                    'plane_slope': 30,
                    'init_velocity': [0, 0, 0, 0, 0, 0],
                    'constant_force': [0, 0, 0, 0, 0, 0]
                    'init_angle': 0
                }
            ]

        """
        list_of_bodies = []
        for attr, value in self.__dict__.items():
            if isinstance(value, Body):
                mass_dict_list = value.get_description()
                list_of_bodies.extend(mass_dict_list)
            elif isinstance(value, list) and all(
                isinstance(item, Entity) for item in value
            ):
                for entity_item in value:
                    mass_dict_list = entity_item.get_description()
                    list_of_bodies.extend(mass_dict_list)
            elif isinstance(value, list) and all(
                isinstance(item, Body) for item in value
            ):
                for body_item in value:
                    mass_dict_list = body_item.get_description()
                    list_of_bodies.extend(mass_dict_list)

        for body in list_of_bodies:
             body["description"] = (
                f"The {body['body_type']} named {body['name']} is a component of the entity {self.name}. "
                f'{body["description"]}'
            )

        return list_of_bodies

    def get_constant_forces(self) -> Dict:
        # Used to store the constant forces of all sub-objects
        constant_force_dict = {}
        
        # Iterate over all attributes of the current object
        for attr, value in self.__dict__.items():
            # If the attribute is of type Body, directly get constant_force_dict
            if isinstance(value, Body) or isinstance(value, Entity):
                constant_force_dict.update(value.get_constant_forces())
            
            if isinstance(value, Entity):
                constant_force_dict.update(value.get_constant_forces())
            elif isinstance(value, Body):
                constant_force_dict.update(value.get_constant_forces())
                if value.child_bodies != []: # if the body has child bodies also get their constant forces
                    for child_body in value.child_bodies:
                        constant_force_dict.update(child_body.get_constant_forces())


            # If the attribute is a list, check the type of its elements
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, Entity):
                        constant_force_dict.update(item.get_constant_forces())
                    elif isinstance(item, Body):
                        constant_force_dict.update(item.get_constant_forces())
                        if item.child_bodies != []: # if the body has child bodies also get their constant forces
                            for child_body in item.child_bodies:
                                constant_force_dict.update(child_body.get_constant_forces())

            # If the attribute is a dictionary, check the type of its values
            elif isinstance(value, dict):
                for key, item in value.items():
                    if isinstance(item, Body) or isinstance(item, Entity):
                        constant_force_dict.update(item.get_constant_forces())

        return constant_force_dict

    def get_springs(self) -> List:
        # Used to store all springs from sub-objects
        springs = []
        springs.extend(self.springs)

        # Iterate over all attributes of the current object
        for attr, value in self.__dict__.items():
            # If the attribute is of type Body, directly get its springs
            if isinstance(value, Body) or isinstance(value, Entity):
                springs.extend(value.get_springs())

            # If the attribute is a list, check the type of its elements
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, Body) or isinstance(item, Entity):
                        springs.extend(item.get_springs())

            # If the attribute is a dictionary, check the type of its values
            elif isinstance(value, dict):
                for item in value.values():
                    if isinstance(item, Body) or isinstance(item, Entity):
                        springs.extend(item.get_springs())

        return springs

    def get_resolution_coefficients(self) -> List:
        return self.resolution_coefficient_list
    
    def get_friction_coefficients(self) -> List:
        return self.friction_coefficient_list
    
    def get_actuator(self) -> Actuator:
        # if actuator is an attribute of the entity, return it
        if hasattr(self, 'actuator'):
            return self.actuator
        else:
            return None

    def get_init_velocities(self) -> Dict:
        # Initialize a dictionary to store the initial velocities of all sub-objects
        init_velocity_dict = {}

        # Iterate over all the attributes of the current object
        for attr, value in self.__dict__.items():
            # If the attribute is of type Body, directly update the init_velocity_dict with the body object's init velocity dictionary
            if isinstance(value, Entity):
                init_velocity_dict.update(value.get_init_velocities())
            elif isinstance(value, Body):
                init_velocity_dict.update(value.get_init_velocities())
                if value.child_bodies != []: # if the body has child bodies also get their initial velocities
                    for child_body in value.child_bodies:
                        init_velocity_dict.update(child_body.get_init_velocities())

            # # If the attribute is a list, check the type of each element
            # elif isinstance(value, list):
            #     for item in value:
            #         # If the list element is a Body, update the dictionary with the body's initial velocity dictionary
            #         if isinstance(item, Body) or isinstance(item, Entity):
            #             init_velocity_dict.update(item.get_init_velocities())

            # TODO: mimic get_constant_forces() to get init_velocity_dict from the child bodies
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, Entity):
                        init_velocity_dict.update(item.get_init_velocities())
                    elif isinstance(item, Body):
                        init_velocity_dict.update(item.get_init_velocities())
                        if item.child_bodies != []: # if the body has child bodies also get their initial velocities
                            for child_body in item.child_bodies:
                                init_velocity_dict.update(child_body.get_init_velocities())

            # If the attribute is a dictionary, check the type of each value
            elif isinstance(value, dict):
                for key, item in value.items():
                    # If the dictionary value is a Body, update the dictionary with the body's initial velocity dictionary
                    if isinstance(item, Body) or isinstance(item, Entity):
                        init_velocity_dict.update(item.get_init_velocities())

        # Return the dictionary containing the initial velocities of all sub-objects
        return init_velocity_dict

    def get_attraction_forces(self) -> List[Tuple[str, str, str, float]]:
        """
        Return a list of tuples describing pairwise attraction forces in this entity.
        Each tuple should look like:
            (body_A_name, body_B_name, force_type, default_force_value)
        By default, return an empty list for entities that do not define attraction forces.
        """
        return []

    def randomize_constant_forces(
        self,
        force_limit: List[float] = [1, 1, 1],
    ):
        # 80% chance to have None constant_force, and 20% chance to have a random constant_force
        self.constant_force = {}
        for force_type in ConstantForceType():
            if random.random() < 0.2:
                self.constant_force[force_type] = [
                    random.uniform(-force_limit[0] / 2, force_limit[0] / 2),
                    random.uniform(-force_limit[1] / 2, force_limit[1] / 2),
                    random.uniform(-force_limit[2] / 2, force_limit[2] / 2),
                    0,
                    0,
                    0,
                ]

        if len(self.constant_force) == 0:
            self.constant_force = None

    def randomize_parameters(
        self,
        degree_of_randomization: DegreeOfRandomization = DegreeOfRandomization.DEFAULT,
        reinitialize_instance=False,
        **kwargs,
    ):
        pass

    def get_sensor_list(self) -> List[Sensor]:
        # Initialize an empty list to collect all sensors
        sensor_list = []
        # Iterate through each attribute in the object's dictionary
        for attr, value in self.__dict__.items():
            # If the value is an instance of Body, directly call its get_sensor_list
            if isinstance(value, Body):
                sensor_list.extend(value.get_sensor_list())
            # If the value is a list, check if it contains Entity instances and recursively collect their sensors
            elif isinstance(value, list) and all(
                isinstance(item, Entity) for item in value
            ):
                for entity in value:
                    sensor_list.extend(entity.get_sensor_list())
            elif isinstance(value, list) and all(
                isinstance(item, Body) for item in value
            ):
                for body in value:
                    sensor_list.extend(body.get_sensor_list())
        return sensor_list

    def get_bodies(self) -> Dict:
        # Used to store the body names of all sub-objects
        body_dict = {}  # {body_name: {}, ...}

        # Iterate over all attributes of the current object
        for attr, value in self.__dict__.items():
            # If the attribute is of type Body, directly get the body name
            if isinstance(value, Entity):
                body_dict.update(value.get_bodies())

            elif isinstance(value, Body):
                if value.child_bodies != []:
                    for child_body in value.child_bodies:
                        body_dict.update(child_body.get_bodies())
                else:
                    body_dict.update(value.get_bodies())

            # If the attribute is a list, check the type of its elements
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, Entity):
                        body_dict.update(item.get_bodies())
                    elif isinstance(item, Body):
                        if item.child_bodies != []:
                            for child_body in item.child_bodies:
                                body_dict.update(child_body.get_bodies())
                        else:
                            body_dict.update(item.get_bodies())

            # If the attribute is a dictionary, check the type of its values
            elif isinstance(value, dict):
                for key, item in value.items():
                    if isinstance(item, Entity):
                        body_dict.update(item.get_bodies())
                    elif isinstance(item, Body):
                        if item.child_bodies != []:
                            for child_body in item.child_bodies:
                                body_dict.update(child_body.get_bodies())
                        else:
                            body_dict.update(item.get_bodies())

        return body_dict
    
    def connecting_point_nl(self, cd: ConnectingDirection, cp: ConnectingPoint, csi: int):
        '''
        Converts connecting info to meaningful natural language

        Input: 
            cd: ConnectingDirection
            cp: ConnectingPoint
            csi: ConnectingPointSeqId
        '''

        # placeholder function
        return cp.value

    def get_shortcut(self):
        return False
    
def get_all_geoms_in_entity(entity: Entity):
    """
    Get all geoms in the body.
    """
    geoms = list(entity.geoms)
    
    # Find all attributes of type Body in body
    for attr in dir(entity):
        if isinstance(getattr(entity, attr), Entity):
            # Recursively get geoms from the nested entity
            nested_entity = getattr(entity, attr)
            nested_geoms = get_all_geoms_in_body(nested_entity)
            geoms.extend(nested_geoms)
        elif isinstance(getattr(entity, attr), Body):
            # Recursively get geoms from the nested body
            nested_body = getattr(entity, attr)
            nested_geoms = get_all_geoms_in_body(nested_body)
            geoms.extend(nested_geoms)
        elif isinstance(getattr(entity, attr), list):
            entities = [item for item in getattr(entity, attr) if isinstance(item, Entity)]
            bodies = [item for item in getattr(entity, attr) if isinstance(item, Body) and item not in entities]
            
            for entity_item in entities:
                nested_geoms = get_all_geoms_in_entity(entity_item)
                geoms.extend(nested_geoms)
            
            for body_item in bodies:
                nested_geoms = get_all_geoms_in_body(body_item)
                geoms.extend(nested_geoms)
    
    return geoms