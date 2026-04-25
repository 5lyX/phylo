from enum import Enum
import os
import yaml
"""
constant variables
"""

def _load_rendering_mode():
    """read rendering_mode from config.yaml"""
    try:
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "config.yaml")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                rendering_mode = config.get('rendering_mode', False)
                print(f"[Constants] Read rendering_mode from config.yaml: {rendering_mode}")
                return rendering_mode
        else:
            print(f"[Constants] config.yaml not found at {config_path}, using default rendering_mode=False")
            return False
    except Exception as e:
        print(f"[Constants] Error reading config.yaml: {e}, using default rendering_mode=False")
        return False

RENDERING_MODE = _load_rendering_mode()

if not RENDERING_MODE:
    print("[Constants] Initializing constants for NON-RENDERING MODE")
    # the default material for the geom
    DEFAULT_MATERIAL = "pulley"
    # the default rgba for the geom
    DEFAULT_RGBA = (1, 1, 1, 1)
    # the default radius for the pulley
    DEFAULT_PULLEY_RADIUS = 0.05
    # the default height for the pulley
    DEFAULT_PULLEY_HEIGHT = 0.05
    DEFAULT_ROPE_THICKNESS = 0.01
    DEFAULT_ROPE_LENGTH = 10
    DEFAULT_MASS_SIZE = 0.1
    # spring constant
    DEFAULT_SPRING_THICKNESS = 0.01
    # length of the plane
    DEFAULT_PLANE_LENGTH = 100
    DEFAULT_COLLISION_PLANE_LENGTH = 100
    DEFAULT_PLANE_THICKNESS = 0.1
    DEFAULT_PLANE_WIDTH = 0.4
    # height of the prism
    DEFAULT_PRISM_HEIGHT = 5
    DEFAULT_PRISM_THICKNESS = 0.1
    DEFAULT_PRISM_WIDTH = 0.3
    # length of the stacked mass
    STACKED_MASS_START_LENGTH = DEFAULT_PLANE_LENGTH / 5
    DEFAULT_SPHERE_RADIUS = 0.1
    DEFAULT_POLYGON_SIDES = 6
    DEFAULT_POLYGON_RADIUS = 0.1
    DEFAULT_CYLINDER_RADIUS = 0.05
    DEFAULT_CYLINDER_HEIGHT = 0.2
    DEFAULT_RGBA = (0.8, 0.8, 0.8, 1.0)

    DEFAULT_DISC_RADIUS = 1
    DEFAULT_DISC_HEIGHT = 0.01

    DEFAULT_BAR_LENGTH = 0.5
    DEFAULT_BAR_THICKNESS = 0.01
    DEFAULT_MIN_COLLISION_BODY_DISTANCE = 0.3

    DEFAULT_BOX_PADDING_LENGTH = 20.0
    DEFAULT_BOX_PADDING_WIDTH = 0.5
    DEFAULT_BOX_PADDING_HEIGHT = 20.0
    DEFAULT_MASS_BOX_PLANE_TOP_MASS_SIZE = 0.1
    DEFAULT_MASS_BOX_PLANE_PULLEY_OFFSET = 0.1
    MESH_MASS_OFFSET = 1

    COLLISION_BODY_POSITION_RANGE = {
        "min": -10,
        "max": 10
    }
    MASS_POSITION_SCALE = 0.2

else: # rendering mode
    print("[Constants] Initializing constants for RENDERING MODE")
    # the default material for the geom
    DEFAULT_MATERIAL = "pulley"
    # the default rgba for the geom
    DEFAULT_RGBA = (1, 1, 1, 1)
    # the default radius for the pulley
    DEFAULT_PULLEY_RADIUS = 0.05
    # the default height for the pulley
    DEFAULT_PULLEY_HEIGHT = 0.05
    DEFAULT_ROPE_THICKNESS = 0.01
    DEFAULT_ROPE_LENGTH = 1.5
    DEFAULT_MASS_SIZE = 0.1
    # spring constant
    DEFAULT_SPRING_THICKNESS = 0.01
    # length of the plane
    DEFAULT_PLANE_LENGTH = 3
    DEFAULT_COLLISION_PLANE_LENGTH = 5
    DEFAULT_PLANE_THICKNESS = 0.1
    DEFAULT_PLANE_WIDTH = 4  # 0.4
    # height of the prism
    DEFAULT_PRISM_HEIGHT = 0.5
    DEFAULT_PRISM_THICKNESS = 0.1
    DEFAULT_PRISM_WIDTH = 0.5 # 0.15
    # length of the stacked mass
    STACKED_MASS_START_LENGTH = DEFAULT_PLANE_LENGTH / 5
    DEFAULT_SPHERE_RADIUS = 0.1
    DEFAULT_POLYGON_SIDES = 6
    DEFAULT_POLYGON_RADIUS = 0.1
    DEFAULT_CYLINDER_RADIUS = 0.05
    DEFAULT_CYLINDER_HEIGHT = 0.2
    DEFAULT_RGBA = (0.8, 0.8, 0.8, 1.0)

    DEFAULT_DISC_RADIUS = 1
    DEFAULT_DISC_HEIGHT = 0.01

    DEFAULT_BAR_LENGTH = 0.5
    DEFAULT_BAR_THICKNESS = 0.01
    DEFAULT_MIN_COLLISION_BODY_DISTANCE = 0.3

    DEFAULT_BOX_PADDING_LENGTH = 1 # 0.5
    DEFAULT_BOX_PADDING_WIDTH = 0.5 # 0.3
    DEFAULT_BOX_PADDING_HEIGHT = 1 # 0.5
    DEFAULT_MASS_BOX_PLANE_TOP_MASS_SIZE = 0.1
    DEFAULT_MASS_BOX_PLANE_PULLEY_OFFSET = 0.1
    MESH_MASS_OFFSET = 1

    COLLISION_BODY_POSITION_RANGE = {
        "min": -3,
        "max": 3
    }
    MASS_POSITION_SCALE = 0.3

# TODO: uncomment the following lines for correctness test/ visualization test
# DEFAULT_ROPE_LENGTH = 1
# DEFAULT_PLANE_LENGTH = 5

GEOM_FIXED_SOURCES_PATH = "sim/geom_sources/fixed_sources"
GEOM_GENERATED_SOURCES_PATH = "sim/geom_sources/generated_sources"
REPO_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class DirectionsEnum(Enum):
    """
    A class to define the directions of the pulleys.
    """

    DEFAULT = 0
    USE_LEFT = 1
    USE_RIGHT = 2
    USE_BOTH = 3


class ConnectingPoint(Enum):
    DEFAULT = "default"
    SIDE_1 = "side_1"
    SIDE_2 = "side_2"
    TOP = "top"
    BOTTOM = "bottom"
    LEFT = "left"
    RIGHT = "right"
    SURROUNDING = "surrounding"


class ConnectingDirection(Enum):
    DEFAULT = "default"
    LEFT_TO_RIGHT = "left_to_right"
    RIGHT_TO_LEFT = "right_to_left"
    INNER_TO_OUTER = "inner_to_outer"
    OUTER_TO_INNER = "outer_to_inner"


class PulleyGroupEntityStartPoint(Enum):
    TOP_FIXED_PULLEY = "top_fixed_pulley"
    BOTTOM_MOVABLE_PULLEY_TOP_SITE = "bottom_movable_pulley_top_site"


class DegreeOfRandomization(Enum):
    DEFAULT = 0
    NON_STRUCTURAL = 1
    EASY = 2
    MEDIUM = 3
    HARD = 4


class SensorType(Enum):
    ACC = "acc"
    FORCE = "force"
    TENDONLIMITFRC = "tendonlimitfrc"


def iterable(cls):
    def __iter__(cls_inner):
        return iter(
            getattr(cls_inner, attr)
            for attr in dir(cls_inner)
            if not attr.startswith("__") and not callable(getattr(cls_inner, attr))
        )
    
    def from_value(cls_inner, value):
        for k, v in  vars(cls_inner).items():
            if not k.startswith('_') and v == value:
                return getattr(cls_inner, k)
            
        raise ValueError(f"Value {value} not found in {cls_inner.__name__}")

    cls.__iter__ = classmethod(__iter__)
    cls.from_value = classmethod(from_value)
    return cls


@iterable
class ConstantForceType:
    PULLEY = "pulley"
    MASS = "mass"
    PRISM = "prism"
    SPHERE = "sphere"

@iterable
class HangOption(Enum):
    HANG_LEFT = "hang_left"
    HANG_RIGHT = "hang_right"
    HANG_BOTH = "hang_both"

@iterable
class InitVelocityType:
    MASS = "mass"
    SPHERE = "sphere"
    PRISM = "prism"
    DISK = "disk"

@iterable
class SpringDirection:
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"

ConnectingPointSeqId = int
