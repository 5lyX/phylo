from .base_bodies import *
from .geom_bodies import *

class SliderWithArch(Body):
    """
    A specialized Body class to represent a slider with an arch attached to its left side.
    """

    def __init__(
        self,
        name: str,
        pos: Tuple[float, float, float] = (0, 0, 0),
        slide_length: float = 4.0,  # Length of the slider
        arch_radius: float = 0.5,  # Radius of the arch
        arch_stl_path: str = f"{GEOM_FIXED_SOURCES_PATH}/round_arch.stl",
        quat: Tuple[float, float, float, float] = (1, 0, 0, 0),
        mass: float = 1.0,
        rgba: Tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1),
        **kwargs,
    ) -> None:
        super().__init__(name, pos, quat, **kwargs)

        arch_size = (arch_radius, arch_radius, arch_radius)  # Size of the arch
        arch_size_adjusted = (
            arch_radius,
            arch_radius,
            arch_radius * 1.2,
        )  # height adjusted size of the arch
        size = (
            slide_length,
            1,
            arch_radius * 0.2,  # should be larger than 0.2
        )  # the height should be higher than the arch thickness

        # Create the slider block geom
        slider_geom = Geom(
            name=f"{name}.slider",
            geom_type="box",
            size=size,  # MuJoCo uses half-sizes
            pos=(0, 0, 0),  # Centered at the body's origin
            mass=mass,
            rgba=rgba,
        )
        self.add_geom(slider_geom)

        # Position the arch relative to the slider
        # Assuming the arch is designed to fit on the left side of the slider
        arch_pos = (
            -size[0] + arch_size[0] * 2,
            0,
            size[2] + arch_size[2] * 2,
        )  # Positioning the arch to the left side

        # Create the arch geom using the STL mesh
        arch_geom = Geom(
            name=f"{name}.arch",
            geom_type="sdf",  # Use the mesh as an SDF, to avoid convex hull issues
            mesh="round_arch",
            pos=arch_pos,
            quat=(1, 0, 0, 0),
            mass=mass,
            rgba=rgba,
            size=arch_size_adjusted,
            plugin="sdf",  # Add a plugin for the mesh to solve issues caused by convex hull
        )
        self.add_geom(arch_geom)

        # Add a joint if needed
        self.add_joint(Joint("free", (1, 1, 1), f"{name}.joint"))

        # Assets for the mesh
        self.assets = f"""
        <asset>
            <mesh name="round_arch" file="{arch_stl_path}"/>
        </asset>\n
        """

    def to_xml(self) -> str:
        """
        Convert the body and its components to an XML string.

        Example xml
        <body name="slider_with_arch" pos="0 0 0" quat="1 0 0 0">
        <geom name="slider_with_arch_slider" type="box" pos="0 0 0" size="4 1 0.1" rgba="0.5 0.5 0.5 1" mass="1.0" quat="0 0 0 1" />
        <geom name="slider_with_arch_arch" type="mesh" pos="-3 0 1.1" size="0.5 0.5 0.5" rgba="0.5 0.5 0.5 1" mass="1.0" quat="1 0 0 0" mesh="round_arch">
            <plugin instance="sdf" />
        </geom>
        <joint name="slider_with_arch_joint" type="free" axis="1 1 1" />
        </body>
        """
        body_xml = f"""<body name="{self.name}" pos="{' '.join(map(str, self.pos))}" quat="{' '.join(map(str, self.quat))}">\n"""
        for geom in self.geoms:
            body_xml += geom.to_xml() + "\n"
        for site in self.sites:
            body_xml += site.to_xml() + "\n"
        for joint in self.joints:
            body_xml += joint.to_xml() + "\n"
        body_xml += "</body>"
        return body_xml

class Rocket(Body):
    def __init__(
            self,
            name: str,
            pos: Tuple[float, float, float],
            material: str = DEFAULT_MATERIAL,
            rgba: Tuple[float, float, float, float] = DEFAULT_RGBA,
            mass: float = 1.0,
            plugin: str = "sdf",
            quat: Tuple[float, float, float, float] = (1, 0, 0, 0),
            **kwargs
        ):
        super().__init__(name, pos, quat, **kwargs)
        self.body_type = "rocket"

        # 1. Add Cosmetic Geom
        rocket_material = DEFAULT_MATERIAL
        self.rocket_geom = RocketGeom(
            name=f"{name}.rocket_geom",
            quat=(0, 0, 0.7071068, 0.7071068),
            material=rocket_material,
            rgba=rgba,
            plugin=plugin,
        )

        # 2. Add Physics Geom
        radius, height = 0.002, 0.03 # Dummy values, REVISIT 
        self.collision_geom = Cylinder(
            name=f"{name}.collision_geom",
            pos=(0, 0, height/2),
            radius=radius,
            height=height,
            material=material,
            rgba=(0, 0, 0, 0),
            mass=mass,
        )

        self.collision_geom.sites = []

        self.add_child_body(self.rocket_geom)
        self.add_child_body(self.collision_geom)

        # 3. Add Joint
        self.add_joint(Joint("free", (0, 0, 0), f"{name}.joint"))