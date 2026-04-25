from .base_bodies import *
from .mass import *
import math

def add_xz_planar_joint(body, plane_slope: float = 0.0):
    """
    Add planar joint for movement in X-Z plane (vertical plane).
    This replaces single linear joints with two orthogonal slide joints.
    
    Args:
        body: The body object to add joints to
        plane_slope: The slope angle of the plane in degrees (rotation around Y-axis)
    """
    # Convert plane_slope from degrees to radians
    theta = math.radians(plane_slope)
    
    # For X-Z plane movement:
    # axis1: X-axis direction (always horizontal)
    axis1 = (1, 0, 0)
    # axis2: Z-axis direction, but adjusted for plane slope (rotation around Y-axis)
    axis2 = (math.sin(theta), 0, math.cos(theta))
    
    # Clear existing joints and add two slide joints
    body.joints = []
    body.add_joint(Joint("slide", axis1, f"{body.name}.joint-x"))
    body.add_joint(Joint("slide", axis2, f"{body.name}.joint-z"))


class Spring(Body):
    """
    Represents a spring modeled using a tendon with a spatial element.
    """

    def __init__(
        self,
        name: str,
        left_site: Site = None,
        right_site: Site = None,
        stiffness: float = 1000.0,
        springlength: float = 0.3,
        damping: float = 0.0,
    ) -> None:
        super().__init__(name)
        self.left_site = left_site
        self.right_site = right_site
        self.stiffness = stiffness
        self.springlength = springlength
        self.damping = damping
        self.spatial = self.create_spatial()

    def create_spatial(self) -> Spatial:
        spatial = Spatial(
            name=self.name,
            width=DEFAULT_SPRING_THICKNESS,
            rgba=(0, 1, 0, 1),
            stiffness=self.stiffness,
            springlength=self.springlength,
            damping=self.damping,
        )
        if self.left_site:
            spatial.add_element(self.left_site.create_spatial_site())
        else:
            # Create a base mass for the missing left site
            base_mass = Mass(
                name=f"{self.name}.left_base_mass",
                positions=[(0, 0, 0)],
                mass_value=0.01,  # Very light mass
                padding_size_x=0.01,
                size_y=0.01,
                size_z=0.01,
                joint_option=("free", (0, 0, 0)),
            )
            spatial.add_element(base_mass.center_site.create_spatial_site())
            self.left_site = (
                base_mass.center_site
            )  # Update left_site to the base mass site
        if self.right_site:
            spatial.add_element(self.right_site.create_spatial_site())
        else:
            # Create a base mass for the missing right site
            base_mass = Mass(
                name=f"{self.name}.right_base_mass",
                positions=[(0, 0, 0)],
                mass_value=0.01,  # Very light mass
                padding_size_x=0.01,
                size_y=0.01,
                size_z=0.01,
                joint_option=("free", (0, 0, 0)),
            )
            spatial.add_element(base_mass.center_site.create_spatial_site())
            self.right_site = (
                base_mass.center_site
            )  # Update right_site to the base mass site
        return spatial

    def to_xml(self) -> str:
        """
        Convert the spring to its corresponding XML representation.
        """
        return self.spatial.to_xml()

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

class FixedSpring(Body):
    """
    Represents a fixed spring in a specified direction with a mass at the other end.
    """

    def __init__(
        self,
        name: str,
        pos: Tuple[float, float, float],
        slope: float,
        k: float,
        original_length: float,
        mass_value: float = 0.1,
        mass_rgba: Tuple[float, float, float, float] = (0, 1, 0, 1),
        constant_force: Optional[Dict[str, List[Union[List, float]]]] = None,
        damping: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(name, pos, **kwargs)
        self.slope = slope
        self.k = k
        self.original_length = original_length

        frame = Frame(
            np.zeros((3,)), axis_angles=np.array((0, -self.slope, 0)), degrees=True
        )  # counterclockwise convention
        
        endpoint_pos = tuple(
            frame.rel2global(
                np.array((-self.original_length, 0, 0))
            )
        )

        self.fixed_site = Site(f"{name}.fixed_site", (endpoint_pos), body_name=name)
        self.add_site(self.fixed_site)

        self.tray_mass = Mass(
            name=f"{name}.tray_mass",
            positions=[(0, 0, 0)],
            mass_value=mass_value,
            rgba=mass_rgba,
            disable_gravity=False,
            constant_force=constant_force,
            joint_option=None,  # We'll add custom planar joints instead
        )
        # set the position of the mass
        self.tray_mass.set_pose(pos=(0, 0, 0), quat=frame.quat)
        # Add X-Z planar joint for the tray mass
        add_xz_planar_joint(self.tray_mass, 0.0)
        self.add_child_body(self.tray_mass)

        # create a spring between the fixed site and the mass
        spring = Spring(
            name=f"{name}.spring",
            left_site=self.fixed_site,
            right_site=self.tray_mass.center_site,
            stiffness=k,
            springlength=original_length,
            damping=damping,
        )
        self.springs.append(spring)

    def get_connecting_tendon_sequences(
        self,
        direction: ConnectingDirection = ConnectingDirection.DEFAULT,
        connecting_option: Any = None,
    ) -> List[TendonSequence]:
        """
        Return the fixed site and the mass site.
        """
        return [TendonSequence(
            elements=[self.tray_mass.center_site.create_spatial_site()],
            description="Tendon sequence connecting the fixed site and the mass site",
            name=f"{self.name}.connecting_tendon"
        )]

class SpringBlock(Body):
    class ConnectOption(Enum):
        MASS_CENTER = "mass_center"
        SURROUNDING_SITES = "surrounding_sites"

    def __init__(
        self,
        name: str,
        pos: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        original_lengths: List[float] = [0.5],
        connecting_angles: List[float] = [0.0],
        stiffnesses: List[float] = [1.0],
        connecting_distances: List[float] = [1.5],
        **kwargs,
    ):
        super().__init__(name, pos, **kwargs)
        self.original_lengths = original_lengths
        self.connecting_angles = connecting_angles
        self.connecting_distances = connecting_distances
        self.stiffnesses = stiffnesses
        self.springs = []
        self.connecting_sites = []

        # Only one tray_mass in the center
        self.tray_mass = Mass(
            name=f"{self.name}.tray_mass",
            positions=[(0, 0, 0)],
            mass_value=0.1,
            rgba=(0, 1, 0, 1),
            disable_gravity=False,
        )

        self.add_child_body(self.tray_mass)

        # Create the springs and connecting sites
        self.create_springs()
        self.create_connecting_sites()

    def create_springs(self):
        for i, (angle, distance, original_length, stiffness) in enumerate(
            zip(self.connecting_angles, self.connecting_distances, self.original_lengths, self.stiffnesses)
        ):
            spring_name = f"{self.name}.spring-{i}"

            frame = Frame(
                np.zeros((3,)), axis_angles=np.array((0, -angle, 0)), degrees=True
            )

            endpoint_pos = tuple(
                frame.rel2global(np.array((-distance, 0, 0)))
            )

            fixed_site = Site(
                f"{spring_name}.fixed_site", endpoint_pos, body_name=self.name
            )
            self.add_site(fixed_site)

            spring = Spring(
                name=f"{spring_name}.spring",
                left_site=fixed_site,
                right_site=self.tray_mass.center_site,
                stiffness=stiffness,
                springlength=original_length,
            )

            self.springs.append(spring)

    def create_connecting_sites(self):
        for i, (angle, distance) in enumerate(
            zip(self.connecting_angles, self.connecting_distances)
        ):
            frame = Frame(
                np.zeros((3,)), axis_angles=np.array((0, angle, 0)), degrees=True
            )

            site_pos = tuple(frame.rel2global(np.array((distance, 0, 0))))

            connecting_site = Site(
                f"{self.name}.connecting_site-{i}", site_pos, body_name=self.name
            )
            self.add_site(connecting_site)
            self.connecting_sites.append(connecting_site)

    def get_connecting_tendon_sequences(
        self,
        direction: ConnectingDirection = ConnectingDirection.DEFAULT,
        connecting_option: ConnectOption = ConnectOption.MASS_CENTER,
    ) -> List[TendonSequence]:
        """
        Get the sites and geoms that tendons can connect to.
        """
        connecting_tendon_sequences = []
        if connecting_option == self.ConnectOption.MASS_CENTER:
            connecting_tendon_sequences = [
                TendonSequence(
                    elements=[self.tray_mass.center_site.create_spatial_site()],
                    description=f"Tendon sequence connecting the tray mass center site to the spring",
                    name=f"{self.name}.tray_mass_center_tendon"
                )
            ]
        elif connecting_option == self.ConnectOption.SURROUNDING_SITES:
            connecting_tendon_sequences = [
                TendonSequence(
                    elements=[self.tray_mass.center_site.create_spatial_site(), site.create_spatial_site()],
                    description=f"Tendon sequence connecting the tray mass center site to the spring",
                    name=f"{self.name}.tray_mass_center_tendon"
                )
                for site in self.connecting_sites
            ]
            if (
                direction == ConnectingDirection.RIGHT_TO_LEFT
                or direction == ConnectingDirection.OUTER_TO_INNER
            ):
                connecting_tendon_sequences.reverse()

        return connecting_tendon_sequences

class FixedMass(Body):
    """
    Represents a mass fixed in position (no joints).
    """

    def __init__(
        self,
        name: str,
        positions: List[Tuple[float, float, float]],
        mass_value: float = 1.0,
        size_x: float = 0.1,
        size_y: float = 0.1,
        size_z: float = 0.1,
        rgba: Tuple[float, float, float, float] = (1, 0, 0, 1),
        **kwargs,
    ) -> None:
        pos_x = positions[0][0]
        pos_y = positions[0][1]
        pos_z = positions[0][2]
        super().__init__(name, (pos_x, pos_y, pos_z), **kwargs)
        self.add_geom(
            Geom(
                name=f"{name}.geom",
                geom_type="box",
                pos=(0, 0, 0),
                size=(size_x, size_y, size_z),
                rgba=rgba,
                mass=mass_value,
            )
        )
        # add a center site
        self.center_site = Site(f"{name}.center_site", (0, 0, 0), body_name=name)
        self.add_site(self.center_site)

    def get_connecting_tendon_sequences(
        self,
        direction: ConnectingDirection = ConnectingDirection.DEFAULT,
        connecting_option: Any = None,
    ) -> List[TendonSequence]:
        """
        Get the sites and geoms that tendons can connect to.
        """
        return [TendonSequence(
            elements=[self.center_site.create_spatial_site()],
            description="Tendon sequence connecting the center site",
            name=f"{self.name}.connecting_tendon"
        )]

class SpringMass(Body):
    """
    Represents a combination of masses connected by springs.
    """

    def __init__(
        self,
        name: str,
        mass_values: List[float],
        mass_positions: List[float],
        spring_configs: List[Dict[str, float]],  # Each dict has 'k' and 'original_length'
        # left_spring: Optional[Dict[str, float]] = None,
        # right_spring: Optional[Dict[str, float]] = None,
        rgba: Tuple[float, float, float, float] = (0, 1, 0, 1),
        constant_force: Optional[Dict[str, List[Union[List, float]]]] = None,
        damping: float = 0.0,
        pos: Tuple[float, float, float] = (0, 0, 0),
        **kwargs,
    ) -> None:
        self.mass_values = mass_values
        self.mass_positions = mass_positions
        self.pos = pos
        self.spring_configs = spring_configs
        self.constant_force = constant_force
        self.damping = damping
        # self.left_spring = left_spring
        # self.right_spring = right_spring

        super().__init__(name, pos, **kwargs)
        self.masses = []


        # Create mass bodies
        for i, (mass_value, mass_pos) in enumerate(zip(self.mass_values, self.mass_positions)):
            current_constant_force = {}
            if (
                self.constant_force is not None
                and self.constant_force.get(ConstantForceType.MASS) is not None
            ):
                # if constant_force["mass"] is a list, then use the ith element
                if isinstance(self.constant_force[ConstantForceType.MASS][0], list):
                    current_constant_force[ConstantForceType.MASS] = self.constant_force[
                        ConstantForceType.MASS
                    ][i]
                else:
                    current_constant_force[ConstantForceType.MASS] = self.constant_force[
                        ConstantForceType.MASS
                    ]

            mass_body = Mass(
                name=f"{name}.mass-{i}",
                positions=[(0, 0, 0)],
                mass_value=mass_value,
                rgba=rgba,
                constant_force=current_constant_force,
                joint_option=None,  # We'll add custom planar joints instead
            )
            mass_body.set_pose(pos=(mass_pos, 0, 0))
            # Add X-Z planar joint for the mass (no slope for SpringMass)
            add_xz_planar_joint(mass_body, 0.0)
            self.masses.append(mass_body)
            self.add_child_body(mass_body)

        # Connect masses with joints representing springs
        for i in range(len(self.masses) - 1):
            mass1 = self.masses[i]
            mass2 = self.masses[i + 1]
            self.springs.append(
                Spring(
                    name=f"{self.name}.spring-{i}",
                    left_site=mass1.center_site,
                    right_site=mass2.center_site,
                    stiffness=self.spring_configs[i]["k"],
                    springlength=self.spring_configs[i]["original_length"],
                    damping=self.damping,
                )
            )

    def get_connecting_tendon_sequences(
        self, direction: ConnectingDirection, connecting_option: Any = None
    ) -> List[TendonSequence]:
        """
        Get the sites and geoms that tendons can connect to.
        """
        connecting_tendon_sequences = [
            TendonSequence(
                elements=[self.masses[0].center_site.create_spatial_site()],
                description=f"Tendon sequence connecting the mass {self.masses[0].name} center site",
                name=f"{self.name}.mass_center_tendon"
            ),
            TendonSequence(
                elements=[self.masses[-1].center_site.create_spatial_site()],
                description=f"Tendon sequence connecting the mass {self.masses[-1].name} center site",
                name=f"{self.name}.mass_center_tendon"
            )
        ]
        if (
            direction == ConnectingDirection.RIGHT_TO_LEFT
            or direction == ConnectingDirection.OUTER_TO_INNER
        ):
            connecting_tendon_sequences.reverse()
        return connecting_tendon_sequences