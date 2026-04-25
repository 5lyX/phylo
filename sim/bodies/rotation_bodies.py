from .base_bodies import *
from .pulley_bodies import *
from .geom_bodies import *


class Pendulum(Body):
    def __init__(
        self,
        name: str,
        pos: Tuple[float, float, float],
        rope_length: float = 1.0,
        mass_value: float = 1.0,
        angle: float = 0.0,
        init_velocity: Optional[Dict[str, List[Union[List, float]]]] = None,
        **kwargs,
    ):
        super().__init__(name, pos, **kwargs)
        if not hasattr(self, 'init_velocity_dict') or self.init_velocity_dict == {}:
            self.rope_length = rope_length
            self.mass_value = mass_value
            self.angle = angle
            self.init_velocity_dict = init_velocity
            # if init_velocity and InitVelocityType.SPHERE in init_velocity:
            #     self.init_velocity_dict[self.name] = init_velocity[InitVelocityType.SPHERE]
        
        self.body_type = "pendulum"
        self.top_site = Site(f"{name}.top_site", (0, 0, 0), body_name=name)
        self.sites.append(self.top_site)
        sphere_pos = (
            self.rope_length * math.sin(math.radians(self.angle)),
            0,
            -self.rope_length * math.cos(math.radians(self.angle)),
        )

        # Create the mass
        self.sphere = Sphere(
            name=f"{name}.sphere",
            pos=sphere_pos,
            radius=0.1,
            mass=self.mass_value,
            rgba=(0, 1, 0, 1),
            joint_option=("free", (0, 0, 0)),
            init_velocity=init_velocity,
        )
        self.add_child_body(self.sphere)

    def get_ready_tendon_sequences(self, direction: ConnectingDirection) -> List[TendonSequence]:
        """
        Get the sites and geoms that tendons can connect to for a pendulum.
        """
        inner_tendon = TendonSequence(
            elements=[
                self.sphere.site.create_spatial_site(),
                self.top_site.create_spatial_site()
            ],
            description=f"A tendon sequence connecting the pendulum mass to the top site",
            name="pendulum_tendon"
        )
        return [inner_tendon]



class DiskRack(Body):
    def __init__(
        self,
        name: str,
        pos: Tuple[float, float, float],
        radius: float = DEFAULT_DISC_RADIUS,
        height: float = DEFAULT_DISC_HEIGHT,
        angle: float = 0.0,
        bar_gap: float = 2 * DEFAULT_SPHERE_RADIUS,
        bar_thickness: float = DEFAULT_BAR_THICKNESS,
        x_offset: float = 0.0,
        **kwargs,
    ):
        super().__init__(name, pos, **kwargs)

        self.radius = radius
        self.height = height
        self.angle = angle
        self.bar_gap = bar_gap
        self.bar_thickness = bar_thickness
        self.x_offset = x_offset

        # Create the disk
        self.disc = Disc(
            name=f"{name}.disc",
            pos=(0, 0, 0),
            radius=radius,
            height=height,
        )
        self.add_child_body(self.disc)

        half_gap = bar_gap / 2.0
        half_bar = bar_thickness / 2.0
        bar_length = 2 * radius

        bar1_x = x_offset + (half_gap + half_bar)
        bar2_x = x_offset - (half_gap + half_bar)

        # We want the bar to extend along the y-axis, while the bar's local axis extends along the x-axis.
        # Align the x-axis with the positive y-axis using a quaternion: rotate 90 degrees around the z-axis
        # 90 degrees = π/2, sin(45°) = √2/2
        quat_y = (math.sqrt(2) / 2, 0.0, 0.0, math.sqrt(2) / 2)

        bar1 = Bar(
            name=f"{name}.bar1",
            pos=(
                bar1_x,
                -radius,
                self.disc.height + self.bar_gap / 2,
            ),  # Assume that the bar_gap is always 2 * radius
            length=bar_length,
            width=bar_thickness,
            height=bar_thickness,
            quat=quat_y,
        )
        bar2 = Bar(
            name=f"{name}.bar2",
            pos=(bar2_x, -radius, self.disc.height + self.bar_gap / 2),
            length=bar_length,
            width=bar_thickness,
            height=bar_thickness,
            quat=quat_y,
        )

        self.rotation_joint = Joint(
            name=f"{self.name}.rotation_joint",
            joint_type="hinge",
            axis=(0, 0, 1),
            pos=(0, 0, 0),
        )
        self.add_joint(self.rotation_joint)

        self.add_child_body(bar1)
        self.add_child_body(bar2)

        left_site = Site(
            f"{name}.left_site",
            (x_offset, -radius, DEFAULT_DISC_HEIGHT + DEFAULT_SPHERE_RADIUS),
            body_name=name,
        )
        right_site = Site(
            f"{name}.right_site",
            (x_offset, radius, DEFAULT_DISC_HEIGHT + DEFAULT_SPHERE_RADIUS),
            body_name=name,
        )
        self.add_site(left_site)
        self.add_site(right_site)

        self.left_site = left_site
        self.right_site = right_site

        self.set_quat_with_angle(angle)


class DiskRackWithSphere(DiskRack):
    def __init__(
        self,
        name: str,
        pos: Tuple[float, float, float],
        radius: float = DEFAULT_DISC_RADIUS,
        height: float = DEFAULT_DISC_HEIGHT,
        angle: float = 45.0,
        bar_gap: float = 2 * DEFAULT_SPHERE_RADIUS,
        bar_thickness: float = DEFAULT_BAR_THICKNESS,
        x_offset: float = 0.0,
        sphere_radius: float = DEFAULT_SPHERE_RADIUS,
        sphere_mass: float = 1.0,
        sphere_offset_along_track: float = 0.0,  # Offset along the y-axis (when angle=0, the rack is along the y-axis)
        **kwargs,
    ):
        super().__init__(
            name, pos, radius, height, angle, bar_gap, bar_thickness, x_offset, **kwargs,
        )

        # When angle=0, the sphere's position is (x_offset, sphere_offset_along_track, 0)
        self.sphere = Sphere(
            name=f"{name}.sphere",
            pos=(x_offset, sphere_offset_along_track, self.disc.height + sphere_radius),
            radius=sphere_radius,
            rgba=(1, 0, 0, 1),
            mass=sphere_mass,
        )
        self.add_child_body(self.sphere)