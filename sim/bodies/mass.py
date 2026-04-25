from .base_bodies import *


class Mass(Body):
    """
    A specialized Body class to represent a mass element in Mujoco XML.
    """

    offset = 0.05

    def __init__(
        self,
        name: str,
        positions: List[Tuple[float, float, float]],
        use_bottom_site: bool = False,
        padding_z: float = 0.0,
        padding_size_x: float = DEFAULT_MASS_SIZE,
        size_y: float = DEFAULT_MASS_SIZE,
        size_z: float = DEFAULT_MASS_SIZE,
        mass_value: float = 1.0,
        rgba: Tuple[float, float, float, float] = None,
        quat: Tuple[float, float, float, float] = (1, 0, 0, 0),
        joint_option: Tuple[str, Tuple[float, float, float]] = ("free", (1, 1, 1)),
        condim: str = "1",
        conaffinity: str = "1",
        slope: float = 0,  # degrees
        constant_force: Optional[Dict[str, List[Union[List, float]]]] = None,
        init_velocity: Optional[Dict[str, List[Union[List, float]]]] = None,
        disable_gravity=False,
    ) -> None:
        self.disable_gravity = disable_gravity

        # adjust the mass position according to the site_positions
        pos_x = sum(p[0] for p in positions) / len(positions)
        pos_y = sum(p[1] for p in positions) / len(positions)
        pos_z = sum(p[2] for p in positions) / len(positions) - padding_z
        size_x = (
            max(p[0] for p in positions) - min(p[0] for p in positions)
        ) / 2 + padding_size_x

        super().__init__(name, (pos_x, pos_y, pos_z), quat)
        self.body_type = "block"
        # set external constant force
        if (
            constant_force is not None
            and constant_force.get(ConstantForceType.MASS) is not None
        ):
            if isinstance(constant_force[ConstantForceType.MASS][0], list):
                constant_force[ConstantForceType.MASS] = constant_force[
                    ConstantForceType.MASS
                ][0]
            else:
                constant_force[ConstantForceType.MASS] = constant_force[
                    ConstantForceType.MASS
                ]
            self.constant_force_dict[self.name] = constant_force[ConstantForceType.MASS]

        # set initial velocity
        if (
            init_velocity is not None
            and init_velocity.get(InitVelocityType.MASS) is not None
        ):
            if isinstance(init_velocity[InitVelocityType.MASS][0], list):
                init_velocity[InitVelocityType.MASS] = init_velocity[
                    InitVelocityType.MASS
                ][0]
            else:
                init_velocity[InitVelocityType.MASS] = init_velocity[
                    InitVelocityType.MASS
                ]
            self.init_velocity_dict[self.name] = init_velocity[InitVelocityType.MASS]

        self.size = (size_x, size_y, size_z)
        self.top_sites = []
        self.center_site = Site(
            f"{self.name}.center", (0, 0, 0), (1, 0, 0, 0), body_name=name
        )
        self.left_site = Site(
            f"{self.name}.left", (-self.size[0], 0, 0), (1, 0, 0, 0), body_name=name
        )
        self.right_site = Site(
            f"{self.name}.right", (self.size[0], 0, 0), (1, 0, 0, 0), body_name=name
        )
        self.bottom_site = Site(
            f"{self.name}.bottom", (0, 0, -size_z), (1, 0, 0, 0), body_name=name
        )
        self.sensor_site = Site(
            f"{self.name}.sensor", (0, 0, 0), (1, 0, 0, 0), body_name=name
        )
        self.use_bottom_site = use_bottom_site
        self.bottom_connecting_site = Site(
            f"{self.name}.bottom_connecting",
            (0, 0, -size_z - self.offset - 3),
            (1, 0, 0, 0),
            body_name=name,
        )

        # Add a box geom to represent the mass
        self.add_geom(
            Geom(
                name=f"{name}.geom",
                geom_type="box",
                pos=(0, 0, 0),
                size=(size_x, size_y, size_z),
                rgba=(
                    random.choice(
                        [
                            (1.0, 0.5, 0.5, 1.0),
                            (1.0, 0.5, 1.0, 1.0),
                            (0.5, 0.5, 1.0, 1.0),
                        ]
                    )
                    if rgba is None
                    else rgba
                ),
                mass=mass_value,
                condim=condim,
                conaffinity=conaffinity,
            )
        )
        self.mass_value = mass_value
        self.slope = slope

        # Add sites at the top and bottom of the mass
        self.add_sites(positions, pos_x, size_z)
        # Add a free joint for the mass
        if joint_option:
            self.add_joint(Joint(joint_option[0], joint_option[1], f"{name}.joint"))

    def update_mass(self, mass_value: float) -> None:
        """
        Update the mass value of the mass element.
        """
        self.geoms[0].mass = mass_value
        self.mass_value = mass_value

    def add_sites(
        self, positions: List[Tuple[float, float, float]], pos_x: float, size_z: float
    ) -> None:
        """
        Add sites at specific positions and a sensor site at the center.
        """
        for i, p in enumerate(positions):
            pos_site_x = pos_x - p[0]
            site = Site(
                f"{self.name}.top-{i + 1}",
                (pos_site_x, 0, size_z),
                (1, 0, 0, 0),
                body_name=self.name,
            )
            self.add_site(site)
            self.top_sites.append(site)
        self.add_site(self.sensor_site)
        self.add_site(self.center_site)
        self.add_site(self.left_site)
        self.add_site(self.right_site)
        self.add_site(self.bottom_site)

    def get_second_connecting_tendon_sequences(
        self, direction: ConnectingDirection = ConnectingDirection.DEFAULT
    ) -> TendonSequence:
        """
        Get the sites and geoms that tendons can connect to.
        """
        if not self.use_bottom_site:
            self.use_bottom_site = True  # always turn on the bottom site when get_second_connecting_tendon_sequences is called
        if self.use_bottom_site:
            if direction == ConnectingDirection.OUTER_TO_INNER:
                tendons = [
                    [
                        self.bottom_connecting_site.create_spatial_site(),
                        self.bottom_site.create_spatial_site(),
                    ]
                ]
            else:
                tendons = [
                    [
                        self.bottom_site.create_spatial_site(),
                        self.bottom_connecting_site.create_spatial_site(),
                    ]
                ]
            return tendons
        raise ValueError("use_bottom_site is not used. No bottom site to connect to.")

    def get_connecting_tendon_sequences(
        self,
        direction: ConnectingDirection = ConnectingDirection.DEFAULT,
        connecting_option: Any = None,
    ) -> List[TendonSequence]:
        """
        Get the sites and geoms that tendons can connect to.
        """
        connecting_tendon_sequences = []
        if direction == ConnectingDirection.DEFAULT or direction == ConnectingDirection.INNER_TO_OUTER or direction == ConnectingDirection.OUTER_TO_INNER:
            connecting_tendon_sequences = [
                TendonSequence(
                    elements=[site.create_spatial_site()],
                    description=f"Tendon sequence connecting mass {self.name} top site to {site.name}",
                )
                for site in self.top_sites
            ]
            connecting_tendon_sequences.reverse()
        elif direction == ConnectingDirection.LEFT_TO_RIGHT:
            connecting_tendon_sequences = [
                TendonSequence(
                    elements=[self.left_site.create_spatial_site()],
                    description=f"Tendon sequence connecting mass {self.name} left site to {self.left_site.name}",
                ),
                TendonSequence(
                    elements=[self.right_site.create_spatial_site()],
                    description=f"Tendon sequence connecting mass {self.name} right site to {self.right_site.name}",
                ),
            ]
        elif direction == ConnectingDirection.RIGHT_TO_LEFT:
            connecting_tendon_sequences = [
                TendonSequence(
                    elements=[self.right_site.create_spatial_site()],
                    description=f"Tendon sequence connecting mass {self.name} right site to {self.right_site.name}",
                ),
                TendonSequence(
                    elements=[self.left_site.create_spatial_site()],
                    description=f"Tendon sequence connecting mass {self.name} left site to {self.left_site.name}",
                ),
            ]
        return connecting_tendon_sequences

    def add_planar_joint(self, plane_slope: float):
        theta = math.radians(plane_slope)
        axis1 = (math.cos(theta), 0, math.sin(theta))
        axis2 = (0, 1, 0)
        self.add_joint(Joint("slide", axis1, f"{self.name}.joint1"))
        self.add_joint(Joint("slide", axis2, f"{self.name}.joint2"))

    def to_xml(self) -> str:
        body_xml = f"""<body name="{self.name}" pos="{' '.join(map(str, self.pos))}" quat="{' '.join(map(str, self.quat))}"{' gravcomp="1"' if self.disable_gravity else ''}>\n"""
        for geom in self.geoms:
            body_xml += geom.to_xml() + "\n"
        for site in self.sites:
            body_xml += site.to_xml() + "\n"
        for joint in self.joints:
            body_xml += joint.to_xml() + "\n"
        body_xml += "</body>"
        if self.use_bottom_site:
            body_xml += f"""<body name="{self.name}.fixed_site" pos="{' '.join(map(str, self.pos))}" quat="{' '.join(map(str, self.quat))}">"""
            body_xml += self.bottom_connecting_site.to_xml() + "\n"
            body_xml += "</body>"
        return body_xml
