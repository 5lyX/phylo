from .base_bodies import *
from .mass import *
from .plane import *
from .composed_bodies import *


class TriangularPrismBox(Body):
    """
    A specialized Body class to represent a mass in the shape of a triangular prism, using 3 box geoms.
    """

    # thickness = 0.01  # the thickness of the box building the prism
    thickness = DEFAULT_PRISM_THICKNESS

    def __init__(
        self,
        name: str,
        positions: Tuple[float, float, float],
        size: float = 0.2,  # the length of the prism in the y direction
        height: float = 0.1,
        slopeL: float = 30,  # degrees
        slopeR: float = 60,  # degrees
        mass_value: float = 1.0,
        rgba: Tuple[float, float, float, float] = (0.70588235, 0.62352941, 0.8, 1),
        quat: Tuple[float, float, float, float] = (1, 0, 0, 0),
        joint_option: Tuple[str, Tuple[float, float, float]] = ("free", (1, 1, 1)),
        condim: str = "1",
        constant_force: Optional[Dict[str, List[float]]] = None,
        init_velocity: Optional[Dict[str, List[float]]] = None,
    ) -> None:
        pos_x = positions[0]
        pos_y = positions[1]
        pos_z = positions[2]
        super().__init__(name, (pos_x, pos_y, pos_z), quat)

        self.slopeL = slopeL
        self.slopeR = slopeR
        self.size = size
        self.height = height
        self.mass_value = mass_value
        self.rgba = rgba
        self.body_type = "prism"

        # create the .obj file
        vertex = self._create_prism()

        # calculate the pos of each box
        geom_bottom, geom_left, geom_right = self._cal_geom(vertex)

        self.geom_quat = [geom_bottom[2], geom_left[2], geom_right[2]]
        self.right_site_pos = (
            self.height / math.tan(math.radians(self.slopeR)),
            0,
            -self.height,
        )
        self.left_site_pos = (
            -self.height / math.tan(math.radians(self.slopeL)),
            0,
            -self.height,
        )
        self.left_site = Site(
            f"{self.name}.left", self.left_site_pos, (1, 0, 0, 0), body_name=name
        )
        self.right_site = Site(
            f"{self.name}.right", self.right_site_pos, (1, 0, 0, 0), body_name=name
        )
        self.sensor_site = Site(
            f"{self.name}.sensor",
            (0, 0, -self.height / 2),
            (1, 0, 0, 0),
            body_name=name,
        )

        # Add a geom to represent the mass
        self.add_geom(
            Geom(
                name=f"{name}.geom_bottom",
                geom_type="box",
                pos=geom_bottom[0],
                size=geom_bottom[1],
                rgba=rgba,
                mass=mass_value/3,
                quat=geom_bottom[2],
                condim=condim,
            )
        )
        self.add_geom(
            Geom(
                name=f"{name}.geom_left",
                geom_type="box",
                pos=geom_left[0],
                size=geom_left[1],
                rgba=rgba,
                mass=mass_value/3,
                quat=geom_left[2],
                condim=condim,
            )
        )

        self.add_geom(
            Geom(
                name=f"{name}.geom_right",
                geom_type="box",
                pos=geom_right[0],
                size=geom_right[1],
                rgba=rgba,
                mass=mass_value/3,
                quat=geom_right[2],
                condim=condim,
            )
        )

        # TODO: add sites on the sloped faces
        self.add_sites()
        # Add a free joint for the mass
        self.add_joint(Joint(joint_option[0], joint_option[1], f"{name}.joint"))

        if constant_force and ConstantForceType.PRISM in constant_force:
            self.constant_force_dict[f"{name}"] = constant_force[
                ConstantForceType.PRISM
            ]
        if init_velocity and InitVelocityType.SPHERE in init_velocity:
            self.init_velocity_dict[f"{name}"] = init_velocity[InitVelocityType.SPHERE]
        
    def _cal_geom(
        self, vertex: List[Tuple[float, float, float]]
    ) -> List[Tuple[float, float, float]]:
        """
        Calculate the position, size and quat of each box building prism.
        """
        v1, v2, v3, v4, v5, v6 = vertex
        x = np.linalg.norm(np.array(v1) - np.array(v4))
        y = np.linalg.norm(np.array(v1) - np.array(v2))
        z = np.linalg.norm(np.array(v1) - np.array(v3))
        w = np.linalg.norm(np.array(v2) - np.array(v3))

        size_bottom = (w / 2, x / 2, self.thickness)
        size_left = (y / 2, x / 2, self.thickness)
        size_right = (z / 2, x / 2, self.thickness)

        pos_bottom = (-(v3[0] + v5[0]) / 2, (v3[1] + v5[1]) / 2, (v3[2] + v5[2]) / 2)
        pos_left = (-(v1[0] + v5[0]) / 2, (v1[1] + v5[1]) / 2, (v1[2] + v5[2]) / 2)
        pos_right = (-(v1[0] + v6[0]) / 2, (v1[1] + v6[1]) / 2, (v1[2] + v6[2]) / 2)

        quat_bottom = (1, 0, 0, 0)
        quat_left = tuple(
            Frame.euler_to_quaternion(np.array([0, -self.slopeL, 0]), degrees=True)
        )
        quat_right = tuple(
            Frame.euler_to_quaternion(np.array([0, self.slopeR, 0]), degrees=True)
        )

        geom_bottom = [pos_bottom, size_bottom, quat_bottom]
        geom_left = [pos_left, size_left, quat_left]
        geom_right = [pos_right, size_right, quat_right]

        return geom_bottom, geom_left, geom_right

    def add_sites(self) -> None:
        """
        Add sites at specific positions and a sensor site at the center.
        """
        self.add_site(self.sensor_site)
        self.add_site(self.left_site)
        self.add_site(self.right_site)
        self.add_site(
            Site(f"{self.name}.top", (0, 0, 0), (1, 0, 0, 0), body_name=self.name)
        )

    def pos_on_left_slope(
        self,
        x: float,  # [0,1], 0 is the left edge, 1 is the right edge
        y: float,  # [-1,1], -1 is the inner edge, 1 is the outer edge
        z_padding: float = 0,  # padding in the z direction of the side (upwards) to make sure the mass is not in the ground
    ) -> Tuple[float, float, float]:
        """
        Calculate the position on the left slope of the prism. (global coordinates)
        """
        x_pos = -x * self.height / math.tan(
            math.radians(self.slopeL)
        ) - z_padding * math.sin(math.radians(self.slopeL))
        y_pos = y * self.size / 2
        z_pos = -x * self.height + z_padding * math.cos(
            math.radians(self.slopeL)
        )  # similar triangles
        local_pos = (x_pos, y_pos, z_pos)

        # local to global

        frame = Frame(origin=np.array(self.pos), quat=np.array(self.quat))
        pos, quat = frame.rel2global(
            local_pos,
            quat=Frame.euler_to_quaternion(
                np.array([0, -self.slopeL, 0]), degrees=True
            ),
        )

        return local_pos, tuple(pos), tuple(quat)

    def pos_on_right_slope(
        self,
        x: float,  # [0,1], 0 is the left edge, 1 is the right edge
        y: float,  # [-1,1], -1 is the inner edge, 1 is the outer edge
        z_padding: float = 0,  # padding in the z direction (upwards) to make sure the mass is not in the ground
    ) -> Tuple[float, float, float]:
        """
        Calculate the position on the right slope of the prism. (global coordinates)
        """
        x_pos = x * self.height / math.tan(
            math.radians(self.slopeR)
        ) + z_padding * math.sin(math.radians(self.slopeR))
        y_pos = y * self.size / 2
        z_pos = -x * self.height + z_padding * math.cos(
            math.radians(self.slopeR)
        )  # similar triangles
        local_pos = (x_pos, y_pos, z_pos)

        # local to global
        frame = Frame(origin=np.array(self.pos), quat=np.array(self.quat))
        global_pos, quat = frame.rel2global(
            local_pos,
            quat=Frame.euler_to_quaternion(np.array([0, self.slopeR, 0]), degrees=True),
        )
        return local_pos, tuple(global_pos), tuple(quat)

    def _create_prism(self) -> None:
        # Tan of the angles
        slope1 = math.radians(self.slopeL)
        slope2 = math.radians(self.slopeR)

        tan1 = math.tan(slope1)
        tan2 = math.tan(slope2)

        y_pos = self.size / 2

        # vertex
        v1 = (0, -y_pos, 0)
        v2 = (self.height / tan1, -y_pos, -self.height)
        v3 = (-self.height / tan2, -y_pos, -self.height)

        v4 = (0, y_pos, 0)
        v5 = (self.height / tan1, y_pos, -self.height)
        v6 = (-self.height / tan2, y_pos, -self.height)

        return [v1, v2, v3, v4, v5, v6]


class TriangularPrism(Body):
    """
    A specialized Body class to represent a mass in the shape of a triangular prism. (using mesh)
    """

    def __init__(
        self,
        name: str,
        positions: Tuple[float, float, float],
        size: float = 0.2,
        height: float = 0.1,
        slopeL: float = 30,  # degrees
        slopeR: float = 60,  # degrees
        mass_value: float = 1.0,
        rgba: Tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1),
        quat: Tuple[float, float, float, float] = (1, 0, 0, 0),
    ) -> None:
        pos_x = positions[0]
        pos_y = positions[1]
        pos_z = positions[2]
        super().__init__(name, (pos_x, pos_y, pos_z), quat)

        self.slopeL = slopeL
        self.slopeR = slopeR
        self.size = size
        self.height = height
        self.mass_value = mass_value
        self.rgba = rgba
        self.body_type = "prism"

        # create the .obj file
        self._create_prism_obj()

        # Add a geom to represent the mass
        self.add_geom(
            Geom(
                name=f"{name}.geom",
                geom_type="mesh",
                pos=(0, 0, 0),
                rgba=rgba,
                mass=mass_value,
                mesh=f"{name}",
            )
        )
        # TODO: add sites on the sloped faces
        self.add_sites()
        # Add a free joint for the mass
        self.add_joint(Joint("free", (1, 1, 1), f"{name}.joint"))

        # TODO: deal with <assets> in more decent way
        self.assets = f"""
        <asset>
            <mesh name="{name}" file="./{name}.obj"/>
        </asset>\n
        """

    def add_sites(self) -> None:
        """
        Add sites at specific positions and a sensor site at the center.
        """

        self.add_site(
            Site(
                f"{self.name}.sensor",
                (0, 0, -self.size / 2),
                (1, 0, 0, 0),
                body_name=self.name,
            )
        )
        self.add_site(
            Site(
                f"{self.name}.left",
                self.pos_on_left_slope(0.5, 0)[0],
                (1, 0, 0, 0),
                body_name=self.name,
            )
        )
        self.add_site(
            Site(
                f"{self.name}.right",
                self.pos_on_right_slope(0.5, 0)[0],
                (1, 0, 0, 0),
                body_name=self.name,
            )
        )
        self.add_site(
            Site(f"{self.name}.top", (0, 0, 0), (1, 0, 0, 0), body_name=self.name)
        )

    def pos_on_left_slope(
        self,
        x: float,  # [0,1], 0 is the left edge, 1 is the right edge
        y: float,  # [-1,1], -1 is the inner edge, 1 is the outer edge
        z_padding: float = 0,  # padding in the z direction (upwards) to make sure the mass is not in the ground
    ) -> Tuple[float, float, float]:
        """
        Calculate the position on the left slope of the prism. (global coordinates)
        """
        x_pos = -x * self.height / math.tan(
            math.radians(self.slopeL)
        ) - z_padding * math.sin(math.radians(self.slopeL))
        y_pos = y * self.size / 2
        z_pos = -x * self.height + z_padding * math.cos(
            math.radians(self.slopeL)
        )  # similar triangles
        local_pos = (x_pos, y_pos, z_pos)

        # local to global

        frame = Frame(origin=np.array(self.pos).copy(), quat=np.array(self.quat).copy())
        pos, quat = frame.rel2global(
            local_pos,
            quat=Frame.euler_to_quaternion(
                np.array([0, -self.slopeL, 0]), degrees=True
            ),
        )
        return local_pos, tuple(pos), tuple(quat)

    def pos_on_right_slope(
        self,
        x: float,  # [0,1], 0 is the left edge, 1 is the right edge
        y: float,  # [-1,1], -1 is the inner edge, 1 is the outer edge
        z_padding: float = 0,  # padding in the z direction (upwards) to make sure the mass is not in the ground
    ) -> Tuple[float, float, float]:
        """
        Calculate the position on the right slope of the prism. (global coordinates)
        """
        x_pos = x * self.height / math.tan(
            math.radians(self.slopeR)
        ) + z_padding * math.sin(math.radians(self.slopeR))
        y_pos = y * self.size / 2
        z_pos = -x * self.height + z_padding * math.cos(
            math.radians(self.slopeR)
        )  # similar triangles
        local_pos = (x_pos, y_pos, z_pos)

        # local to global
        frame = Frame(origin=np.array(self.pos), quat=np.array(self.quat))
        global_pos, quat = frame.rel2global(
            (x_pos, y_pos, z_pos),
            quat=Frame.euler_to_quaternion(np.array([0, self.slopeR, 0]), degrees=True),
        )
        return local_pos, tuple(global_pos), tuple(quat)

    def _create_prism_obj(self) -> None:
        # Tan of the angles
        slope1 = math.radians(self.slopeL)
        slope2 = math.radians(self.slopeR)

        tan1 = math.tan(slope1)
        tan2 = math.tan(slope2)

        y_pos = self.size / 2

        # vertex
        v1 = (0, -y_pos, 0)
        v2 = (self.height / tan1, -y_pos, -self.height)
        v3 = (-self.height / tan2, -y_pos, -self.height)

        v4 = (0, y_pos, 0)
        v5 = (self.height / tan1, y_pos, -self.height)
        v6 = (-self.height / tan2, y_pos, -self.height)

        # Faces (triangular sides and rectangular sides)
        faces = [
            (1, 3, 2),  # Bottom triangle
            (4, 5, 6),  # Top triangle
            (1, 2, 4),  # Side face
            (2, 5, 4),
            (2, 3, 5),  # Side face
            (3, 6, 5),
            (3, 1, 6),  # Side face
            (1, 4, 6),
        ]

        # Writing to the .obj file
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "xml_output/asset/"
        )
        os.makedirs(output_dir, exist_ok=True)
        with open(output_dir + self.name + ".obj", "w") as f:
            # Write vertices
            f.write("# Vertices\n")
            for v in [v1, v2, v3, v4, v5, v6]:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")

            # Write faces (faces are 1-indexed in .obj format)
            f.write("# Faces\n")
            for face in faces:
                f.write(f"f {' '.join(str(i) for i in face)}\n")


class MassPrismPlane(Body):
    """
    Represents a mass on a prism on a plane.
    """

    offset = 0.05

    def __init__(
        self,
        name: str,
        plane_slope: float,  # degrees
        prism_left_slope: float,  # degrees
        prism_right_slope: float,  # degrees
        block_mass_value: float,  # mass value of the mass on the prism
        prism_mass_value: float,  # mass value of the prism
        use_left_site: DirectionsEnum = DirectionsEnum.USE_LEFT,  # the position of the prism on the plane, True is on the left edge, False is on the right edge
        use_prism_left: bool = True,
        positions: List[Tuple[float, float, float]] = [],  # positions to be aligned to
        padding_z: float = 0.0,  # padding additional site in the z direction (downwards)
        condim: str = "1",  # 1 means frictionless
        constant_force: Optional[Dict[str, List[Union[List, float]]]] = None,
        init_velocity: Optional[Dict[str, List[Union[List, float]]]] = None,
        **kwargs,
    ) -> None:
        super().__init__(name, quat=(1, 0, 0, 0), **kwargs)
        self.use_left_site = use_left_site
        self.plane_slope = plane_slope
        self.use_prism_left = use_prism_left
        self.plane = Plane(
            name=name + ".plane",
            pos=(0, 0, 0),
            size=(DEFAULT_PLANE_LENGTH, DEFAULT_PLANE_WIDTH, DEFAULT_PLANE_THICKNESS),
            quat=Frame.euler_to_quaternion(np.array([0, -plane_slope, 0]), degrees=True),
            condim=condim,
            site_padding=TriangularPrismBox.thickness,  # padding in the z direction (upwards) to make site parallel to the center of the mass on the prism
        )

        mass_slope = prism_left_slope if use_prism_left else prism_right_slope
        self.mass_slope = mass_slope

        self.mass = Mass(
            name=name + ".mass",
            positions=[(0, 0, 0)],
            mass_value=block_mass_value,
            slope=mass_slope,
            constant_force=(
                {ConstantForceType.MASS: constant_force[ConstantForceType.MASS]}
                if constant_force and ConstantForceType.MASS in constant_force
                else None
            ),
            init_velocity=(
                {InitVelocityType.MASS: init_velocity[InitVelocityType.MASS]}
                if init_velocity and InitVelocityType.MASS in init_velocity
                else None
            ),
        )

        self.prism = TriangularPrismBox(
            name=name + ".prism",
            positions=(0, 0, 0),
            size=DEFAULT_PRISM_WIDTH,
            height=DEFAULT_PRISM_HEIGHT,
            slopeL=prism_left_slope,
            slopeR=prism_right_slope,
            mass_value=prism_mass_value,
            condim=condim,
            constant_force=(
                {ConstantForceType.PRISM: constant_force[ConstantForceType.PRISM]}
                if constant_force and ConstantForceType.PRISM in constant_force
                else None
            ),
        )

        self.plane_x = 0  # [-1,1], -1 is the left edge, 1 is the right edge
        self.plane_y = (
            0  # The prism is always on the center of the plane in the y direction
        )
        self.prism_x = 0.5  # [0,1], 0 is the left edge, 1 is the right edge
        self.prism_y = (
            0  # The mass is always on the center of the prism side in the y direction
        )

        local_pos, global_triangular_prism_pos, triangular_prism_quat = (
            self.plane.pos_on_top(
                0,
                0,
                z_padding=self.prism.height
                + TriangularPrismBox.thickness,  # prism_hight has take TriangularPrismBox.thickness into account
            )
        )

        # set the position of the prism
        self.prism.set_pose(global_triangular_prism_pos, triangular_prism_quat)

        if use_prism_left:
            local_pos, global_mass_pos, mass_quat = self.prism.pos_on_left_slope(
                self.prism_x,
                0,
                z_padding=TriangularPrismBox.thickness + self.mass.size[2],
            )
        else:
            local_pos, global_mass_pos, mass_quat = self.prism.pos_on_right_slope(
                self.prism_x,
                0,
                z_padding=TriangularPrismBox.thickness + self.mass.size[2],
            )

        # set the position of the mass
        self.mass.set_pose(global_mass_pos, mass_quat)

        # set the left or right site of the plane as the origin of the system
        self.origin_pos = None  # to be set in the align_pose function
        self.add_additional_sites(
            use_left_site=self.use_left_site,
            positions=positions,
            z_padding=padding_z,
        )

    # TODO: change the structure
    def get_masses_quality(self) -> List[dict]:
        """
        Get the quality of each mass used for symbolic regression.
        """
        list_of_masses = []
        mass_dict_prism = self.prism.get_masses_quality()[
            0
        ]  # use only one geom in the prism

        mass_dict_mass = self.mass.get_masses_quality()[0]  # only one geom in the mass
        mass_dict_plane = self.plane.get_masses_quality()[
            0
        ]  # only one geom in the plane

        mass_dict_prism["slope"] = (
            self.prism.slopeL if self.use_prism_left else self.prism.slopeR
        )

        mass_dict_plane["slope"] = self.plane_slope
        list_of_masses.append(mass_dict_prism)
        list_of_masses.append(mass_dict_mass)

        return list_of_masses
    
    def get_description(self, simDSL2nlq = False) -> List[dict]:
        """
        Get the description of the mass, prism and plane.
        """
        # TODO: should we include friction in the description?
        mass_dict_prism = self.prism.get_description()[0]
        mass_dict_mass = self.mass.get_description()[0]

        # block information
        mass_dict_mass["prism_slope"] = (
            self.prism.slopeL if self.use_prism_left else self.prism.slopeR
        )
        mass_dict_mass["prism_name"] = self.prism.name
        mass_dict_mass["plane_slope"] = self.plane_slope
        mass_dict_mass["plane_name"] = self.plane.name
        mass_dict_mass["description"] = (
            f"{mass_dict_mass['description']} "
            f"It rests on a prism named {mass_dict_mass['prism_name']} "
            f"with a slope of {mass_dict_mass['prism_slope']} degrees. "
            f"The prism is on a plane named {mass_dict_mass['plane_name']} "
            f"with a slope of {mass_dict_mass['plane_slope']} degrees."
        )

        # prism information
        mass_dict_prism["plane_slope"] = self.plane_slope
        mass_dict_prism["plane_name"] = self.plane.name
        mass_dict_prism["mass_name"] = mass_dict_mass["name"]
        mass_dict_prism["mass_mass"] = mass_dict_mass["mass"]
        mass_dict_prism["description"] = (
            f"{mass_dict_prism['description']} "
            f"The prism is on a plane named {mass_dict_prism['plane_name']} "
            f"with a slope of {mass_dict_prism['plane_slope']} degrees. "
            f"There is a block named {mass_dict_prism['mass_name']} "
            f"with a mass of {mass_dict_prism['mass_mass']} kg on the prism."
        )

        return [mass_dict_prism, mass_dict_mass]
    

    def add_additional_sites(
        self,
        use_left_site: DirectionsEnum = DirectionsEnum.USE_LEFT,
        positions: List[Tuple[float, float, float]] = [],
        z_padding: float = 3,
    ) -> None:
        """
        Add helper sites above the original sites on the plane, which can be used to connect moveable pulleys.
        """
        pos_x = sum(p[0] for p in positions) / len(positions)
        pos_y = sum(p[1] for p in positions) / len(positions)
        pos_z = sum(p[2] for p in positions) / len(positions)

        target_pos = (pos_x, pos_y, pos_z)
        self.align_pose(
            target_pos=target_pos,
            use_left_site=use_left_site,
            displacement_z=-(z_padding),
        )

        # add additional sites
        frame = Frame(origin=np.array(self.plane.pos), quat=np.array(self.plane.quat))
        self.sites_to_connect = []
        for i, pos in enumerate(positions):
            pos_new = (
                pos[0],
                pos[1],
                pos[2] - z_padding + self.offset,
            )  # global position
            pos_local = frame.global2rel(pos_new)
            self.plane.add_site(
                Site(
                    f"{self.plane.name}.additional_site-{i}",
                    tuple(pos_local),
                    (1, 0, 0, 0),
                    body_name=self.plane.name,
                )
            )
            self.sites_to_connect.append(self.plane.sites[-1])

    def align_pose(
        self,
        target_pos: Tuple[float, float, float],
        use_left_site: DirectionsEnum = DirectionsEnum.USE_LEFT,
        displacement_x: float = 0,
        displacement_y: float = 0,
        displacement_z: float = 0,
    ) -> None:
        """
        Change the pose of the mass and plane.
        """
        frame = Frame(origin=np.array(self.plane.pos), quat=np.array(self.plane.quat))
        if use_left_site == DirectionsEnum.USE_LEFT:
            local_original_site_pos = np.array(self.plane.left_site.pos)
            displacement = np.array(target_pos) - frame.rel2global(
                local_original_site_pos
            )
        else:  # use right site
            local_original_site_pos = np.array(self.plane.right_site.pos)
            displacement = np.array(target_pos) - frame.rel2global(
                local_original_site_pos
            )

        displacement += tuple(
            np.array([displacement_x, displacement_y, displacement_z])
        )

        # move all bodies
        self.plane.move(displacement)
        self.prism.move(displacement)
        self.mass.move(displacement)

        # add child bodies
        self.add_child_body(self.prism)
        self.add_child_body(self.mass)
        self.add_child_body(self.plane)

        # set the origin position
        self.origin_pos = target_pos + tuple(
            np.array([displacement_x, displacement_y, displacement_z])
        )


    def get_connecting_tendon_sequences(
        self,
        direction: ConnectingDirection = ConnectingDirection.DEFAULT,
        connecting_option: Any = None,
    ) -> List[TendonSequence]:
        """
        Get the tendon sequence for the mass on the prism on the plane.
        """
        tendons = []
        if self.use_left_site == DirectionsEnum.USE_LEFT:
            for site in self.sites_to_connect:
                tendons.append(
                    TendonSequence(
                        elements=[
                            self.prism.left_site.create_spatial_site(),
                            self.plane.left_site.create_spatial_site(),
                            site.create_spatial_site(),
                        ],
                        description=f"Tendon sequence connecting prism left site to plane left site to {site.name}",
                    )
                )

            if (
                direction in [
                    ConnectingDirection.INNER_TO_OUTER,
                    ConnectingDirection.DEFAULT,
                ]
            ):
                return tendons
            else:  # ConnectingDirection.OUTER_TO_INNER
                return [reverse_tendon_sequence(sequence) for sequence in tendons]
        else:  # use right site
            for site in self.sites_to_connect:
                tendons.append(
                    TendonSequence(
                        elements=[
                            self.prism.right_site.create_spatial_site(),
                            self.plane.right_site.create_spatial_site(),
                            site.create_spatial_site(),
                        ],
                        description=f"Tendon sequence connecting prism right site to plane right site to {site.name}",
                    )
                )

            if direction == ConnectingDirection.INNER_TO_OUTER:
                return tendons
            else:  # ConnectingDirection.OUTER_TO_INNER
                return [reverse_tendon_sequence(sequence) for sequence in tendons]

    def get_second_connecting_tendon_sequences(
        self, direction: ConnectingDirection
    ) -> List[TendonSequence]:
        """
        Get the tendon sequence from the other side of the plane to the mass on the plane.
        """
        tendons = []
        if self.use_left_site == DirectionsEnum.USE_LEFT:
            # if use left site, then this function is to get the tendon sequence from the right site to the mass
            inner_tendon_seq = TendonSequence(
                elements=[
                    self.prism.right_site.create_spatial_site(),
                    self.plane.right_site.create_spatial_site(),
                ],
                description=f"Tendon sequence connecting prism right site to plane right site",
            )
            tendons.append(inner_tendon_seq)
        else:  # use right site
            # if use right site, then this function is to get the tendon sequence from the left site to the mass
            inner_tendon_seq = TendonSequence(
                elements=[
                    self.prism.left_site.create_spatial_site(),
                    self.plane.left_site.create_spatial_site(),
                ],
                description=f"Tendon sequence connecting prism left site to plane left site",
            )
            tendons.append(inner_tendon_seq)

        if direction == ConnectingDirection.INNER_TO_OUTER:
            return tendons
        else:  # ConnectingDirection.OUTER_TO_INNER
            return [reverse_tendon_sequence(sequence) for sequence in tendons]


    def to_xml(self) -> str:
        """
        Convert the mass, prism and plane to an XML string.
        """
        xml = self.mass.to_xml() + "\n"
        xml += self.prism.to_xml() + "\n"
        xml += self.plane.to_xml() + "\n"
        return xml

    def get_sensor_list(self) -> List[Sensor]:
        """
        Get the sensors of the mass and prism.
        """
        sensors = self.mass.get_sensor_list()
        sensors += self.prism.get_sensor_list()
        return sensors


class MassPlane(Body):
    """
    Represents a mass on a plane.
    """

    offset = 0.05  # offset for the additional sites

    def __init__(
        self,
        name: str,
        plane_slope: float,  # degrees
        mass_values: List[float],  # mass value of the mass on the plane
        positions: List[Tuple[float, float, float]] = [
            (0, 0, 0)
        ],  # positions to be aligned to
        padding_z: float = 0,  # padding in the z direction (downwards)
        use_left_site: DirectionsEnum = DirectionsEnum.USE_LEFT,
        condim: str = "1",  # 1 means frictionless
        constant_force: Optional[Dict[str, List[Union[List, float]]]] = None,
        init_velocity: Optional[
            Dict[str, List[Union[List, float]]]
        ] = None,  # init_velocity should not be involved in this scenario
        **kwargs,
        # TODO: add a param to define how masses are connected, e.g., using tendon or they are physically connected together
    ) -> None:
        # use the quat from kwargs if provided, otherwise use default value
        quat = kwargs.pop('quat', (1, 0, 0, 0))
        super().__init__(name, quat=quat, **kwargs)
        self.use_left_site = use_left_site
        self.plane_slope = plane_slope
        self.plane = Plane(
            name=name + ".plane",
            size=(DEFAULT_PLANE_LENGTH, DEFAULT_PLANE_WIDTH, DEFAULT_PLANE_THICKNESS),
            quat=Frame.euler_to_quaternion(np.array([0, -plane_slope, 0]), degrees=True),
            condim=condim,
            site_padding=DEFAULT_MASS_SIZE,  # default size of Mass
        )

        self.mass_values = mass_values

        self.masses = []
        object_padding = 0.4
        for i, mass_value in enumerate(mass_values):
            current_constant_force = {}
            if (
                constant_force is not None
                and constant_force.get(ConstantForceType.MASS) is not None
            ):
                # if constant_force["mass"] is a list, then use the ith element
                if isinstance(constant_force[ConstantForceType.MASS][0], list):
                    current_constant_force[ConstantForceType.MASS] = constant_force[
                        ConstantForceType.MASS
                    ][i]
                else:
                    current_constant_force[ConstantForceType.MASS] = constant_force[
                        ConstantForceType.MASS
                    ]
            # init_velocity should not be involved in this scenario
            self.masses.append(
                Mass(  # use DEFAULT_MASS_SIZE as the size of the mass
                    name=name + ".mass" + str(i),
                    positions=[(0, 0, 0)],
                    mass_value=mass_value,
                    slope=plane_slope,
                    constant_force=current_constant_force,
                )
            )
            x_loc = (i - len(mass_values) // 2) * object_padding
            local_pos, global_mass_pos, mass_quat = self.plane.pos_on_top(
                x_loc, 0, z_padding=self.masses[-1].size[2]
            )

            # set the position of the mass
            self.masses[-1].set_pose(global_mass_pos, mass_quat)
            self.add_child_body(self.masses[-1])    # add child bodies

        # set the left or right site of the plane as the origin of the system
        self.origin_pos = None  # to be set in the align_pose function
        self.add_additional_sites(
            use_left_site=self.use_left_site,
            positions=positions,
            z_padding=padding_z,
        )

    def get_masses_quality(self) -> List[dict]:
        """
        Get the quality of each mass used for symbolic regression.
        """
        list_of_masses = []
        for mass in self.masses:
            mass_dict = mass.get_masses_quality()[0]  # only one geom per mass
            list_of_masses.append(mass_dict)
        # if self.plane_slope != 0:  # even 0 I think we should include still include the plane for symbolic regression
        plane_dict = self.plane.get_masses_quality()[0]  # only one geom in the plane
        plane_dict["slope"] = self.plane_slope
        list_of_masses.append(plane_dict)
        return list_of_masses
    
    def get_description(self, simDSL2nlq = False) -> List[dict]:
        """
        Get the description of each body for variable matching.
        """
        # TODO: should we include friction in the description?
        list_of_masses = []
        for mass in self.masses:
            mass_dict = mass.get_description()[0]
            mass_dict["plane_slope"] = self.plane_slope
            mass_dict["plane_name"] = self.plane.name
            mass_dict["description"] = (
                f"{mass_dict['description']} "
                f"It rests on a plane named {mass_dict['plane_name']} "
                f"with a slope of {mass_dict['plane_slope']} degrees."
            )
            list_of_masses.append(mass_dict)
        # we don't need to include the plane in the description
        return list_of_masses

    def set_pose(
        self, pos: Tuple[float, float, float], quat: Tuple[float, float, float, float]
    ) -> None:
        """
        Set the position and orientation of the MassPlane. quat is not used.
        """
        displacement = tuple(np.array(pos) - np.array(self.origin_pos))

        # move all bodies
        self.move(displacement)

    def move(self, displacement: Tuple[float, float, float]) -> None:
        """
        Move the MassPlane by the given displacement.
        """
        self.plane.move(displacement)
        for mass in self.masses:
            mass.move(displacement)

    def add_additional_sites(
        self,
        use_left_site: DirectionsEnum = DirectionsEnum.USE_LEFT,
        positions: List[Tuple[float, float, float]] = [],
        z_padding: float = 3,
    ) -> None:
        """
        Add helper sites above the original sites on the plane, which can be used to connect moveable pulleys.
        """
        pos_x = sum(p[0] for p in positions) / len(positions)
        pos_y = sum(p[1] for p in positions) / len(positions)
        pos_z = sum(p[2] for p in positions) / len(positions)

        target_pos = (pos_x, pos_y, pos_z)
        self.align_pose(
            target_pos=target_pos,
            use_left_site=use_left_site,
            displacement_z=-(z_padding),
        )

        # add additional sites
        frame = Frame(origin=np.array(self.plane.pos), quat=np.array(self.plane.quat))
        self.sites_to_connect = []
        for i, pos in enumerate(positions):
            pos_new = (
                pos[0],
                pos[1],
                pos[2] - z_padding + self.offset,
            )  # global position
            pos_local = frame.global2rel(pos_new)
            self.plane.add_site(
                Site(
                    f"{self.plane.name}.additional_site-{i}",
                    tuple(pos_local),
                    (1, 0, 0, 0),
                    body_name=self.plane.name,
                )
            )
            self.sites_to_connect.append(self.plane.sites[-1])

    def get_second_connecting_tendon_sequences(
        self, direction: ConnectingDirection
    ) -> TendonSequence:
        """
        Only get the tendon sequence from the other side of the plane to the mass on the plane.
        """
        tendons = []
        if self.use_left_site == DirectionsEnum.USE_LEFT:
            # if use left site, then this function is to get the tendon sequence from the right site to the mass
            inner_tendon_seq = TendonSequence(
                elements=[
                    self.masses[-1].right_site.create_spatial_site(),
                    self.plane.right_site.create_spatial_site(),
                ],
                description=f"Tendon sequence connecting prism right site to plane right site",
            )
            tendons.append(inner_tendon_seq)
        else:  # use right site
            # if use right site, then this function is to get the tendon sequence from the left site to the mass
            inner_tendon_seq = TendonSequence(
                elements=[
                    self.masses[0].left_site.create_spatial_site(),
                    self.plane.left_site.create_spatial_site(),
                ],
                description=f"Tendon sequence connecting prism left site to plane left site",
            )
            tendons.append(inner_tendon_seq)

        if direction == ConnectingDirection.INNER_TO_OUTER or direction == ConnectingDirection.DEFAULT:
            return tendons
        else:  # ConnectingDirection.OUTER_TO_INNER
            return [reverse_tendon_sequence(sequence) for sequence in tendons]

    def get_ready_tendon_sequences(self, direction: ConnectingDirection) -> List[TendonSequence]:
        tendons = []
        for i in range(len(self.masses) - 1):
            inner_tendon = TendonSequence(
                elements=[
                    self.masses[i].right_site.create_spatial_site(),
                    self.masses[i + 1].left_site.create_spatial_site()
                ],
                description=f"A tendon sequence connecting mass-{i} to mass-{i+1}",
                name=f"tendon_{i}"
            )
            tendons.append(inner_tendon)
        
        return tendons

    def get_connecting_tendon_sequences(
        self,
        direction: ConnectingDirection = ConnectingDirection.DEFAULT,
        connecting_option: Any = None,
    ) -> List[TendonSequence]:
        """
        Get the tendon sequence for the mass on the plane.
        """
        tendons = []
        if (
            self.use_left_site == DirectionsEnum.USE_LEFT
            or self.use_left_site == DirectionsEnum.USE_BOTH
        ):
            for site in self.sites_to_connect:
                inner_tendon = [
                    self.masses[0].left_site.create_spatial_site(),
                    self.plane.left_site.create_spatial_site(),
                    site.create_spatial_site(),
                ]
                tendons.append(TendonSequence(elements=inner_tendon, description=f"Tendon sequence connecting mass-0 to the plane to the site", name=f"{self.name}.connecting_tendon"))
        else:  # DirectionsEnum.USE_RIGHT
            for site in self.sites_to_connect:
                inner_tendon = [
                    self.masses[-1].right_site.create_spatial_site(),
                    self.plane.right_site.create_spatial_site(),
                    site.create_spatial_site(),
                ]
                tendons.append(TendonSequence(elements=inner_tendon, description=f"Tendon sequence connecting mass-{len(self.masses)-1} to the plane to the site", name=f"{self.name}.connecting_tendon"))

        if (
            direction == ConnectingDirection.INNER_TO_OUTER
            or direction == ConnectingDirection.DEFAULT
        ):
            return tendons
        else:  # ConnectingDirection.OUTER_TO_INNER
            return [reverse_tendon_sequence(sequence) for sequence in tendons]

    def to_xml(self) -> str:
        """
        Convert the mass and plane to an XML string.
        """
        xml = ""
        for mass in self.masses:
            xml += mass.to_xml() + "\n"
        xml += self.plane.to_xml() + "\n"
        return xml

    def align_pose(
        self,
        target_pos: Tuple[float, float, float],
        use_left_site: DirectionsEnum = DirectionsEnum.USE_LEFT,
        displacement_x: float = 0,
        displacement_y: float = 0,
        displacement_z: float = 0,
    ) -> None:
        """
        Change the pose of the mass and plane.
        """
        frame = Frame(origin=np.array(self.plane.pos), quat=np.array(self.plane.quat))
        if use_left_site == DirectionsEnum.USE_LEFT:
            local_original_site_pos = np.array(self.plane.left_site.pos)
            displacement = np.array(target_pos) - frame.rel2global(
                local_original_site_pos
            )
        else:  # DirectionsEnum.USE_RIGHT
            local_original_site_pos = np.array(self.plane.right_site.pos)
            displacement = np.array(target_pos) - frame.rel2global(
                local_original_site_pos
            )

        displacement += tuple(
            np.array([displacement_x, displacement_y, displacement_z])
        )

        # move all bodies
        self.move(displacement)

        # define the origin of the system
        self.origin_pos = target_pos + tuple(
            np.array([displacement_x, displacement_y, displacement_z])
        )

    def get_sensor_list(self) -> List[Sensor]:
        """
        Get the sensors of the masses.
        """
        sensors = []
        for mass in self.masses:
            sensors += mass.get_sensor_list()
        return sensors
