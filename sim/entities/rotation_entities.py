from .base_entities import *
from sim.bodies.mass import Mass
import math
from sim.utils import replace_all, rotated_axes_from_quaternion

class RigidRotationEntity(Entity):

    randomization_levels = {
        DegreeOfRandomization.EASY: {
            "num_bodies": {"min": 1, "max": 1},
            "body_types": ["cylinder", "bar"],
            "position_range": {"min": (-0.1, -0.1, -0.1), "max": (0.1, 0.1, 0.1)},
            "mass_range": {"min": 1.0, "max": 3.0},
            "radius_range": {"min": 0.1, "max": 0.3},
            "height_range": {"min": 0.3, "max": 0.5},
            "length_range": {"min": 0.5, "max": 1.0},
            "width_range": {"min": 0.1, "max": 0.2},
            "joint_position_range": {
                "min": (-0.05, -0.05, -0.05),
                "max": (0.05, 0.05, 0.05),
            },
            "axis_mode": "fixed_y",
        },
        DegreeOfRandomization.MEDIUM: {
            "num_bodies": {"min": 1, "max": 3},
            "body_types": ["cylinder", "bar", "sphere"],
            "position_range": {"min": (-0.5, -0.5, -0.5), "max": (0.5, 0.5, 0.5)},
            "mass_range": {"min": 0.5, "max": 5.0},
            "radius_range": {"min": 0.1, "max": 0.5},
            "height_range": {"min": 0.2, "max": 1.0},
            "length_range": {"min": 0.5, "max": 2.0},
            "width_range": {"min": 0.1, "max": 0.5},
            "joint_position_range": {
                "min": (-0.2, -0.2, -0.2),
                "max": (0.2, 0.2, 0.2),
            },
            "axis_mode": "random_standard",
        },
        DegreeOfRandomization.HARD: {
            "num_bodies": {"min": 2, "max": 4},
            "body_types": ["cylinder", "bar", "sphere", "polygonalprism"],
            "position_range": {"min": (-1.0, -1.0, -1.0), "max": (1.0, 1.0, 1.0)},
            "mass_range": {"min": 0.1, "max": 10.0},
            "radius_range": {"min": 0.05, "max": 0.8},
            "height_range": {"min": 0.1, "max": 2.0},
            "length_range": {"min": 0.3, "max": 3.0},
            "width_range": {"min": 0.05, "max": 0.5},
            "joint_position_range": {
                "min": (-0.5, -0.5, -0.5),
                "max": (0.5, 0.5, 0.5),
            },
            "axis_mode": "random_quat",
        },
    }

    def __init__(
        self,
        name: str,
        pos: Tuple[float, float, float] = (0, 0, 0),
        quat: Tuple[float, float, float, float] = (1, 0, 0, 0),
        rigid_bodies: List[Dict[str, Any]] = None,
        joint: Dict[str, Any] = None,
        **kwargs,
    ):
        """
        RigidRotationEntity is used to create one or more rigid bodies and implement a rotation around a specific axis via a hinge joint.

        Parameters:
        - name: The name of the entity
        - pos: The position of the entity
        - quat: The rotation quaternion of the entity
        - rigid_bodies: A list of dictionaries to define the rigid bodies to be created. Each dictionary may include the following keys:
            {
                "body_type": "Cylinder" or "Bar" or other supported types,
                "pos": (x, y, z),
                "quat": (qw, qx, qy, qz) optional
                "radius": ... for Cylinder and similar types
                "height": ... for Cylinder or Bar
                "length": ... for Bar
                "width": ... for Bar
                ... other parameters for geometries
            }
        - joint: A dictionary to specify the joint information for the axis of rotation, for example:
            {
                "position": (px, py, pz),  # The position of the joint (relative to the entity's local coordinate system)
                "axis": (ax, ay, az) or
                "quat": (qw, qx, qy, qz) # Used to represent the direction of the rotation axis
            }

        If the joint specifies the axis, it will directly use the axis as the direction vector of the hinge joint.
        If a quaternion is provided, the direction of the axis will be extracted from the quaternion.
        """
        if rigid_bodies is None:
            rigid_bodies = []

        self.rigid_bodies = rigid_bodies
        self.created_bodies = []
        self.joint = joint
        super().__init__(name, pos, quat, entity_type=self.__class__.__name__, **kwargs)

        # Create and add rigid bodies from rigid_bodies
        self._create_rigid_bodies()

        # Create and add the hinge joint for rotation around the given axis
        if self.joint is not None:
            self._create_rotation_joint()

    def _create_rigid_bodies_old(self):
        """
        Create rigid bodies based on rigid_bodies and add them to the current entity.
        """
        # For multiple bodies, we add them as child bodies to this Entity
        # The same approach applies to a single body
        for i, body_spec in enumerate(self.rigid_bodies):
            body_type = body_spec.get("body_type", "Cylinder")  # Default to Cylinder
            body_pos = body_spec.get("pos", (0, 0, 0))
            body_quat = body_spec.get("quat", (1, 0, 0, 0))

            thickness = body_spec.get("thickness", 0.0)  # TODO: thickness cannot be applied to every body type, we need to only consider the mesh

            if body_type.lower() == "cylinder":
                radius = body_spec.get("radius", 0.1)
                height = body_spec.get("height", 0.5)
                mass = body_spec.get("mass", 1.0)
                rgba = body_spec.get("rgba", (0.5, 0.5, 0.5, 1))
                body = Cylinder(
                    name=f"{self.name}.cylinder-{i}",
                    pos=(0, 0, 0),
                    radius=radius,
                    height=height,
                    mass=mass,
                    rgba=rgba,
                    thickness=thickness,
                )
                body.set_pose(body_pos, body_quat)
            elif body_type.lower() == "bar":
                length = body_spec.get("length", 1.0)
                width = body_spec.get("width", 0.1)
                height = body_spec.get("height", 0.1)
                end_pos = body_spec.get("end_pos", None)
                mass = body_spec.get("mass", 1.0)
                body = Bar(
                    name=f"{self.name}.bar-{i}",
                    pos=body_pos,
                    length=length,
                    width=width,
                    height=height,
                    quat=body_quat,
                    end_pos=end_pos,
                    mass=mass,
                )
            elif body_type.lower() == "sphere":
                radius = body_spec.get("radius", 0.1)
                mass = body_spec.get("mass", 1.0)
                rgba = body_spec.get("rgba", (0.5, 0.5, 0.5, 1))
                body = Sphere(  # TODO: sphere initialization is not mesh, if we need thickness we need to consider the mesh
                    name=f"{self.name}.sphere-{i}",
                    pos=body_pos,
                    radius=radius,
                    mass=mass,
                    rgba=rgba,
                    quat=body_quat,
                    thickness=thickness,
                )
            elif body_type.lower() == "polygonalprism":
                sides = body_spec.get("sides", 6)
                radius = body_spec.get("radius", 0.1)
                height = body_spec.get("height", 0.5)
                mass = body_spec.get("mass", 1.0)
                rgba = body_spec.get("rgba", (0.5, 0.5, 0.5, 1))
                body = PolygonalPrism(
                    name=f"{self.name}.polygonalprism-{i}",
                    pos=(0, 0, 0),
                    sides=sides,
                    radius=radius,
                    height=height,
                    mass=mass,
                    rgba=rgba,
                )
                body.set_pose(body_pos, body_quat)
            else:
                raise ValueError(f"Unsupported body_type: {body_type}")

            body.joints = []  # Reset the quaternion to avoid double
            self.add_child_body(body)
            self.created_bodies.append(body)

    def _create_rigid_bodies(self):
        """
        Create rigid bodies based on rigid_bodies and add them to the current entity.
        """
        # For multiple bodies, we add them as child bodies to this Entity
        # The same approach applies to a single body
        body_class_mapping = {
            "cylinder": Cylinder,
            "bar": Bar,
            "sphere": Sphere,
            "polygonalprism": PolygonalPrism,
            "disc": Disc,
            "hemisphere": Hemisphere,
            "bowl": Bowl,
            "sphere_with_hole": SphereWithHole,
        }

        self.body_class_mapping = body_class_mapping

        for i, body_spec in enumerate(self.rigid_bodies):
            # 1) Extract the body_type and convert to lowercase to match the mapping
            body_type = body_spec.get("body_type", "cylinder").lower()

            # 2) Check if the body_type is supported in body_class_mapping
            if body_type not in self.body_class_mapping:
                raise ValueError(f"Unsupported body_type: {body_type}")

            # 3) Get the actual geometry class
            body_class = self.body_class_mapping[body_type]

            # 4) Make a copy of body_spec to avoid modifying the original dict
            spec_copy = dict(body_spec)

            # 5) Handle the name: if not provided by user, generate one like "myEntity.hemisphere-0"
            generated_name = f"{self.name}.{body_type}-{i}"
            spec_copy.setdefault("name", generated_name)

            # 6) Read pos/quat and remove them from spec to prevent duplication errors in constructor
            #    (Whether this is needed depends on whether the geometry class constructor accepts pos/quat)
            body_pos = spec_copy.pop("pos", (0, 0, 0))
            body_quat = spec_copy.pop("quat", (1, 0, 0, 0))

            # 7) The remaining items in spec_copy are the other arguments to be passed to the geometry constructor
            try:
                body = body_class(**spec_copy)
            except:
                body = body_class(pos = body_pos, quat = body_quat, **spec_copy)

            # 8) Set pose (pos & quat). Some geometry classes might have their own set_pose() or set_position()
            #    If pos/quat are handled in the constructor, this step can be skipped. Depends on class design.
            body.set_pose(body_pos, body_quat)

            if body.joints:
                body.joints = []  # Reset the joints to avoid double

            # 9) Now the body is a valid child body. Add it to the entity
            self.add_child_body(body)
            self.created_bodies.append(body)

    def _create_rotation_joint(self):
        """
        Create a hinge joint based on the joint parameters to achieve rotation around the specified axis.
        The joint must include "position" and either "axis" or "quat".
        """
        joint_position = self.joint.get("position", (0, 0, 0))
        axis = self.joint.get("axis", None)
        quat = self.joint.get("quat", None)

        if axis is None:
            # If axis is not provided, attempt to extract the rotation axis from quat
            if quat is None:
                # If neither is provided, default to rotation around the Y-axis
                axis = (0, 1, 0)
            else:
                # quat = (w,x,y,z)
                w, xq, yq, zq = quat
                angle = 2 * math.acos(w)
                s = math.sqrt(1 - w * w) if (1 - w * w) > 0 else 0
                if s < 1e-8:
                    # When quat is close to (1,0,0,0) or has negligible rotation, arbitrarily choose an axis
                    axis = (1, 0, 0)
                else:
                    axis = (xq / s, yq / s, zq / s)

        # Create the hinge joint
        # The Joint initialization assumes: Joint(type, axis, name, pos=...) already supports the pos parameter
        joint_obj = Joint(
            joint_type="hinge", axis=axis, name="{self.name}.joint", pos=joint_position
        )
        self.add_joint(joint_obj)

    def randomize_parameters(
        self,
        degree_of_randomization: DegreeOfRandomization = DegreeOfRandomization.NON_STRUCTURAL,
        reinitialize_instance: bool = False,
        **kwargs,
    ):
        """
        - EASY: Single rotational body, simple axis (Z-axis), small parameter range
        - MEDIUM: 1-3 objects, moderate parameter range, random standard axis
        - HARD: 2-4 objects, large parameter range, random quaternion axis
        - NON_STRUCTURAL: Fine-tuning existing parameters
        """
        randomization_levels = {
            DegreeOfRandomization.EASY: {
                "num_bodies": {"min": 1, "max": 1},
                "body_types": ["cylinder", "bar"],
                "position_range": {"min": (-0.1, -0.1, -0.1), "max": (0.1, 0.1, 0.1)},
                "mass_range": {"min": 1.0, "max": 3.0},
                "radius_range": {"min": 0.1, "max": 0.3},
                "height_range": {"min": 0.3, "max": 0.5},
                "length_range": {"min": 0.5, "max": 1.0},
                "width_range": {"min": 0.1, "max": 0.2},
                "joint_position_range": {
                    "min": (-0.05, -0.05, -0.05),
                    "max": (0.05, 0.05, 0.05),
                },
                "axis_mode": "fixed_y",
                "body_quat_mode": "identity",  # no rotation
                "body_thickness_mode": "solid",
            },
            DegreeOfRandomization.MEDIUM: {
                "num_bodies": {"min": 1, "max": 3},
                "body_types": ["cylinder", "bar", "sphere"],
                "position_range": {"min": (-0.5, -0.5, -0.5), "max": (0.5, 0.5, 0.5)},
                "mass_range": {"min": 0.5, "max": 5.0},
                "radius_range": {"min": 0.1, "max": 0.5},
                "height_range": {"min": 0.2, "max": 1.0},
                "length_range": {"min": 0.5, "max": 2.0},
                "width_range": {"min": 0.1, "max": 0.5},
                "joint_position_range": {
                    "min": (-0.2, -0.2, -0.2),
                    "max": (0.2, 0.2, 0.2),
                },
                "axis_mode": "random_standard",
                "body_quat_mode": "random_z",
                "body_thickness_mode": "random_shell",
            },
            DegreeOfRandomization.HARD: {
                "num_bodies": {"min": 2, "max": 4},
                "body_types": [
                    "cylinder", "bar", "sphere", "polygonalprism",
                    "disc", "hemisphere", "bowl", "sphere_with_hole"
                ],
                "position_range": {"min": (-1.0, -1.0, -1.0), "max": (1.0, 1.0, 1.0)},
                "mass_range": {"min": 0.1, "max": 10.0},

                "radius_range": {"min": 0.05, "max": 0.8},
                "height_range": {"min": 0.1, "max": 2.0},
                "length_range": {"min": 0.3, "max": 3.0},
                "width_range": {"min": 0.05, "max": 0.5},

                "joint_position_range": {
                    "min": (-0.5, -0.5, -0.5),
                    "max": (0.5, 0.5, 0.5),
                },
                "axis_mode": "random_quat",
                "body_quat_mode": "random_quat",
                "body_thickness_mode": "random_shell",

                # hemisphere
                "hemisphere_radius_range": (0.1, 0.5),
                "hemisphere_thickness_range": (0.1, 0.75),

                # bowl
                "bowl_radius_range": (0.1, 0.5),
                "bowl_height_range": (0.05, 0.2),
                "bowl_thickness_range": (0.1, 0.75),

                # sphere_with_hole
                "sphere_with_hole_radius_range": (0.2, 0.5),
                "sphere_with_hole_hole_radius_range": (0.05, 0.15),
                "sphere_with_hole_hole_position_range": (-0.2, 0.2),
                "sphere_with_hole_thickness_range": (0.1, 0.75),

                # disc
                "disc_radius_range": (0.1, 0.4),
                "disc_thickness_range": (0.0, 0.3),
            },
        }

        self.randomization_levels = randomization_levels  # Defined at the beginning of the class
    
        # If it is DEFAULT, randomly choose EASY/MEDIUM/HARD
        if degree_of_randomization == DegreeOfRandomization.DEFAULT:
            degree_of_randomization = random.choice([
                DegreeOfRandomization.EASY,
                DegreeOfRandomization.MEDIUM,
                DegreeOfRandomization.HARD,
            ])

        if degree_of_randomization in randomization_levels:
            params = randomization_levels[degree_of_randomization]

            # ---------------------------
            # 1) Randomly generate the number of bodies required
            # ---------------------------
            num_bodies = random.randint(
                params["num_bodies"]["min"], params["num_bodies"]["max"]
            )

            new_specs = []
            for _ in range(num_bodies):
                # Select body_type
                body_type = random.choice(params["body_types"])

                # Random position
                min_pos = params["position_range"]["min"]
                max_pos = params["position_range"]["max"]
                pos = (
                    round(random.uniform(min_pos[0], max_pos[0]), 2),
                    round(random.uniform(min_pos[1], max_pos[1]), 2),
                    round(random.uniform(min_pos[2], max_pos[2]), 2),
                )

                # Random mass
                mass = round(random.uniform(
                    params["mass_range"]["min"], params["mass_range"]["max"]
                ), 2)

                # Prepare spec
                spec = {
                    "body_type": body_type,
                    "pos": pos,
                    "mass": mass,
                }

                # ----------------------------
                # 2) Subdivide parameters based on body_type
                # ----------------------------
                if body_type in ["cylinder", "sphere", "polygonalprism", "disc", "hemisphere", "bowl", "sphere_with_hole"]:
                    # For most bodies that need radius, randomly choose radius first
                    # But the specific range depends on whether it is HARD. If it is EASY/MEDIUM, get it from "radius_range"
                    # If it is HARD, get it from "radius_range" or the specific "hemisphere_radius_range"
                    if body_type in ["cylinder", "sphere", "polygonalprism", "disc"]:
                        # Use params["radius_range"]
                        r = random.uniform(params["radius_range"]["min"], params["radius_range"]["max"])
                        spec["radius"] = round(r, 2)
                    
                    # hemisphere
                    if body_type == "hemisphere":
                        # Use hemisphere_radius_range
                        rmin, rmax = params["hemisphere_radius_range"]
                        spec["radius"] = round(random.uniform(rmin, rmax), 2)
                        # thickness
                        tmin, tmax = params["hemisphere_thickness_range"]
                        spec["thickness"] = round(random.uniform(tmin, tmax), 2)

                    # bowl
                    elif body_type == "bowl":
                        rmin, rmax = params["bowl_radius_range"]
                        spec["radius"] = round(random.uniform(rmin, rmax), 2)
                        hmin, hmax = params["bowl_height_range"]
                        spec["height"] = round(random.uniform(hmin, hmax), 2)
                        tmin, tmax = params["bowl_thickness_range"]
                        spec["thickness"] = round(random.uniform(tmin, tmax), 2)

                    # sphere_with_hole
                    elif body_type == "sphere_with_hole":
                        rmin, rmax = params["sphere_with_hole_radius_range"]
                        spec["radius"] = round(random.uniform(rmin, rmax), 2)
                        hrmin, hrmax = params["sphere_with_hole_hole_radius_range"]
                        spec["hole_radius"] = round(random.uniform(hrmin, hrmax), 2)
                        hpmin, hpmax = params["sphere_with_hole_hole_position_range"]
                        spec["hole_position"] = round(random.uniform(hpmin, hpmax), 2)
                        thmin, thmax = params["sphere_with_hole_thickness_range"]
                        spec["thickness"] = round(random.uniform(thmin, thmax), 2)

                    # disc special case: can also randomly choose thickness
                    if body_type == "disc":
                        dtmin, dtmax = params["disc_thickness_range"]
                        spec["thickness"] = round(random.uniform(dtmin, dtmax), 2)

                # bar handled separately: because bar has length, width, and height instead of radius
                if body_type == "bar":
                    spec["length"] = round(random.uniform(
                        params["length_range"]["min"], params["length_range"]["max"]
                    ), 2)
                    spec["width"] = round(random.uniform(
                        params["width_range"]["min"], params["width_range"]["max"]
                    ), 2)
                    spec["height"] = round(random.uniform(
                        params["height_range"]["min"], params["height_range"]["max"]
                    ), 2)

                # If cylinder or polygonalprism, need height
                if body_type in ["cylinder", "polygonalprism"]:
                    spec["height"] = round(random.uniform(
                        params["height_range"]["min"], params["height_range"]["max"]
                    ), 2)

                # ----------------------------
                # 3) Random thickness strategy (optional)
                # ----------------------------
                # If the body_type requires thickness, but we can also decide whether it's solid or shell based on body_thickness_mode
                if body_type in ["sphere_with_hole", "hemisphere", "bowl"]:  # For example
                    btm = params.get("body_thickness_mode", "solid")
                    if btm == "solid" or (btm == "random_shell" and random.random() < 0.7):
                        spec["thickness"] = -1  # Solid

                # ----------------------------
                # 4) Random quaternion rotation
                # ----------------------------
                body_quat_mode = params.get("body_quat_mode", "identity")
                if body_quat_mode == "identity":
                    spec["quat"] = (1, 0, 0, 0)

                elif body_quat_mode == "random_z":
                    yaw_angle = round(random.uniform(0, 2 * math.pi), 2)
                    cz = math.cos(yaw_angle / 2.0)
                    sz = math.sin(yaw_angle / 2.0)
                    spec["quat"] = (cz, 0, 0, sz)

                elif body_quat_mode == "random_quat":
                    axis = [round(random.uniform(-1, 1), 2) for _ in range(3)]
                    axis_len = math.sqrt(axis[0]**2 + axis[1]**2 + axis[2]**2)
                    if axis_len < 1e-6:
                        axis = (1, 0, 0)
                    else:
                        axis = (axis[0]/axis_len, axis[1]/axis_len, axis[2]/axis_len)
                    angle = round(random.uniform(0, 2 * math.pi), 2)
                    w = math.cos(angle / 2.0)
                    x = math.sin(angle / 2.0) * axis[0]
                    y = math.sin(angle / 2.0) * axis[1]
                    z = math.sin(angle / 2.0) * axis[2]
                    spec["quat"] = (w, x, y, z)

                # ----------------------------
                # 5) Normalizationn
                # ----------------------------
                if body_type in ["sphere_with_hole", "hemisphere", "bowl"]:
                    if "thickness" in spec: 
                        spec["thickness"] = round(spec["thickness"] * spec["radius"], 2)
                    if "hole_radius" in spec:
                        spec["hole_radius"] = round(spec["hole_radius"] * spec["radius"], 2)
                    if "hole_position" in spec:
                        spec["hole_position"] = round(spec["hole_position"] * spec["radius"], 2)
                
                new_specs.append(spec)

            # Put the newly generated rigid_bodies into self.rigid_bodies
            self.rigid_bodies = new_specs

            # ----------------------------
            # 6) Randomize joint parameters
            # ----------------------------
            self.joint = {}
            self.joint["position"] = tuple(
                round(random.uniform(_min, _max), 2)
                for _min, _max in zip(
                    params["joint_position_range"]["min"], params["joint_position_range"]["max"]
                )
            )

            if params["axis_mode"] == "fixed_y":
                self.joint["axis"] = (0, 1, 0)
            elif params["axis_mode"] == "random_standard":
                self.joint["axis"] = random.choice([(1, 0, 0), (0, 1, 0)])
            elif params["axis_mode"] == "random_quat":
                angle = round(random.uniform(0, 2 * math.pi), 2)
                axis = [round(random.uniform(-1, 1), 2) for _ in range(3)]
                axis_len = math.sqrt(sum(x*x for x in axis))
                if axis_len < 1e-6:
                    axis = (0, 1, 0)
                else:
                    axis = tuple(x / axis_len for x in axis)
                self.joint["quat"] = (
                    math.cos(angle / 2),
                    math.sin(angle / 2) * axis[0],
                    math.sin(angle / 2) * axis[1],
                    math.sin(angle / 2) * axis[2],
                )
                # Either axis or quat, not both. If quat is generated, delete axis.
                if "axis" in self.joint:
                    del self.joint["axis"]

        elif degree_of_randomization == DegreeOfRandomization.NON_STRUCTURAL:
            # ----------------------------
            # Non-structural: Make slight adjustments to existing rigid_bodies
            # ----------------------------
            for spec in self.rigid_bodies:
                # mass
                if "mass" in spec:
                    spec["mass"] = round(spec["mass"] * random.uniform(0.9, 1.1), 2)

                # Slight adjustment in position
                px, py, pz = spec.get("pos", (0, 0, 0))
                spec["pos"] = (
                    round(px + random.uniform(-0.05, 0.05), 2),
                    round(py + random.uniform(-0.05, 0.05), 2),
                    round(pz + random.uniform(-0.05, 0.05), 2),
                )

                # If radius exists, apply random multiplication
                if "radius" in spec:
                    spec["radius"] = round(spec["radius"] * random.uniform(0.9, 1.1), 2)
                # If thickness exists
                if "thickness" in spec:
                    spec["thickness"] = round(spec["thickness"] * random.uniform(0.9, 1.1), 2)
                # If height exists
                if "height" in spec:
                    spec["height"] = round(spec["height"] * random.uniform(0.9, 1.1), 2)
                # If length/width exists
                if "length" in spec:
                    spec["length"] = round(spec["length"] * random.uniform(0.9, 1.1), 2)
                if "width" in spec:
                    spec["width"] = round(spec["width"] * random.uniform(0.9, 1.1), 2)
                # If sphere_with_hole
                if "hole_radius" in spec:
                    spec["hole_radius"] = round(spec["hole_radius"] * random.uniform(0.9, 1.1), 2)
                if "hole_position" in spec:
                    spec["hole_position"] = round(spec["hole_position"] + random.uniform(-0.05, 0.05), 2)

            # Slight adjustment to joint
            jpx, jpy, jpz = self.joint.get("position", (0, 0, 0))
            self.joint["position"] = (
                round(jpx + random.uniform(-0.05, 0.05), 2),
                round(jpy + random.uniform(-0.05, 0.05), 2),
                round(jpz + random.uniform(-0.05, 0.05), 2),
            )

            if "axis" in self.joint:
                ax, ay, az = self.joint["axis"]
                ax += round(random.uniform(-0.1, 0.1), 2)
                ay += round(random.uniform(-0.1, 0.1), 2)
                az += round(random.uniform(-0.1, 0.1), 2)
                length = math.sqrt(ax**2 + ay**2 + az**2)
                if length < 1e-6:
                    length = 1e-6
                self.joint["axis"] = (ax / length, ay / length, az / length)

        # If reinitialization is needed
        if reinitialize_instance:
            self.reinitialize()

    def generate_entity_yaml(
        self,
        use_random_parameters: bool = False,
        degree_of_randomization: DegreeOfRandomization = DegreeOfRandomization.NON_STRUCTURAL,
    ) -> dict:
        """
        Export entity parameters (bodies and joint) to a dict.
        """
        if use_random_parameters:
            self.randomize_parameters(
                degree_of_randomization, reinitialize_instance=True
            )

        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "position": list(self.pos),
            "parameters": {"rigid_bodies": self.rigid_bodies, "joint": self.joint},
        }

    def get_parameters(self) -> List[dict]:
        """
        Return details about all created rigid bodies (e.g. mass and names).
        """
        param_list = []
        for body in self.created_bodies:
            for geom in body.geoms:
                param_list.append(
                    {
                        "body_name": body.name,
                        "geom_name": geom.name,
                        "mass": float(geom.mass),
                    }
                )
        return param_list

    def to_xml(self) -> str:
        """
        Convert the entity to an XML string.
        """
        body_xml = f"""<body name="{self.name}" pos="{' '.join(map(str, self.pos))}" quat="{' '.join(map(str, self.quat))}">"""
        # Add joints
        for joint in self.joints:
            body_xml += joint.to_xml() + "\n"

        # Add child bodies (each child body contains its own geom, etc.)
        for body in self.child_bodies:
            body_xml += body.to_xml() + "\n"

        body_xml += "</body>"
        return body_xml

    def get_nlq(self, symbolic = False):
        description = (
            f"In a system, there are {len(self.rigid_bodies)} rigid bodies that rotate about a hinge joint. "
            f"These bodies are welded together, so they always move as one rigid assembly. "
            f"The rigid bodies are:\n"
        )

        sym_dict = {}
        body_descriptions = []

        for idx, body in enumerate(self.rigid_bodies):
            body_type = body.get("body_type", "cylinder").lower()
            pretty_type = "polygonal cylinder" if body_type == "polygonalprism" else body_type
            body_name = self.created_bodies[idx].name if idx < len(self.created_bodies) else f"{body_type}_{idx}"

            mass_key = f"<mass>{idx}"
            sym_dict[mass_key] = body.get("mass", 1.0)

            body_pos = body.get("pos", (0, 0, 0))
            body_quat = body.get("quat", (1, 0, 0, 0))
            new_axes = [tuple(axis) for axis in rotated_axes_from_quaternion(body_quat)]

            body_description = (
                f" A {pretty_type} called '{body_name}' has a mass of {mass_key} kg."
            )

            # Frame-origin wording is aligned with how body frames are defined in geometry constructors.
            frame_origin_desc = "Its body-frame origin"
            orientation_desc = ""

            if body_type == "bar":
                length = body.get("length", 1.0)
                width = body.get("width", 0.1)
                height = body.get("height", 0.1)
                body_description += f" It has length {length} m, width {width} m, and height {height} m."
                frame_origin_desc = "Its body-frame origin (the center of the face at the negative end of the local length axis)"
                orientation_desc = (
                    f" The local length, width, and height axes point along {new_axes[0]}, {new_axes[1]}, and {new_axes[2]}, respectively."
                )
            elif body_type == "cylinder":
                radius = body.get("radius", 0.1)
                height = body.get("height", 0.5)
                body_description += f" It has radius {radius} m and height {height} m."
                frame_origin_desc = "Its body-frame origin (the cylinder center)"
                orientation_desc = f" Its cylinder axis is along {new_axes[-1]}."
            elif body_type == "sphere":
                radius = body.get("radius", 0.1)
                body_description += f" It has radius {radius} m."
                frame_origin_desc = "Its body-frame origin (the sphere center)"
            elif body_type == "polygonalprism":
                sides = body.get("sides", 6)
                radius = body.get("radius", 0.1)
                height = body.get("height", 0.5)
                body_description += (
                    f" Its cross-section is a regular polygon with {sides} sides and circumscribing radius {radius} m, "
                    f"and it has height {height} m."
                )
                frame_origin_desc = "Its body-frame origin (the prism center)"
                orientation_desc = f" Its prism axis is along {new_axes[0]}."
            elif body_type == "disc":
                radius = body.get("radius", 0.1)
                thickness = 2 * body.get("height", 0.01)
                body_description += f" It has radius {radius} m and thickness {thickness} m."
                frame_origin_desc = "Its body-frame origin (the disc center)"
                orientation_desc = f" Its disc normal is along {new_axes[-1]}."
            elif body_type == "hemisphere":
                radius = body.get("radius", 0.1)
                thickness = body.get("thickness", 0.01)
                thickness_desc = "it is solid" if thickness < 0 else f"its shell thickness is {thickness:.2f} m"
                body_description += f" It has radius {radius} m and {thickness_desc}."
                frame_origin_desc = "Its body-frame origin (the center of the parent sphere / base-circle center for the default cut)"
                orientation_desc = (
                    f" The axis perpendicular to the circular base, pointing into the solid cap, is along "
                    f"{tuple([-x for x in new_axes[-1]])}."
                )
            elif body_type == "bowl":
                radius = body.get("radius", 0.1)
                cut_height = body.get("height", 0.1)
                thickness = body.get("thickness", 0.01)
                thickness_desc = "it is solid" if thickness < 0 else f"its shell thickness is {thickness:.2f} m"
                body_description += (
                    f" It has radius {radius} m and {thickness_desc}. "
                    f"The cutout plane is at height {cut_height} m from the parent sphere center."
                )
                frame_origin_desc = "Its body-frame origin (the center of the parent sphere)"
                orientation_desc = (
                    f" The axis perpendicular to the bowl opening, pointing into the solid cap, is along "
                    f"{tuple([-x for x in new_axes[-1]])}."
                )
            elif body_type == "sphere_with_hole":
                radius = body.get("radius", 0.1)
                hole_radius = body.get("hole_radius", 0.05)
                hole_position = body.get("hole_position", 0.0)
                thickness = body.get("thickness", 0.01)
                thickness_desc = "it is solid" if thickness < 0 else f"its shell thickness is {thickness:.2f} m"
                hole_pos = tuple(round(x * hole_position, 2) for x in new_axes[0])
                body_description += (
                    f" It has radius {radius} m and {thickness_desc}. "
                    f"The spherical hole has radius {hole_radius} m and is centered at {hole_pos} m from the body-frame origin."
                )
                frame_origin_desc = "Its body-frame origin (the center of the parent sphere before hole subtraction)"

            body_description += (
                f" {frame_origin_desc} is at position {body_pos} m in the entity frame."
                f"{orientation_desc} The listed position/orientation describe the body frame (not necessarily the COM).\n"
            )
            body_descriptions.append(body_description)

        if self.joint is not None and len(self.joints) > 0:
            joint_description = (
                f"The hinge joint is located at {self.joint['position']} m in the entity frame. "
                f"It allows rotation about axis {self.joints[0].axis} (unit vector)."
            )
        else:
            joint_description = "No hinge joint is defined."

        full_description = description + "".join(body_descriptions) + joint_description
        if symbolic:
            return full_description, sym_dict
        return replace_all(full_description, sym_dict)

    def get_question(self, sub_entity: str, quantity: str) -> str:
        idx = int(sub_entity.split('-')[-1])
        body = self.rigid_bodies[idx]
        body_type = body.get("body_type", "cylinder")
        body_name = body_type.lower() + f'_{idx}'
        return f"What is the {quantity} of the {body_name} in the system?"

class BarPlaneSupport(Entity):
    """
    Place a Bar on a horizontal Plane that can rotate around the y-axis and is supported by a vertical pillar (also a Bar).
    - plane_slope: No longer used, defaults to 0 degrees (i.e., keeping the Plane horizontal).
    - bar_angle: The angle of the Bar's rotation around the global y-axis (in degrees).
    - bar_length, bar_width, bar_height: Dimensions of the Bar, where length aligns with the x-axis, width with the y-axis, and height with the z-axis.
    - support_ratio: The proportion of the Bar's length (along the x-axis) where the pillar contacts the bottom surface of the Bar, typically within the range [3/5, 1].
    - support_width, support_thickness: Dimensions of the pillar's (essentially another Bar) cross-section.

    Assumptions:
      1. The Plane remains horizontal without tilting.
      2. In the coordinate system, it is not important whether the z-axis points "up" or "down"; the key is that the Bar rotates around the y-axis in this example.
      3. The bottom surface refers to the surface where y = -width/2 in the Bar's local coordinate system.
    """

    def __init__(
        self,
        name: str,
        pos: Tuple[float, float, float] = (0, 0, 0),
        quat: Tuple[float, float, float, float] = (1, 0, 0, 0),
        bar_angle: float = 30.0,  # The rotation angle of the Bar around the y-axis (degrees)
        bar_length: float = 1.0,
        bar_width: float = 0.04,  # Assume the y direction represents "thickness/width"
        bar_height: float = 0.02,  # Assume the z direction represents "height"
        support_ratio: float = 0.5,  # The proportion of the Bar's length where the pillar contacts it
        support_width: float = 0.02,  # The "width" of the pillar's cross-section
        support_thickness: float = 0.02,
        plane_angle: float = 0.0,  # The angle between the Plane and the global coordinate system (degrees)
        **kwargs,
    ):
        # Parent class initialization
        super().__init__(
            name=name,
            pos=pos,
            quat=quat,
            entity_type=self.__class__.__name__,
            **kwargs,
        )
        self.set_quat_with_angle(plane_angle)
        self.bar_angle = bar_angle
        self.bar_length = bar_length
        self.bar_width = bar_width
        self.bar_height = bar_height
        self.support_ratio = support_ratio
        self.support_width = support_width
        self.support_thickness = support_thickness
        self.plane_angle = plane_angle

        # 1. Create the Plane: Assume horizontal placement (no rotation)
        self.plane = Plane(
            name=f"{self.name}.plane",
            pos=pos,  # Align the center of the Plane with the Entity's pos
            quat=(1.0, 0.0, 0.0, 0.0),  # No tilt => unit quaternion
            size=(1.0, 1.0, 0.01),  # Example only
        )

        # 2. Create the Bar: Rotate around the y-axis by bar_angle
        bar_quat = self._quat_from_yaxis_rotation(-self.bar_angle)
        # Assume the default aligns the Bar's "left end" (local x=0) with the Plane's center, and raise it slightly above the Plane
        bar_start_pos = (
            pos[0],
            pos[1],
            pos[2]
            + self.plane.size[2]
            + self.bar_height / 2 * math.cos(math.radians(self.bar_angle)),
        )
        self.bar = Bar(
            name=f"{self.name}.bar",
            pos=bar_start_pos,
            length=self.bar_length,
            width=self.bar_width,
            height=self.bar_height,
            quat=bar_quat,
        )

        self.bar.joints = []  # No joints required
        self.bar.add_joint(
            Joint(joint_type="free", axis=(1, 1, 1), name=f"{self.name}.joint")
        )

        column_top_pos = (
            (self.support_ratio * self.bar_length)
            * math.cos(math.radians(self.bar_angle)),
            # - self.support_width / 2,  # x
            0.0,  # self.bar_width / 2,
            (self.support_ratio * self.bar_length)
            * math.sin(math.radians(self.bar_angle)),  # z => center
        )

        column_buttom_pos = (
            (self.support_ratio * self.bar_length)
            * math.cos(math.radians(self.bar_angle)),
            # - self.support_width / 2,  # x
            0.0,  # self.bar_width / 2,
            0,
        )

        x_offset = + self.bar_height/2*math.sin(math.radians(self.bar_angle)) + self.support_thickness/2
        y_offset = - self.bar_height/2*math.cos(math.radians(self.bar_angle))

        column_top_pos = (
            column_top_pos[0] + bar_start_pos[0] + x_offset,
            column_top_pos[1] + bar_start_pos[1],
            column_top_pos[2] + bar_start_pos[2] + y_offset,
        )
        column_buttom_pos = (
            column_buttom_pos[0] + bar_start_pos[0] + x_offset,
            column_buttom_pos[1] + bar_start_pos[1],
            column_buttom_pos[2]#  + bar_start_pos[2],
        )

        # 5. Create the pillar using the Bar class, letting it automatically calculate length and orientation from bottom to top
        #    In the Bar constructor, pass: pos=bottom, end_pos=top
        self.support = Bar(
            name=f"{self.name}.support",
            pos=column_top_pos,
            end_pos=column_buttom_pos,  # Let the Bar class calculate orientation & length
            width=self.support_width,
            height=self.support_thickness,
        )

    @staticmethod
    def _quat_from_yaxis_rotation(
        angle_deg: float,
    ) -> Tuple[float, float, float, float]:
        """
        Simple quaternion for rotation around the y-axis by angle_deg
        Assume the y-axis is vertical; bar_angle=0 => Bar aligns with the x-axis.
        """
        theta = math.radians(angle_deg)
        half = theta * 0.5
        qw = math.cos(half)
        qx = 0.0
        qy = math.sin(half)
        qz = 0.0
        return (qw, qx, qy, qz)

    def to_xml(self) -> str:
        """
        Combine the Plane, Bar, and support into a single body node
        """
        xml_str = (
            f'<body name="{self.name}" '
            f'pos="{self.pos[0]} {self.pos[1]} {self.pos[2]}" '
            f'quat="{self.quat[0]} {self.quat[1]} {self.quat[2]} {self.quat[3]}">\n'
        )
        # Plane
        xml_plane = self.plane.to_xml()
        xml_str += "  " + xml_plane.replace("\n", "\n  ") + "\n"

        # Bar
        xml_bar = self.bar.to_xml()
        xml_str += "  " + xml_bar.replace("\n", "\n  ") + "\n"

        # Support
        xml_support = self.support.to_xml()
        xml_str += "  " + xml_support.replace("\n", "\n  ") + "\n"

        xml_str += "</body>"
        return xml_str

    def randomize_parameters(
        self,
        degree_of_randomization: DegreeOfRandomization = DegreeOfRandomization.NON_STRUCTURAL,
        reinitialize_instance: bool = False,
        **kwargs,
    ):
        self.bar_length = round(random.uniform(0.5, 2.0), 2)
        self.bar_angle = round(random.uniform(0, 90), 2)
        self.support_ratio = round(random.uniform(0.6, 1.0), 2)

        # Re-initialize if needed
        if reinitialize_instance:
            self.reinitialize()

    def get_parameters(self) -> Dict[str, Union[float, Tuple]]:
        """
        Return key parameter information for this entity
        """
        return {}
    
    def generate_entity_yaml(self, use_random_parameters: bool = False, degree_of_randomization: DegreeOfRandomization = DegreeOfRandomization.NON_STRUCTURAL) -> dict:
        """
        Generate a dictionary representation of the entity's parameters.
        """
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "position": list(self.pos),
            "parameters": {
                "bar_angle": self.bar_angle,
                "bar_length": self.bar_length,
                "bar_width": self.bar_width,
                "bar_height": self.bar_height,
                "support_ratio": self.support_ratio,
            }
        }

    def get_nlq(self, symbolic = False):
        """
        Generate a natural language query (NLQ) description of the entity.
        """

        sym_dict = {}
        
        sym_dict["<angle>1"] = self.bar_angle
        angle = "<angle>1"

        sym_dict["<length>1"] = self.bar_length
        length = "<length>1"
        

        description = (
            f"A bar is placed on a horizontal plane, supported by a vertical pillar."
            f" The bar makes an angle of {angle} degrees with the horizontal initially."
            f" The bar has a length of {length} m."
            f" The pillar has a support ratio of {self.support_ratio}, meaning the length of the bar between the ground and pillar is of that proportion."
        )

        if not symbolic:
            description = replace_all(description, sym_dict)
        else: return description, sym_dict

        return description
    
    def get_question(self, sub_entity: str, quantity: str) -> str:
        """
        Generate a question based on the entity's parameters.
        """
        return f"What is the {quantity} of the bar in the system?"
        
class RollingPlaneEntity(Entity):
    """
    Contains:
      1) A Plane (default horizontal, quat=(1,0,0,0)),
      2) A mesh_body initialized through a dictionary parameter,
      3) The entire entity rotates around the y-axis controlled by the slope,
      4) The angle of mesh_body controls its rotation around the y-axis on the plane,
      5) Reads the real STL of mesh_body and calculates min_z, then positions it to "just touch" the plane.
      6) (Optional) A Mass, placed on the plane, connected to the mesh_body.

    Parameters
    ----------
    name : str
        Entity name.
    pos : tuple of float
        The (x, y, z) position of this entity in the parent coordinate frame.
    quat : tuple of float
        The (w, x, y, z) quaternion for this entity's orientation.
    slope : float
        Degrees by which the entire entity is tilted around the y-axis.
        Internally calls `self.set_quat_with_angle(self.slope, axis='y')`.
    plane_size : tuple of float
        Size of the plane; typically (length, width, thickness).
    mesh_body : dict
        A dictionary describing the mesh body parameters. Must contain key "body_type".
        For example:
          {
            "body_type": "hemisphere",
            "radius": 0.2,
            "thickness": 0.0,
            ...
          }
    body_angle : float
        Rotation of the mesh body around z-axis in its local frame (relative to the plane).
    mass_spec : dict, optional
        If provided, create a `Mass` block on this plane. Recognized fields include:
          - "name": str, optional
              The mass name; if not provided uses f"{self.name}.mass".
          - "positions": list of (x, y, z) tuples
              Where the mass center is placed (in this entity's local frame). Default is [(0, 0, z_offset + 0.1)].
          - "mass_value": float
              The mass (MuJoCo unit). Default = 1.0.
          - "joint_option": tuple or other joint specification
              For example ("free", (1, 1, 1)) means adding a free joint with that axis.
          - "disable_gravity": bool
              Whether to ignore gravity for this mass. Default = False.
          - "rgba": tuple of float
              The color/transparency, e.g. (1,0,0,1). If None, picks a random color.
          - (Other fields like "use_bottom_site", "padding_z", "padding_size_x" can be recognized as well.)

        NOTE: We do not strictly need "size" in this dict because the `Mass` class
              can compute or default to a bounding box. If you want a round mass,
              you'd need to customize the `Mass` class or pass a "radius" and handle it inside that class.
    """
    def __init__(
            self,
            name: str,
            pos: Tuple[float, float, float] = (0, 0, 0),
            quat: Tuple[float, float, float, float] = (1, 0, 0, 0),
            slope: float = 0.0,  # Used for self.set_quat_with_angle(), controls the entire Entity's tilt around the y-axis
            plane_size: Tuple[float, float, float] = (
                DEFAULT_PLANE_LENGTH,
                DEFAULT_PLANE_WIDTH * 10,
                DEFAULT_PLANE_THICKNESS,
            ),
            mesh_body: Optional[dict] = None,  # A dict describing the mesh body parameters
            body_angle: float = 0.0,  # The rotation of the mesh body relative to the plane around the y-axis
            mass_spec: Optional[dict] = None,  # Optional Mass parameters
            coefficient_of_friction: float = 0.5,
            **kwargs,
    ):
        # ========== 1. Basic Properties ==========
        self.name = name
        self.pos = pos
        self.slope = slope  # Used for later calling set_quat_with_angle()
        self.entity_type = self.__class__.__name__
        self.kwargs = kwargs  # Store additional information

        # The quaternion for the top-level <body>
        self.quat = quat
        
        # mesh body related
        self.mesh_body = mesh_body
        self.body_angle = body_angle
        
        # mass related (optional)
        self.mass_spec = mass_spec
        self.mass_instance = None  # Will be initialized if mass_spec is not None
        
        # Set quaternion based on slope
        self.set_quat_with_angle(self.slope, axis='y')
        self.plane_size = plane_size
        
        # Call parent constructor
        super().__init__(name, pos, self.quat, entity_type=self.__class__.__name__, **kwargs)

        # ========== 2. Create Plane ==========
        # Note: The plane here is at (0,0,0) in local coordinates with quat=(1,0,0,0), always horizontal
        # "Whether it is tilted" is determined by the outermost RollingPlaneEntity's quat
        self.plane = Plane(
            name=f"{self.name}.plane",
            pos=(0, 0, 0),
            quat=(1, 0, 0, 0),
            size=self.plane_size,
        )

        # ========== 3. Check and Create mesh_body ==========
        if self.mesh_body is None or not isinstance(self.mesh_body, dict):
            raise ValueError("RollingPlaneEntity: A valid dict type mesh_body parameter must be provided.")

        # Copy the dictionary to ensure the original data is not modified
        mesh_body_spec = self.mesh_body.copy()
        body_type = mesh_body_spec.get("body_type")
        if not body_type:
            raise ValueError("RollingPlaneEntity: The mesh_body dict must contain the 'body_type' key.")

        # Define a mapping from body_type to the corresponding mesh body class, ensure related classes are correctly imported
        body_class_mapping = {
            "sphere": Sphere,
            "polygonal_prism": PolygonalPrism,
            "cylinder": Cylinder,
            "disc": Disc,
            "bar": Bar,
            "hemisphere": Hemisphere,
            "bowl": Bowl,
            "sphere_with_hole": SphereWithHole,
        }
        if body_type not in body_class_mapping:
            raise ValueError(f"RollingPlaneEntity: Unsupported mesh body type: {body_type}")

        # If no name is provided in the dict, set a default name
        mesh_body_spec.setdefault("name", f"{self.name}.mesh_body")
        # Pass a placeholder for pos, which will be updated later
        mesh_body_spec.setdefault("pos", (0, 0, 0))
        # Instantiate the mesh body object
        self.mesh_body_instance = body_class_mapping[body_type](**mesh_body_spec)
        if body_type == "cylinder":
            self.mesh_body_instance.set_horizontal()

        # ========== 3.1 Adjust mesh_body local pose based on body_angle ==========
        # Note: mesh_body_instance.add_rotation(...) assumes axis='z'
        mesh_body_quat = self.mesh_body_instance.add_rotation(self.body_angle, axis='y')  # note, the axis is always applied upon the original axis, not the axis after rotation

        # ========== 3.2 Position mesh_body to "just touch" the plane ==========
        z_min = self._approximate_min_z(self.mesh_body_instance, self.body_angle)
        # The top surface of the plane is at z = plane_size[2] (assuming no extra offset for the plane)
        # So, we need to lift the mesh_body to z = plane_size[2] - z_min
        z_offset = self.plane_size[2] - z_min
        mesh_body_pos = (0.0, 0.0, z_offset)

        # Set the mesh_body's local pos & quat
        self.mesh_body_instance.set_pose(pos=mesh_body_pos)
        self.mesh_body_instance.joints = []  # clear the original joint
        self.mesh_body_instance.add_joint(
            Joint(joint_type="free", axis=(1, 1, 1), name=f"{self.name}.joint")
        )
        
        # (Optional) Add a "center site" to mesh_body for later connecting to mass with tendon
        if not any(site.name.endswith("center") for site in self.mesh_body_instance.sites):
            center_site = Site(
                name=f"{self.mesh_body_instance.name}.center",
                pos=(0, 0, 0),  # Center at the local origin of the body
                quat=(1, 0, 0, 0),
                body_name=self.mesh_body_instance.name,
            )
            self.mesh_body_instance.add_site(center_site)

        if self.mass_spec is not None: print("self.mass_spec is not None.")
        # ========== 4. (Optional) Create a mass if mass_spec is not None ==========
        if self.mass_spec is not None:
            # Example of placing the mass block at the same horizontal position as mesh_body and lifted up a bit.
            # Adjust coordinates/size/mass as needed.
            positions = self.mass_spec.get(
                "positions",
                [(-MESH_MASS_OFFSET, 0, z_offset)],  # Default positions: just above mesh_body by 0.1
            )

            # Other parameters can be customized via mass_spec, e.g., mass_value, rgba, etc.
            # Use defaults if not specified
            ms = self.mass_spec.copy()
            ms.setdefault("name", f"{self.name}.mass")
            ms.setdefault("positions", positions)
            ms.setdefault("mass_value", 1.0)
            mass_size = (z_offset - self.plane_size[2])

            self.mass_instance = Mass(
                name=ms["name"],
                positions=ms["positions"],
                joint_option=ms.get("joint_option", ("free", (1, 1, 1))),
                padding_size_x=ms.get("padding_size_x", mass_size),
                size_y=ms.get("size_y", mass_size),
                size_z=ms.get("size_z", mass_size),
            )

        self.coefficient_of_friction = coefficient_of_friction
        if self.coefficient_of_friction > 1e-2:
            self.friction_coefficient_list = [
                (
                    f"{self.name}.plane.geom",
                    mesh_body_spec["name"] + ".geom",
                    self.coefficient_of_friction,
                    FrictionType.ROLLING.value,
                ),
                (
                    f"{self.name}.plane.geom",
                    f"{self.name}.mass.geom",
                    self.coefficient_of_friction,
                    FrictionType.DEFAULT.value,
                )
            ]

    def _approximate_min_z(self, mesh_body, angle: float) -> float:
        """
        A simple approximation for common bodies like "hemisphere, bowl, cylinder".
        For complex meshes, use _get_min_z_after_rotation() to read STL more precisely.
        """
        btype = getattr(mesh_body, "body_type", "")
        # Estimate min_z based on radius
        # Typically, hemisphere/bowl types' bottom surface isn't affected by rotation around the y-axis (due to symmetry)
        if btype == "hemisphere":
            radius = getattr(mesh_body, "radius", 1.0)
            return -radius
        elif btype == "bowl":
            radius = getattr(mesh_body, "radius", 1.0)
            return -radius
        elif btype == "cylinder":
            radius = getattr(mesh_body, "radius", 1.0)
            return -radius
        else:
            # if mesh_body has radius as attribute, use it
            if hasattr(mesh_body, "radius"):
                return -mesh_body.radius
            # If unknown, simply return -0.5
            return -0.5
        
    def get_ready_tendon_sequences(self, direction: ConnectingDirection) -> List[TendonSequence]:
        tendons = []
        if self.mass_instance:
            tendons.append(TendonSequence(
                elements=[
                    self.mesh_body_instance.center_site.create_spatial_site(),
                    self.mass_instance.center_site.create_spatial_site()
                ],
                description=f"A tendon sequence connecting {self.mesh_body_instance.name}.center to {self.mass_instance.name}.center.",
                name=f"{self.name}.tendon_to_mass"
            ))
        return tendons

    def to_xml(self) -> str:
        """
        Generate XML containing plane + mesh_body (+ optional mass).
        Note: To define <tendon>, implement at the top-level <mujoco> element.
        This method only combines the bodies into a single <body> element.
        """
        xml_str = (
            f'<body name="{self.name}" '
            f'pos="{self.pos[0]} {self.pos[1]} {self.pos[2]}" '
            f'quat="{self.quat[0]} {self.quat[1]} {self.quat[2]} {self.quat[3]}">\n'
        )
        # 1) Plane
        plane_xml = self.plane.to_xml()
        plane_xml = "\n".join("  " + ln for ln in plane_xml.split("\n"))
        xml_str += plane_xml + "\n"
        
        # 2) mesh_body
        body_xml = self.mesh_body_instance.to_xml()
        body_xml = "\n".join("  " + ln for ln in body_xml.split("\n"))
        xml_str += body_xml + "\n"
        
        # 3) Optional mass
        if self.mass_instance is not None:
            mass_xml = self.mass_instance.to_xml()
            mass_xml = "\n".join("  " + ln for ln in mass_xml.split("\n"))
            xml_str += mass_xml + "\n"
            
        xml_str += "</body>"
        return xml_str

    def get_parameters(self) -> Dict[str, Union[float, Tuple]]:
        """
        Return core parameters.
        """
        return {}  # TODO(yangmin): check whether this is needed

    def randomize_parameters(
            self,
            degree_of_randomization: DegreeOfRandomization = DegreeOfRandomization.DEFAULT,
            reinitialize_instance: bool = False,
            **kwargs,
    ):
        """
        Randomize parameters of the RollingPlaneEntity based on different difficulty levels (modifies only the parameter dictionary self.mesh_body):

        - EASY:
            * plane's slope is fixed to 0;
            * mesh body's rotation angle (body_angle) can only be chosen from [30, 45, 60];
            * only "hemisphere" type is allowed, and the randomization range for radius and thickness is small.
        - MEDIUM:
            * plane's slope is randomly selected from [30, 45];
            * mesh body's rotation angle is still chosen from [30, 45, 60];
            * "hemisphere" or "bowl" types are allowed, with "bowl" randomizing height and thickness in addition to radius.
        - HARD:
            * plane's slope is continuously randomized between 20 and 60 degrees;
            * mesh body's rotation angle is randomly chosen between 20 and 70 degrees;
            * all types are allowed (including "sphere", "polygonal_prism", "cylinder", "disc", "bar",
              "hemisphere", "bowl", "sphere_with_hole"), with a larger range of randomization for each parameter.
        - NON_STRUCTURAL:
            * Only slight adjustments are made to current parameters without changing mesh_body type (i.e., structure remains unchanged).
            
        To randomize mass parameters, similar logic can be added here.
        """
        import random

        # If DEFAULT is passed, randomly select from EASY/MEDIUM/HARD
        if degree_of_randomization == DegreeOfRandomization.DEFAULT:
            degree_of_randomization = random.choice([
                DegreeOfRandomization.EASY,
                DegreeOfRandomization.MEDIUM,
                DegreeOfRandomization.HARD,
            ])

        # Define parameter configurations for different difficulty levels
        randomization_levels = {
            DegreeOfRandomization.EASY: {
                "plane_slope": 0,  # Fixed to 0 degrees
                "body_angle_choices": [30, 45, 60],
                "allowed_body_types": ["hemisphere", "bowl", "cylinder"],
                "hemisphere_radius": (0.1, 0.2),
                "hemisphere_thickness": (-0.05, 0.0),
                "bowl_radius": (0.1, 0.2),
                "bowl_height": (0.05, 0.1),
                "bowl_thickness": (0.05, 0.2),
                "cylinder_radius": (0.1, 0.2),
                "cylinder_height": (0.2, 0.4),
                "round_decimals": 2,
                "use_mass": False,
                "coefficient_of_friction": (0.2, 0.7),
            },
            DegreeOfRandomization.MEDIUM: {
                "plane_slope_choices": [30, 45],
                "body_angle_choices": [30, 45, 60],
                "allowed_body_types": ["hemisphere", "bowl", "cylinder"],
                "hemisphere_radius": (0.1, 0.3),
                "hemisphere_thickness": (-0.1, 0.0),
                "bowl_radius": (0.1, 0.3),
                "bowl_height": (0.05, 0.15),
                "bowl_thickness": (0.05, 0.25),
                "cylinder_radius": (0.1, 0.3),
                "cylinder_height": (0.3, 0.6),
                "round_decimals": 2,
                "use_mass": False,
                "coefficient_of_friction": (0.2, 0.7),
            },
            DegreeOfRandomization.HARD: {
                "plane_slope_range": (20, 60),
                "body_angle_range": (20, 70),
                "allowed_body_types": [
                    "sphere", "cylinder", "hemisphere", "bowl", "sphere_with_hole"
                ],
                "hemisphere_radius": (0.1, 0.3),
                "hemisphere_thickness": (-0.2, 0.0),
                "bowl_radius": (0.1, 0.3),
                "bowl_height": (0.05, 0.2),
                "bowl_thickness": (0.05, 0.3),
                "cylinder_radius": (0.1, 0.3),
                "cylinder_height": (0.3, 0.6),
                "sphere_with_hole_radius": (0.2, 0.3),
                "sphere_with_hole_hole_radius": (0.05, 0.15),
                "sphere_with_hole_hole_position": (-0.2, 0.2),
                "sphere_with_hole_thickness": (0.0, 0.2),
                # For other types, use default parameters
                "default_radius": (0.1, 0.3),
                "default_thickness": (0.05, 0.3),
                "round_decimals": 1,
                "use_mass": True,
                "coefficient_of_friction": (0.2, 0.7),
            },
        }
        self.randomization_levels = randomization_levels  # Save configuration (for debugging if needed)

        # Structural randomization: Allows changing mesh_body type and parameters (modifies only self.mesh_body dictionary)
        if degree_of_randomization in [
            DegreeOfRandomization.EASY,
            DegreeOfRandomization.MEDIUM,
            DegreeOfRandomization.HARD,
        ]:
            params = randomization_levels[degree_of_randomization]
            decimals = params["round_decimals"]

            # Randomly set the slope of the plane
            if degree_of_randomization == DegreeOfRandomization.EASY:
                new_slope = params["plane_slope"]
            elif degree_of_randomization == DegreeOfRandomization.MEDIUM:
                new_slope = random.choice(params["plane_slope_choices"])
            else:  # HARD
                new_slope = round(random.uniform(*params["plane_slope_range"]), decimals)
            self.slope = new_slope
            self.set_quat_with_angle(self.slope)

            # Randomly set the mesh body's rotation angle (body_angle)
            if "body_angle_choices" in params and degree_of_randomization in [DegreeOfRandomization.EASY, DegreeOfRandomization.MEDIUM]:
                new_body_angle = random.choice(params["body_angle_choices"])
                self.body_angle = new_body_angle
            elif "body_angle_range" in params:
                new_body_angle = round(random.uniform(*params["body_angle_range"]), decimals)
                self.body_angle = new_body_angle
            else:
                print(f"No body_angle_choices or body_angle_range provided for {self.name}")
                self.body_angle = 0

            # Randomly choose mesh_body type
            allowed_types = params["allowed_body_types"]
            new_body_type = random.choice(allowed_types)

            # Construct a new mesh_body parameter dictionary
            new_mesh_body_spec = {
                "body_type": new_body_type,
                "name": f"{self.name}.mesh_body",
                "pos": (0, 0, 0)  # Placeholder, will be updated by reinitialize or other logic later
            }

            # Randomize specific parameters based on type
            if new_body_type == "hemisphere":
                new_mesh_body_spec["radius"] = round(
                    random.uniform(*params["hemisphere_radius"]), decimals
                )
                new_mesh_body_spec["thickness"] = round(
                    random.uniform(*params["hemisphere_thickness"]), decimals
                )
            elif new_body_type == "bowl":
                new_mesh_body_spec["radius"] = round(
                    random.uniform(*params["bowl_radius"]), decimals
                )
                new_mesh_body_spec["height"] = round(
                    random.uniform(*params["bowl_height"]), decimals
                )
                new_mesh_body_spec["thickness"] = round(
                    random.uniform(*params["bowl_thickness"]), decimals
                )
            elif new_body_type == "sphere_with_hole":
                new_mesh_body_spec["radius"] = round(
                    random.uniform(*params["sphere_with_hole_radius"]), decimals
                )
                new_mesh_body_spec["hole_radius"] = round(
                    random.uniform(*params["sphere_with_hole_hole_radius"]), decimals
                )
                new_mesh_body_spec["hole_position"] = round(
                    random.uniform(*params["sphere_with_hole_hole_position"]), decimals
                )
                new_mesh_body_spec["thickness"] = round(
                    random.uniform(*params["sphere_with_hole_thickness"]), decimals
                )
            elif new_body_type == "cylinder":
                new_mesh_body_spec["radius"] = round(
                    random.uniform(*params["cylinder_radius"]), decimals
                )
                new_mesh_body_spec["height"] = round(
                    random.uniform(*params["cylinder_height"]), decimals
                )
            else:
                # Use default random parameters for other types
                new_mesh_body_spec["radius"] = round(
                    random.uniform(*params.get("default_radius", (0.1, 0.5))), decimals
                )
                if "default_thickness" in params:
                    new_mesh_body_spec["thickness"] = round(
                        random.uniform(*params["default_thickness"]), decimals
                    )

            # Update the parameter dictionary (self.mesh_body is the external record of the parameters)
            self.mesh_body = new_mesh_body_spec

            if params["use_mass"]:
                self.mass_spec = {
                    "mass_value": round(random.uniform(0.1, 10.0), 2),
                }
            
            self.coefficient_of_friction = round(random.uniform(*params["coefficient_of_friction"]), decimals)

        # Non-structural randomization: Only slight adjustments are made to current parameters (structure remains unchanged)
        elif degree_of_randomization == DegreeOfRandomization.NON_STRUCTURAL:
            self.slope = round(self.slope * random.uniform(0.95, 1.05), 2)
            self.body_angle = round(min(90, max(0, self.body_angle + random.uniform(-5, 5))), 2)
            # Slightly adjust parameters in the self.mesh_body dictionary (without changing body_type)
            if isinstance(self.mesh_body, dict):
                if "radius" in self.mesh_body:
                    self.mesh_body["radius"] = round(
                        self.mesh_body["radius"] * random.uniform(0.95, 1.05), 2
                    )
                if "thickness" in self.mesh_body:
                    self.mesh_body["thickness"] = round(
                        self.mesh_body["thickness"] * random.uniform(0.95, 1.05), 2
                    )
                if self.mesh_body.get("body_type") == "bowl" and "height" in self.mesh_body:
                    self.mesh_body["height"] = round(
                        self.mesh_body["height"] * random.uniform(0.95, 1.05), 2
                    )
                if self.mesh_body.get("body_type") == "sphere_with_hole":
                    if "hole_radius" in self.mesh_body:
                        self.mesh_body["hole_radius"] = round(
                            self.mesh_body["hole_radius"] * random.uniform(0.95, 1.05), 2
                        )
                    if "hole_position" in self.mesh_body:
                        self.mesh_body["hole_position"] = round(
                            self.mesh_body["hole_position"] + random.uniform(-0.1, 0.1), 2
                        )
            self.set_quat_with_angle(self.slope)
            # Note: We do not modify mesh_body_instance here, as we are only concerned with changes to the parameters (self.mesh_body)

        if reinitialize_instance:
            self.reinitialize()

    def generate_entity_yaml(
            self,
            use_random_parameters: bool = False,
            degree_of_randomization: DegreeOfRandomization = DegreeOfRandomization.NON_STRUCTURAL,
    ) -> dict:
        if use_random_parameters:
            self.randomize_parameters(degree_of_randomization, reinitialize_instance=True)

        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "position": list(self.pos),
            "parameters": {
                "slope": self.slope,
                # "plane_size": list(self.plane_size),
                "mesh_body": self.mesh_body,
                "body_angle": self.body_angle,
                "mass_spec": self.mass_spec,
                "coefficient_of_friction": self.coefficient_of_friction,
            },
        }

    def get_nlq(self, symbolic=False):
        """
        Generate a natural language query (NLQ) description of the entity.
        """
        body_type_mapping = {
            "sphere": "sphere",
            "polygonal_prism": "regular polygonal cylinder",
            "cylinder": "cylinder",
            "disc": "disc",
            "bar": "bar",
            "hemisphere": "hemisphere",
            "bowl": "bowl (spherical cap cut from a sphere at arbitrary height (h <= 2r))",
            "sphere_with_hole": "sphere with a spherical hole cutout from it",
        }

        sym_dict = {}
        sym_dict["<angle>1"] = self.slope
        sym_dict["<angle>2"] = self.body_angle

        slope = "<angle>1"
        angle = "<angle>2"
        mesh_body = self.mesh_body
        body_type = body_type_mapping[mesh_body["body_type"]]

        # —— 1. Basic description of the plane + mesh body —— 
        description = (
            f"A smooth plane is tilted by {slope}° around its local y-axis relative to the horizontal. "
            f"A {body_type} sits on it and is then rotated by {angle}° about that same y-axis so its bottom just touches the plane."
        )

        # —— 2. Size description for various geometric bodies —— 
        body_description = ""
        if mesh_body["body_type"] in ["hemisphere", "sphere", "bowl", "sphere_with_hole"]:
            sym_dict["<radius>1"] = mesh_body["radius"]
            radius = "<radius>1"
            extra = ""
            if mesh_body["body_type"] == "bowl":
                sym_dict["<height>1"] = mesh_body["height"]
                height = "<height>1"
                extra = f" The cutout plane is at a height of {height} m from the center."
            elif mesh_body["body_type"] == "sphere_with_hole":
                sym_dict["<radius>2"] = mesh_body["hole_radius"]
                sym_dict["<x>1"] = mesh_body["hole_position"]
                hole_radius = "<radius>2"
                hole_position = "(<x>1, 0, 0)"
                extra = (
                    f" It has a spherical hole of radius {hole_radius} m, "
                    f"cut out at position {hole_position} m from the center."
                )
            body_description = (
                f" It has a radius of {radius} m.{extra} It is placed so its curved surface just touches the plane."
            )

        elif mesh_body["body_type"] == "cylinder":
            sym_dict["<radius>1"] = mesh_body["radius"]
            sym_dict["<height>1"] = mesh_body["height"]
            radius, height = "<radius>1", "<height>1"
            body_description = (
                f" It has a height of {height} m and a radius of {radius} m, placed so its curved surface just touches the plane."
            )

        elif mesh_body["body_type"] == "polygonal_prism":
            sym_dict["<radius>1"] = mesh_body["radius"]
            sym_dict["<height>1"] = mesh_body["height"]
            radius, height = "<radius>1", "<height>1"
            n_sides = mesh_body["sides"]
            body_description = (
                f" It has height {height} m, {n_sides} sides, and a circumscribing radius {radius} m; its slanted side just touches the plane."
            )

        elif mesh_body["body_type"] == "bar":
            sym_dict.update({
                "<length>1": mesh_body["length"],
                "<width>1": mesh_body["width"],
                "<height>1": mesh_body["height"],
            })
            length, width, height = "<length>1", "<width>1", "<height>1"
            body_description = (
                f" It measures {length}×{width}×{height} m and just touches the plane."
            )

        elif mesh_body["body_type"] == "disc":
            sym_dict["<radius>1"] = mesh_body["radius"]
            radius = "<radius>1"
            body_description = (
                f" It has a radius of {radius} m and just touches the plane."
            )

        # Append the geometric description
        description += body_description

        # —— 3. Optional mass + tendon description —— 
        if self.mass_spec is not None:
            sym_dict["<length>1"] = self.mass_spec.get("offset", MESH_MASS_OFFSET)
            sym_dict["<mass>1"] = self.mass_spec.get("mass_value", 1.0)
            distance = "<length>1"
            mass_val = "<mass>1"

            description += (
                f" A mass of {mass_val} kg is placed {distance} m uphill from the mesh body along the inclined plane, "
                f"directly behind it in the downhill direction, at the same elevation. "
                f"The tendon connecting them runs parallel to the plane."
            )

        # —— 4. Return —— 
        if not symbolic:
            return replace_all(description, sym_dict)
        else:
            return description, sym_dict


    def get_question(self, sub_entity: str, quantity: str) -> str:
        """
        Generate a question based on the entity's parameters.
        """

        body_type_mapping = {
            "sphere": "sphere",
            "polygonal_prism": "polygonal cylinder",
            "cylinder": "cylinder",
            "disc": "disc",
            "bar": "bar",
            "hemisphere": "hemisphere",
            "bowl": "bowl",
            "sphere_with_hole": "sphere with the hole",
        }

        mesh_body = self.mesh_body
        body_type = body_type_mapping[mesh_body["body_type"]]

        return f"What is the {quantity} of the {body_type} in the system?"

class PendulumEntity(Pendulum, Entity):
    def __init__(
        self,
        name: str,
        pos: Tuple[float, float, float],
        rope_length: float = 1.0,
        mass_value: float = 1.0,
        angle: float = 0.0,
        init_velocity_dict: Optional[Dict[str, List[Union[List, float]]]] = None,
        **kwargs,
    ):
        self.pos = pos
        self.rope_length = rope_length
        self.mass_value = mass_value
        self.angle = angle
        # init_velocity = None if init_velocity is None else init_velocity # {InitVelocityType.from_value(init_velocity["type"]): init_velocity["velocity"]}
        self.init_velocity_dict = init_velocity_dict
        super().__init__(
            name=name,
            pos=pos,
            rope_length=rope_length,
            mass_value=mass_value,
            angle=angle,
            init_velocity_dict=init_velocity_dict,
            entity_type=self.__class__.__name__,
            **kwargs,
        )       

    def randomize_parameters(
        self,
        degree_of_randomization: DegreeOfRandomization = DegreeOfRandomization.DEFAULT,
        reinitialize_instance: bool = False,
        **kwargs,
    ):
        # TODO: check whether there are differences between the two randomize_parameters methods in this entity.
        if degree_of_randomization == DegreeOfRandomization.DEFAULT:
            self.rope_length = round(random.uniform(0.5, 2.0), 2)
            self.mass_value = round(random.uniform(0.1, 10.0), 2)
            self.angle = round(random.uniform(0, 90), 2)
            # self.init_velocity = {
            #     "type": InitVelocityType.SPHERE,
            #     "velocity": [round(random.uniform(-1, 1), 2), round(random.uniform(-1, 1), 2), 0],
            # }  # TODO(yangmin): use angle & value instead of global velocity
            self.init_velocity_dict = {
                InitVelocityType.SPHERE: [round(random.uniform(-1, 1), 2), round(random.uniform(-1, 1), 2), 0]
            }
        else:
            self.rope_length = round(random.uniform(0.5, 2.0), 2)
            self.mass_value = round(random.uniform(0.1, 10.0), 2)
            self.angle = round(random.uniform(0, 90), 2)

            if self.init_velocity_dict in [None, {}]:
                # self.init_velocity = {
                #     "type": InitVelocityType.SPHERE,
                #     "velocity": [round(random.uniform(-1, 1), 2), round(random.uniform(-1, 1), 2), 0],
                # }
                self.init_velocity_dict = {
                    InitVelocityType.SPHERE: [round(random.uniform(-1, 1), 2), round(random.uniform(-1, 1), 2), 0]
                }
        
        if reinitialize_instance:
            self.reinitialize()

    def generate_entity_yaml(
        self,
        use_random_parameters: bool = False,
        degree_of_randomization: DegreeOfRandomization = DegreeOfRandomization.NON_STRUCTURAL,
    ) -> dict:
        if use_random_parameters:
            self.randomize_parameters(degree_of_randomization, reinitialize_instance=True) # (degree_of_randomization==DegreeOfRandomization.NON_STRUCTURAL))

        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "position": list(self.pos),
            "parameters": {
                "rope_length": self.rope_length,
                "mass_value": self.mass_value,
                "angle": self.angle,
                "init_velocity_dict": self.init_velocity_dict,
            },
        }

    def get_nlq(self, symbolic = False):
        """
        Generate a natural language question (NLQ) description of the entity.
        """
        sym_dict = {}
        
        sym_dict["<angle>1"] = self.angle
        angle = "<angle>1"

        sym_dict["<length>1"] = self.rope_length
        length = "<length>1"

        sym_dict["<mass>1"] = self.mass_value
        mass = "<mass>1"

        sym_dict["<vx>1"], sym_dict["<vy>1"] = tuple(self.init_velocity_dict[InitVelocityType.SPHERE][:-1])
        velocity = "(<vx>1, <vy>1)"

        description = (
            f"A pendulum with a rope length of {length} m is attached to a fixed point in the ceiling on one end "
            f"and a bob of mass {mass} kg on the other end. The bob has an initial velocity of "
            f"{velocity} m/s along the global x and y directions (i.e., in the horizontal plane with z as vertical), "
            f"and the rope initially makes an angle of {angle} degrees with the vertical."
        )

        if not symbolic:
            description = replace_all(description, sym_dict)
        else: return description, sym_dict

        return description
    
    def get_question(self, sub_entity: str, quantity: str) -> str:
        """
        Generate a question based on the entity's parameters.
        """
        return f"What is the {quantity} of the bob in the system?"

class DiskRackWithSphereEntity(DiskRackWithSphere, Entity):

    # Parameter configuration template (analysis in comments below)
    randomization_levels = {
        DegreeOfRandomization.EASY: {
            "sphere_radius": (0.1, 0.3),  # Small ball size range
            "disk_radius_base": 0.5,  # Fixed base radius
            "disk_radius_offset": (0.1, 0.3),  # Radius fluctuation range
            "disk_height": (0.2, 0.5),  # Thicker disks are more stable
            "disk_angle": (0, 30),  # Small tilt angles
            "bar_gap_multiplier": (2.5, 3.0),  # Larger bar gaps
            "bar_thickness": (0.15, 0.3),  # Thicker bars
            "x_offset": (-0.5, 0.5),  # Small offset range
            "sphere_mass": (0.5, 1.5),  # Medium mass range
            "sphere_offset": (-0.5, 0.5),  # Small ball position offset
            "round_decimals": 2,  # Value precision
        },
        DegreeOfRandomization.MEDIUM: {
            "sphere_radius": (0.2, 0.4),
            "disk_radius_base": 0.4,
            "disk_radius_offset": (0.2, 0.6),
            "disk_height": (0.1, 0.8),
            "disk_angle": (0, 60),
            "bar_gap_multiplier": (2.0, 3.0),
            "bar_thickness": (0.1, 0.4),
            "x_offset": (-1.0, 1.0),
            "sphere_mass": (0.2, 2.0),
            "sphere_offset": (-1.0, 1.0),
            "round_decimals": 2,
        },
        DegreeOfRandomization.HARD: {
            "sphere_radius": (0.1, 0.5),
            "disk_radius_base": 0.3,
            "disk_radius_offset": (0.3, 1.7),
            "disk_height": (0.05, 1.5),
            "disk_angle": (0, 90),
            "bar_gap_multiplier": (2.0, 4.0),
            "bar_thickness": (0.05, 0.5),
            "x_offset": (-2.0, 2.0),
            "sphere_mass": (0.1, 5.0),
            "sphere_offset": (-2.0, 2.0),
            "round_decimals": 1,  # Lower precision increases randomness
        },
    }

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
        initial_angular_velocity: float = 0.0,
        **kwargs,
    ):
        self.radius = radius
        self.height = height
        self.angle = angle
        self.bar_gap = bar_gap
        self.bar_thickness = bar_thickness
        self.x_offset = x_offset
        self.sphere_radius = sphere_radius
        self.sphere_mass = sphere_mass
        self.sphere_offset_along_track = sphere_offset_along_track
        self.initial_angular_velocity = initial_angular_velocity
        super().__init__(
            name=name,
            pos=pos,
            radius=radius,
            height=height,
            angle=angle,
            bar_gap=bar_gap,
            bar_thickness=bar_thickness,
            x_offset=x_offset,
            sphere_radius=sphere_radius,
            sphere_mass=sphere_mass,
            sphere_offset_along_track=sphere_offset_along_track,
            entity_type=self.__class__.__name__,
            **kwargs,
        )
        # DiskRackWithSphereEntity (Entity)
        self.constant_velocity_actuator = Actuator(
            name=f"{self.name}.constant_velocity_actuator",
            actuator_type="velocity",
            kv=1000,
            velocity=self.initial_angular_velocity,
        )
        self.constant_velocity_actuator.joint = self.rotation_joint.name
        self.actuator = self.constant_velocity_actuator

    def randomize_parameters(
        self,
        degree_of_randomization: DegreeOfRandomization = DegreeOfRandomization.DEFAULT,
        reinitialize_instance: bool = False,
        **kwargs,
    ):
        """
        Randomize parameters based on difficulty levels
        - EASY: Simple structure configuration, small parameter range, low complexity
        - MEDIUM: Medium random range, allows some extreme values for parameters
        - HARD: Maximum random range, strong coupling between parameters
        - NON_STRUCTURAL: Keep structure, only fine-tune values
        """
        # Parameter configuration template (analysis in comments below)
        randomization_levels = {
            DegreeOfRandomization.EASY: {
                "sphere_radius": (0.1, 0.3),  # Small ball size range
                "disk_radius_base": 0.5,  # Fixed base radius
                "disk_radius_offset": (0.1, 0.3),  # Radius fluctuation range
                "disk_height": (0.2, 0.5),  # Thicker disks are more stable
                "disk_angle": (0, 30),  # Small tilt angles
                "bar_gap_multiplier": (2.5, 3.0),  # Larger bar gaps
                "bar_thickness": (0.15, 0.3),  # Thicker bars
                "x_offset": (-0.5, 0.5),  # Small offset range
                "sphere_mass": (0.5, 1.5),  # Medium mass range
                "sphere_offset": (-0.5, 0.5),  # Small ball position offset
                "round_decimals": 2,  # Value precision
                "angular_velocity": (0, 4.0),
            },
            DegreeOfRandomization.MEDIUM: {
                "sphere_radius": (0.2, 0.4),
                "disk_radius_base": 0.4,
                "disk_radius_offset": (0.2, 0.6),
                "disk_height": (0.1, 0.8),
                "disk_angle": (0, 60),
                "bar_gap_multiplier": (2.0, 3.0),
                "bar_thickness": (0.1, 0.4),
                "x_offset": (-1.0, 1.0),
                "sphere_mass": (0.2, 2.0),
                "sphere_offset": (-1.0, 1.0),
                "round_decimals": 2,
                "angular_velocity": (3.0, 6.0),
            },
            DegreeOfRandomization.HARD: {
                "sphere_radius": (0.1, 0.5),
                "disk_radius_base": 0.3,
                "disk_radius_offset": (0.3, 1.7),
                "disk_height": (0.05, 1.5),
                "disk_angle": (0, 90),
                "bar_gap_multiplier": (2.0, 4.0),
                "bar_thickness": (0.05, 0.5),
                "x_offset": (-2.0, 2.0),
                "sphere_mass": (0.1, 5.0),
                "sphere_offset": (-2.0, 2.0),
                "round_decimals": 1,  # Lower precision increases randomness
                "angular_velocity": (4.0, 8.0),
            },
        }

        self.randomization_levels = randomization_levels

        # Handle DEFAULT case
        if degree_of_randomization == DegreeOfRandomization.DEFAULT:
            degree_of_randomization = random.choice(
                [
                    DegreeOfRandomization.EASY,
                    DegreeOfRandomization.MEDIUM,
                    DegreeOfRandomization.HARD,
                ]
            )

        if degree_of_randomization in randomization_levels:
            params = randomization_levels[degree_of_randomization]
            decimals = params["round_decimals"]

            # Sphere parameters (core parameters affecting other parameter ranges)
            sphere_r = random.uniform(*params["sphere_radius"])
            sphere_r = round(sphere_r, decimals)

            # Disk radius (dynamically calculated to ensure safety)
            min_disk_r = max(params["disk_radius_base"], sphere_r + 0.05)
            disk_r = min_disk_r + random.uniform(*params["disk_radius_offset"])
            disk_r = round(min(disk_r, 2.0), decimals)  # Keep upper limit 2.0

            # New: Angular velocity randomization
            angular_velocity = random.uniform(*params["angular_velocity"])
            angular_velocity = round(angular_velocity, decimals)

            # Other parameter generation
            disk_h = round(random.uniform(*params["disk_height"]), decimals)
            disk_angle = round(random.uniform(*params["disk_angle"]), decimals)
            bar_gap = round(
                sphere_r * random.uniform(*params["bar_gap_multiplier"]), decimals
            )
            bar_thickness = round(random.uniform(*params["bar_thickness"]), decimals)
            x_offset = round(random.uniform(*params["x_offset"]), decimals)
            sphere_m = round(random.uniform(*params["sphere_mass"]), decimals)
            sphere_offset = round(random.uniform(*params["sphere_offset"]), decimals)

            # Update instance parameters
            self.radius = disk_r
            self.height = disk_h
            self.angle = disk_angle
            self.bar_gap = bar_gap
            self.bar_thickness = bar_thickness
            self.x_offset = x_offset
            self.sphere_radius = sphere_r
            self.sphere_mass = sphere_m
            self.sphere_offset_along_track = sphere_offset
            self.initial_angular_velocity = angular_velocity

        elif degree_of_randomization == DegreeOfRandomization.NON_STRUCTURAL:
            # Non-structural fine-tuning (retain 10% fluctuation)
            self.radius = round(self.radius * random.uniform(0.95, 1.05), 2)
            self.height = round(self.height * random.uniform(0.9, 1.1), 2)
            self.angle = round(min(90, max(0, self.angle + random.uniform(-5, 5))), 2)
            self.bar_gap = round(self.bar_gap * random.uniform(0.9, 1.1), 2)
            self.bar_thickness = round(self.bar_thickness * random.uniform(0.9, 1.1), 2)
            self.x_offset = round(self.x_offset + random.uniform(-0.1, 0.1), 2)
            self.sphere_radius = round(self.sphere_radius * random.uniform(0.95, 1.05), 2)
            self.sphere_mass = round(self.sphere_mass * random.uniform(0.9, 1.1), 2)
            self.sphere_offset_along_track = round(self.sphere_offset_along_track + random.uniform(-0.2, 0.2), 2)

        if reinitialize_instance:
            self.reinitialize()

    def generate_entity_yaml(
        self,
        use_random_parameters: bool = False,
        degree_of_randomization: DegreeOfRandomization = DegreeOfRandomization.NON_STRUCTURAL,
    ) -> dict:
        if use_random_parameters:
            self.randomize_parameters(degree_of_randomization, reinitialize_instance=True)

        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "position": list(self.pos),
            "parameters": {
                "radius": self.radius,
                "height": self.height,
                "angle": self.angle,
                "bar_gap": self.bar_gap,
                "bar_thickness": self.bar_thickness,
                "x_offset": self.x_offset,
                "sphere_radius": self.sphere_radius,
                "sphere_mass": self.sphere_mass,
                "sphere_offset_along_track": self.sphere_offset_along_track,
            },
        }
