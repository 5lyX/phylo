from sim.bodies import ConnectingDirection, ConnectingPoint, ConnectingPointSeqId, TendonSequence
from .mass_entities import *
import ipdb
from sim.utils import replace_all
import math

st = ipdb.set_trace


class TwoSideMassPlane(MassPlane, Entity):

    randomization_levels = {
        DegreeOfRandomization.EASY: {
            "num_masses": {"min": 1, "max": 1},
            "mass_value": {"min": 0.1, "max": 5.0},
            "plane_slope": {"min": 0, "max": 0},
            "use_left_site_options": [
                DirectionsEnum.USE_LEFT,
                DirectionsEnum.USE_RIGHT,
            ],
            "min_distance": 1.0,
            "position_range": (-2, 2),  # Restrict position range to ensure simplicity
            "coefficient_of_friction": {"min": 0, "max": 0},
        },
        DegreeOfRandomization.MEDIUM: {
            "num_masses": {"min": 1, "max": 3},
            "mass_value": {"min": 0.5, "max": 10.0},
            "plane_slope": {"min": 10, "max": 30},
            "use_left_site_options": [
                DirectionsEnum.USE_LEFT,
                DirectionsEnum.USE_RIGHT,
            ],
            "min_distance": 0.75,
            "position_range": (-4, 4),
            "coefficient_of_friction": {"min": 0, "max": 0},
        },
        DegreeOfRandomization.HARD: {
            "num_masses": {"min": 2, "max": 5},
            "mass_value": {"min": 0.1, "max": 20.0},
            "plane_slope": {"min": 20, "max": 70},
            "use_left_site_options": [
                DirectionsEnum.USE_LEFT,
                DirectionsEnum.USE_RIGHT,
            ],
            "min_distance": 0.5,
            "position_range": (-5, 5),
            "coefficient_of_friction": {"min": 0.2, "max": 0.7},
        },
    }

    def __init__(
        self,
        name: str,
        pos: Tuple[float, float, float] = (0, 0, 0),
        mass_values: List[float] = [1.0],
        mass_positions: List[Tuple[float, float, float]] = [(0, 0, 0)],
        plane_slope: float = 0,
        padding_z: float = 0,
        use_left_site: DirectionsEnum = DirectionsEnum.USE_LEFT,
        condim: str = "1",
        coefficient_of_friction: float = 0.0,
        constant_force: Optional[Dict[str, List[Union[List, float]]]] = None,
        **kwargs,
    ) -> None:
        self.plane_slope = plane_slope
        self.mass_values = mass_values
        self.padding_z = padding_z
        self.use_left_site = use_left_site
        self.condim = condim
        self.coefficient_of_friction = coefficient_of_friction
        self.mass_positions = mass_positions
        
        # 创建旋转四元数来旋转整个entity（参考ComplexCollisionPlane的旋转方式）
        theta = math.radians(-plane_slope)
        qx = 0.0
        qy = math.sin(theta / 2)
        qz = 0.0
        qw = math.cos(theta / 2)
        entity_quat = (qw, qx, qy, qz)
        
        super().__init__(
            name=name,
            pos=pos,
            mass_values=mass_values,
            positions=self.mass_positions,
            plane_slope=plane_slope,
            padding_z=padding_z,
            use_left_site=use_left_site,
            condim=condim,
            constant_force=constant_force,
            entity_type=self.__class__.__name__,
            quat=entity_quat,  # 设置entity的全局旋转
            **kwargs,
        )
        
        if self.coefficient_of_friction > 1e-2:
            self.friction_coefficient_list = [
                (
                    f"{name}.plane.geom",
                    f"{name}.mass{idx}.geom",
                    self.coefficient_of_friction,
                ) for idx, m in enumerate(mass_values)
            ]

    def get_connecting_tendon_sequence(
        self,
        direction: ConnectingDirection,
        connecting_point: ConnectingPoint = ConnectingPoint.DEFAULT,
        connecting_point_seq_id: Optional[ConnectingPointSeqId] = None,
        use_sidesite: bool = False,
    ) -> TendonSequence:
        if connecting_point == ConnectingPoint.DEFAULT or connecting_point == ConnectingPoint.LEFT:
            sequences = self.get_connecting_tendon_sequences(direction)
            if sequences:
                return sequences[0]
            else:
                return []
        else:
            sequences = self.get_second_connecting_tendon_sequences(direction)
            if sequences:
                return sequences[0]
            else:
                return []

    def randomize_parameters(
        self,
        degree_of_randomization: DegreeOfRandomization = DegreeOfRandomization.DEFAULT,
        reinitialize_instance: bool = False,
        **kwargs,
    ):
        """
        - EASY: Single mass, zero slope
        - MEDIUM: 1-3 masses, slope between 10-30 degrees
        - HARD: 2-5 masses, slope between 20-70 degrees
        - NON_STRUCTURAL: Slight adjustments to existing parameters
        """
        import math
        import random

        randomization_levels = {
            DegreeOfRandomization.EASY: {
                "num_masses": {"min": 1, "max": 1},
                "mass_value": {"min": 0.1, "max": 5.0},
                "plane_slope": {"min": 0, "max": 0},
                "use_left_site_options": [
                    DirectionsEnum.USE_LEFT,
                    DirectionsEnum.USE_RIGHT,
                ],
                "min_distance": 1.0,
                "position_range": (-2, 2),  # Restrict position range to ensure simplicity
                "coefficient_of_friction": {"min": 0, "max": 0},
            },
            DegreeOfRandomization.MEDIUM: {
                "num_masses": {"min": 1, "max": 3},
                "mass_value": {"min": 0.5, "max": 10.0},
                "plane_slope": {"min": 10, "max": 30},
                "use_left_site_options": [
                    DirectionsEnum.USE_LEFT,
                    DirectionsEnum.USE_RIGHT,
                ],
                "min_distance": 0.75,
                "position_range": (-4, 4),
                "coefficient_of_friction": {"min": 0, "max": 0},
            },
            DegreeOfRandomization.HARD: {
                "num_masses": {"min": 2, "max": 5},
                "mass_value": {"min": 0.1, "max": 20.0},
                "plane_slope": {"min": 20, "max": 70},
                "use_left_site_options": [
                    DirectionsEnum.USE_LEFT,
                    DirectionsEnum.USE_RIGHT,
                ],
                "min_distance": 0.5,
                "position_range": (-5, 5),
                "coefficient_of_friction": {"min": 0.2, "max": 0.7},
            },
        }

        self.randomization_levels = randomization_levels

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

            # Generate the number of masses
            num_masses = random.randint(
                params["num_masses"]["min"], params["num_masses"]["max"]
            )

            # Generate mass values with two decimal places
            self.mass_values = [
                round(
                    random.uniform(
                        params["mass_value"]["min"], params["mass_value"]["max"]
                    ),
                    2,
                )
                for _ in range(num_masses)
            ]

            # Generate plane slope
            self.plane_slope = round(
                random.uniform(
                    params["plane_slope"]["min"], params["plane_slope"]["max"]
                ),
                2,
            )

            # Randomly select connection direction
            self.use_left_site = random.choice(params["use_left_site_options"])

            # Generate mass positions
            self.mass_positions = []
            pos_min, pos_max = params["position_range"]
            attempts_limit = 100

            for _ in range(num_masses):
                for attempt in range(attempts_limit):
                    x = round(random.uniform(pos_min, pos_max), 2)
                    y = round(random.uniform(pos_min, pos_max), 2)
                    new_pos = (x, y, 0)  # Z-axis is handled by padding_z

                    # Check minimum distance constraint
                    if all(
                        math.hypot(existing[0] - x, existing[1] - y)
                        >= params["min_distance"]
                        for existing in self.mass_positions
                    ):
                        self.mass_positions.append(new_pos)
                        break
                else:
                    raise RuntimeError(
                        f"Failed to generate valid positions within {attempts_limit} attempts"
                    )
                
            self.coefficient_of_friction = round(
                random.uniform(
                    params["coefficient_of_friction"]["min"],
                    params["coefficient_of_friction"]["max"],
                ),
                2,
            )

        elif degree_of_randomization == DegreeOfRandomization.NON_STRUCTURAL:
            # Non-structural adjustment: Maintain structure, slightly tweak values
            self.mass_values = [
                max(0.1, round(mv * random.uniform(0.8, 1.2), 2))  # Increase variation range
                for mv in self.mass_values
            ]
            # Adjust slope while keeping it within valid limits
            self.plane_slope = round(
                max(0, min(90, self.plane_slope + random.uniform(-10, 10))), 2
            )

            # Keep existing positions but add slight random perturbations
            self.mass_positions = [
                (
                    round(pos[0] + random.uniform(-0.5, 0.5), 2),
                    round(pos[1] + random.uniform(-0.5, 0.5), 2),
                    pos[2],
                )
                for pos in self.mass_positions
            ]

        else:
            raise ValueError(f"Unsupported randomization level: {degree_of_randomization}")

        # Reinitialize instance if required
        if reinitialize_instance:
            self.reinitialize()

    def generate_entity_yaml(
        self,
        use_random_parameters: bool = False,
        degree_of_randomization: DegreeOfRandomization = DegreeOfRandomization.NON_STRUCTURAL,
    ) -> dict:
        entity_dict = {
            "name": self.name,
            "type": self.__class__.__name__,
            "position": list(self.pos),
            "parameters": {},
        }

        if use_random_parameters:
            self.randomize_parameters(degree_of_randomization)

        entity_params = dict()
        if self.plane_slope != 0:
            entity_params["plane_slope"] = self.plane_slope
        entity_params["mass_values"] = self.mass_values
        entity_params["use_left_site"] = self.use_left_site.name
        entity_params["coefficient_of_friction"] = self.coefficient_of_friction
        entity_dict["parameters"] = entity_params
        if self.constant_force and len(self.constant_force) > 0:
            entity_dict["parameters"]["constant_force"] = self.constant_force
        return round_floats(entity_dict)

    def get_parameters(self) -> List[dict]:
        """
        Get the parameters of the entity in a list of dictionaries
        """
        mass_dict_list = self.get_masses_quality()
        return mass_dict_list

    def to_xml(self) -> str:
        """
        Convert the body and its components to an XML string.
        """
        body_xml = (
            f"""<body name="{self.name}" pos="{' '.join(map(str, self.pos))}">\n"""
        )
        body_xml += super().to_xml()
        body_xml += "</body>"
        return body_xml

    def get_description(self, simDSL2nlq=False):
        if not simDSL2nlq: return super().get_description()

        mass_descriptions = []
        for idx, m in enumerate(self.masses):
            mass_description = {
                "name": m.name,
                "body_type": "block",
            }

            constant_frc_str = ""
            if (
                hasattr(m, "constant_force_dict")
                and m.name in m.constant_force_dict
            ):
                constant_frc_str = f"A constant force of {m.constant_force_dict[m.name]} N acts on it."

            mass_description["description"] = (
                f"A block named {m.name} has a mass of {m.mass_value} Kg and"
                f" it rests on a plane inclined at an angle {self.plane_slope} degrees."
                f" {constant_frc_str}"
            )

            mass_descriptions.append(mass_description)

        return mass_descriptions

    def get_nlq(self, symbolic = False):
        sym_dict = {}
        
        description = (
            f"In a system called '{self.name}', a block of mass {sum([float(m.mass_value) for m in self.masses])} kg rests on a plane inclined at an angle of {self.plane_slope} degrees."
        )
        if symbolic:
            description = (
                f"In a system called '{self.name}', a block of mass <mass>1 kg rests on a plane inclined at an angle of <angle>1 degrees."
            )
            sym_dict = {
                "<mass>1": sum([float(m.mass_value) for m in self.masses]),
                "<angle>1": self.plane_slope
            }

        constant_frc_str = ""
        for idx, m in enumerate(self.masses):
            if idx not in [0, len(self.masses) - 1]: continue
            if (
                hasattr(m, "constant_force_dict")
                and m.name in m.constant_force_dict
            ):
                if not symbolic:
                    constant_frc_str += f"An external force of {m.constant_force_dict[m.name]} N acts on its {['right', 'left'][idx == 0]} side. "
                else:
                    constant_frc_str += f"An external force of <force>{idx + 1} N acts on its {['right', 'left'][idx == 0]} side. "
                    sym_dict[f"<force>{idx+1}"] = m.constant_force_dict[m.name]

        if self.coefficient_of_friction > 1e-2:
            if not symbolic:
                description += f" The coefficient of friction between the block and the plane is {self.coefficient_of_friction}."
            else:
                description += f" The coefficient of friction between the block and the plane is <friction>1."
                sym_dict["<friction>1"] = self.coefficient_of_friction

        if symbolic: return f"{description} {constant_frc_str}", sym_dict
        return f"{description} {constant_frc_str}"
    
    def connecting_point_nl(self, cd, cp, csi, first=False):
        """
        Get the connecting point for the mass on the plane.
        
        Args:
            cd (ConnectingDirection): Connecting direction
            cp (ConnectingPoint): Connecting point
            csi (ConnectingPointSeqId): Connecting point sequence ID
            
        Returns:
            str: Natural language description of the connecting point
        """
        side = ['right', 'left'][cp == ConnectingPoint.LEFT]
        if cd == ConnectingDirection.INNER_TO_OUTER:
            description = (
                f"A string connected to the {side} side of the block in '{self.name}'"
                f" runs parallel to the plane to wrap around a pulley on the {side} side of the plane,"
                f" and extends outward"
            )
        else:
            description = (
                f" to connect to the {side} side of the block in '{self.name}'"
                f" after wrapping around a pulley on the {side} side of the plane,"
                f" remaining parallel to the plane."
            )      

        return description
    
    def get_question(self, sub_entity: str, quantity: str) -> str:
        """
        Get a question related to the entity
        
        Inputs:
            sub_entity: str
            quantity: str
            
        Returns:
            str
        """

        idx = int(sub_entity[4:]) # mass{i} -> i
        
        modifier = ""
        if quantity == "net_force":
            modifier = f" x {self.mass_values[idx]} / {sum(self.mass_values)}"

        question = (
            f"What is the {quantity}{modifier} of the block resting on the plane in the system '{self.name}'"
        )   
        
        return question
    
class MassPrismPlaneEntity(MassPrismPlane, Entity):

    randomization_levels = {
        DegreeOfRandomization.EASY: {
            "plane_slope": {"fixed": 0},  # Fixed zero slope
            "prism_slope_options": [
                (30, 60),
                (60, 30),
            ],  # Symmetrical angle combinations
            "block_mass": {
                "min": 0.5,
                "max": 3.0,
                "decimal": 1,
            },  # Smaller mass range
            "prism_mass": {"min": 1.0, "max": 5.0, "decimal": 1},
            "use_left_prob": 0.8,  # Bias towards left-side connection
            "use_prism_left_prob": 0.8,  # Bias towards left-side prism
            "coefficient_of_friction": {"min": 0, "max": 0},
        },
        DegreeOfRandomization.MEDIUM: {
            "plane_slope": {"min": 10, "max": 30},  # Medium slope range
            "prism_slopes": {
                "left": {"min": 20, "max": 50},
                "right": {"min": 20, "max": 50},
            },
            "block_mass": {"min": 0.2, "max": 8.0, "decimal": 2},
            "prism_mass": {"min": 0.5, "max": 10.0, "decimal": 2},
            "use_left_prob": 0.5,  # Fully random
            "use_prism_left_prob": 0.5,
            "coefficient_of_friction": {"min": 0, "max": 0},
        },
        DegreeOfRandomization.HARD: {
            "plane_slope": {"min": 20, "max": 70},  # Large slope range
            "prism_slopes": {
                "left": {"min": 10, "max": 80},  # Extreme angle range
                "right": {"min": 10, "max": 80},
            },
            "block_mass": {"min": 0.1, "max": 20.0, "decimal": 2},
            "prism_mass": {"min": 0.1, "max": 20.0, "decimal": 2},
            "use_left_prob": 0.5,
            "use_prism_left_prob": 0.5,
            "asymmetric_slope": True,  # Allow asymmetric angles
            "coefficient_of_friction": {"min": 0.2, "max": 0.7},
        },
    }

    def __init__(
        self,
        name: str,
        pos: Tuple[float, float, float] = (0, 0, 0),
        plane_slope: float = 0.0,  # degrees
        prism_left_slope: float = 30.0,  # degrees
        prism_right_slope: float = 60.0,  # degrees
        block_mass_value: float = 1.0,  # mass value of the mass on the prism
        prism_mass_value: float = 1.0,  # mass value of the prism
        use_left_site: DirectionsEnum = DirectionsEnum.USE_LEFT,
        use_prism_left: bool = True,
        # padding_z: float = 0.0,  # padding in the z direction (downwards)
        condim: str = "1",  # 1 means frictionless
        constant_force: Optional[Dict[str, List[Union[List, float]]]] = None,
        coefficient_of_friction: List[float] = [0.0, 0.0],
        **kwargs,
    ) -> None:
        # Store parameters
        self.plane_slope = plane_slope
        self.prism_left_slope = prism_left_slope
        self.prism_right_slope = prism_right_slope
        self.block_mass_value = block_mass_value
        self.prism_mass_value = prism_mass_value
        self.use_left_site = use_left_site
        self.use_prism_left = use_prism_left
        self.condim = condim
        self.coefficient_of_friction = coefficient_of_friction
        super().__init__(
            name=name,
            pos=pos,
            plane_slope=plane_slope,
            prism_left_slope=prism_left_slope,
            prism_right_slope=prism_right_slope,
            block_mass_value=block_mass_value,
            prism_mass_value=prism_mass_value,
            use_left_site=use_left_site,
            use_prism_left=use_prism_left,
            positions=[(0, 0, 0)],
            condim=condim,
            entity_type=self.__class__.__name__,
            constant_force=constant_force,
            **kwargs,
        )

        self.friction_coefficient_list = []
        if self.coefficient_of_friction[0] > 1e-2:
            self.friction_coefficient_list.append(
                (
                    f"{name}.plane.geom",
                    f"{name}.prism.geom_bottom",
                    self.coefficient_of_friction[0],
                )
            )
        if self.coefficient_of_friction[1] > 1e-2:
            self.friction_coefficient_list.append(
                (
                    f"{name}.mass.geom",
                    f"{name}.prism.geom_{['right', 'left'][self.use_prism_left]}",
                    self.coefficient_of_friction[1],
                )
            )

    def get_connecting_tendon_sequence(
        self,
        direction: ConnectingDirection,
        connecting_point: ConnectingPoint = ConnectingPoint.DEFAULT,
        connecting_point_seq_id: Optional[ConnectingPointSeqId] = None,
        use_sidesite: bool = False,
    ) -> TendonSequence:
        """
        Get the tendon sequence for the mass on the prism on the plane.
        """
        if connecting_point == ConnectingPoint.DEFAULT or connecting_point == ConnectingPoint.LEFT:
            sequences = self.get_connecting_tendon_sequences(direction)
            if sequences:
                return sequences[0]
            else:
                return []
        else:
            sequences = self.get_second_connecting_tendon_sequences(direction)
            if sequences:
                return sequences[0]
            else:
                return []

    def randomize_parameters(
        self,
        degree_of_randomization: DegreeOfRandomization = DegreeOfRandomization.DEFAULT,
        reinitialize_instance: bool = False,
        **kwargs,
    ):
        import random

        randomization_levels = {
            DegreeOfRandomization.EASY: {
                "plane_slope": {"fixed": 0},  # Fixed zero slope
                "prism_slope_options": [
                    (30, 60),
                    (60, 30),
                ],  # Symmetrical angle combinations
                "block_mass": {
                    "min": 0.5,
                    "max": 3.0,
                    "decimal": 1,
                },  # Smaller mass range
                "prism_mass": {"min": 1.0, "max": 5.0, "decimal": 1},
                "use_left_prob": 0.8,  # Bias towards left-side connection
                "use_prism_left_prob": 0.8,  # Bias towards left-side prism
                "coefficient_of_friction": {"min": 0, "max": 0},
            },
            DegreeOfRandomization.MEDIUM: {
                "plane_slope": {"min": 10, "max": 30},  # Medium slope range
                "prism_slopes": {
                    "left": {"min": 20, "max": 50},
                    "right": {"min": 20, "max": 50},
                },
                "block_mass": {"min": 0.2, "max": 8.0, "decimal": 2},
                "prism_mass": {"min": 0.5, "max": 10.0, "decimal": 2},
                "use_left_prob": 0.5,  # Fully random
                "use_prism_left_prob": 0.5,
                "coefficient_of_friction": {"min": 0, "max": 0},
            },
            DegreeOfRandomization.HARD: {
                "plane_slope": {"min": 20, "max": 70},  # Large slope range
                "prism_slopes": {
                    "left": {"min": 10, "max": 80},  # Extreme angle range
                    "right": {"min": 10, "max": 80},
                },
                "block_mass": {"min": 0.1, "max": 20.0, "decimal": 2},
                "prism_mass": {"min": 0.1, "max": 20.0, "decimal": 2},
                "use_left_prob": 0.5,
                "use_prism_left_prob": 0.5,
                "asymmetric_slope": True,  # Allow asymmetric angles
                "coefficient_of_friction": {"min": 0.2, "max": 0.7},
            },
        }

        self.randomization_levels = randomization_levels

        # Handle default randomization level
        if degree_of_randomization == DegreeOfRandomization.DEFAULT:
            degree_of_randomization = random.choice(
                [
                    DegreeOfRandomization.EASY,
                    DegreeOfRandomization.MEDIUM,
                    DegreeOfRandomization.HARD,
                ]
            )

        # Structured randomization
        if degree_of_randomization in randomization_levels:
            params = randomization_levels[degree_of_randomization]

            # Plane slope setup
            if "fixed" in params["plane_slope"]:
                self.plane_slope = params["plane_slope"]["fixed"]
            else:
                self.plane_slope = round(
                    random.uniform(
                        params["plane_slope"]["min"], params["plane_slope"]["max"]
                    ),
                    2,
                )

            # Prism angle generation
            if "prism_slope_options" in params:  # EASY mode
                left, right = random.choice(params["prism_slope_options"])
                self.prism_left_slope = left
                self.prism_right_slope = right
            else:  # MEDIUM/HARD mode
                self.prism_left_slope = round(
                    random.uniform(
                        params["prism_slopes"]["left"]["min"],
                        params["prism_slopes"]["left"]["max"],
                    ),
                    2,
                )
                self.prism_right_slope = round(
                    random.uniform(
                        params["prism_slopes"]["right"]["min"],
                        params["prism_slopes"]["right"]["max"],
                    ),
                    2,
                )
                # HARD mode allows forced asymmetry
                if params.get("asymmetric_slope", False):
                    while abs(self.prism_left_slope - self.prism_right_slope) < 15:
                        self.prism_left_slope = round(
                            random.uniform(
                                params["prism_slopes"]["left"]["min"],
                                params["prism_slopes"]["left"]["max"],
                            ),
                            2,
                        )

            # Mass value generation
            def gen_mass(config):
                value = random.uniform(config["min"], config["max"])
                return round(value, config["decimal"])

            self.block_mass_value = gen_mass(params["block_mass"])
            self.prism_mass_value = gen_mass(params["prism_mass"])

            # Connection direction control
            self.use_left_site = (
                DirectionsEnum.USE_LEFT
                if random.random() < params["use_left_prob"]
                else DirectionsEnum.USE_RIGHT
            )
            self.use_prism_left = random.random() < params["use_prism_left_prob"]

            self.coefficient_of_friction = [
                round(
                    random.uniform(
                        params["coefficient_of_friction"]["min"],
                        params["coefficient_of_friction"]["max"],
                    ),
                    2,
                ) for _ in range(2)
            ]

        # Unstructured randomization
        elif degree_of_randomization == DegreeOfRandomization.NON_STRUCTURAL:
            # Mass fine-tuning (±20%)
            self.block_mass_value = round(
                self.block_mass_value * random.uniform(0.8, 1.2), 2
            )
            self.prism_mass_value = round(
                self.prism_mass_value * random.uniform(0.8, 1.2), 2
            )

            # Slope fine-tuning (±15 degrees)
            self.plane_slope = round(self.plane_slope + random.uniform(-15, 15), 2)
            self.prism_left_slope = round(
                self.prism_left_slope + random.uniform(-15, 15), 2
            )
            self.prism_right_slope = round(
                self.prism_right_slope + random.uniform(-15, 15), 2
            )

            # Maintain physical reasonableness
            self.plane_slope = max(0, min(90, self.plane_slope))
            self.prism_left_slope = max(0, min(90, self.prism_left_slope))
            self.prism_right_slope = max(0, min(90, self.prism_right_slope))

        else:
            raise ValueError(
                f"Unsupported randomization level: {degree_of_randomization}"
            )

        if reinitialize_instance:
            self.reinitialize()

    def generate_entity_yaml(
        self,
        use_random_parameters: bool = False,
        degree_of_randomization: DegreeOfRandomization = DegreeOfRandomization.NON_STRUCTURAL,
    ) -> dict:
        entity_dict = {
            "name": self.name,
            "type": self.__class__.__name__,
            "position": list(self.pos),
            "parameters": {},
        }

        if use_random_parameters:
            self.randomize_parameters(degree_of_randomization)

        entity_dict["parameters"] = {
            "plane_slope": self.plane_slope,
            "prism_left_slope": self.prism_left_slope,
            "prism_right_slope": self.prism_right_slope,
            "block_mass_value": self.block_mass_value,
            "prism_mass_value": self.prism_mass_value,
            "use_left_site": self.use_left_site.name,
            "use_prism_left": self.use_prism_left,
            "coefficient_of_friction": self.coefficient_of_friction,
        }
        if self.constant_force and len(self.constant_force) > 0:
            entity_dict["parameters"]["constant_force"] = self.constant_force
        return round_floats(entity_dict)

    def get_parameters(self) -> List[dict]:
        """
        Get the parameters of the entity in a list of dictionaries
        """
        mass_dict_list = self.get_masses_quality()
        return mass_dict_list

    def to_xml(self) -> str:
        """
        Convert the body and its components to an XML string.
        """
        body_xml = (
            f"""<body name="{self.name}" pos="{' '.join(map(str, self.pos))}">\n"""
        )
        body_xml += super().to_xml()
        body_xml += "</body>"
        return body_xml

    def get_description(self, simDSL2nlq=False):
        if not simDSL2nlq:
            return super().get_description()

        descriptions = []

        block_description = {
            "name": self.mass.name,
            "body_type": "block",
        }

        block_description["description"] = (
            f"A block named {self.mass.name} has a mass of {self.mass.mass_value} Kg is placed on a movable prism called {self.prism.name} on the {['right', 'left'][self.use_prism_left]} side."
        )

        prism_description = {
            "name": self.prism.name,
            "body_type": "prism",
        }

        prism_description["description"] = (
            f"A movable incline named {self.prism.name} has a mass of {self.prism.mass_value} Kg, and it rests on a plane inclined at an angle {self.plane_slope} degrees."
            f" The prism is inclined at an angle of {self.mass_slope} degrees on the {['right', 'left'][self.use_prism_left]} side."
        )

        mass_descriptions = [block_description, prism_description]

        descriptions += mass_descriptions

        return descriptions

    def get_nlq(self, symbolic = False):
        sym_dict = {}
        
        description = (
            f"a block of mass {self.mass.mass_value} kg rests on the {['right', 'left'][self.use_prism_left]} side a movable prism of mass {self.prism.mass_value} Kg."
            f" The prism makes an angle {self.mass_slope} degrees with the plane, which in turn makes an angle {self.plane_slope} degrees with horizontal."
        )

        if symbolic:
            description = (
                f"a block of mass <mass>1 kg rests on the {['right', 'left'][self.use_prism_left]} side a movable prism of mass <mass>2 Kg."
                f" The prism makes an angle <angle>1 degrees with the plane, which in turn makes an angle <angle>2 degrees with horizontal."
            )
            sym_dict.update(
                {
                    "<mass>1": self.mass.mass_value,
                    "<mass>2": self.prism.mass_value,
                    "<angle>1": self.mass_slope,
                    "<angle>2": self.plane_slope
                }
            )

        mass_constant_frc_str = ""
        if (
            hasattr(self.mass, "constant_force_dict")
            and self.mass.name in self.mass.constant_force_dict
        ):
            mass_constant_frc_str = f"A constant force of {self.mass.constant_force_dict[self.mass.name]} N acts on the block."
            if symbolic:
                mass_constant_frc_str = f"A constant force of <force>1 N acts on the block."
                sym_dict["<force>1"] = self.mass.constant_force_dict[self.mass.name]

        prism_constant_frc_str = ""
        if (
            hasattr(self.prism, "constant_force_dict")
            and self.prism.name in self.prism.constant_force_dict
        ):
            prism_constant_frc_str = f"A constant force of {self.prism.constant_force_dict[self.prism.name]} N acts on the prism."
            if symbolic:
                prism_constant_frc_str = f"A constant force of <force>2 N acts on the prism."
                sym_dict["<force>2"] = self.prism.constant_force_dict[self.prism.name]

        constant_frc_str = f"{mass_constant_frc_str} {prism_constant_frc_str}"

        if self.coefficient_of_friction[0] > 1e-2:
            if not symbolic:
                description += f" The coefficient of friction between the prism and the plane is {self.coefficient_of_friction[0]}."
            else:
                description += f" The coefficient of friction between the prism and the plane is <friction>1."
                sym_dict["<friction>1"] = self.coefficient_of_friction[0]
        if self.coefficient_of_friction[1] > 1e-2:
            if not symbolic:
                description += f" The coefficient of friction between the block and the prism is {self.coefficient_of_friction[1]}."
            else:
                description += f" The coefficient of friction between the block and the prism is <friction>2."
                sym_dict["<friction>2"] = self.coefficient_of_friction[1]

        if symbolic: return f"In a system called {self.name}, {description} {constant_frc_str}", sym_dict 
        return f"In a system called {self.name}, {description} {constant_frc_str}"
    
    def connecting_point_nl(self, cd, cp, csi, first=False):
        """
        Get the connecting point for the mass on the prism on the plane.
        
        Args:
            cd (ConnectingDirection): Connecting direction
            cp (ConnectingPoint): Connecting point
            csi (ConnectingPointSeqId): Connecting point sequence ID
            
        Returns:
            str: Natural language description of the connecting point
        """

        side = ['right', 'left'][cp == ConnectingPoint.LEFT]
        if cd == ConnectingDirection.INNER_TO_OUTER:
            description = (
                f"A string connected to the {side} side of the prism in '{self.name}'"
                f" runs parallel to the plane to wrap around a pulley on the {side} side of the plane,"
                f" and extends outward"
            )
        else:
            description = (
                f" to connect to the {side} side of the prism in '{self.name}'"
                f" after wrapping around a pulley on the {side} side of the plane,"
                f" remaining parallel to the plane."
            )      

        return description
    
    def get_question(self, sub_entity: str, quantity: str) -> str:
        """
        Get a question related to the entity
        
        Inputs:
            sub_entity: str
            quantity: str
            
        Returns:
            str
        """

        body = sub_entity # mass or prism
        if body == "mass": body = "block"

        question = (
            f"What is the {quantity} of the {body} in the system '{self.name}'"
        )
        
        return question

    def get_shortcut(self):
        # self.prism.add_joint(Joint('fixed', (0,0,1), f'{self.prism.name}.shortcut_fixed_joint'))
        self.prism.joints = []
        return True

class MassPrismPulleyPlane(Entity):
    """
    Similar to the original MassPrismPlane class, but directly implemented as an Entity.
    No need for logic like get_connecting_tendon_sequence, etc.
    
    Features:
    1. Place a mass block (Mass) on the left and right inclined planes of the prism (Prism).
    2. Add two top sites (similar to pulleys) at the top sides of the prism, called left_top_site and right_top_site.
    3. Connect mass_1, left_top_site, right_top_site, and mass_2 together using a ready_tendon.
    4. randomize_parameters only needs to randomize plane_slope, prism_left_slope, prism_right_slope, block_mass_value, etc.
    """

    randomization_levels = {
        DegreeOfRandomization.EASY: {
            "plane_slope": {"min": 0, "max": 0},
            "prism_mass_value": {"min": 2, "max": 20},
            "left_mass_value": {"min": 1, "max": 5},
            "right_mass_value": {"min": 1, "max": 5},
            "prism_left_slope": [30, 45, 60],
            "prism_right_slope": [30, 45, 60],
            "coefficient_of_friction": {"min": 0, "max": 0},
        },
        DegreeOfRandomization.MEDIUM: {
            "plane_slope": {"min": 0, "max": 0},
            "prism_mass_value": {"min": 2, "max": 20},
            "left_mass_value": {"min": 1, "max": 5},
            "right_mass_value": {"min": 1, "max": 5},
            "prism_left_slope": {"min": 15, "max": 75},
            "prism_right_slope": {"min": 15, "max": 75},
            "coefficient_of_friction": {"min": 0, "max": 0},
        },
        DegreeOfRandomization.HARD: {
            "plane_slope": {"min": 0, "max": 60},
            "prism_mass_value": {"min": 2, "max": 20},
            "left_mass_value": {"min": 1, "max": 5},
            "right_mass_value": {"min": 1, "max": 5},
            "prism_left_slope": {"min": 15, "max": 75},
            "prism_right_slope": {"min": 15, "max": 75},
            "coefficient_of_friction": {"min": 0.2, "max": 0.7},
        },
    }

    def __init__(
        self,
        name: str,
        plane_slope: float = 0.0,          # Plane angle
        prism_left_slope: float = 30.0,    # Left inclined plane angle of the triangular prism
        prism_right_slope: float = 60.0,   # Right inclined plane angle of the triangular prism
        block_left_mass_value: float = 1.0,     # Mass of each mass block
        block_right_mass_value: float = 1.0,     # Mass of each mass block
        prism_mass_value: float = 1.0,     # Mass of the triangular prism itself
        pos: Tuple[float, float, float] = (0, 0, 0),
        condim: str = "1",                # 1 indicates no friction, can be modified as needed
        constant_force: Optional[Dict[str, List[Union[List, float]]]] = None,
        init_velocity: Optional[Dict[str, List[Union[List, float]]]] = None,
        coefficient_of_friction: List[float] = [0.0, 0.0, 0.0],
        **kwargs,
    ) -> None:
        """
        Directly inherits from Entity and combines Plane / TriangularPrismBox / Mass_1 / Mass_2 inside.
        """
        self.plane_slope = plane_slope
        self.prism_left_slope = prism_left_slope
        self.prism_right_slope = prism_right_slope
        self.block_left_mass_value = block_left_mass_value
        self.block_right_mass_value = block_right_mass_value
        self.prism_mass_value = prism_mass_value
        self.condim = condim

        super().__init__(name=name, pos=pos, entity_type=self.__class__.__name__, **kwargs)
        
        # (1) Create the inclined plane
        self.plane = Plane(
            name=f"{name}.plane",
            pos=(0, 0, 0),
            size=(DEFAULT_PLANE_LENGTH, DEFAULT_PLANE_WIDTH, DEFAULT_PLANE_THICKNESS),
            quat=Frame.euler_to_quaternion(np.array([0, -plane_slope, 0]), degrees=True),
            condim=self.condim,
        )
        # Calculate the position of the triangular prism on the plane using plane.pos_on_top
        _, global_prism_pos, prism_quat = self.plane.pos_on_top(
            0, 0,
            z_padding=DEFAULT_PRISM_HEIGHT + TriangularPrismBox.thickness
        )

        # (2) Create the triangular prism and place it on the plane
        self.prism = TriangularPrismBox(
            name=f"{name}.prism",
            positions=(0, 0, 0),
            size=DEFAULT_PRISM_WIDTH,
            height=DEFAULT_PRISM_HEIGHT,
            slopeL=self.prism_left_slope,
            slopeR=self.prism_right_slope,
            mass_value=self.prism_mass_value,
            condim=self.condim,
            constant_force=(
                {ConstantForceType.PRISM: constant_force[ConstantForceType.PRISM]}
                if (constant_force and ConstantForceType.PRISM in constant_force) else None
            ),
        )
        self.prism.set_pose(global_prism_pos, prism_quat)

        # (3) Create two mass blocks mass_1 and mass_2, placed on the left and right slopes respectively
        # mass_1 on the left slope
        self.mass_1 = Mass(
            name=f"{name}.mass_left",
            positions=[(0, 0, 0)],
            mass_value=self.block_left_mass_value,
            slope=self.prism_left_slope,
            constant_force=(
                {ConstantForceType.MASS: constant_force[ConstantForceType.MASS]}
                if (constant_force and ConstantForceType.MASS in constant_force) else None
            ),
            init_velocity=(
                {InitVelocityType.MASS: init_velocity[InitVelocityType.MASS]}
                if (init_velocity and InitVelocityType.MASS in init_velocity) else None
            ),
        )
        # Example uses x=0.5, y=0.0, z_padding = block thickness + TriangularPrismBox thickness
        _, global_m1_pos, m1_quat = self.prism.pos_on_left_slope(
            x=0.5,
            y=0.0,
            z_padding=TriangularPrismBox.thickness + self.mass_1.size[2]
        )
        self.mass_1.set_pose(global_m1_pos, m1_quat)

        # mass_2 on the right slope
        self.mass_2 = Mass(
            name=f"{name}.mass_right",
            positions=[(0, 0, 0)],
            mass_value=self.block_right_mass_value,
            slope=self.prism_right_slope,
            constant_force=(
                {ConstantForceType.MASS: constant_force[ConstantForceType.MASS]}
                if (constant_force and ConstantForceType.MASS in constant_force) else None
            ),
            init_velocity=(
                {InitVelocityType.MASS: init_velocity[InitVelocityType.MASS]}
                if (init_velocity and InitVelocityType.MASS in init_velocity) else None
            ),
        )
        _, global_m2_pos, m2_quat = self.prism.pos_on_right_slope(
            x=0.5,
            y=0.0,
            z_padding=TriangularPrismBox.thickness + self.mass_2.size[2]
        )
        self.mass_2.set_pose(global_m2_pos, m2_quat)

        # (4) Add top pulley sites on both sides of the prism
        # Assume we define them at a local x=1 (rightmost of the left slope) or x=0 (leftmost of the right slope), then raise by DEFAULT_MASS_SIZE
        # 1) left_top_site
        _, global_lt_pos, _ = self.prism.pos_on_left_slope(
            x=0.0,  # Leftmost end (changed from 1.0 to 0.0)
            y=0.0,
            z_padding=TriangularPrismBox.thickness + DEFAULT_MASS_SIZE
        )
        local_lt_pos = Frame(origin=np.array(self.prism.pos), quat=self.prism.quat).global2rel(global_lt_pos)
        self.left_top_site = Site(
            name=f"{self.prism.name}.left_top_site",
            pos=tuple(local_lt_pos),
            quat=(1, 0, 0, 0),
            body_name=self.prism.name,
        )
        self.prism.add_site(self.left_top_site)

        # 2) right_top_site
        _, global_rt_pos, _ = self.prism.pos_on_right_slope(
            x=0.0,  # Leftmost end
            y=0.0,
            z_padding=TriangularPrismBox.thickness + DEFAULT_MASS_SIZE
        )
        local_rt_pos = Frame(origin=np.array(self.prism.pos), quat=self.prism.quat).global2rel(global_rt_pos)
        self.right_top_site = Site(
            name=f"{self.prism.name}.right_top_site",
            pos=tuple(local_rt_pos),
            quat=(1, 0, 0, 0),
            body_name=self.prism.name,
        )
        self.prism.add_site(self.right_top_site)

        # Add plane, prism, mass_1, mass_2 as child bodies
        self.add_child_body(self.plane)
        self.add_child_body(self.prism)
        self.add_child_body(self.mass_1)
        self.add_child_body(self.mass_2)

        self.friction_coefficient_list = []
        self.coefficient_of_friction = coefficient_of_friction

        if self.coefficient_of_friction[0] > 1e-2:
            self.friction_coefficient_list.append(
                (
                    f"{self.name}.plane.geom",
                    f"{self.name}.prism.geom_bottom",
                    self.coefficient_of_friction[0],
                )
            )
        if self.coefficient_of_friction[1] > 1e-2:
            self.friction_coefficient_list.append(
                (
                    f"{self.name}.mass_left.geom",
                    f"{self.name}.prism.geom_left",
                    self.coefficient_of_friction[1],
                )
            )
        if self.coefficient_of_friction[2] > 1e-2:
            self.friction_coefficient_list.append(
                (
                    f"{self.name}.mass_right.geom",
                    f"{self.name}.prism.geom_right",
                    self.coefficient_of_friction[2],
                )
            )

    def get_ready_tendon_sequences(self, direction: ConnectingDirection) -> List[TendonSequence]:
        """
        Get the ready_tendon sequence.
        """
        return [
            TendonSequence(
                elements=[
                    self.mass_1.center_site.create_spatial_site(),
                    self.left_top_site.create_spatial_site(),
                    self.right_top_site.create_spatial_site(),
                    self.mass_2.center_site.create_spatial_site(),
                ],
                description="Tendon connecting mass_1 -> left_top_site -> right_top_site -> mass_2",
                name = f"{self.name}.ready_tendon",
            )
        ]

    def to_xml(self) -> str:
        """
        Combine the XML of plane, prism, mass_1, mass_2.
        If you also want to write the tendon sequence to XML, it needs to be handled here or elsewhere.
        """
        xml = []
        xml.append(self.plane.to_xml())
        xml.append(self.prism.to_xml())
        xml.append(self.mass_1.to_xml())
        xml.append(self.mass_2.to_xml())
        # If tendon information needs to be output, you can also append self.ready_tendon.to_xml() here.
        # xml.append(self.ready_tendon.to_xml())
        return "\n".join(xml)

    def randomize_parameters(self, 
                            degree_of_randomization: DegreeOfRandomization = DegreeOfRandomization.DEFAULT,
                            reinitialize_instance: bool = False,
                            **kwargs,
                            ):
        """
        Randomize only the plane angle (plane_slope), the angles of the left and right slopes of the triangular prism (prism_left_slope/prism_right_slope), and the block mass (block_mass_value).
        This is just an example, and the range of values can be adjusted as needed.
        """
        # self.plane_slope = round(random.uniform(0.0, 70.0), 2)
        # self.prism_left_slope = round(random.uniform(0.0, 80.0), 2)
        # self.prism_right_slope = round(random.uniform(0.0, 80.0), 2)
        # self.block_left_mass_value = round(random.uniform(0.5, 5.0), 2)
        # self.block_right_mass_value = round(random.uniform(0.5, 5.0), 2)
        # self.prism_mass_value = round(random.uniform(0.5, 5.0), 2)

        params = self.randomization_levels[degree_of_randomization]
        self.plane_slope = round(random.uniform(params["plane_slope"]["min"], params["plane_slope"]["max"]), 2)
        if degree_of_randomization == DegreeOfRandomization.EASY:
            self.prism_left_slope = random.choice(params["prism_left_slope"])
            self.prism_right_slope = random.choice(params["prism_right_slope"])
        else:
            self.prism_left_slope = round(random.uniform(params["prism_left_slope"]["min"], params["prism_left_slope"]["max"]), 2)
            self.prism_right_slope = round(random.uniform(params["prism_right_slope"]["min"], params["prism_right_slope"]["max"]), 2)
        self.block_left_mass_value = round(random.uniform(params["left_mass_value"]["min"], params["left_mass_value"]["max"]), 2)
        self.block_right_mass_value = round(random.uniform(params["right_mass_value"]["min"], params["right_mass_value"]["max"]), 2)
        self.prism_mass_value = round(random.uniform(params["prism_mass_value"]["min"], params["prism_mass_value"]["max"]), 2)
        self.coefficient_of_friction = [
            round(random.uniform(params["coefficient_of_friction"]["min"], params["coefficient_of_friction"]["max"]), 2)
            for _ in range(3)
        ]

        # To make the changes take effect immediately, you can recreate or reset the entity
        if reinitialize_instance:
            self.reinitialize()

    def generate_entity_yaml(
        self,
        use_random_parameters: bool = False,
        degree_of_randomization: DegreeOfRandomization = DegreeOfRandomization.NON_STRUCTURAL,
    ) -> dict:
        """
        Generate the YAML configuration for the entity.
        """
        entity_dict = {
            "name": self.name,
            "type": self.__class__.__name__,
            "position": list(self.pos),
            "parameters": {},
        }

        if use_random_parameters:
            self.randomize_parameters()

        entity_dict["parameters"] = {
            "plane_slope": self.plane_slope,
            "prism_left_slope": self.prism_left_slope,
            "prism_right_slope": self.prism_right_slope,
            "block_left_mass_value": self.block_left_mass_value,
            "block_right_mass_value": self.block_right_mass_value,
            "prism_mass_value": self.prism_mass_value,
            "condim": self.condim,
            "coefficient_of_friction": self.coefficient_of_friction,
        }
        return round_floats(entity_dict)
    
    def get_parameters(self):
        return super().get_parameters()

    def get_description(self, simDSL2nlq=False):
        return super().get_description(simDSL2nlq)
    
    def get_nlq(self, symbolic=False):
        """
        Provide a simplified description, similar to the common patterns in the code above.
        """
        sym_dict = {}

        slope = "<angle>1"
        left_slope = "<angle>2"
        right_slope = "<angle>3"
        mass = "<mass>1"
        left_mass = "<mass>2"
        right_mass = "<mass>3"
        coeff_bot = "<friction>1"
        coeff_left = "<friction>2"
        coeff_right = "<friction>3"

        sym_dict.update(
            {
                slope: self.plane_slope,
                mass: self.prism_mass_value,
                left_mass: self.block_left_mass_value,
                right_mass: self.block_right_mass_value,
                left_slope: self.prism_left_slope,
                right_slope: self.prism_right_slope,
                coeff_bot: self.coefficient_of_friction[0],
                coeff_left: self.coefficient_of_friction[1],
                coeff_right: self.coefficient_of_friction[2],
            }
        )
        
        desc = (
            f"In a system called '{self.name}', a large movable prism of mass {mass} kg rests on a plane inclined at {slope} degrees. "
            f"The prism is inclined at an angle of {left_slope} degrees on the left side and {right_slope} degrees on the right side. "
        )
        
        desc += (
            f"Two blocks of mass {left_mass} kg and {right_mass} kg are resting on the left and right sides of the prism, respectively. "
            f"A rope connects the left block to the right after passing over a pulley attached to the top edge of the prism. "
        )

        desc += (
            f"The coefficient of friction between the prism and the plane is {coeff_bot}. " if self.coefficient_of_friction[0] > 1e-2 else ""
            f"The coefficient of friction between the left block and the prism is {coeff_left}. " if self.coefficient_of_friction[1] > 1e-2 else ""
            f"The coefficient of friction between the right block and the prism is {coeff_right}. " if self.coefficient_of_friction[2] > 1e-2 else ""
        )
        
        if not symbolic:
            desc = replace_all(desc, sym_dict)
            return desc

        return desc, sym_dict

    def connecting_point_nl(self, cd, cp, csi):
        raise NotImplementedError("MassPrismPulleyPlane is not supposed to have connections.")
    
    def get_question(self, sub_entity: str, quantity: str) -> str:
        """
        Return a question related to the scene, for example:
        - "What is the tension in the rope on the left side?"
        - "What is the net force on the small mass on top?"
        Here, sub_entity distinguishes between top_mass or the mass hanging on the left/right side.
        """
        if sub_entity == "prism":
            return f"What is the {quantity} of the prism in the system '{self.name}'"
        elif sub_entity == "mass_left":
            return f"What is the {quantity} of the block on the left face of the prism in the system '{self.name}'"
        elif sub_entity == "mass_right":
            return f"What is the {quantity} of the block on the right face of the prism in the system '{self.name}'"
        else:
            raise ValueError(f"Unknown sub_entity: {sub_entity}. Expected 'prism', 'mass_left', or 'mass_right'.")

    def get_shortcut(self):
        # self.prism.add_joint(Joint('fixed', (0,0,1), f'{self.prism.name}.shortcut_fixed_joint'))
        self.prism.joints = []
        return True

class StackedMassPlane(Entity):
    """
    Represents a stack of masses on a plane.
    """

    randomization_levels = {
        DegreeOfRandomization.EASY: {
            "num_masses": {"min": 2, "max": 2},  # Fixed 2 mass
            "mass_value": {
                "min": 0.5,
                "max": 3.0,
                "decimal": 1,
            },  # Smaller mass range
            "plane_slope": {"fixed": 0},  # Fixed zero slope
            "coefficient_of_friction": {"min": 0, "max": 0},
        },
        DegreeOfRandomization.MEDIUM: {
            "num_masses": {"min": 2, "max": 3},
            "mass_value": {"min": 0.2, "max": 8.0, "decimal": 2},
            "plane_slope": {"min": 10, "max": 30},
            "coefficient_of_friction": {"min": 0, "max": 0},
        },
        DegreeOfRandomization.HARD: {
            "num_masses": {"min": 3, "max": 5},  # Up to 5 masses
            "mass_value": {"min": 0.1, "max": 15.0, "decimal": 2},
            "plane_slope": {"min": 20, "max": 50},  # Steeper slope
            "coefficient_of_friction": {"min": 0.2, "max": 0.7},
        },
    }

    offset = 0.05  # TODO: make this in constants.py

    def __init__(
        self,
        name: str,
        pos: Tuple[float, float, float] = (0, 0, 0),
        quat: Tuple[float, float, float, float] = (1, 0, 0, 0),
        plane_slope: float = 0.0,  # degrees
        mass_values: List[float] = [1],  # mass value of the mass on the plane
        padding_z: float = 0,  # padding in the z direction (downwards)
        use_left_site: DirectionsEnum = DirectionsEnum.USE_LEFT,
        condim: str = "1",  # 1 means frictionless
        constant_force: Optional[Dict[str, List[Union[List, float]]]] = None,
        init_randomization_degree: DegreeOfRandomization = None,
        coefficient_of_friction: List[float] | None = None,
        **kwargs,
    ) -> None:
        self.mass_values = mass_values
        self.use_left_site = use_left_site
        self.plane_slope = plane_slope

        # Coeff of friction
        num_masses = len(mass_values)
        if coefficient_of_friction is None: 
            self.coefficient_of_friction = [0.0 for _ in range(num_masses)]
        else: self.coefficient_of_friction = coefficient_of_friction
        
        self.friction_coefficient_list = []
        if self.coefficient_of_friction[0] > 1e-2:
            self.friction_coefficient_list.append(
                (
                    f"{name}.plane.geom",
                    f"{name}.mass0.geom",
                    self.coefficient_of_friction[0],
                )
            )
        
        for i in range(1, num_masses):
            try: self.coefficient_of_friction[i]
            except: st()
            if self.coefficient_of_friction[i] > 1e-2:
                self.friction_coefficient_list.append(
                    (
                        f"{name}.mass{i-1}.geom",
                        f"{name}.mass{i}.geom",
                        self.coefficient_of_friction[i],
                    )
                )
        
        super().__init__(
            name,
            pos=pos,
            quat=quat,
            entity_type=self.__class__.__name__,
            init_randomization_degree=init_randomization_degree,
            **kwargs,
        )
        self.initialize_connecting_points(
            connection_constraints={
                ConnectingPoint.LEFT: len(self.mass_values),
                ConnectingPoint.RIGHT: len(self.mass_values),
            }
        )

        self.plane = Plane(
            name=name + ".plane",
            size=(DEFAULT_PLANE_LENGTH, DEFAULT_PLANE_WIDTH, DEFAULT_PLANE_THICKNESS),
            quat=Frame.euler_to_quaternion(np.array([0, -plane_slope, 0]), degrees=True),
            condim=condim,
            site_padding=DEFAULT_MASS_SIZE,  # default size of Mass
        )
        self.stacked_left_sites = []
        self.stacked_right_sites = []

        self.masses = []
        mass_z_padding = 0.0
        plane_height = self.plane.size[2]
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
            self.masses.append(
                Mass(  # use DEFAULT_MASS_SIZE as the size of the mass
                    name=name + ".mass" + str(i),
                    positions=[(0, 0, 0)],
                    mass_value=mass_value,
                    slope=plane_slope,
                    padding_size_x=STACKED_MASS_START_LENGTH / (2**i),
                    size_z=DEFAULT_MASS_SIZE / (2**i),
                    size_y=DEFAULT_MASS_SIZE / (2**i),
                    constant_force=current_constant_force,
                )
            )
            local_pos, global_mass_pos, mass_quat = self.plane.pos_on_top(
                0, 0, z_padding=self.masses[-1].size[2] + mass_z_padding
            )

            # create two additional sites for the plane on the left and right, with the hight of the mass
            self.stacked_left_sites.append(
                Site(
                    f"{self.plane.name}.left-{i}",
                    (
                        -self.plane.size[0] - DEFAULT_MASS_SIZE / 2,
                        0,
                        self.masses[-1].size[2] + mass_z_padding + plane_height,
                    ),
                    (1, 0, 0, 0),
                    body_name=name,
                )
            )
            self.stacked_right_sites.append(
                Site(
                    f"{self.plane.name}.right-{i}",
                    (
                        self.plane.size[0] + DEFAULT_MASS_SIZE / 2,
                        0,
                        self.masses[-1].size[2] + mass_z_padding + plane_height,
                    ),
                    (1, 0, 0, 0),
                    body_name=name,
                )
            )
            # add the sites to the plane
            self.plane.add_site(self.stacked_left_sites[-1])
            self.plane.add_site(self.stacked_right_sites[-1])

            # set the position of the mass
            mass_z_padding += self.masses[-1].size[2] * 2

            # set the position of the mass
            self.masses[-1].set_pose(global_mass_pos, mass_quat)

        # set the left or right site of the plane as the origin of the system
        self.origin_pos = None  # to be set in the align_pose function

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
        if (
            use_left_site == DirectionsEnum.USE_LEFT
            or use_left_site == DirectionsEnum.USE_BOTH
        ):
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

    def to_xml(self) -> str:
        """
        Convert the mass and plane to an XML string.
        """
        # xml = ""
        xml = f"""<body name="{self.name}" pos="{' '.join(map(str, self.pos))}" quat="{' '.join(map(str, self.quat))}">"""
        for mass in self.masses:
            xml += mass.to_xml() + "\n"
        xml += self.plane.to_xml() + "\n"
        xml += "</body>"
        return xml

    def get_connecting_tendon_sequence(
        self,
        direction: ConnectingDirection,
        connecting_point: ConnectingPoint = ConnectingPoint.DEFAULT,
        connecting_point_seq_id: Optional[ConnectingPointSeqId] = None,
        use_sidesite: bool = False,
    ) -> TendonSequence:
        """
        Get the tendon sequence for the mass on the plane.
        """
        sequences = []
        for i, mass in enumerate(self.masses):
            if connecting_point == ConnectingPoint.LEFT:
                site = self.stacked_left_sites[i]
                inner_tendon = [
                    mass.left_site.create_spatial_site(),
                    site.create_spatial_site(),
                ]
                sequences.append(inner_tendon)
            elif connecting_point == ConnectingPoint.RIGHT:
                site = self.stacked_right_sites[i]
                inner_tendon = [
                    mass.right_site.create_spatial_site(),
                    site.create_spatial_site(),
                ]
                sequences.append(inner_tendon)
            else:
                raise ValueError(f"Unsupported connecting_point: {connecting_point}")

        if direction == ConnectingDirection.INNER_TO_OUTER:
            if connecting_point_seq_id is None:
                return TendonSequence(
                    elements=sequences[0],
                    description=f"Tendon sequence for connecting point {connecting_point}",
                    name=f"{self.name}.connecting_tendon"
                )
            return TendonSequence(
                elements=sequences[int(connecting_point_seq_id) - 1],
                description=f"Tendon sequence for connecting point {connecting_point}",
                name=f"{self.name}.connecting_tendon"
            )
        else:
            if connecting_point_seq_id is None:
                return TendonSequence(
                    elements=sequences[0][::-1],
                    description=f"Tendon sequence for connecting point {connecting_point}",
                    name=f"{self.name}.connecting_tendon"
                )
            return TendonSequence(
                elements=sequences[int(connecting_point_seq_id) - 1][::-1],
                description=f"Tendon sequence for connecting point {connecting_point}",
                name=f"{self.name}.connecting_tendon"
            )

    def randomize_parameters(
        self,
        degree_of_randomization: DegreeOfRandomization = DegreeOfRandomization.DEFAULT,
        reinitialize_instance: bool = False,
        **kwargs,
    ):
        import random

        randomization_levels = {
            DegreeOfRandomization.EASY: {
                "num_masses": {"min": 2, "max": 2},  # Fixed 1 mass
                "mass_value": {
                    "min": 0.5,
                    "max": 3.0,
                    "decimal": 1,
                },  # Smaller mass range
                "plane_slope": {"fixed": 0},  # Fixed zero slope
                "coefficient_of_friction": {"min": 0, "max": 0},
            },
            DegreeOfRandomization.MEDIUM: {
                "num_masses": {"min": 2, "max": 3},
                "mass_value": {"min": 0.2, "max": 8.0, "decimal": 2},
                "plane_slope": {"min": 10, "max": 30},
                "coefficient_of_friction": {"min": 0, "max": 0},
            },
            DegreeOfRandomization.HARD: {
                "num_masses": {"min": 3, "max": 5},  # Up to 5 masses
                "mass_value": {"min": 0.1, "max": 15.0, "decimal": 2},
                "plane_slope": {"min": 20, "max": 50},  # Steeper slope
                "coefficient_of_friction": {"min": 0.2, "max": 0.7},
            },
        }

        self.randomization_levels = randomization_levels

        # Handle default randomization level
        if degree_of_randomization == DegreeOfRandomization.DEFAULT:
            degree_of_randomization = random.choice(
                [
                    DegreeOfRandomization.EASY,
                    DegreeOfRandomization.MEDIUM,
                    DegreeOfRandomization.HARD,
                ]
            )

        # Structured randomization
        if degree_of_randomization in randomization_levels:
            params = randomization_levels[degree_of_randomization]

            # Generate number of masses
            num_masses = random.randint(
                params["num_masses"]["min"], params["num_masses"]["max"]
            )

            # Generate mass values
            self.mass_values = [
                round(
                    random.uniform(
                        params["mass_value"]["min"], params["mass_value"]["max"]
                    ),
                    params["mass_value"]["decimal"],
                )
                for _ in range(num_masses)
            ]

            # Set plane slope
            if "fixed" in params["plane_slope"]:
                self.plane_slope = params["plane_slope"]["fixed"]
            else:
                self.plane_slope = round(
                    random.uniform(
                        params["plane_slope"]["min"], params["plane_slope"]["max"]
                    ),
                    2,
                )

            self.coefficient_of_friction = [
                round(
                    random.uniform(
                        params["coefficient_of_friction"]["min"],
                        params["coefficient_of_friction"]["max"],
                    ),
                    2,
                )
                for _ in range(num_masses)
            ]
            
        # Unstructured randomization
        elif degree_of_randomization == DegreeOfRandomization.NON_STRUCTURAL:
            # Mass value fine-tuning (±15%)
            self.mass_values = [
                max(0.1, round(m * random.uniform(0.85, 1.15), 2))
                for m in self.mass_values
            ]

            # Slope fine-tuning (±10 degrees)
            self.plane_slope = round(
                max(0, min(90, self.plane_slope + random.uniform(-10, 10))), 2
            )

        else:
            raise ValueError(
                f"Unsupported randomization level: {degree_of_randomization}"
            )

        # Reinitialize instance if needed
        if reinitialize_instance:
            self.reinitialize()

    def generate_entity_yaml(
        self,
        use_random_parameters: bool = False,
        degree_of_randomization: DegreeOfRandomization = DegreeOfRandomization.NON_STRUCTURAL,
    ) -> dict:
        entity_dict = {
            "name": self.name,
            "type": self.__class__.__name__,
            "position": list(self.pos),
            "parameters": {},
        }

        if use_random_parameters:
            self.randomize_parameters(degree_of_randomization)

        entity_dict["parameters"] = {
            "plane_slope": self.plane_slope,
            "mass_values": self.mass_values,
            "use_left_site": self.use_left_site.name,
            "coefficient_of_friction": self.coefficient_of_friction,
        }
        return entity_dict

    def get_parameters(self) -> List[dict]:
        """
        Get the parameters of the entity in a list of dictionaries
        """
        mass_dict_list = self.get_masses_quality()
        return mass_dict_list

    def get_description(self, simDSL2nlq=False):
        if not simDSL2nlq:
            return super().get_description()

        descriptions = []
        for idx, mass_val in enumerate(self.mass_values):
            below = (
                f"a plane with slope {self.plane_slope} degrees"
                if idx == 0
                else f"the block {self.mass}_mass{idx - 1}"
            )
            constant_frc_str = (
                ""
                if self.constant_force is None
                or ConstantForceType.MASS not in self.constant_force
                else f" The block is subject to a constant force of {self.constant_force[ConstantForceType.MASS][idx]} N."
            )
            description = {
                "name": self.name + f"_mass{idx}",
                "type": "Mass",
                "mass_value": mass_val,
                "description": (
                    f"A long block named {self.name+f'_mass{idx}'} has a mass {mass_val} Kg, and rests on {below}."
                    f"{constant_frc_str}"
                ),
            }
            descriptions.append(description)

        return descriptions

    def get_nlq(self, symbolic = False):
        sym_dict = {}
        
        description = (
            f"In a system called '{self.name}', {len(self.mass_values)} blocks are stacked on top of each other, resting on a plane with slope {self.plane_slope} degress."
            f" The masses of the blocks are {convert_list_to_natural_language(self.mass_values)} kg in bottom-to-top order."
        )

        if symbolic:
            description = (
                f"In a system called '{self.name}', {len(self.mass_values)} blocks are stacked on top of each other, resting on a plane with slope <angle>1 degress."
                f" The masses of the blocks are {convert_list_to_natural_language([f'<mass>{i + 1}' for i in range(len(self.mass_values))])} kg in bottom-to-top order."
            )
            sym_dict["<angle>1"] = self.plane_slope
            sym_dict.update({f"<mass>{idx}": mass_val for idx, mass_val in enumerate(self.mass_values, 1)})

        for idx, mass_val in enumerate(self.mass_values):
            constant_frc_str = (
                ""
                if self.constant_force is None
                or ConstantForceType.MASS not in self.constant_force
                else f" An external force of {self.constant_force[ConstantForceType.MASS][idx]} N on the {(['1st', '2nd', '3rd'] + [str(i+1) + 'th' for i in range(3, len(self.mass_values))])[idx]} block from the bottom."
            )

            # assume constant force not present for now

            description += constant_frc_str
        
        fric_coeff_counter = 0
        if self.coefficient_of_friction[0] > 1e-2:
            fric_coeff_counter += 1
            fric_desc = ""
            fric_desc = (
                f" The coefficient of friction between the plane and the 1st block is {self.coefficient_of_friction[0]}. "
            )
            if symbolic:
                fric_desc = (
                    f" The coefficient of friction between the plane and the 1st block is <friction>{fric_coeff_counter}. "
                )
                sym_dict[f"<friction>{fric_coeff_counter}"] = self.coefficient_of_friction[0]
            description += fric_desc
        helper_str = (['1st', '2nd', '3rd'] + [str(i+1) + 'th' for i in range(3, len(self.mass_values))])
        for i in range(1, len(self.mass_values)):
            fric_desc = ""
            if self.coefficient_of_friction[i] > 1e-2:
                fric_coeff_counter += 1
                fric_desc = (
                    f" The coefficient of friction between the {helper_str[i-1]} block and the {helper_str[i]} block is {self.coefficient_of_friction[i]}. "
                )
                if symbolic:
                    fric_desc = (
                        f" The coefficient of friction between the {helper_str[i-1]} block and the {helper_str[i]} block is <friction>{fric_coeff_counter}. "
                    )
                    sym_dict[f"<friction>{fric_coeff_counter}"] = self.coefficient_of_friction[i]
            description += fric_desc

        if symbolic: return description, sym_dict
        return description
    
    def connecting_point_nl(self, cd, cp, csi, first=False):
        """
        Get the connecting point for the mass on the plane.
        
        Args:
            cd (ConnectingDirection): Connecting direction
            cp (ConnectingPoint): Connecting point
            csi (ConnectingPointSeqId): Connecting point sequence ID
            
        Returns:
            str: Natural language description of the connecting point
        """

        side = ['right', 'left'][cp == ConnectingPoint.LEFT]
        try:
            idx = (['1st', '2nd', '3rd'] + [str(i+1) + 'th' for i in range(3, len(self.mass_values))])[(csi or 1) - 1]
        except: 
            st()
        if cd == ConnectingDirection.INNER_TO_OUTER:
            description = (
                f"A string connected to the {side} side of the {idx} block from the bottom in '{self.name}'"
                f" runs parallel to the plane to wrap around a pulley on the {side} side of the plane,"
                f" and extends outward"
            )
        else:
            description = (
                f" to connect to the {side} side of the {idx} block from the bottom in '{self.name}'"
                f" after wrapping around a pulley on the {side} side of the plane,"
                f" remaining parallel to the plane."
            )      

        return description
    
    def get_question(self, sub_entity, quantity):
        """
        Get a question related to the entity
        
        Inputs:
            sub_entity: str
            quantity: str
            
        Returns:
            str
        """
        
        try: idx = int(sub_entity[4:])
        except: st()
        idx = (['1st', '2nd', '3rd'] + [str(i+1) + 'th' for i in range(3, len(self.mass_values))])[idx]
        
        question = (
            f"What is the {quantity} of the {idx} block from the bottom in the system '{self.name}'"
        )   
        
        return question

class MassBoxPlaneEntity(Entity):
    """
    A box is placed on a plane, with fixed pulleys on the left and right of the box's top (the Site acts as the pulley location).
    A small mass called top_mass is placed at the center of the box's top, and you can decide through hang_option whether it is connected to the left or right pulley.
    One or more masses can be hung below.
    """

    # Randomization levels can be adjusted as needed
    randomization_levels = {
        DegreeOfRandomization.EASY: {
            "plane_slope": {"min": -30, "max": 30},
            "hang_options": [HangOption.HANG_RIGHT],  # only hang on the right side
            "box_mass_value": {"min": 1, "max": 2},
            "top_mass_value": {"min": 0.5, "max": 1},
            "right_mass_value": {"min": 0.1, "max": 0.5},  # only keep the right side mass
            "coefficient_of_friction": {"min": 0, "max": 0},
        },
        DegreeOfRandomization.MEDIUM: {
            "plane_slope": {"min": -30, "max": 30},
            "hang_options": [HangOption.HANG_RIGHT],  # only hang on the right side
            "box_mass_value": {"min": 1, "max": 2},
            "top_mass_value": {"min": 0.5, "max": 1},
            "right_mass_value": {"min": 0.1, "max": 0.5},  # only keep the right side mass
            "coefficient_of_friction": {"min": 0, "max": 0},
        },
        DegreeOfRandomization.HARD: {
            "plane_slope": {"min": -30, "max": 30},
            "hang_options": [HangOption.HANG_RIGHT],  # only hang on the right side
            "box_mass_value": {"min": 1, "max": 2},
            "top_mass_value": {"min": 0.5, "max": 1},
            "right_mass_value": {"min": 0.1, "max": 0.5},  # only keep the right side mass
            "coefficient_of_friction": {"min": 0.2, "max": 0.7},
        },
    }

    def __init__(
        self,
        name: str,
        pos: Tuple[float, float, float] = (0, 0, 0),
        plane_slope: float = 0.0,
        box_half_length: float = DEFAULT_BOX_PADDING_LENGTH,
        box_half_width: float = DEFAULT_BOX_PADDING_WIDTH,
        box_half_height: float = DEFAULT_BOX_PADDING_HEIGHT,
        hang_option: HangOption = HangOption.HANG_RIGHT,
        box_mass_value: float = 10.0,
        top_mass_value: float = 1.0,
        right_mass_value: float = 1.0,
        condim: str = "1",  # frictionless
        constant_force: Optional[Dict[str, List[Union[List, float]]]] = None,
        coefficient_of_friction: List[float] = [0.0, 0.0, 0.0, 0.0],
        **kwargs,
    ) -> None:
        
        self.plane_slope = plane_slope
        self.box_half_length = box_half_length
        self.box_half_width  = box_half_width
        self.box_half_height = box_half_height
        self.top_mass_size = DEFAULT_MASS_BOX_PLANE_TOP_MASS_SIZE
        self.hang_option = hang_option
        self.box_mass_value = box_mass_value
        self.top_mass_value = top_mass_value
        self.right_mass_value = right_mass_value
        self.condim = condim
        self.constant_force = constant_force

        # create quaternion to rotate the entire entity (参考ComplexCollisionPlane的旋转方式)
        theta = math.radians(-plane_slope)
        qx = 0.0
        qy = math.sin(theta / 2)
        qz = 0.0
        qw = math.cos(theta / 2)
        entity_quat = (qw, qx, qy, qz)

        super().__init__(
            name=name,
            pos=pos,
            quat=entity_quat,  # set the global rotation of the entity
            entity_type=self.__class__.__name__,
            **kwargs,
        )

        # 1) Create the plane (for illustration only, no slope processing)
        #    If your Plane class needs a slope, you can fill it in here
        self.plane = Plane(
            name=f"{name}.plane",
            size=(DEFAULT_PLANE_LENGTH, DEFAULT_PLANE_WIDTH, DEFAULT_PLANE_THICKNESS),
            quat=(1, 0, 0, 0),  # Temporarily no slope handling
            condim=self.condim,
        )

        # 2) Create the box (cart). Assuming you have a Box class available, otherwise, you can simplify using Mass from mass_entities.
        #    Below is for illustration, representing a rigid body with dimensions box_half_length * box_half_width * box_half_height
        #    You could also write it as: self.box = Mass(...) if your system uses Mass to represent any shape of a rigid body
        self.box = Mass(
            name=f"{name}.box",
            mass_value=self.box_mass_value,
            positions=[(0, 0, self.box_half_height + DEFAULT_PLANE_THICKNESS)],
            padding_size_x=self.box_half_length,
            size_y=self.box_half_width,
            size_z=self.box_half_height,
        )

        # 3) Create a pulley site on the top-left and top-right corners of the box
        #    Calculate the positions roughly as: left_pulley_x = -box_half_length/2 - DEFAULT_MASS_BOX_PLANE_PULLEY_OFFSET
        #                                       right_pulley_x =  box_half_length/2 + DEFAULT_MASS_BOX_PLANE_PULLEY_OFFSET
        #    Height can be set as box_half_height + DEFAULT_MASS_BOX_PLANE_PULLEY_OFFSET (or adjusted slightly based on your needs)
        self.pulley_offset = DEFAULT_MASS_BOX_PLANE_PULLEY_OFFSET
        self.left_pulley_site = Site(
            name=f"{self.box.name}.pulley_left",
            pos=(
                -self.box_half_length - self.pulley_offset,
                0.0,
                self.box_half_height + self.plane.size[2]
            ),
            quat=(1, 0, 0, 0),
            body_name=self.box.name,
        )
        self.right_pulley_site = Site(
            name=f"{self.box.name}.pulley_right",
            pos=(
                self.box_half_length + self.pulley_offset,
                0.0,
                self.box_half_height + self.plane.size[2]
            ),
            quat=(1, 0, 0, 0),
            body_name=self.box.name,
        )
        # Add the sites to the box
        self.box.add_site(self.left_pulley_site)
        self.box.add_site(self.right_pulley_site)

        # add the right connect site (on the right side of the plane, at the same height as the box center)
        self.right_connect_site = Site(
            name=f"{self.plane.name}.right_connect",
            pos=(
                self.plane.size[0] + DEFAULT_MASS_SIZE,  # 右侧位置：plane.size[0] + BoxSize
                0.0,
                self.plane.size[2] + self.box_half_height  # 平面上方，与盒子中心同高度
            ),
            quat=(1, 0, 0, 0),
            body_name=self.plane.name,
        )
        self.plane.add_site(self.right_connect_site)

        self.top_mass = Mass(
            name=f"{name}.top_mass",
            positions=[(0, 0, self.box_half_height * 2 + self.top_mass_size + DEFAULT_PLANE_THICKNESS)],  # Initially set to (0,0,0), later adjust with set_pose/move
            mass_value=self.top_mass_value,
            padding_size_x=self.top_mass_size,
            size_y=self.top_mass_size,
            size_z=self.top_mass_size,
            constant_force=(
                constant_force if constant_force else {}
            ),
        )

        self.right_hanging_mass = None

        if self.hang_option in [HangOption.HANG_RIGHT]:
            self.right_hanging_mass = Mass(
                name=f"{name}.right_hanging_mass",
                positions=[(self.box_half_length + self.pulley_offset, 0, (self.box_half_height + self.top_mass_size + DEFAULT_PLANE_THICKNESS))],
                mass_value=self.right_mass_value,
                padding_size_x=self.top_mass_size,
                size_y=self.top_mass_size,
                size_z=self.top_mass_size,
            )

        for body in [self.plane, self.box, self.top_mass, self.right_hanging_mass]:
            if body:
                self.add_child_body(body)

        self.friction_coefficient_list = []
        self.coefficient_of_friction = coefficient_of_friction
        if self.coefficient_of_friction[0] > 1e-2:
            self.friction_coefficient_list.append(
                (
                    f"{self.name}.plane.geom",
                    f"{self.name}.box.geom",
                    self.coefficient_of_friction[0],
                )
            )
        if self.coefficient_of_friction[1] > 1e-2:
            self.friction_coefficient_list.append(
                (
                    f"{self.name}.box.geom",
                    f"{self.name}.top_mass.geom",
                    self.coefficient_of_friction[1],
                )
            )
        if self.coefficient_of_friction[2] > 1e-2 and self.right_hanging_mass:
            self.friction_coefficient_list.append(
                (
                    f"{self.name}.top_mass.geom",
                    f"{self.name}.right_hanging_mass.geom",
                    self.coefficient_of_friction[2],
                )
            )
        if self.coefficient_of_friction[3] > 1e-2:
            self.friction_coefficient_list.append(
                (
                    f"{self.name}.box.geom",
                    f"{self.name}.right_hanging_mass.geom",
                    self.coefficient_of_friction[3],
                )
            )

    def randomize_parameters(
        self,
        degree_of_randomization: DegreeOfRandomization = DegreeOfRandomization.DEFAULT,
        reinitialize_instance: bool = False,
        **kwargs,
    ):
        params = self.randomization_levels[degree_of_randomization]
        # 4) hang_options
        self.hang_option = random.choice(params["hang_options"])
        self.plane_slope = random.uniform(params["plane_slope"]["min"], params["plane_slope"]["max"])
        self.box_mass_value = random.uniform(params["box_mass_value"]["min"], params["box_mass_value"]["max"])
        self.top_mass_value = random.uniform(params["top_mass_value"]["min"], params["top_mass_value"]["max"])
        self.right_mass_value = random.uniform(params["right_mass_value"]["min"], params["right_mass_value"]["max"])
        self.coefficient_of_friction = [
            round(
                random.uniform(
                    params["coefficient_of_friction"]["min"],
                    params["coefficient_of_friction"]["max"],
                ),
                2,
            )
            for _ in range(4)
        ]
        
        if reinitialize_instance:
            self.reinitialize()
    
    def get_ready_tendon_sequences(self, direction: ConnectingDirection) -> List[TendonSequence]:
        """
        Return all possible tendon sequences.
        """
        seqs = []
        if self.hang_option in [HangOption.HANG_RIGHT]:
            seq = [
                self.top_mass.center_site.create_spatial_site(),
                self.right_pulley_site.create_spatial_site(),
                self.right_hanging_mass.center_site.create_spatial_site(),
            ]
            seqs.append(TendonSequence(
                elements=seq,
                description="Right side tendon sequence",
                name=f"{self.name}.right_tendon",
            ))
        return seqs
    
    def get_connecting_tendon_sequence(
        self,
        direction: ConnectingDirection,
        connecting_point: ConnectingPoint = ConnectingPoint.DEFAULT,
        connecting_point_seq_id: int | None = None,
        use_sidesite: bool = False,
    ) -> TendonSequence:
        if connecting_point == ConnectingPoint.RIGHT:
            sequence = [
                self.box.center_site.create_spatial_site(),
                self.right_connect_site.create_spatial_site(),
            ]
            if direction == ConnectingDirection.OUTER_TO_INNER:
                sequence.reverse()
            return TendonSequence(
                elements=sequence,
                description="Right connecting tendon sequence",
                name=f"{self.name}.right_connect_tendon",
            )
        else:
            raise ValueError(f"Invalid connecting point: {connecting_point}")
    
    def generate_entity_yaml(
        self,
        use_random_parameters: bool = False,
        degree_of_randomization: DegreeOfRandomization = DegreeOfRandomization.NON_STRUCTURAL,
    ) -> dict:
        """
        Output a serializable dict for external saving as yaml.
        """
        entity_dict = {
            "name": self.name,
            "type": self.__class__.__name__,
            "position": list(self.pos),
            "parameters": {},
        }

        if use_random_parameters:
            self.randomize_parameters(degree_of_randomization)

        entity_dict["parameters"] = {
            "plane_slope": self.plane_slope,
            "box_half_length": self.box_half_length,
            "box_half_width": self.box_half_width,
            "box_half_height": self.box_half_height,
            "hang_option": self.hang_option.name if self.hang_option else None,
            "box_mass_value": self.box_mass_value,
            "top_mass_value": self.top_mass_value,
            "right_mass_value": self.right_mass_value,
            "coefficient_of_friction": self.coefficient_of_friction,
        }
        if self.constant_force and len(self.constant_force) > 0:
            entity_dict["parameters"]["constant_force"] = self.constant_force

        return round_floats(entity_dict)

    def get_description(self, simDSL2nlq=False):
        """
        Return a natural language description of the current scene.
        simDSL2nlq=False means a normal description; True indicates a more detailed description for simDSL2nlq.
        """
        if not simDSL2nlq:
            # Simple description
            desc = (
                f"This is a system named {self.name}, containing a plane (slope={self.plane_slope} degrees), "
                f"a box of size {self.box_half_length}x{self.box_half_width}x{self.box_half_height}, "
                f"and a top mass of size {self.top_mass_size}. "
                f"Hang option is {self.hang_option.name}."
            )
            return desc
        else:
            # Further natural language description, which can be detailed as needed
            descriptions = []
            descriptions.append(
                f"A plane called {self.plane.name} has a slope of {self.plane_slope} degrees (though effectively used as horizontal)."
            )
            descriptions.append(
                f"A box (like a car) named {self.box.name} is placed on the plane, with length={self.box_half_length}, width={self.box_half_width}, height={self.box_half_height}."
            )
            descriptions.append(
                f"Two fixed pulleys are located at the top-left and top-right corners of the box."
            )
            descriptions.append(
                f"On top of the box, there is a small mass named {self.top_mass.name}, with an edge size of {self.top_mass_size}."
            )
            if self.hang_option == HangOption.HANG_RIGHT:
                descriptions.append(
                    "A single rope goes from the small mass to the right pulley, and a mass is hanging below that pulley."
                )
            else:
                descriptions.append("No rope is connected in the system.")

            return "\n".join(descriptions)

    def get_nlq(self, symbolic=False):
        """
        Provide a simplified description, similar to the common patterns in the code above.
        """
        sym_dict = {}

        slope = "<angle>1"
        mass = "<mass>1"
        top_mass = "<mass>2"

        sym_dict.update(
            {
                slope: self.plane_slope,
                mass: self.box_mass_value,
                top_mass: self.top_mass_value,
            }
        )
        
        desc = (
            f"In a system called '{self.name}', a long movable box of mass {mass} kg rests on a plane inclined at {slope} degrees. "
            f"A small block of mass {top_mass} kg is placed on its top, allowing it to slide freely on the box. "
        )
        
        if self.hang_option == HangOption.HANG_RIGHT:
            right_mass = "<mass>3"
            sym_dict[right_mass] = self.right_mass_value

            desc += (
                f"A pulley is fixed to the right edge of the box. "
                f"A rope is attached to the right side of the top block, passes over the pulley, "
                f"and connects to a hanging block of mass {right_mass} kg on the right side. "
                f"The hanging block maintains contact with the side edge of the box."
            )
        else:
            desc += (
                f"No rope is connected in the system."
            )

        if self.coefficient_of_friction[0] > 1e-2:
            friction = "<friction>1"
            sym_dict[friction] = self.coefficient_of_friction[0]
            desc += (
                f" The coefficient of friction between the plane and the box is {friction}."
            )
        if self.coefficient_of_friction[1] > 1e-2:
            friction = "<friction>2"
            sym_dict[friction] = self.coefficient_of_friction[1]
            desc += (
                f" The coefficient of friction between the box and the top block is {friction}."
            )
        if self.coefficient_of_friction[2] > 1e-2 and self.right_hanging_mass:
            friction = "<friction>3"
            sym_dict[friction] = self.coefficient_of_friction[2]
            desc += (
                f" The coefficient of friction between the top block and the right hanging block is {friction}."
            )
        if self.coefficient_of_friction[3] > 1e-2:
            friction = "<friction>4"
            sym_dict[friction] = self.coefficient_of_friction[3]
            desc += (
                f" The coefficient of friction between the box and the right hanging block is {friction}."
            )
        
        if not symbolic:
            desc = replace_all(desc, sym_dict)
            return desc

        return desc, sym_dict

    def connecting_point_nl(self, cd, cp, csi, first=False):
        """
        Return the connecting point description in natural language.
        """
        if cp == ConnectingPoint.RIGHT:
            if cd == ConnectingDirection.INNER_TO_OUTER:
                description = (
                    f"A string connected to the right side of the box in '{self.name}'"
                    f" extends outward"
                )
            else:
                description = (
                    f" to connect to the right side of the box in '{self.name}'"
                )
            return description
        else:
            return "default connection point"
    
    def get_question(self, sub_entity: str, quantity: str) -> str:
        """
        Return a question related to the scene, for example:
        - "What is the tension in the rope on the left side?"
        - "What is the net force on the small mass on top?"
        Here, sub_entity distinguishes between top_mass or the mass hanging on the left/right side.
        """
        if sub_entity == "box":
            return f"What is the {quantity} of the long box in the system '{self.name}'"
        elif sub_entity == "top_mass":
            return f"What is the {quantity} of the block on top of the box in the system '{self.name}'"
        elif sub_entity == "right_hanging_mass":
            return f"What is the {quantity} of the block hanging from the right pulley in the system '{self.name}'"
        else:
            raise ValueError(f"Unknown sub_entity: {sub_entity}. Expected 'top_mass', 'right_hanging_mass', or 'box'.")

    def get_shortcut(self):
        # self.box.add_joint(Joint('fixed', (0,0,1), f'{self.box.name}.shortcut_fixed_joint'))
        self.box.joints = []
        return True