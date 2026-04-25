from .base_entities import *
import ipdb
from sim.utils import replace_all

class FixedPulleyEntity(FixedPulley, Entity):
    def __init__(
        self,
        name: str,
        pos: Tuple[float, float, float],
        constant_force: Optional[Dict[str, List[Union[List, float]]]] = None,
        offset: float = 0,
        entity_type: str = "FixedPulleyEntity",
        **kwargs,
    ) -> None:
        super().__init__(
            name=name,
            constant_force=constant_force,
            entity_type=entity_type,
            pos=pos,
            offset=offset,
            **kwargs,
        )

    def get_connecting_tendon_sequence(
        self,
        direction: ConnectingDirection = "",
        connecting_point: ConnectingPoint = ConnectingPoint.DEFAULT,
        connecting_point_seq_id: Optional[ConnectingPointSeqId] = None,
        use_sidesite: bool = False,
    ) -> TendonSequence:
        if not self.site:
            raise ValueError("No site available for FixedPulleyEntity")
        
        return TendonSequence(
            elements=[self.site.create_spatial_site()],
            description="Tendon sequence for FixedPulleyEntity",
            name=f"{self.name}.connecting_tendon"
        )

    def generate_entity_yaml(
        self,
        use_random_parameters: bool = False,
        degree_of_randomization: DegreeOfRandomization = DegreeOfRandomization.NON_STRUCTURAL,
    ) -> dict:
        data = {
            "name": self.name,
            "type": self.__class__.__name__,
            "position": list(self.pos),
            "parameters": {},
        }
        return round_floats(data)

    def get_description(self, simDSL2nlq=False):
        if simDSL2nlq:
            # ipdb.set_trace()
            if not hasattr(self, "force") or self.force is None or self.force == 0:
                description = f"There is a fixed point on the ceiling called {self.site.get_body_name()}."
            else:
                description = f"A pulley called {self.site.get_body_name()} is pulled with a constant force of {self.force} N."
            return [
                {
                    "name": self.site.get_body_name(),
                    "body_type": "site",
                    "description": description,
                }
            ]

        return super().get_description()

    def get_nlq(self, symbolic = False):
        description = f"'{self.name}' is a point on a fixed support."

        return description
    
    def connecting_point_nl(self, cd, cp, csi, first=False):
        """
        Get the natural language question for the connecting point of the entity.
        
        Args:
            cd: ConnectingDirection
            cp: ConnectingPoint
            csi: ConnectingPointSeqId
            
        Returns:
            str: Natural language question for the connecting point of the entity.
        """
        
        if cd == ConnectingDirection.INNER_TO_OUTER:
            description = (
                f"A string extends from the fixed point '{self.name}' outward"
            )
        elif cd == ConnectingDirection.OUTER_TO_INNER:
            description = (
                f"to connect to the fixed point '{self.name}'."
            )
        else:
            print("Unsupported connecting direction. Supported directions are INNER_TO_OUTER and OUTER_TO_INNER.")
            description = super().connecting_point_nl(cd, cp, csi)

        return description

class ConstantVelocityPuller(Entity):
    def __init__(
        self,
        name: str,
        pos: Tuple[float, float, float],
        velocity: float = 1.0,
        **kwargs,
    ):
        self.velocity = velocity
        self.joint_name = f"{name}.slide_x"
        self.box_name = f"{name}.box"
        self.pulley_name = f"{name}.pulley"
        self.actuator = None
        super().__init__(name=name, entity_type=self.__class__.__name__, pos=pos, **kwargs)
        self.mass_pos = (self.pos[0], self.pos[1], self.pos[2])
        self.create_body()
        self.create_actuator()

    def create_body(self):
        self.box = Mass(
            name=self.box_name,
            positions=[self.mass_pos],
            mass_value=1.0,
            conaffinity="1",
        )
        
        # Clear all joints and add a new one
        self.box.joints = []
        
        self.box.add_joint(
            Joint(
                name=self.joint_name,
                joint_type="slide",
                axis=(1, 0, 0),
            )
        )
        self.pulley = FixedPulley(
            name=self.pulley_name,
            pos=self.pos,
            offset=0,
        )
        self.add_child_body(self.box)
        self.add_child_body(self.pulley)

    def create_actuator(self):
        # 注意 joint 而不是 tendon！
        self.actuator = Actuator(
            name=f"{self.name}.actuator",
            actuator_type="velocity",
            kv=1000,
            velocity=self.velocity,
        )
        self.actuator.joint = self.joint_name  # 动态赋予 joint 名称

    def get_actuator(self) -> Actuator:
        return self.actuator
    
    def get_connecting_tendon_sequence(
        self,
        direction: ConnectingDirection = "",
        connecting_point: ConnectingPoint = ConnectingPoint.DEFAULT,
        connecting_point_seq_id: Optional[ConnectingPointSeqId] = None,
        use_sidesite: bool = False,
    ) -> TendonSequence:
        # sequences = self.box.get_connecting_tendon_sequences(direction, connecting_point) # + self.pulley.get_connecting_tendon_sequences(direction, connecting_point)
        # sequences[0].add_element(self.pulley.site.create_spatial_site())
        # elements = []
        # for sequence in sequences:
        #     elements.extend(sequence.get_elements())

        elements = [self.pulley.site.create_spatial_site(), self.box.center_site.create_spatial_site()]

        if direction == ConnectingDirection.INNER_TO_OUTER:
            elements.reverse()
        elif direction in [
            ConnectingDirection.LEFT_TO_RIGHT, 
            ConnectingDirection.RIGHT_TO_LEFT
            ]:
            elements.append(self.pulley.site.create_spatial_site())
            
            self.velocity /= 2
            self.velocity = round(self.velocity, 2)
            self.actuator.velocity = self.velocity

        return TendonSequence(
            elements=elements,
            description="Tendon sequence for ConstantVelocityPuller",
            name=f"{self.name}.connecting_tendon"
        )

    def generate_entity_yaml(
        self,
        use_random_parameters: bool = False,
        degree_of_randomization: DegreeOfRandomization = DegreeOfRandomization.NON_STRUCTURAL,
    ) -> dict:
        if use_random_parameters:
            self.randomize_parameters(degree_of_randomization)

        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "position": list(self.pos),
            "parameters": {
                "velocity": self.velocity,
            },
        }

    def randomize_parameters(
        self,
        degree_of_randomization: DegreeOfRandomization = DegreeOfRandomization.DEFAULT,
        reinitialize_instance: bool = False,
        **kwargs,
    ):
        self.velocity = round(random.uniform(0.1, 5.0), 2)
        if reinitialize_instance:
            self.reinitialize()

    def get_nlq(self, symbolic = False):
        sym_dict= {}

        sym_dict["<vx>1"] = self.velocity
        description = f"'{self.name}' pulls the string attached to it with a constant velocity of <vx>1 m/s (therefore shrinking the length of string at this constant rate)."

        if not symbolic:
            description = replace_all(description, sym_dict)

            return description

        return description, sym_dict
    
    def connecting_point_nl(self, cd, cp, csi, first=False):
        """
        Get the natural language question for the connecting point of the entity.
        
        Args:
            cd: ConnectingDirection
            cp: ConnectingPoint
            csi: ConnectingPointSeqId
            
        Returns:
            str: Natural language question for the connecting point of the entity.
        """
        
        if cd == ConnectingDirection.INNER_TO_OUTER:
            description = (
                f"The string pulled by '{self.name}' extends outward"
            )
        elif cd == ConnectingDirection.OUTER_TO_INNER:
            description = (
                f"and is pulled by '{self.name}'."
            )
        elif cd in [
            ConnectingDirection.LEFT_TO_RIGHT,
            ConnectingDirection.RIGHT_TO_LEFT,
        ]:
            opening = f"to be"
            ending = f" to connect to another system."
            if first:
                opening = f"The string that is"
                ending = f""
            
            description = (
                f"{opening} pulled by '{self.name}' extends to the "
                f"{['left', 'right'][cd == ConnectingDirection.LEFT_TO_RIGHT]} side{ending}"
            )
        else:
            print("Unsupported connecting direction. Supported directions are INNER_TO_OUTER, OUTER_TO_INNER, LEFT_TO_RIGHT and RIGHT_TO_LEFT.")
            description = super().connecting_point_nl(cd, cp, csi)

        return description

class ConstantForceFixedPulley(FixedPulleyEntity):
    def __init__(
        self,
        name: str,
        pos: Tuple[float, float, float],
        constant_force: Optional[Dict[str, List[Union[List, float]]]] = None,
        force: float = 0,
        offset: float = 0,
        **kwargs,
    ) -> None:
        self.force = force
        self.actuator = None
        super().__init__(
            name=name,
            entity_type=self.__class__.__name__,
            constant_force=constant_force,
            pos=pos,
            offset=offset,
            **kwargs,
        )
        self.create_actuator()

    def create_actuator(self) -> None:
        self.actuator = Actuator(
            name=f"{self.name}.actuator",
            actuator_type="general",
            gainprm=0.0,
            biasprm=-self.force,
            ctrllimited=False,
            ctrlrange=(0.0, 0.0),
        )

    def randomize_parameters(
        self,
        degree_of_randomization: DegreeOfRandomization = DegreeOfRandomization.DEFAULT,
        reinitialize_instance=False,
        **kwargs,
    ):
        self.force = round(random.uniform(0.1, 10.0), 2)

        if reinitialize_instance:
            self.reinitialize()

    def set_actuator_tendon_spatial(self, tendon_spatial_name: str):
        self.actuator.tendon = tendon_spatial_name

    def get_actuator(self) -> Actuator:
        return self.actuator

    def generate_entity_yaml(
        self,
        use_random_parameters: bool = False,
        degree_of_randomization: DegreeOfRandomization = DegreeOfRandomization.NON_STRUCTURAL,
    ) -> dict:

        if use_random_parameters:
            self.randomize_parameters(degree_of_randomization)

        data = {
            "name": self.name,
            "type": self.__class__.__name__,
            "position": list(self.pos),
            "parameters": {
                "force": self.force,
            },
        }
        return round_floats(data)
    
    def get_nlq(self, symbolic = False):
        sym_dict= {}

        description = f"'{self.name}' pulls the string attached to it with a constant force of {self.force} N."

        if symbolic:
            description = f"'{self.name}' pulls the string attached to it with a constant force of <force>1 N."
            sym_dict["<force>1"] = self.force

            return description, sym_dict

        return description
    
    def connecting_point_nl(self, cd, cp, csi, first=False):
        """
        Get the natural language question for the connecting point of the entity.
        
        Args:
            cd: ConnectingDirection
            cp: ConnectingPoint
            csi: ConnectingPointSeqId
            
        Returns:
            str: Natural language question for the connecting point of the entity.
        """
        
        if cd == ConnectingDirection.INNER_TO_OUTER:
            description = (
                f"The string pulled by '{self.name}' extends outward"
            )
        elif cd == ConnectingDirection.OUTER_TO_INNER:
            description = (
                f"and is pulled by '{self.name}'."
            )
        else:
            print("Unsupported connecting direction. Supported directions are INNER_TO_OUTER and OUTER_TO_INNER.")
            description = super().connecting_point_nl(cd, cp, csi)

        return description

class ComplexMovablePulley(Entity):
    def __init__(
        self,
        name: str,
        pos: Tuple[float, float, float],
        pulley_radius: float = DEFAULT_PULLEY_RADIUS,
        rope_length: float = DEFAULT_ROPE_LENGTH,
        pulley_mass: float = 1,
        winding_direction: str = "down",
        constant_force: Optional[Dict[str, List[Union[List, float]]]] = None,
        init_randomization_degree: DegreeOfRandomization = None,
        **kwargs,
    ):
        self.pulley_radius = pulley_radius
        self.rope_length = rope_length
        self.pulley_mass = pulley_mass
        self.winding_direction = winding_direction
        super().__init__(
            name=name,
            pos=pos,
            entity_type=self.__class__.__name__,
            constant_force=constant_force,
            init_randomization_degree=init_randomization_degree,
            **kwargs,
        )

        # Create the movable pulley at the entity's position
        self.movable_pulley = MovablePulley(
            f"{name}.movable_pulley",
            (0, 0, 0),
            mass=self.pulley_mass,
            constant_force=(
                {ConstantForceType.PULLEY: constant_force[ConstantForceType.PULLEY]}
                if constant_force and ConstantForceType.PULLEY in constant_force
                else None
            ),
        )

        # Add top and bottom connect sites to the movable pulley
        site_offset = pulley_radius
        self.top_connect_site = Site(
            f"{name}.top_connect_site",
            (0, 0, site_offset),
            body_name=self.movable_pulley.name,
        )
        self.movable_pulley.add_site(self.top_connect_site)
        self.bottom_connect_site = Site(
            f"{name}.bottom_connect_site",
            (0, 0, -site_offset),
            body_name=self.movable_pulley.name,
        )
        self.movable_pulley.add_site(self.bottom_connect_site)

        # Create top and bottom fixed pulleys
        top_fixed_pulley_pos = (0, 0, rope_length)
        self.top_fixed_pulley = FixedPulley(
            f"{name}.top_fixed_pulley", top_fixed_pulley_pos, offset=0
        )

        bottom_fixed_pulley_pos = (0, 0, -rope_length)
        self.bottom_fixed_pulley = FixedPulley(
            f"{name}.bottom_fixed_pulley", bottom_fixed_pulley_pos, offset=0
        )

        # Create left and right fixed pulleys
        horizontal_offset = pulley_radius  # Adjust as needed

        # winding_direction: "down" or "up"
        if winding_direction == "down":
            vertical_offset = rope_length
        elif winding_direction == "up":
            vertical_offset = -rope_length
        else:
            raise ValueError(f"Unsupported winding direction: {winding_direction}")

        left_fixed_pulley_pos = (-horizontal_offset, 0, vertical_offset)
        self.left_fixed_pulley = FixedPulley(
            f"{name}.left_fixed_pulley", left_fixed_pulley_pos, offset=0
        )

        right_fixed_pulley_pos = (horizontal_offset, 0, vertical_offset)
        self.right_fixed_pulley = FixedPulley(
            f"{name}.right_fixed_pulley", right_fixed_pulley_pos, offset=0
        )

    def get_connecting_tendon_sequence(
        self,
        direction: ConnectingDirection,
        connecting_point: ConnectingPoint = ConnectingPoint.DEFAULT,
        connecting_point_seq_id: Optional[ConnectingPointSeqId] = None,
        use_sidesite: bool = False,
    ) -> TendonSequence:
        if connecting_point == ConnectingPoint.DEFAULT:
            sequence = [
                self.left_fixed_pulley.site.create_spatial_site(),
                *self.movable_pulley.generate_spatial_elements(
                    use_sidesite=use_sidesite
                ),
                self.right_fixed_pulley.site.create_spatial_site(),
            ]
        elif connecting_point == ConnectingPoint.TOP:
            sequence = [
                self.top_connect_site.create_spatial_site(),
                self.top_fixed_pulley.site.create_spatial_site(),
            ]
        elif connecting_point == ConnectingPoint.BOTTOM:
            sequence = [
                self.bottom_connect_site.create_spatial_site(),
                self.bottom_fixed_pulley.site.create_spatial_site(),
            ]
        else:
            raise ValueError(f"Unsupported connecting point: {connecting_point}")

        if (
            direction == ConnectingDirection.OUTER_TO_INNER
            or direction == ConnectingDirection.RIGHT_TO_LEFT
        ):
            sequence.reverse()
        elif (
            direction != ConnectingDirection.INNER_TO_OUTER
            and direction != ConnectingDirection.LEFT_TO_RIGHT
        ):
            raise ValueError(f"Unsupported direction: {direction}")

        return TendonSequence(
            elements=sequence,
            description=f"Tendon sequence for connecting point {connecting_point}",
            name=f"{self.name}.connecting_tendon"
        )


    def to_xml(self) -> str:
        """
        Convert the entity and its components to an XML string.
        """
        body_xml = (
            f"""<body name="{self.name}" pos="{' '.join(map(str, self.pos))}">\n"""
        )
        body_xml += self.movable_pulley.to_xml() + "\n"
        body_xml += self.top_fixed_pulley.to_xml() + "\n"
        body_xml += self.bottom_fixed_pulley.to_xml() + "\n"
        body_xml += self.left_fixed_pulley.to_xml() + "\n"
        body_xml += self.right_fixed_pulley.to_xml() + "\n"
        body_xml += "</body>"
        return body_xml

    def randomize_parameters(
        self,
        degree_of_randomization: DegreeOfRandomization = DegreeOfRandomization.DEFAULT,
        reinitialize_instance=False,
        **kwargs,
    ):
        # self.pulley_radius = round(random.uniform(0.05, 0.5), 3)
        # self.rope_length = round(random.uniform(0.5, 5.0), 2)
        self.pulley_mass = round(random.uniform(0.1, 2.0), 2)
        if degree_of_randomization == DegreeOfRandomization.DEFAULT:
            # self.randomize_constant_forces([self.pulley_mass / 2] * 3)
            self.winding_direction = random.choice(
                ["up", "down"]
            )  # TODO: up direction is not always meaningful

        if reinitialize_instance:
            self.reinitialize()

    def generate_entity_yaml(
        self,
        use_random_parameters: bool = False,
        degree_of_randomization: DegreeOfRandomization = DegreeOfRandomization.NON_STRUCTURAL,
    ) -> dict:
        data = {
            "name": self.name,
            "type": self.__class__.__name__,
            "position": list(self.pos),
            "parameters": {},
        }

        if use_random_parameters:
            self.randomize_parameters(degree_of_randomization)

        data["parameters"] = {
            "pulley_mass": self.pulley_mass,
        }
        if self.constant_force and len(self.constant_force) > 0:
            data["parameters"]["constant_force"] = self.constant_force
        return round_floats(data)

    def get_description(self, simDSL2nlq=False):
        if not simDSL2nlq:
            return super().get_description(simDSL2nlq)

        raise NotImplementedError(
            "Description for Complex Movable Pulley not implemented yet"
        )

class MassWithFixedPulley(Entity):

    randomization_levels = {
        DegreeOfRandomization.EASY: {
            "mass_type_options": ["Mass"],
            "mass_config": {
                "num_masses": {"min": 1, "max": 1},
                "mass_range": {"min": 0.5, "max": 3.0, "decimal": 1},
            },
            "plane_config": {"slope": {"fixed": 0}},
            "prism_config": {
                "left_slope": {"fixed": 30},
                "right_slope": {"fixed": 60},
            },
            "prism_mass_range": {"min": 1.0, "max": 3.0},
            "use_left_prob": 0.8,
        },
        DegreeOfRandomization.MEDIUM: {
            "mass_type_options": ["Mass", "MassPlane"],
            "mass_config": {
                "Mass": {
                    "num_masses": {"min": 1, "max": 1},
                    "mass_range": {"min": 0.5, "max": 5.0, "decimal": 2},
                },
                "MassPlane": {
                    "num_masses": {"min": 1, "max": 2},
                    "mass_range": {"min": 0.5, "max": 8.0, "decimal": 2},
                    "plane_slope": {"min": 10, "max": 30},
                },
            },
            "prism_config": {
                "left_slope": {"fixed": 30},
                "right_slope": {"fixed": 60},
            },
            "prism_mass_range": {"min": 0.5, "max": 5.0},
            "use_left_prob": 0.5,
        },
        DegreeOfRandomization.HARD: {
            "mass_type_options": ["Mass", "MassPlane", "MassPrismPlane"],
            "mass_config": {
                "Mass": {
                    "num_masses": {"min": 1, "max": 2},
                    "mass_range": {"min": 0.1, "max": 10.0, "decimal": 2},
                },
                "MassPlane": {
                    "num_masses": {"min": 2, "max": 3},
                    "mass_range": {"min": 0.1, "max": 15.0, "decimal": 2},
                    "plane_slope": {"min": 20, "max": 50},
                },
                "MassPrismPlane": {
                    "num_masses": {"min": 1, "max": 2},
                    "mass_range": {"min": 0.1, "max": 20.0, "decimal": 2},
                    "prism_slopes": {
                        "left": {"min": 20, "max": 70},
                        "right": {"min": 20, "max": 70},
                    },
                    "min_slope_diff": 15,
                },
            },
            "prism_mass_range": {"min": 0.1, "max": 10.0},
            "use_left_prob": 0.5,
        },
    }

    def __init__(
        self,
        name: str,
        pos: Tuple[float, float, float],  # it has to be below z=0
        mass_type: str = "Mass",  # "Mass", "MassPlane", or "MassPrismPlane"
        mass_values: List[float] = [
            1.0
        ],  # mass value of the block on the plane or prism
        prism_mass_value: float = 1.0,  # mass value of the prism
        use_left_site: DirectionsEnum = DirectionsEnum.USE_LEFT,
        use_prism_left: bool = True,
        plane_slope: float = 0,  # degrees
        prism_left_slope: float = 30,  # degrees
        prism_right_slope: float = 60,  # degrees
        condim: str = "1",  # 1 means frictionless
        constant_force: Optional[Dict[str, List[Union[List, float]]]] = None,
        init_randomization_degree: DegreeOfRandomization = None,
        **kwargs,
    ) -> None:
        self.mass_type = mass_type
        self.mass_values = mass_values
        self.prism_mass_value = prism_mass_value
        self.use_left_site = use_left_site
        self.use_prism_left = use_prism_left
        self.plane_slope = plane_slope
        self.prism_left_slope = prism_left_slope
        self.prism_right_slope = prism_right_slope
        self.condim = condim
        super().__init__(
            name=name,
            pos=pos,
            entity_type=self.__class__.__name__,
            constant_force=constant_force,
            init_randomization_degree=init_randomization_degree,
            **kwargs,
        )  # move the body to the mass position
        # Adding a fixed pulley at this site

        self.fixed_pulley = FixedPulley(
            name=f"{name}.fixed_pulley", pos=(0, 0, DEFAULT_ROPE_LENGTH), offset=0
        )

        # Adding a site on the mass for the fixed pulley
        self.mass = create_mass_body(
            name=f"{name}.mass",
            mass_type=mass_type,
            positions=[(0, 0, 0)],
            mass_values=mass_values,
            plane_slope=plane_slope,
            prism_left_slope=prism_left_slope,
            prism_right_slope=prism_right_slope,
            prism_mass_value=prism_mass_value,
            use_left_site=use_left_site,
            use_prism_left=use_prism_left,
            padding_z=DEFAULT_PULLEY_RADIUS + 2 * DEFAULT_MASS_SIZE,
            condim=condim,
            constant_force=constant_force,
        )

    def get_connecting_tendon_sequence(
        self,
        direction: ConnectingDirection,
        connecting_point: ConnectingPoint = ConnectingPoint.DEFAULT,
        connecting_point_seq_id: Optional[ConnectingPointSeqId] = None,
        use_sidesite: bool = False,
    ) -> TendonSequence:
        if (
            connecting_point == ConnectingPoint.LEFT
            or connecting_point == ConnectingPoint.DEFAULT
        ):
            # ConnectingPoint.LEFT means default connecting point of MassPlane or MassPrismPlane
            # it could be from either the left or right site of the plane
            tendon_sequence = self.mass.get_connecting_tendon_sequences()[0]
            tendon_sequence.add_element(self.fixed_pulley.site.create_spatial_site())
        elif connecting_point == ConnectingPoint.RIGHT:
            if self.mass_type == "Mass":  # Mass has only one free site
                raise ValueError(
                    "Unsupported connecting_point: RIGHT is only for MassPlane or MassPrismPlane"
                )
            else:
                tendon_sequence = self.mass.get_second_connecting_tendon_sequences(
                    direction=direction
                )[0]
        else:
            raise ValueError(f"Unsupported connecting_point: {connecting_point}")

        if direction == ConnectingDirection.INNER_TO_OUTER:
            return TendonSequence(description=f"A tendon sequence connecting mass to the fixed pulley", name=f"{self.name}.connecting_tendon", children=[tendon_sequence])
        else:  # ConnectingDirection.OUTER_TO_INNER
            tendon_sequence = TendonSequence(description=f"A tendon sequence connecting the fixed pulley to the mass", name=f"{self.name}.connecting_tendon", children=[tendon_sequence])
            tendon_sequence.reverse()
            return tendon_sequence

    def get_ready_tendon_sequences(self, direction: ConnectingDirection) -> List[TendonSequence]:
        """
        Get the tendon sequence connecting masses if there are multiple masses for the mass body.
        """
        return self.mass.get_ready_tendon_sequences(direction)

    def to_xml(self) -> str:
        """
        Convert the body and its components to an XML string.
        """
        body_xml = (
            f"""<body name="{self.name}" pos="{' '.join(map(str, self.pos))}">\n"""
        )
        body_xml += self.fixed_pulley.to_xml() + "\n"
        body_xml += self.mass.to_xml() + "\n"
        body_xml += "</body>"
        return body_xml

    def randomize_parameters(
        self,
        degree_of_randomization: DegreeOfRandomization = DegreeOfRandomization.DEFAULT,
        reinitialize_instance: bool = False,
        **kwargs,
    ):
        import random

        # Define configuration parameters for different difficulty levels
        randomization_levels = {
            DegreeOfRandomization.EASY: {
                "mass_type_options": ["Mass"],
                "mass_config": {
                    "num_masses": {"min": 1, "max": 1},
                    "mass_range": {"min": 0.5, "max": 3.0, "decimal": 1},
                },
                "plane_config": {"slope": {"fixed": 0}},
                "prism_config": {
                    "left_slope": {"fixed": 30},
                    "right_slope": {"fixed": 60},
                },
                "prism_mass_range": {"min": 1.0, "max": 3.0},
                "use_left_prob": 0.8,
            },
            DegreeOfRandomization.MEDIUM: {
                "mass_type_options": ["Mass", "MassPlane"],
                "mass_config": {
                    "Mass": {
                        "num_masses": {"min": 1, "max": 1},
                        "mass_range": {"min": 0.5, "max": 5.0, "decimal": 2},
                    },
                    "MassPlane": {
                        "num_masses": {"min": 1, "max": 2},
                        "mass_range": {"min": 0.5, "max": 8.0, "decimal": 2},
                        "plane_slope": {"min": 10, "max": 30},
                    },
                },
                "prism_config": {
                    "left_slope": {"fixed": 30},
                    "right_slope": {"fixed": 60},
                },
                "prism_mass_range": {"min": 0.5, "max": 5.0},
                "use_left_prob": 0.5,
            },
            DegreeOfRandomization.HARD: {
                "mass_type_options": ["Mass", "MassPlane", "MassPrismPlane"],
                "mass_config": {
                    "Mass": {
                        "num_masses": {"min": 1, "max": 2},
                        "mass_range": {"min": 0.1, "max": 10.0, "decimal": 2},
                    },
                    "MassPlane": {
                        "num_masses": {"min": 2, "max": 3},
                        "mass_range": {"min": 0.1, "max": 15.0, "decimal": 2},
                        "plane_slope": {"min": 20, "max": 50},
                    },
                    "MassPrismPlane": {
                        "num_masses": {"min": 1, "max": 2},
                        "mass_range": {"min": 0.1, "max": 20.0, "decimal": 2},
                        "prism_slopes": {
                            "left": {"min": 20, "max": 70},
                            "right": {"min": 20, "max": 70},
                        },
                        "min_slope_diff": 15,
                    },
                },
                "prism_mass_range": {"min": 0.1, "max": 10.0},
                "use_left_prob": 0.5,
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

            # 1. Choose mass type
            self.mass_type = random.choice(params["mass_type_options"])

            # 2. Generate mass parameters
            if self.mass_type == "MassPrismPlane":
                type_config = params["mass_config"]["MassPrismPlane"]
            elif self.mass_type == "MassPlane":
                type_config = params["mass_config"]["MassPlane"]
            else:
                type_config = (
                    params["mass_config"]["Mass"]
                    if "Mass" in params["mass_config"]
                    else params["mass_config"]
                )

            # Generate number of masses
            num_masses = random.randint(
                type_config["num_masses"]["min"], type_config["num_masses"]["max"]
            )

            # Generate mass values
            self.mass_values = [
                round(
                    random.uniform(
                        type_config["mass_range"]["min"],
                        type_config["mass_range"]["max"],
                    ),
                    type_config["mass_range"]["decimal"],
                )
                for _ in range(num_masses)
            ]

            # 3. Generate plane parameters
            if self.mass_type in ["MassPlane", "MassPrismPlane"]:
                if "plane_slope" in type_config:
                    self.plane_slope = round(
                        random.uniform(
                            type_config["plane_slope"]["min"],
                            type_config["plane_slope"]["max"],
                        ),
                        2,
                    )
                else:
                    self.plane_slope = 0

            # 4. Generate prism parameters
            if self.mass_type == "MassPrismPlane":
                # Generate prism angles
                attempts = 0
                while attempts < 100:
                    left = round(
                        random.uniform(
                            type_config["prism_slopes"]["left"]["min"],
                            type_config["prism_slopes"]["left"]["max"],
                        ),
                        2,
                    )
                    right = round(
                        random.uniform(
                            type_config["prism_slopes"]["right"]["min"],
                            type_config["prism_slopes"]["right"]["max"],
                        ),
                        2,
                    )
                    if abs(left - right) >= type_config.get("min_slope_diff", 0):
                        self.prism_left_slope = left
                        self.prism_right_slope = right
                        break
                    attempts += 1
                else:
                    raise ValueError("Unable to generate valid prism angles")

            # 5. Generate other parameters
            self.prism_mass_value = round(
                random.uniform(
                    params["prism_mass_range"]["min"], params["prism_mass_range"]["max"]
                ),
                2,
            )

            self.use_left_site = (
                DirectionsEnum.USE_LEFT
                if random.random() < params["use_left_prob"]
                else DirectionsEnum.USE_RIGHT
            )
            self.use_prism_left = random.choice([True, False])

        # Unstructured randomization
        elif degree_of_randomization == DegreeOfRandomization.NON_STRUCTURAL:
            # Mass value fine-tuning (±15%)
            self.mass_values = [
                max(0.1, round(m * random.uniform(0.85, 1.15), 2))
                for m in self.mass_values
            ]

            # Prism mass fine-tuning
            self.prism_mass_value = max(
                0.1, round(self.prism_mass_value * random.uniform(0.8, 1.2), 2)
            )

            # Slope fine-tuning
            if self.mass_type == "MassPlane":
                self.plane_slope = round(
                    max(0, min(90, self.plane_slope + random.uniform(-10, 10))), 2
                )
            elif self.mass_type == "MassPrismPlane":
                self.prism_left_slope = round(
                    max(0, min(90, self.prism_left_slope + random.uniform(-15, 15))), 2
                )
                self.prism_right_slope = round(
                    max(0, min(90, self.prism_right_slope + random.uniform(-15, 15))), 2
                )

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
            "mass_type": self.mass_type,
            "mass_values": self.mass_values,
            "prism_mass_value": self.prism_mass_value,
            "use_left_site": self.use_left_site.name,
            "use_prism_left": self.use_prism_left,
            "plane_slope": self.plane_slope,
            "prism_left_slope": self.prism_left_slope,
            "prism_right_slope": self.prism_right_slope,
        }
        return round_floats(entity_dict)

    def get_mass_description(self):
        mass_descriptions = []

        if self.mass_type == "Mass":
            mass_description = {
                "name": self.mass.name,
                "body_type": "block",
            }

            constant_frc_str = ""
            if (
                hasattr(self.mass, "constant_force_dict")
                and self.mass.name in self.mass.constant_force_dict
            ):
                constant_frc_str = f"A constant force of {self.mass.constant_force_dict[self.mass.name]} N acts on it."

            mass_description["description"] = (
                f"A block named {self.mass.name} has a mass of {self.mass.mass_value} Kg, and is suspended in the air"
                f" with the help of a light string."
                f" {constant_frc_str}"
            )

            mass_descriptions.append(mass_description)

        elif self.mass_type == "MassPlane":
            for idx, m in enumerate(self.mass.masses):
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
                    f" it rests on a plane inclined at an angle {self.mass.plane_slope} degrees."
                    f" {constant_frc_str}"
                )

                mass_descriptions.append(mass_description)

        elif self.mass_type == "MassPrismPlane":
            block_description = {
                "name": self.mass.mass.name,
                "body_type": "block",
            }

            constant_frc_str = ""
            if (
                hasattr(self.mass.mass, "constant_force_dict")
                and self.mass.mass.name in self.mass.mass.constant_force_dict
            ):
                constant_frc_str = f"A constant force of {self.mass.mass.constant_force_dict[self.mass.mass.name]} N acts on it."

            block_description["description"] = (
                f"A block named {self.mass.mass.name} has a mass of {self.mass.mass.mass_value} Kg rests on a movable prism called {self.mass.prism.name} on the {['right', 'left'][self.mass.use_prism_left]} side."
                f" {constant_frc_str}"
            )

            prism_description = {
                "name": self.mass.prism.name,
                "body_type": "prism",
            }

            constant_frc_str = ""
            if (
                hasattr(self.mass.prism, "constant_force_dict")
                and self.mass.prism.name in self.mass.prism.constant_force_dict
            ):
                constant_frc_str = f"A constant force of {self.mass.prism.constant_force_dict[self.mass.prism.name]} N acts on it."

            prism_description["description"] = (
                f"A movable incline named {self.mass.prism.name} has a mass of {self.mass.prism.mass_value} Kg, and it rests on a plane inclined at an angle {self.mass.plane_slope} degrees."
                f" The prism is inclined at an angle of {self.mass.mass_slope} degrees on the {['right', 'left'][self.mass.use_prism_left]} side."
                f" {constant_frc_str}"
            )

            mass_descriptions = [block_description, prism_description]

        return mass_descriptions

    def get_description(self, simDSL2nlq=False):
        if not simDSL2nlq:
            return super().get_description(simDSL2nlq)

        descriptions = []

        mass_descriptions = self.get_mass_description()

        pulley_description = {
            "name": self.fixed_pulley.name,
            "body_type": "site",
            "description": (
                f"A fixed pulley / site is named {self.fixed_pulley.name}."
            ),
        }

        descriptions += mass_descriptions
        descriptions.append(pulley_description)

        return descriptions

    def get_nlq(self, symbolic = False):
        sym_dict = {}

        constant_frc_str = ""
        if self.mass_type == "Mass":
            description = f"a block of mass {self.mass_values[0]} kg is suspended in the air with the help of a light string that goes over a pulley."

            if symbolic:
                description = f"a block of mass <mass>1 kg is suspended in the air with the help of a light string that goes over a pulley."
                sym_dict["<mass>1"] = self.mass_values[0]
            
            if (
                hasattr(self.mass, "constant_force_dict")
                and self.mass.name in self.mass.constant_force_dict
            ):
                constant_frc_str = f"A constant force of {self.mass.constant_force_dict[self.mass.name]} N acts on it."
        elif self.mass_type == "MassPlane":

            # Can have multiple blocks
            masses = [m for m in self.mass.masses]
            values = [str(m.mass_value) for m in masses]
            if len(masses) == 1:
                opening = f"a block of mass {values[0]} kg rests"
            else:
                opening = f"{len(masses)} blocks, each connected to the next with a light string, of masses {convert_list_to_natural_language(values)} kg respectively, rest" ## HERE
            
            if symbolic:
                if len(masses) == 1:
                    opening = f"a block of mass <mass>2 kg rests"
                    sym_dict["<mass>2"] = values[0]
                else:
                    opening = f"{len(masses)} blocks, each connected to the next with a light string, of masses {convert_list_to_natural_language([f'<mass>{i+2}' for i in range(len(values))])} kg respectively, rest"
                    sym_dict.update(
                        {
                            f'<mass>{idx}': val
                            for idx, val in enumerate(values, 2)
                        }
                    )
            
            description = f"{opening} on a plane inclined at an angle of {self.plane_slope} degrees with horizontal."
            if symbolic:
                description = f"{opening} on a plane inclined at an angle of <angle>1 degrees with horizontal."
                sym_dict["<angle>1"] = self.plane_slope

            constant_frc_str = ""
            for i, m in enumerate(masses):
                if (
                    hasattr(m, "constant_force_dict")
                    and m.name in m.constant_force_dict
                ):
                    constant_frc_str += f"A constant force of {m.constant_force_dict[m.name]} N acts on the {(['1st', '2nd', '3rd'] +[str(j) + 'th' for j in range(4, len(masses))])[i]} block. "

        elif self.mass_type == "MassPrismPlane":
            description = (
                f"a block of mass {self.mass_values[0]} kg rests on the {['right', 'left'][self.mass.use_prism_left]} side a movable prism of mass {self.mass.prism.mass_value} Kg."
                f" The prism makes an angle {self.mass.mass_slope} degrees with the plane, which in turn makes an angle {self.mass.plane_slope} degrees with horizontal."
            )

            if symbolic:
                description = (
                    f"a block of mass <mass>3 kg rests on the {['right', 'left'][self.mass.use_prism_left]} side a movable prism of mass <mass>4 Kg."
                    f" The prism makes an angle <angle>2 degrees with the plane, which in turn makes an angle <angle>1 degrees with horizontal."
                )
                sym_dict.update(
                    {
                        "<mass>3": self.mass_values[0],
                        "<mass>4": self.mass.prism.mass_value,
                        "<angle>1": self.mass.plane_slope,
                        "<angle>2": self.mass.mass_slope,
                    }
                )

            mass_constant_frc_str = ""
            if (
                hasattr(self.mass.mass, "constant_force_dict")
                and self.mass.mass.name in self.mass.mass.constant_force_dict
            ):
                mass_constant_frc_str = f"A constant force of {self.mass.mass.constant_force_dict[self.mass.mass.name]} N acts on the block."

            prism_constant_frc_str = ""
            if (
                hasattr(self.mass.prism, "constant_force_dict")
                and self.mass.prism.name in self.mass.prism.constant_force_dict
            ):
                prism_constant_frc_str = f"A constant force of {self.mass.prism.constant_force_dict[self.mass.prism.name]} N acts on the prism."

            constant_frc_str = f"{mass_constant_frc_str} {prism_constant_frc_str}"
        
        else: 
            raise KeyError(f"Unknown mass type {self.mass_type}")
        
        description = f"In a system called '{self.name}', {description} {constant_frc_str}"

        if symbolic: return description, sym_dict
        return description

    def connecting_point_nl(self, cd, cp, csi, first=False):
        """
        Return the connecting point description in natural language.

        Args:
            cd (ConnectingDirection): Connecting direction
            cp (ConnectingPoint): Connecting point
            csi (ConnectingPointSeqId): Connecting point sequence ID

        Returns:
            str: Connecting point description in natural language
        """
        
        if self.mass_type == "Mass":
            if cd == ConnectingDirection.INNER_TO_OUTER:
                description = (
                    f"The string that connects the block to an overhead pulley in '{self.name}'"
                    f" extends outward"
                )
            else:
                description = (
                    f"to connect to the block in '{self.name}' after wrapping over an overhead pulley."
                )
        else:
            primary_is_left = self.use_left_site == DirectionsEnum.USE_LEFT
            side = "left" if primary_is_left else "right"
            if cp == ConnectingPoint.RIGHT:
                side = "right" if primary_is_left else "left"

            if self.mass_type == "MassPlane":
                if cd == ConnectingDirection.INNER_TO_OUTER:
                    description = (
                        f"The string that connects the {'1st ' if len(self.mass_values) > 1 else ''}block in '{self.name}' runs parallel to the plane to wrap around"
                        f" a pulley on the {side} side of the plane, and then extends outward"
                    )
                else:
                    description = (
                        f"to connect to the {'1st ' if len(self.mass_values) > 1 else ''}block in '{self.name}', parallel to the plane, after wrapping around"
                        f" a pulley on the {side} side of the plane."
                    )
            elif self.mass_type == "MassPrismPlane":
                if cd == ConnectingDirection.INNER_TO_OUTER:
                    description = (
                        f"The string that connects the prism in '{self.name}' runs parallel to the plane to wrap around"
                        f" a pulley on the {side} side of the plane, and then extends outward"
                    )
                else:
                    description = (
                        f"to connect to the prism in '{self.name}', parallel to the plane, after wrapping around"
                        f" a pulley on the {side} side of the plane."
                    )
            else:
                raise KeyError(f"Unknown mass type {self.mass_type}")

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

        # sub_entity can be "mass", "mass_plane.mass", "mass_prism_plane.mass/prism"

        if sub_entity in ["mass", "mass_prism_plane.mass"]:
            question = (
                f"What is the {quantity} of the block in the system '{self.name}'"
            )
        elif sub_entity == "mass_prism_plane.prism":
            question = (
                f"What is the {quantity} of the prism in the system '{self.name}'"
            )
        elif sub_entity[:15] == "mass_plane.mass":
            idx = int(sub_entity[15:])
            idx = (['1st', '2nd', '3rd'] + [f'{i}th' for i in range(4, len(self.mass_values))])[idx]
            idx += ''
            question = (
                f"What is the {quantity} of the {idx if len(self.mass_values) > 1 else ''}block in the system '{self.name}'"
            )
        
        return question

    def get_shortcut(self):
        if isinstance(self.mass, MassPrismPlane):
            # self.mass.prism.add_joint(Joint('fixed', (0,0,1), f'{self.mass.prism.name}.shortcut_fixed_joint'))
            self.mass.prism.joints = []
            return True
        return False

class MassWithReversedMovablePulley(Entity):  # TODO(yangmin): please confirm MassWithReversedMovablePulley only has one movable pulley
    """
    There is a movable pulley system with reverse winding.
    A mass block with a fixed pulley is attached to a movable pulley on each side of the left and right.
    There is a fixed pulley above the movable pulley.
    """

    def __init__(
        self,
        name: str,
        pos: Tuple[float, float, float],
        mass_values: List[float] = [1.0, 1.0],
        mass_type: str = "Mass",  # 仅支持 "Mass"
        pulley_mass: float = 1.0,
        condim: str = "1",
        constant_force: Optional[Dict[str, List[Union[List, float]]]] = None,
        init_randomization_degree: DegreeOfRandomization = None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            pos=pos,
            entity_type=self.__class__.__name__,
            constant_force=constant_force,
            init_randomization_degree=init_randomization_degree,
            **kwargs,
        )

        # Create the left and right MassWithFixedPulley entities
        mass_offset = DEFAULT_PULLEY_RADIUS * 5

        self.left_mass_with_pulley = MassWithFixedPulley(
            name=f"{name}.left_mass_with_fixed_pulley",
            pos=(-mass_offset, 0, -DEFAULT_MASS_SIZE),
            mass_type=mass_type,
            mass_values=[mass_values[0]],
            condim=condim,
            constant_force=constant_force,
        )

        self.right_mass_with_pulley = MassWithFixedPulley(
            name=f"{name}.right_mass_with_fixed_pulley",
            pos=(mass_offset, 0, -DEFAULT_MASS_SIZE),
            mass_type=mass_type,
            mass_values=[mass_values[1]],
            condim=condim,
            constant_force=constant_force,
        )

        # Create a movable pulley for reverse winding with winding direction set to 'up'
        self.movable_pulley = MovablePulley(
            name=f"{name}.movable_pulley",
            pos=(0, 0, 0),
            mass=pulley_mass,
            winding_direction="up",
            constant_force=(
                {ConstantForceType.PULLEY: constant_force[ConstantForceType.PULLEY]}
                if constant_force and ConstantForceType.PULLEY in constant_force
                else None
            ),
        )

        self.mass_type = mass_type

        # Create a fixed pulley above the movable pulley
        self.fixed_pulley_above = FixedPulley(
            name=f"{name}.fixed_pulley_above",
            pos=(0, 0, DEFAULT_ROPE_LENGTH),
            offset=0,
        )
        self.left_pulley_below = FixedPulley(
            name=f"{name}.left_pulley_below",
            pos=(-DEFAULT_PULLEY_RADIUS, 0, -DEFAULT_ROPE_LENGTH),
            offset=0,
        )
        self.right_pulley_below = FixedPulley(
            name=f"{name}.right_pulley_below",
            pos=(DEFAULT_PULLEY_RADIUS, 0, -DEFAULT_ROPE_LENGTH),
            offset=0,
        )
 
    def get_ready_tendon_sequences(self, direction: ConnectingDirection) -> List[TendonSequence]:
        """
        Returns the tendon sequence that has been prepared to connect the tendons inside the component.
        The sequence includes:
        - Tendon sequence of left mass
        - Space element of movable pulley
        - Tendon sequence of the right mass
        """
        # Get tendon sequence of left mass block
        left_tendon_sequence = self.left_mass_with_pulley.get_connecting_tendon_sequence(
            direction=ConnectingDirection.INNER_TO_OUTER
        )
        # Get the tendon sequence of the right mass
        right_tendon_sequence = self.right_mass_with_pulley.get_connecting_tendon_sequence(
            direction=ConnectingDirection.OUTER_TO_INNER
        )

        # Space element of movable pulley
        movable_pulley_sequence = self.movable_pulley.generate_spatial_elements()

        # Create root sequence
        root_sequence = TendonSequence(
            description="Prepared tendon sequence connecting left and right masses through the movable pulley",
            name=f"{self.name}.ready_tendon"
        )
        
        # Create child sequence for left mass
        left_tendon_sequence = TendonSequence(
            elements=left_tendon_sequence.get_elements() + [self.left_pulley_below.site.create_spatial_site()],
            description="Tendon sequence for the left mass with pulley",
            name=f"{self.name}.left_tendon"
        )
        root_sequence.add_child(left_tendon_sequence)
        
        # Create child sequence for movable pulley
        pulley_tendon_sequence = TendonSequence(
            elements=movable_pulley_sequence,
            description="Tendon sequence for the movable pulley",
            name=f"{self.name}.pulley_tendon"
        )
        root_sequence.add_child(pulley_tendon_sequence)
        
        # Create child sequence for right mass
        right_tendon_sequence = TendonSequence(
            elements=[self.right_pulley_below.site.create_spatial_site()] + right_tendon_sequence.get_elements(),
            description="Tendon sequence for the right mass with pulley",
            name=f"{self.name}.right_tendon"
        )
        root_sequence.add_child(right_tendon_sequence)
        
        if direction == ConnectingDirection.OUTER_TO_INNER:
            root_sequence.reverse()

        return [root_sequence]

    def get_connecting_tendon_sequence(
        self,
        direction: ConnectingDirection,
        connecting_point: ConnectingPoint = ConnectingPoint.TOP,
        connecting_point_seq_id: Optional[ConnectingPointSeqId] = None,
        use_sidesite: bool = False,
    ) -> TendonSequence:
        """
        Return the tendon connection sequence according to the connection point.
        Three connection points:
        - ConnectingPoint.TOP: Connect from the fixed pulley at the top to the movable pulley
        - ConnectingPoint.SIDE_1: Tendon sequence of the left mass block
        - ConnectingPoint.SIDE_2: Tendon sequence of the right mass
        """
        if connecting_point == ConnectingPoint.TOP:
            # Connecting sequence from fixed pulley above to movable pulley
            sequence = [
                self.fixed_pulley_above.site.create_spatial_site(),
                self.movable_pulley.mass_site.create_spatial_site(),
            ]
            if direction == ConnectingDirection.INNER_TO_OUTER:
                sequence.reverse()
            return TendonSequence(
                elements=sequence,
                description="Tendon sequence connecting fixed pulley above to movable pulley",
                name=f"{self.name}.top_connecting_tendon"
            )
        elif connecting_point == ConnectingPoint.SIDE_1:
            # Returns the tendon sequence of the left mass
            return self.left_mass_with_pulley.get_connecting_tendon_sequence(
                direction=direction
            )
        elif connecting_point == ConnectingPoint.SIDE_2:
            # Returns the tendon sequence of the right mass
            return self.right_mass_with_pulley.get_connecting_tendon_sequence(
                direction=direction
            )
        else:
            raise ValueError(f"Unsupported connecting_point: {connecting_point}")

    def to_xml(self) -> str:
        """
        Converts the entity and its components to an XML string.
        """
        body_xml = (
            f"""<body name="{self.name}" pos="{' '.join(map(str, self.pos))}">\n"""
        )
        body_xml += self.left_mass_with_pulley.to_xml() + "\n"
        body_xml += self.movable_pulley.to_xml() + "\n"
        body_xml += self.right_mass_with_pulley.to_xml() + "\n"
        body_xml += self.fixed_pulley_above.to_xml() + "\n"
        body_xml += self.left_pulley_below.to_xml() + "\n"
        body_xml += self.right_pulley_below.to_xml() + "\n"
        body_xml += "</body>"
        return body_xml

    def randomize_parameters(
        self,
        degree_of_randomization: DegreeOfRandomization = DegreeOfRandomization.DEFAULT,
        reinitialize_instance=False,
        **kwargs,
    ):
        # Randomize parameters
        self.mass_values = [round(random.uniform(0.1, 10.0), 2) for _ in range(2)]
        self.pulley_mass = round(random.uniform(0.1, 2.0), 2)
        if degree_of_randomization == DegreeOfRandomization.DEFAULT:
            # self.randomize_constant_forces([min(self.mass_values) / 2] * 3)
            self.mass_type = "Mass"  # Currently only "Mass" is supported
        else:
            pass

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
            "mass_values": self.mass_values,
            "mass_type": self.mass_type,
            "pulley_mass": self.pulley_mass,
        }
        if self.constant_force and len(self.constant_force) > 0:
            entity_dict["parameters"]["constant_force"] = self.constant_force
        return round_floats(entity_dict)

    def get_description(self, simDSL2nlq=False):
        # TODO: Implement the description for winding direction

        if not simDSL2nlq:
            return super().get_description(simDSL2nlq)

        descriptions = []
        descriptions += MassWithFixedPulley.get_mass_description(
            self.left_mass_with_pulley
        )
        descriptions += MassWithFixedPulley.get_mass_description(
            self.right_mass_with_pulley
        )

        m_pulley_description = {
            "name": self.movable_pulley.name,
            "body_type": "movable_pulley",
        }

        constant_frc_str = ""
        if (
            hasattr(self.movable_pulley, "constant_force_dict")
            and self.movable_pulley.name in self.movable_pulley.constant_force_dict
        ):
            constant_frc_str = f"A constant force of {self.movable_pulley.constant_force_dict[self.movable_pulley.name]} N acts on it."

        m_pulley_description["description"] = (
            f"A reversed movable pulley (string winds over the pulley) named {self.movable_pulley.name} has a mass {self.movable_pulley.mass_value} Kg."
            f" {constant_frc_str}"
        )

        descriptions += [m_pulley_description]

        return descriptions

    def get_nlq(self, symbolic = False):
        sym_dict = {}

        description = (
            f"In a system called '{self.name}', a movable pulley of mass {self.movable_pulley.mass_value} kg" 
            f" supports two blocks of masses {self.left_mass_with_pulley.mass.mass_value} kg and {self.right_mass_with_pulley.mass.mass_value} kg" 
            f" hanging on its left and right sides respectively, pulling the pulley down."
        )

        if symbolic:
            description = (
                f"In a system called '{self.name}', a movable pulley of mass <mass>1 kg"
                f" supports two blocks of masses <mass>2 kg and <mass>3 kg"
                f" hanging on its left and right sides respectively, pulling the pulley down."
            )
            sym_dict.update(
                {
                    "<mass>1": self.movable_pulley.mass_value,
                    "<mass>2": self.left_mass_with_pulley.mass.mass_value,
                    "<mass>3": self.right_mass_with_pulley.mass.mass_value,
                }
            )

        pulley_constant_frc_str = ""
        if (
            hasattr(self.movable_pulley, "constant_force_dict")
            and self.movable_pulley.name in self.movable_pulley.constant_force_dict
        ):
            pulley_constant_frc_str = f"An external force of {self.movable_pulley.constant_force_dict[self.movable_pulley.name]} N is applied on the pulley."

        # I assume there are no external forces applied on the blocks. Check and get back if this is true.

        description = f"{description} {pulley_constant_frc_str}"

        if symbolic: return description, sym_dict
        return description
    
    def connecting_point_nl(self, cd, cp, csi, first=False):
        """
        Return the connecting point description in natural language.
        
        Args:
            cd (ConnectingDirection): Connecting direction
            cp (ConnectingPoint): Connecting point
            csi (ConnectingPointSeqId): Connecting point sequence ID

        Returns:
            str: Connecting point description in natural language    
        """
        
        if cd == ConnectingDirection.INNER_TO_OUTER:
            description = (
                f"The string that connects the movable pulley in '{self.name}' to an overhead fixed pulley (pulling it upwards)"
                f" extends outward"
            )
        if cd == ConnectingDirection.OUTER_TO_INNER:
            description = (
                f"to connect to the movable pulley in '{self.name}' after wrapping over an overhead fixed pulley"
                f" (pulling the movable pulley upwards)."
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

        # "{side}_mass_with_fixed_pulley.mass", "movable_pulley"

        if '.' in sub_entity:
            sub_entity, body = tuple(sub_entity.split('.'))

            side = sub_entity.split('_')[0]

            question = (
                f"What is the {quantity} of the block hanging in the {side} side of the system '{self.name}'"
            )
        else:
            question = (
                f"What is the {quantity} of the movable pulley in the system '{self.name}'"
            )

        return question

class MassWithMovablePulley(Entity):

    # Define configuration parameters for different difficulty levels
    randomization_levels = {
        DegreeOfRandomization.EASY: {
            "num_pulleys_range": (1, 1),
            "mass_type_options": ["Mass"],  # EASY模式只允许基础Mass类型
            "pulley_mass_range": (0.1, 0.5),
            "mass_config": {  # 结构与其他难度级别统一
                "Mass": {  # 关键修改：添加Mass键
                    "num_masses": (1, 1),
                    "mass_range": (0.5, 3.0),
                    "decimal": 1,
                }
            },
            "plane_config": {"fixed_slope": 0},
            "prism_config": {"fixed_slopes": (30, 60)},
        },
        DegreeOfRandomization.MEDIUM: {
            "num_pulleys_range": (1, 3),  # 1-3 movable pulleys
            "mass_type_options": ["Mass", "MassPlane"],  # Allowing plane mass
            "pulley_mass_range": (0.2, 1.0),  # Medium pulley mass
            "mass_config": {
                "Mass": {
                    "num_masses": (1, 2),  # 1-2 masses
                    "mass_range": (0.5, 5.0),  # Medium mass range
                    "decimal": 2,
                },
                "MassPlane": {
                    "num_masses": (1, 2),  # 1-2 masses
                    "mass_range": (0.5, 8.0),
                    "slope_range": (10, 30),  # Medium slope range
                    "decimal": 2,
                },
            },
            "prism_config": {"slope_range": (20, 45)},  # Prism angle range
        },
        DegreeOfRandomization.HARD: {
            "num_pulleys_range": (2, 5),  # 2-5 movable pulleys
            "mass_type_options": ["Mass", "MassPlane", "MassPrismPlane"],
            "pulley_mass_range": (0.5, 2.0),  # Heavy pulley
            "mass_config": {
                "Mass": {
                    "num_masses": (1, 3),  # Up to 3 masses
                    "mass_range": (0.1, 10.0),
                    "decimal": 2,
                },
                "MassPlane": {
                    "num_masses": (2, 4),  # 2-4 masses
                    "mass_range": (0.1, 15.0),
                    "slope_range": (20, 50),  # Steep slope range
                    "decimal": 2,
                },
                "MassPrismPlane": {
                    "num_masses": (1, 2),
                    "mass_range": (0.1, 20.0),
                    "prism_slopes": (10, 70),  # Wide prism angle range
                    "min_diff": 15,  # Minimum slope difference
                    "decimal": 2,
                },
            },
        },
    }

    def __init__(
        self,
        name: str,
        pos: Tuple[float, float, float],
        num_pulleys: int = 1,
        mass_type: str = "Mass",  # "Mass", "MassPlane", or "MassPrismPlane"
        mass_values: List[float] = [
            1.0
        ],  # mass value of the block on the plane or prism
        prism_mass_value: float = 1,  # mass value of the prism
        use_left_site: DirectionsEnum = DirectionsEnum.USE_LEFT,
        use_prism_left: bool = True,
        plane_slope: float = 0,  # degrees
        prism_left_slope: float = 30,  # degrees
        prism_right_slope: float = 60,  # degrees
        pulley_mass=0.1,
        condim: str = "1",  # 1 means frictionless
        constant_force: Optional[Dict[str, List[Union[List, float]]]] = None,
        init_randomization_degree: DegreeOfRandomization = None,
        **kwargs,
    ):
        num_pulleys = 1  # TEMPORARY< REMOVE LATER
        self.pulley_mass = pulley_mass
        self.num_pulleys = num_pulleys
        self.mass_type = mass_type
        self.mass_values = mass_values
        self.prism_mass_value = prism_mass_value
        self.use_left_site = use_left_site
        self.plane_slope = plane_slope
        self.prism_left_slope = prism_left_slope
        self.prism_right_slope = prism_right_slope
        self.condim = condim

        super().__init__(
            name=name,
            pos=pos,
            entity_type=self.__class__.__name__,
            constant_force=constant_force,
            init_randomization_degree=init_randomization_degree,
            **kwargs,
        )
        self.mass_site_positions = []
        self.movable_pulleys = []
        self.fixed_pulleys = []

        # Spacing between movable pulleys, 2r for movable, 2r for fix
        spacing = 4 * DEFAULT_PULLEY_RADIUS

        last_offset = 0

        # Create evenly spaced movable pulleys
        for i in range(num_pulleys):
            offset = (i - num_pulleys // 2) * spacing
            pulley = MovablePulley(
                f"{name}.movable_pulley-{i}",
                (offset, 0, 0),
                mass=self.pulley_mass,
                constant_force=(
                    {ConstantForceType.PULLEY: constant_force[ConstantForceType.PULLEY]}
                    if constant_force and ConstantForceType.PULLEY in constant_force
                    else None
                ),
            )
            self.movable_pulleys.append(pulley)
            # Add the mass site position for the tendon connects the mass and the movable pulley
            self.mass_site_positions.append((offset, 0, -DEFAULT_MASS_SIZE))
            self.add_site(pulley.mass_site)
            self.add_site(pulley.tendon_site)

            # Create a fixed pulley above every two movable pulleys
            if i > 0:
                fixed_pulley = FixedPulley(
                    f"{name}.fixed_pulley-{i - 1}",
                    ((offset + last_offset) / 2, 0, DEFAULT_ROPE_LENGTH),
                )
                self.fixed_pulleys.append(fixed_pulley)
                self.add_site(fixed_pulley.left_site)
                self.add_site(fixed_pulley.right_site)

            last_offset = offset

        # Adding a site on the mass for the fixed pulley
        self.mass = create_mass_body(
            name=f"{name}.mass",
            mass_type=mass_type,
            positions=self.mass_site_positions,
            mass_values=mass_values,
            plane_slope=plane_slope,
            prism_left_slope=prism_left_slope,
            prism_right_slope=prism_right_slope,
            prism_mass_value=prism_mass_value,
            use_left_site=use_left_site,
            use_prism_left=use_prism_left,
            padding_z=DEFAULT_PULLEY_RADIUS + 2 * DEFAULT_MASS_SIZE,
            condim=condim,
            constant_force=constant_force,
        )

        # add two fixed pulley at the left and right of the mass
        left_fixed_pulley_offset = -(num_pulleys // 2) * spacing - DEFAULT_PULLEY_RADIUS
        right_fixed_pulley_offset = (
            num_pulleys - 1 - num_pulleys // 2
        ) * spacing + DEFAULT_PULLEY_RADIUS
        self.left_fixed_pulley = FixedPulley(
            f"{name}.left_fixed_pulley",
            (left_fixed_pulley_offset, 0, DEFAULT_ROPE_LENGTH),
            offset=0,
        )
        self.right_fixed_pulley = FixedPulley(
            f"{name}.right_fixed_pulley",
            (right_fixed_pulley_offset, 0, DEFAULT_ROPE_LENGTH),
            offset=0,
        )

    def get_ready_tendon_sequences(self, direction: ConnectingDirection) -> List[TendonSequence]:
        """
        Returns the tendon sequence that connects each mass to its corresponding pulley,
        along with the ready tendon sequences inside the mass system.
        """
        tendons = []
        
        # Additional tendons connecting each mass to each pulley
        mass_connecting_sequences = self.mass.get_connecting_tendon_sequences(ConnectingDirection.INNER_TO_OUTER)
        for i, pulley in enumerate(self.movable_pulleys):
            tendon_sequence = TendonSequence(
                elements=mass_connecting_sequences[i].get_elements() + [pulley.mass_site.create_spatial_site()],
                description=f"Tendon sequence connecting mass {i} to its pulley",
                name=f"{self.name}.mass_{i}_pulley_tendon"
            )
            tendons.append(tendon_sequence)
        
        # Add ready tendons inside mass system
        mass_ready_sequence = self.mass.get_ready_tendon_sequences(direction)
        tendons.extend(mass_ready_sequence)
        
        return tendons

    def get_connecting_tendon_sequence(
        self,
        direction: ConnectingDirection,
        connecting_point: ConnectingPoint = ConnectingPoint.DEFAULT,
        connecting_point_seq_id: Optional[ConnectingPointSeqId] = None,
        use_sidesite: bool = False,
    ) -> TendonSequence:
        if (
            connecting_point == ConnectingPoint.DEFAULT
            or connecting_point == ConnectingPoint.LEFT
        ):
            # ConnectingPoint.LEFT means default connecting point of MassPlane or MassPrismPlane
            # it could be from either the left or right site of the plane
            movable_pulley_sequence = [
                self.left_fixed_pulley.site.create_spatial_site(),
                *self.movable_pulleys[0].generate_spatial_elements(
                    use_sidesite=use_sidesite
                ),
            ]
            for i in range(len(self.movable_pulleys) - 1):
                movable_pulley_sequence.append(
                    self.fixed_pulleys[i].left_site.create_spatial_site()
                )
                movable_pulley_sequence.append(
                    self.fixed_pulleys[i].right_site.create_spatial_site()
                )

                movable_pulley_sequence.extend(
                    self.movable_pulleys[i + 1].generate_spatial_elements(
                        use_sidesite=use_sidesite
                    )
                )
            movable_pulley_sequence.append(
                self.right_fixed_pulley.site.create_spatial_site()
            )
            if direction == ConnectingDirection.RIGHT_TO_LEFT:
                movable_pulley_sequence = movable_pulley_sequence[::-1]
            return TendonSequence(
                elements=movable_pulley_sequence,
                description=f"Tendon sequence connecting mass to its pulley",
                name=f"{self.name}.connecting_tendon"
            )

        elif connecting_point == ConnectingPoint.RIGHT:
            movable_pulley_sequence = []
            movable_pulley_sequence = self.mass.get_second_connecting_tendon_sequences(
                direction=direction
            )[
                0
            ]  # for right side, there will be only one tendon sequence
            return TendonSequence(
                elements=movable_pulley_sequence,
                description=f"Tendon sequence connecting mass to its pulley",
                name=f"{self.name}.connecting_tendon"
            )

        else:
            raise ValueError(f"Unsupported connecting_point: {connecting_point}")

    def to_xml(self) -> str:
        """
        Convert the body and its components to an XML string.
        """
        body_xml = (
            f"""<body name="{self.name}" pos="{' '.join(map(str, self.pos))}">\n"""
        )
        body_xml += self.left_fixed_pulley.to_xml()
        for movable_pulley in self.movable_pulleys:
            body_xml += movable_pulley.to_xml() + "\n"
        for fixed_pulley in self.fixed_pulleys:
            body_xml += fixed_pulley.to_xml() + "\n"
        body_xml += self.right_fixed_pulley.to_xml()
        body_xml += self.mass.to_xml() + "\n"
        body_xml += "</body>"
        return body_xml

    def randomize_parameters(
        self,
        degree_of_randomization: DegreeOfRandomization = DegreeOfRandomization.DEFAULT,
        reinitialize_instance: bool = False,
        **kwargs,
    ):

        # Define configuration parameters for different difficulty levels
        randomization_levels = {
            DegreeOfRandomization.EASY: {
                "num_pulleys_range": (1, 1),
                "mass_type_options": ["Mass"],  # EASY模式只允许基础Mass类型
                "pulley_mass_range": (0.1, 0.5),
                "mass_config": {  # 结构与其他难度级别统一
                    "Mass": {  # 关键修改：添加Mass键
                        "num_masses": (1, 1),
                        "mass_range": (0.5, 3.0),
                        "decimal": 1,
                    }
                },
                "plane_config": {"fixed_slope": 0},
                "prism_config": {"fixed_slopes": (30, 60)},
            },
            DegreeOfRandomization.MEDIUM: {
                "num_pulleys_range": (1, 3),  # 1-3 movable pulleys
                "mass_type_options": ["Mass", "MassPlane"],  # Allowing plane mass
                "pulley_mass_range": (0.2, 1.0),  # Medium pulley mass
                "mass_config": {
                    "Mass": {
                        "num_masses": (1, 2),  # 1-2 masses
                        "mass_range": (0.5, 5.0),  # Medium mass range
                        "decimal": 2,
                    },
                    "MassPlane": {
                        "num_masses": (1, 2),  # 1-2 masses
                        "mass_range": (0.5, 8.0),
                        "slope_range": (10, 30),  # Medium slope range
                        "decimal": 2,
                    },
                },
                "prism_config": {"slope_range": (20, 45)},  # Prism angle range
            },
            DegreeOfRandomization.HARD: {
                "num_pulleys_range": (2, 5),  # 2-5 movable pulleys
                "mass_type_options": ["Mass", "MassPlane", "MassPrismPlane"],
                "pulley_mass_range": (0.5, 2.0),  # Heavy pulley
                "mass_config": {
                    "Mass": {
                        "num_masses": (1, 3),  # Up to 3 masses
                        "mass_range": (0.1, 10.0),
                        "decimal": 2,
                    },
                    "MassPlane": {
                        "num_masses": (2, 4),  # 2-4 masses
                        "mass_range": (0.1, 15.0),
                        "slope_range": (20, 50),  # Steep slope range
                        "decimal": 2,
                    },
                    "MassPrismPlane": {
                        "num_masses": (1, 2),
                        "mass_range": (0.1, 20.0),
                        "prism_slopes": (10, 70),  # Wide prism angle range
                        "min_diff": 15,  # Minimum slope difference
                        "decimal": 2,
                    },
                },
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

            # 1. Generate number of pulleys
            self.num_pulleys = random.randint(*params["num_pulleys_range"])
            self.num_pulleys = 1 # TEMPORARY< REMOVE LATER

            # 2. Choose mass type
            self.mass_type = random.choice(params["mass_type_options"])

            # 3. Generate pulley mass
            self.pulley_mass = round(random.uniform(*params["pulley_mass_range"]), 2)

            # 4. Generate mass parameters
            if self.mass_type == "MassPrismPlane":
                type_config = params["mass_config"]["MassPrismPlane"]
                # Generate prism angle difference
                for _ in range(100):
                    left = round(random.uniform(*type_config["prism_slopes"]), 2)
                    right = round(random.uniform(*type_config["prism_slopes"]), 2)
                    if abs(left - right) >= type_config["min_diff"]:
                        self.prism_left_slope, self.prism_right_slope = left, right
                        break
            elif self.mass_type == "MassPlane":
                type_config = params["mass_config"]["MassPlane"]
                self.plane_slope = round(random.uniform(*type_config["slope_range"]), 2)
            else:
                type_config = params["mass_config"]["Mass"]

            # Generate number of masses
            num_masses = random.randint(*type_config["num_masses"])
            # Generate mass values
            self.mass_values = [
                round(
                    random.uniform(*type_config["mass_range"]), type_config["decimal"]
                )
                for _ in range(num_masses)
            ]

            # 5. Generate connection parameters
            self.use_left_site = random.choice(
                [DirectionsEnum.USE_LEFT, DirectionsEnum.USE_RIGHT]
            )
            self.prism_mass_value = round(random.uniform(0.5, 10.0), 2)

        # Unstructured randomization
        elif degree_of_randomization == DegreeOfRandomization.NON_STRUCTURAL:
            # Maintain structure, only adjust values
            self.mass_values = [
                max(0.1, round(m * random.uniform(0.9, 1.1), 2))
                for m in self.mass_values
            ]
            self.pulley_mass = max(
                0.1, round(self.pulley_mass * random.uniform(0.8, 1.2), 2)
            )
            if self.mass_type == "MassPlane":
                self.plane_slope = round(
                    max(0, self.plane_slope + random.uniform(-5, 5)), 2
                )
            elif self.mass_type == "MassPrismPlane":
                self.prism_left_slope = round(
                    max(0, self.prism_left_slope + random.uniform(-10, 10)), 2
                )
                self.prism_right_slope = round(
                    max(0, self.prism_right_slope + random.uniform(-10, 10)), 2
                )

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
            "num_pulleys": self.num_pulleys,
            "mass_type": self.mass_type,
            "mass_values": self.mass_values,
            "prism_mass_value": self.prism_mass_value,
            "plane_slope": self.plane_slope,
            "prism_left_slope": self.prism_left_slope,
            "prism_right_slope": self.prism_right_slope,
            "use_left_site": self.use_left_site.name,
            "pulley_mass": self.pulley_mass,
        }
        if self.constant_force and len(self.constant_force) > 0:
            entity_dict["parameters"]["constant_force"] = self.constant_force
        return round_floats(entity_dict)

    def get_description(self, simDSL2nlq=False):
        if not simDSL2nlq:
            return super().get_description(simDSL2nlq)

        descriptions = MassWithFixedPulley.get_mass_description(self)

        pulley_descriptions = []

        for i, pulley in enumerate(self.movable_pulleys):
            m_pulley_description = {
                "name": pulley.name,
                "body_type": "movable_pulley",
            }

            constant_frc_str = ""
            if (
                hasattr(pulley, "constant_force_dict")
                and pulley.name in pulley.constant_force_dict
            ):
                constant_frc_str = f"A constant force of {pulley.constant_force_dict[pulley.name]} N acts on it."

            m_pulley_description["description"] = (
                f"A movable pulley named {pulley.name} has a mass {pulley.mass_value} Kg."
                f" {constant_frc_str}"
            )

            pulley_descriptions.append(m_pulley_description)

            if i == len(self.movable_pulleys) - 1:
                continue

            pulley = self.fixed_pulleys[i]
            f_pulley_description = {
                "name": pulley.name,
                "body_type": "site",
            }
            f_pulley_description["description"] = (
                f"A fixed pulley / site is named {pulley.name}."
            )

            pulley_descriptions.append(f_pulley_description)

        return descriptions + pulley_descriptions

    def get_nlq(self, symbolic = False):
        sym_dict = {}
        
        description = (
            f"In a system called '{self.name}', a movable pulley of mass {self.movable_pulleys[0].mass_value} kg" 
            f" supports two systems hanging on its either side, pulling the pulley up." 
        )

        if symbolic:
            description = (
                f"In a system called '{self.name}', a movable pulley of mass <mass>1 kg"
                f" supports two systems hanging on its either side, pulling the pulley up."
            )
            sym_dict["<mass>1"] = self.movable_pulleys[0].mass_value

        if self.mass_type == "Mass":
            mass_description = f" It also supports a block of mass {self.mass.mass_value} kg hanging directly below it (pulling the pulley down)."
            if symbolic:
                mass_description = f" It also supports a block of mass <mass>2 kg hanging directly below it (pulling the pulley down)."
                sym_dict["<mass>2"] = self.mass.mass_value
            
            block_constant_frc_str = ""
            if (
                hasattr(self.mass, "constant_force_dict")
                and self.mass.name in self.mass.constant_force_dict
            ):
                block_constant_frc_str = f"An external force of {self.mass.constant_force_dict[self.mass.name]} N is applied on the block."

        elif self.mass_type == "MassPlane":
            mass_description = (
                f" It also supports a block of mass {sum(self.mass.mass_values)} kg resting on a plane inclined at {self.mass.plane_slope} degrees with the horizontal," 
                f" with a string (pulling the pulley down)."
            )
            if symbolic:
                mass_description = (
                    f" It also supports a block of mass <mass>3 kg resting on a plane inclined at <angle>1 degrees with the horizontal," 
                    f" with a string (pulling the pulley down)."
                )
                sym_dict.update(
                    {
                        "<mass>3": sum(self.mass.mass_values),
                        "<angle>1": self.mass.plane_slope
                    }
                )

            net_frc = None

            for mass in self.mass.masses:
                if (
                    hasattr(self.mass, "constant_force_dict")
                    and mass.name in mass.constant_force_dict
                ):
                    net_frc = (
                        mass.constant_force_dict[mass.name] 
                        if net_frc is None else 
                        [i+j for i,j in zip(net_frc, mass.constant_force_dict[mass.name])]                    
                    )
                    
            block_constant_frc_str = ""
            if net_frc is not None:
                block_constant_frc_str = f"An external force of {net_frc} N is applied on the block."

        else: # MassPrismPlane
            side = ['left', 'right'][self.mass.use_prism_left]
            use_left_site = self.mass.use_left_site == DirectionsEnum.USE_LEFT
            mass_description = (
                f" It also supports a subsytem where a block of mass {self.mass.mass.mass_value} kg resting on the {side} side" 
                f" of a movable prism of mass {self.mass.prism.mass_value} kg."
                # f" The prism rests on a plane inclined at {self.mass.plane_slope} degrees with the horizontal."
                f" The prism makes an angle {self.mass.mass_slope} degrees with the plane, which in turn makes an angle {self.mass.plane_slope} degrees with horizontal."
                f" The string connects the movable pulley (pulling it down) to the {['right', 'left'][use_left_site]} side of the prism."
            )

            if symbolic:
                mass_description = (
                    f" It also supports a subsytem where a block of mass <mass>4 kg resting on the {side} side"
                    f" of a movable prism of mass <mass>5 kg."
                    # f" The prism rests on a plane inclined at <angle>2 degrees with the horizontal."
                    f" The prism makes an angle <angle>3 degrees with the plane, which in turn makes an angle <angle>2 degrees with horizontal."
                    f" The string connects the movable pulley (pulling it down) to the {['right', 'left'][use_left_site]} side of the prism."
                )
                sym_dict.update(
                    {
                        "<mass>4": self.mass.mass.mass_value,
                        "<mass>5": self.mass.prism.mass_value,
                        "<angle>2": self.mass.plane_slope,
                        "<angle>3": self.mass.mass_slope
                    }
                )
                    
            block_constant_frc_str = ""
            if hasattr(self.mass.mass, "constant_force_dict") and self.mass.mass.name in self.mass.mass.constant_force_dict:
                block_constant_frc_str = f"An external force of {self.nass.mass.constant_force_dict[self.mass.mass.name]} N is applied on the block."

        pulley_constant_frc_str = ""
        if (
            hasattr(self.movable_pulleys[0], "constant_force_dict")
            and self.movable_pulleys[0].name in self.movable_pulleys[0].constant_force_dict
        ):
            pulley_constant_frc_str = f"An external force of {self.movable_pulleys[0].constant_force_dict[self.movable_pulleys[0].name]} N is applied on the pulley."

        description = f"{description} {mass_description} {pulley_constant_frc_str} {block_constant_frc_str}"
        
        if symbolic: return description, sym_dict

        return description
    
    def connecting_point_nl(self, cd, cp, csi, first=False):
        """
        Return the connecting point description in natural language.
        
        Args:
            cd (ConnectingDirection): Connecting direction
            cp (ConnectingPoint): Connecting point
            csi (ConnectingPointSeqId): Connecting point sequence ID

        Returns:
            str: Connecting point description in natural language    
        """

        if cp == ConnectingPoint.DEFAULT:
            opening = f"to wrap"
            ending = f" to connect to another system."
            if first:
                opening = f"The string that wraps"
                ending = f""
            
            description = (
                f"{opening} around the movable pulley in '{self.name}' and further extends to" 
                f" its {['left', 'right'][cd == ConnectingDirection.LEFT_TO_RIGHT]} side{ending}"
            )
        else:
            modifier = f"block hanging"
            pull_modifier = "downwards"
            if self.mass_type == "MassPlane":
                use_left_site = self.mass.use_left_site == DirectionsEnum.USE_LEFT
                modifier = "block resting on the inclined plane"
                pull_modifier = ["left", "right"][use_left_site]
            elif self.mass_type == "MassPrismPlane":
                use_left_site = self.mass.use_left_site == DirectionsEnum.USE_LEFT
                modifier = "prism on the inclined plane"
                pull_modifier = ["left", "right"][use_left_site]


            if cd == ConnectingDirection.INNER_TO_OUTER:
                description = (
                    f"A string connected to the {modifier} under the movable pulley in '{self.name}' extends outward (pulling the {modifier.split(' ')[0]} {pull_modifier})"
                )
            else:
                description = (
                    f"to connect to the {modifier} under the movable pulley in '{self.name}' (pulling the {modifier.split(' ')[0]} {pull_modifier})."
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

        if sub_entity == "movable_pulley-0":
            question = (
                f"What is the {quantity} of the movable pulley in the system '{self.name}'"
            )
        else:
            # It is the mass.
            if sub_entity == "mass":
                question = (
                    f"What is the {quantity} of the block hanging directly below the movable pulley in the system '{self.name}'"
                )
            elif sub_entity[:15] == "mass_plane.mass":
                idx = int(sub_entity[15:])

                modifier = ""
                if quantity == "net_force":
                    modifier = f" x {self.mass_values[idx]} / {sum(self.mass_values)}"

                question = (
                    f"What is the {quantity}{modifier} of the block resting on the plane in the system '{self.name}'"
                )
            else:
                body = sub_entity.split('.')[-1]
                if body == "mass": body = "block"
                question = (
                    f"What is the {quantity} of the {body} in the system '{self.name}'"
                )

        return question

    def get_shortcut(self):
        if isinstance(self.mass, MassPrismPlane):
            # self.mass.prism.add_joint(Joint('fixed', (0,0,1), f'{self.mass.prism.name}.shortcut_fixed_joint'))
            self.mass.prism.joints = []
            return True
        return False

class PulleyGroupEntity(Entity):
    def __init__(
        self,
        name: str,
        pos: Tuple[float, float, float],
        num_movable_pulleys: int = 2,
        num_fixed_pulleys: int = 2,
        starting_point: PulleyGroupEntityStartPoint = PulleyGroupEntityStartPoint.TOP_FIXED_PULLEY,
        # 'top_fixed_pulley' or 'bottom_movable_pulley_top_site'
        pulley_spacing: float = None,  # 0.5 Adjust the spacing between pulleys
        pulley_radius: float = DEFAULT_PULLEY_RADIUS,
        rope_length: float = DEFAULT_ROPE_LENGTH,
        pulley_mass: float = 1,
        constant_force: Optional[Dict[str, List[Union[List, float]]]] = None,
        init_randomization_degree: DegreeOfRandomization = None,
        **kwargs,
    ):
        self.num_movable_pulleys = num_movable_pulleys
        self.num_fixed_pulleys = num_fixed_pulleys
        self.starting_point = starting_point
        if pulley_spacing is None:
            pulley_spacing = 1 / num_movable_pulleys

        self.pulley_spacing = pulley_spacing
        self.pulley_radius = pulley_radius
        self.rope_length = rope_length
        self.pulley_mass = pulley_mass
        super().__init__(
            name=name,
            pos=pos,
            entity_type=self.__class__.__name__,
            constant_force=constant_force,
            init_randomization_degree=init_randomization_degree,
            **kwargs,
        )
        self.movable_pulleys = []
        self.fixed_pulleys = []

        # Create movable pulleys
        for i in range(num_movable_pulleys):
            y_offset = i * pulley_spacing
            pulley_name = f"{name}.movable_pulley-{i}"
            pulley_pos = (0, y_offset, 0)
            movable_pulley = ComplexMovablePulley(
                name=pulley_name,
                pos=pulley_pos,
                pulley_radius=pulley_radius,
                rope_length=rope_length,
                pulley_mass=self.pulley_mass,
            )
            self.movable_pulleys.append(movable_pulley)

        # Create fixed pulleys
        for i in range(num_fixed_pulleys):
            y_offset = i * pulley_spacing
            pulley_name = f"{name}.fixed_pulley-{i}"
            pulley_pos = (0, y_offset, rope_length)
            fixed_pulley = FixedPulleyEntity(name=pulley_name, pos=pulley_pos)
            self.fixed_pulleys.append(fixed_pulley)

        # Create external fixed pulley
        external_pulley_pos = (
            0,
            max(num_fixed_pulleys, num_movable_pulleys) * pulley_spacing,
            rope_length,
        )
        self.external_fixed_pulley = FixedPulleyEntity(
            name=f"{name}.external_fixed_pulley", pos=external_pulley_pos
        )

    def get_ready_tendon_sequences(self, direction: ConnectingDirection) -> List[TendonSequence]:
        """
        Returns the tendon sequence that connects consecutive movable pulleys.
        """
        tendons = []
        
        for i in range(len(self.movable_pulleys) - 1):
            first_pulley = self.movable_pulleys[i]
            second_pulley = self.movable_pulleys[i + 1]
            
            # Get the bottom connection point of the first movable pulley
            bottom_site = first_pulley.bottom_connect_site.create_spatial_site()
            # Get the top connection point of the second movable pulley
            top_site = second_pulley.top_connect_site.create_spatial_site()
            
            # Create connection sequence
            sequence = [bottom_site, top_site]
            
            if direction == ConnectingDirection.OUTER_TO_INNER:
                sequence.reverse()
            
            tendon_sequence = TendonSequence(
                elements=sequence,
                description=f"Tendon sequence connecting pulley {i} to pulley {i+1}",
                name=f"{self.name}.pulley_{i}_to_pulley_{i+1}"
            )
            tendons.append(tendon_sequence)
        
        return tendons

    def get_connecting_tendon_sequence(
        self,
        direction: ConnectingDirection,
        connecting_point: ConnectingPoint = ConnectingPoint.DEFAULT,
        connecting_point_seq_id: Optional[ConnectingPointSeqId] = None,
        use_sidesite: bool = False,
    ) -> TendonSequence:
        sequence = []

        if self.starting_point == PulleyGroupEntityStartPoint.TOP_FIXED_PULLEY:
            # Start from the first fixed pulley at the top
            for i in range(max(self.num_fixed_pulleys, self.num_movable_pulleys)):
                if i < self.num_fixed_pulleys:
                    # Add the connection point of the fixed pulley
                    fixed_pulley = self.fixed_pulleys[i]
                    sequence.extend(
                        fixed_pulley.get_connecting_tendon_sequence(direction)
                    )
                if i < self.num_movable_pulleys:
                    # Add the default connection point of the movable pulley
                    movable_pulley = self.movable_pulleys[i]
                    sequence.extend(
                        movable_pulley.get_connecting_tendon_sequence(
                            direction, ConnectingPoint.DEFAULT
                        )
                    )
            # Add the external fixed pulley
            sequence.extend(
                self.external_fixed_pulley.get_connecting_tendon_sequence(direction)
            )
        elif (
            self.starting_point
            == PulleyGroupEntityStartPoint.BOTTOM_MOVABLE_PULLEY_TOP_SITE
        ):
            # Start from the top connection point of the first movable pulley at the bottom
            for i in range(max(self.num_movable_pulleys, self.num_fixed_pulleys)):
                if i < self.num_movable_pulleys:
                    # Add the top connection point of the movable pulley
                    movable_pulley = self.movable_pulleys[i]
                    sequence.extend(
                        movable_pulley.get_connecting_tendon_sequence(
                            direction, ConnectingPoint.TOP
                        )
                    )
                if i < self.num_fixed_pulleys:
                    # Add the connection point of the fixed pulley
                    fixed_pulley = self.fixed_pulleys[i]
                    sequence.extend(
                        fixed_pulley.get_connecting_tendon_sequence(direction)
                    )
                if i < self.num_movable_pulleys:
                    # Add the default connection point of the movable pulley
                    movable_pulley = self.movable_pulleys[i]
                    sequence.extend(
                        movable_pulley.get_connecting_tendon_sequence(
                            direction, ConnectingPoint.DEFAULT
                        )
                    )
            # Add the external fixed pulley
            sequence.extend(
                self.external_fixed_pulley.get_connecting_tendon_sequence(direction)
            )
        else:
            raise ValueError(f"Unsupported starting_point: {self.starting_point}")

        if direction == ConnectingDirection.OUTER_TO_INNER:
            sequence = sequence[::-1]

        return TendonSequence(
            elements=sequence,
            description=f"Tendon sequence connecting pulley {i} to pulley {i+1}",
            name=f"{self.name}.connecting_tendon"
        )

    def to_xml(self) -> str:
        """
        Generate an XML string for the entity and its components.
        """
        body_xml = (
            f"""<body name="{self.name}" pos="{' '.join(map(str, self.pos))}">\n"""
        )
        for movable_pulley in self.movable_pulleys:
            body_xml += movable_pulley.to_xml() + "\n"
        for fixed_pulley in self.fixed_pulleys:
            body_xml += fixed_pulley.to_xml() + "\n"
        body_xml += self.external_fixed_pulley.to_xml() + "\n"
        body_xml += "</body>"
        return body_xml

    def randomize_parameters(
        self,
        degree_of_randomization: DegreeOfRandomization = DegreeOfRandomization.DEFAULT,
        reinitialize_instance=False,
        **kwargs,
    ):
        if degree_of_randomization == DegreeOfRandomization.DEFAULT:
            self.num_movable_pulleys = random.randint(1, 5)
            self.num_fixed_pulleys = random.randint(1, 5)
            self.starting_point = random.choice(
                [
                    PulleyGroupEntityStartPoint.TOP_FIXED_PULLEY,
                    PulleyGroupEntityStartPoint.BOTTOM_MOVABLE_PULLEY_TOP_SITE,
                ]
            )
        # self.pulley_spacing = round(random.uniform(0.1, 1.0), 2)
        # self.pulley_radius = round(random.uniform(0.05, 0.5), 3)
        # self.rope_length = round(random.uniform(0.5, 5.0), 2)
        self.pulley_mass = round(random.uniform(0.1, 2.0), 2)

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
            "num_movable_pulleys": self.num_movable_pulleys,
            "num_fixed_pulleys": self.num_fixed_pulleys,
            "starting_point": self.starting_point,
            "pulley_mass": self.pulley_mass,
        }

        return round_floats(entity_dict)
