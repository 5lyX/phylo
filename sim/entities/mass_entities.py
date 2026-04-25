from .pulley_entities import *
import ipdb

st = ipdb.set_trace


class DirectedMass(Entity):
    """
    A specialized Body class to represent a mass element that can only move in a specific direction
    with up to two fixed pulleys attached.
    """

    def __init__(
        self,
        name: str,
        pos: Tuple[float, float, float],
        mass_type: str = "Mass",
        mass_value: float = 10,
        joint_option: Tuple[str, Tuple[float, float, float]] = ("slide", (1, 0, 0)),
        pulley_param1: Optional[PulleyParam] = None,
        pulley_param2: Optional[PulleyParam] = None,
        constant_force: Optional[Dict[str, List[Union[List, float]]]] = None,
        init_randomization_degree: DegreeOfRandomization = None,
        **kwargs,
    ) -> None:
        self.mass_type = mass_type
        self.mass_value = mass_value
        self.joint_option = joint_option
        self.pulley_param1 = pulley_param1
        self.pulley_param2 = pulley_param2
        super().__init__(
            name,
            pos,
            entity_type=self.__class__.__name__,
            constant_force=constant_force,
            init_randomization_degree=init_randomization_degree,
            **kwargs,
        )

        # Initialize the mass inside the class
        if mass_type == "Mass":
            self.mass = create_mass_body(
                name=f"{name}.mass",
                mass_type="Mass",
                positions=[(0, 0, 0)],
                joint_option=joint_option,
                mass_values=[mass_value],
                padding_z=0.05,
            )
        else:
            raise ValueError("Currently only supports mass_type 'Mass'")

        # Initialize pulleys based on pulley_param1 and pulley_param2
        self.fixed_pulleys = []
        self.to_connect = []  # List[Tuple[Site, Site]]

        if self.pulley_param1:
            self.add_pulley(
                angle=self.pulley_param1.angle,
                distance=self.pulley_param1.distance,
                side=self.pulley_param1.side,
                offset=self.pulley_param1.offset,
            )

        if self.pulley_param2:
            self.add_pulley(
                angle=self.pulley_param2.angle,
                distance=self.pulley_param2.distance,
                side=self.pulley_param2.side,
                offset=self.pulley_param2.offset,
            )

    def get_parameters(self) -> List[dict]:
        list_of_parameters = []
        mass_dict = self.mass.get_masses_quality()[0]
        # if self.pulley_param1:
        #     # mass_dict[f"{self.fixed_pulleys[0].name}.init_angle"] = self.pulley_param1.angle
        #     mass_dict[f"tendon_angle_1"] = [
        #         self.mass.sites[0].name,
        #         self.fixed_pulleys[0].site.name,
        #     ]
        # if self.pulley_param2:
        #     # mass_dict[f"{self.fixed_pulleys[1].name}.init_angle"] = self.pulley_param2.angle
        #     mass_dict[f"tendon_angle_2"] = [
        #         self.mass.sites[0].name,
        #         self.fixed_pulleys[1].site.name,
        #     ]
        mass_dict["use_tendon_angle"] = True

        list_of_parameters.append(mass_dict)
        return list_of_parameters

    def add_pulley(
        self,
        angle: float,  # degrees
        distance: float = 0,
        side: str = "top",  # "top", "bottom"
        offset: float = 0,
    ) -> None:
        """
        Calculate the position on the mass at the given angle (global coordinates) and add a fixed pulley there.
        """
        # Force remove offset
        offset = 0

        x = distance * math.cos(math.radians(angle))
        y = 0
        z = distance * math.sin(math.radians(angle))
        if side == "top":
            site = self.mass.sites[0]  # assume only one site on the top of mass
            site_pos = site.pos
            
        else:
            site = self.mass.bottom_site  # adjust as needed
            site_pos = site.pos

        target_pos = (site_pos[0] + x, site_pos[1] + y, site_pos[2] + z + offset)

        frame = Frame(origin=np.array(self.mass.pos), quat=np.array(self.mass.quat))
        global_pos = frame.rel2global(target_pos)

        pulley = FixedPulley(
            name=f"{self.name}.fixed_pulley-{len(self.fixed_pulleys)}",
            pos=global_pos,
            offset=0,
        )
        self.fixed_pulleys.append(pulley)
        self.to_connect.append((site, pulley.site))

    def get_connecting_tendon_sequence(
        self,
        direction: ConnectingDirection,
        connecting_point: ConnectingPoint = ConnectingPoint.DEFAULT,
        connecting_point_seq_id: Optional[ConnectingPointSeqId] = None,
        use_sidesite: bool = False,
    ) -> TendonSequence:
        """
        Get the tendon sequence for a specific connecting point.
        """
        if connecting_point in {ConnectingPoint.DEFAULT, ConnectingPoint.SIDE_1}:
            index = 0
        elif connecting_point == ConnectingPoint.SIDE_2:
            if len(self.to_connect) < 2:
                raise ValueError("Not enough tendon sequences for SIDE_2")
            index = 1
        else:
            raise ValueError(f"Unsupported connecting_point: {connecting_point}")
        
        site, fixed_pulley_site = self.to_connect[index]
        sequence = [
            site.create_spatial_site(),
            fixed_pulley_site.create_spatial_site(),
        ]
        
        if direction == ConnectingDirection.OUTER_TO_INNER:
            sequence.reverse()
        
        return TendonSequence(
            elements=sequence,
            description=f"Tendon sequence for {connecting_point}",
            name=f"{self.name}.connecting_tendon"
        )

    def to_xml(self) -> str:
        """
        Convert the directed mass to an XML string.
        """
        body_xml = (
            f"""<body name="{self.name}" pos="{' '.join(map(str, self.pos))}">\n"""
        )
        body_xml += self.mass.to_xml() + "\n"

        for fixed_pulley in self.fixed_pulleys:
            body_xml += fixed_pulley.to_xml() + "\n"
        body_xml += "</body>"
        return body_xml

    def randomize_parameters(
        self,
        degree_of_randomization: DegreeOfRandomization = DegreeOfRandomization.DEFAULT,
        reinitialize_instance=False,
        **kwargs,
    ):
        # No need to set randomization difficulty level because the structure is always similar
        # Randomize parameters
        mass_min = max(1, self.mass_value - 5)
        mass_max = self.mass_value + 5
        if mass_min > mass_max:
            mass_max = mass_min + 10
        self.mass_value = round(random.uniform(mass_min, mass_max), 2)
        if degree_of_randomization == DegreeOfRandomization.DEFAULT:
            # self.randomize_constant_forces(
            #     [self.mass_value / 2, self.mass_value / 2, self.mass_value / 2]
            # )
            self.joint_option = (
                "slide",
                random.choice([(1, 0, 0), (0, 1, 0), (0, 0, 1)]),
            )
            # Randomize pulley_param1 and pulley_param2
            self.pulley_param1 = PulleyParam(
                angle=random.uniform(30, 60),  # angle in degrees
                distance=random.uniform(0.1, 5.0),  # distance
                side=random.choice(["top", "bottom"]),  # side
                offset=random.uniform(-1.0, 1.0),  # offset
            )
            self.pulley_param2 = PulleyParam(
                angle=random.uniform(120, 150),
                distance=random.uniform(0.1, 5.0),
                side=random.choice(["top", "bottom"]),
                offset=random.uniform(-1.0, 1.0),
            )
        else:

            def get_quadrant(angle):
                return (int(angle) // 90) * 90

            def random_angle_in_same_quadrant(angle):
                # Determine the quadrant of the existing angle
                quadrant_start = get_quadrant(angle)
                padding = 30  # so not too close to the boundary
                return (
                    random.uniform(
                        quadrant_start + padding, quadrant_start + 90 - padding
                    ),
                    quadrant_start,
                )

            # Randomize pulley_param1 and pulley_param2
            if not self.pulley_param1:  # avoid uninitialized issue
                self.pulley_param1 = PulleyParam(
                    angle=random.uniform(10, 80),  # angle in degrees
                    distance=random.uniform(0.1, 5.0),  # distance
                    side=random.choice(["top", "bottom"]),  # side
                    offset=random.uniform(-1.0, 1.0),  # offset
                )
            if not self.pulley_param2:  # avoid uninitialized issue
                self.pulley_param2 = PulleyParam(
                    angle=random.uniform(100, 170),
                    distance=random.uniform(0.1, 5.0),
                    side=random.choice(["top", "bottom"]),
                    offset=random.uniform(-1.0, 1.0),
                )
            # angle_1, quadrant_1 = random_angle_in_same_quadrant(
            #     self.pulley_param1.angle
            # )

            # if self.pulley_param1:
            #     self.pulley_param1 = PulleyParam(
            #         angle=angle_1,
            #         distance=self.pulley_param1.distance * 10,
            #         side=self.pulley_param1.side,
            #         offset=self.pulley_param1.offset,
            #     )
            # if self.pulley_param2:
            #     quadrant_2 = get_quadrant(self.pulley_param2.angle)
            #     angle_2 = ((quadrant_2 + 90) - angle_1) + np.random.normal(
            #         loc=0, scale=5
            #     )
            #     self.pulley_param2 = PulleyParam(
            #         angle=angle_2,
            #         distance=self.pulley_param2.distance * 10,
            #         side=self.pulley_param2.side,
            #         offset=self.pulley_param2.offset,
            #     )

        # Re-initialize if needed
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

        entity_dict["parameters"]["mass_type"] = self.mass_type
        entity_dict["parameters"]["mass_value"] = self.mass_value
        if self.pulley_param1:
            entity_dict["parameters"]["pulley_param1"] = {
                "angle": self.pulley_param1.angle,
                "distance": self.pulley_param1.distance,
                "side": self.pulley_param1.side,
                "offset": self.pulley_param1.offset,
            }
        if self.pulley_param2:
            entity_dict["parameters"]["pulley_param2"] = {
                "angle": self.pulley_param2.angle,
                "distance": self.pulley_param2.distance,
                "side": self.pulley_param2.side,
                "offset": self.pulley_param2.offset,
            }
        if self.constant_force and len(self.constant_force) > 0:
            entity_dict["parameters"]["constant_force"] = self.constant_force
        return round_floats(entity_dict)

    def get_description(self, simDSL2nlq=False):
        if not simDSL2nlq:
            return super().get_description(simDSL2nlq)

        string_str = ""
        if self.pulley_param1 or self.pulley_param2:
            pulley_param = self.pulley_param1 or self.pulley_param2
            string_str = f" It is connected to a light string, making an angle of {pulley_param.angle} degrees with horizontal."

            if self.pulley_param1 and self.pulley_param2:
                string_str = f" It is connected to two light strings, making angle {self.pulley_param1.angle} and {self.pulley_param2.angle} degrees, respectively, with horizontal."

        constant_frc_string = ""
        if self.constant_force and ConstantForceType.MASS in self.constant_force:
            constant_frc = self.constant_force[ConstantForceType.MASS]
            constant_frc_string = f" The block is subject to a constant external force of {constant_frc} N."

        joint_str = ""
        if self.joint_option[0] == "slide":
            axis = self.joint_option[1][:-1]
            slope = np.rad2deg(np.arctan2(axis[1], axis[0]))
            joint_str = f"  It is constrained to only move along"
            if slope == 0:
                joint_str += f" the horizontal axis."
            elif slope == 90:
                joint_str += f" the vertical axis."
            else:
                joint_str += f" an axis making {slope} degrees with horizontal."

        descriptions = []
        description = {
            "name": f"{self.name}.mass",
            "type": "block",
            "mass": self.mass_value,
            "description": (
                f"A block named {self.name}.mass of mass {self.mass_value} Kg is suspended in air."  # TODO(Aryan): adjust description according to the new naming convention
                f"{joint_str}"
                f"{string_str}"
                f"{constant_frc_string}"
            ),
        }

        descriptions.append(description)

        return descriptions

    def get_nlq(self, symbolic = False)->str:
        mass= self.mass_value
        sym_dict = {}

        joint_str = "constrained to only move along"
        if self.joint_option[0] == "slide":
            axis = self.joint_option[1][:-1]
            slope = np.rad2deg(np.arctan2(axis[1], axis[0]))
            if slope == 0:
                joint_str += f" the horizontal axis."
            elif slope == 90:
                joint_str += f" the vertical axis."
            else:
                if not symbolic: joint_str += f" an axis making {slope} degrees with horizontal."
                else: 
                    joint_str += f" an axis making <angle>1 degrees with horizontal."
                    sym_dict['<angle1>'] = slope
        else:
            joint_str = "suspended in air."

        decription = f"a block of mass {mass} Kg is {joint_str}"
        if symbolic: 
            decription = f"a block of mass <mass>1 Kg is {joint_str}"
            sym_dict['<mass>1'] = mass

        string_str = ""
        if self.pulley_param1 or self.pulley_param2:
            pulley_param = self.pulley_param1 or self.pulley_param2

            d = pulley_param.distance

            side = "left"
            if pulley_param is self.pulley_param2:
                side = "right"
            string_str = f" A light string is connected on its {side} side, making an angle of {pulley_param.angle} degrees with horizontal for a length of {d} m before reaching a fixed pulley."
            if symbolic:
                string_str = f" A light string is connected on its {side} side, making an angle of <angle>2 degrees with horizontal for a length of {d} m before reaching a fixed pulley."
                sym_dict['<angle>2'] = pulley_param.angle

            if self.pulley_param1 and self.pulley_param2:
                d1, d2 = self.pulley_param1.distance, self.pulley_param2.distance
                string_str = f" It is connected to two light strings, making angle {self.pulley_param1.angle} and {self.pulley_param2.angle} degrees on its left and right sides, respectively, with horizontal for lengths {d1} and {d2} m before reaching fixed pulleys."
                if symbolic:
                    string_str = (
                            f" It is connected to two light strings, making angle <angle>3"
                            f" and <angle>4 degrees on its left and right sides, respectively, with horizontal"
                            f" for lengths {d1} and {d2} m before reaching fixed pulleys."
                    )
                    sym_dict['<angle>3'] = self.pulley_param1.angle
                    sym_dict['<angle>4'] = self.pulley_param2.angle

        constant_frc_string = ""
        if self.constant_force and ConstantForceType.MASS in self.constant_force:
            constant_frc = self.constant_force[ConstantForceType.MASS]
            constant_frc_string = f" The block is subject to a constant external force of {constant_frc} N."
            if symbolic:
                constant_frc_string = f" The block is subject to a constant external force of <force>1 N."
                sym_dict['<force>1'] = constant_frc

        if symbolic: return f"In a system called {self.name}, {decription}{string_str}{constant_frc_string}", sym_dict
        return f"In a system called {self.name}, {decription}{string_str}{constant_frc_string}"

        # raise NotImplementedError(f'Description not implemented for entity {self.__class__.__module__ + "." + self.__class__.__name__}')

    def connecting_point_nl(self, cd, cp, csi, first=False):
        """
        Returns the connecting point in natural language
        
        Inputs:
            cd: ConnectingDirection
            cp: ConnectingPoint
            csi: ConnectingPointSeqId  

        Returns:
            str
        """

        description = (
            f"{['right', 'left'][cp==ConnectingPoint.SIDE_1]} side of the block in '{self.name}'"
        )    

        if cd == ConnectingDirection.INNER_TO_OUTER:
            modifier = (
                f"The string connected to the {description} extends outwards"
            )
        else:
            modifier = (
                f"to connect to the {description}."
            )

        return modifier
    
    def get_question(self, sub_entity: str, quantity: str) -> str:
        """
        Get a question related to the entity
        
        Inputs:
            sub_entity: str
            quantity: str
            
        Returns:
            str
        """

        if '.' in sub_entity: raise ValueError(f"Sub-entity {sub_entity} of DirectedMass should not contain '.'")

        if sub_entity == 'mass':
            question = (
                f"What is the {quantity} of the block in '{self.name}'"
            )   
        else:
            raise KeyError(f"Sub-entity {sub_entity} not recognized for DirectedMass")

        return question
