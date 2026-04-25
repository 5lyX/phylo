from .base_entities import *
from sim.bodies.mass import Mass
from sim.utils import replace_all

class ThrowingMotionEntity(Entity):
    """
    ThrowingMotionEntity: Used to demonstrate a simple horizontal/projectile motion.
    
    Parameters:
        name (str): Entity name
        initial_angular_velocity (float): Initial angular velocity of the ball (rad/s)
        initial_height (float): Initial height of the ball (m)
        initial_speed (float): Initial speed magnitude of the ball (m/s)
        initial_angle (float): Initial projection angle of the ball (°), angle with the x-axis
        ball_mass (float): Mass of the ball
        ball_radius (float): Radius of the ball
        pos (tuple): World reference position of the entire entity
        quat (tuple): World reference quaternion of the entire entity
        other (kwargs): Other optional parameters

    Note:
        - If initial_angular_velocity is not 0, the ball will rotate and fall down. In this case, the initial_angle and initial_speed are ignored.
        - If initial_angular_velocity is 0, the ball will move in a straight line.
    """

    randomization_levels = {
        DegreeOfRandomization.EASY: {
            "angle_choices": [0, 90, -90],          # Fixed horizontal / vertical projection
            "height_range": (1.0, 2.0),
            "speed_range": (1.0, 3.0),
            "angular_velocity_range": (0.0, 0.0),
            "coefficient_of_friction": (0.0, 0.0),  # Not used in this difficulty level
            "coefficient_of_restitution": (1.0, 1.0),  
        },
        DegreeOfRandomization.MEDIUM: {
            "angle_choices": [30, 45, 60, -30, -45, -60],  
            "height_range": (1.0, 3.0),
            "speed_range": (2.0, 5.0),
            "angular_velocity_range": (0.0, 0.0),
            "coefficient_of_friction": (0.0, 0.0),  # Not used in this difficulty level
            "coefficient_of_restitution": (0, 1.0),
        },
        DegreeOfRandomization.HARD: {
            "angle_range": (10.0, 80.0),
            "height_range": (1.0, 5.0),
            "speed_range": (3.0, 10.0),
            "angular_velocity_range": (-1.0, 1.0),
            "coefficient_of_friction": (0.2, 0.7),  # Rolling questions
            "coefficient_of_restitution": (0, 1.0),
        },
    }

        

    def __init__(
        self,
        name: str,
        initial_height: float = 0,
        initial_speed: float = 0,
        initial_angle: float = 0,
        initial_angular_velocity: float = 0,
        ball_mass: float = 1.0,
        ball_radius: float = 0.1,
        pos=(0, 0, 0),
        quat=(1, 0, 0, 0),
        coefficient_of_friction: float = 0.0,
        resolution_coefficient_list: list = None,
        **kwargs,
    ):
        # Store these parameters for later use in randomize or YAML export
        self.initial_height = initial_height
        self.initial_speed = initial_speed
        self.initial_angle = initial_angle
        self.ball_mass = ball_mass
        self.ball_radius = ball_radius
        self.initial_angular_velocity = initial_angular_velocity

        if resolution_coefficient_list is not None:  # save for description
            self.resolution_coefficient_list = [
                (
                    f"{name}.ball",
                    f"{name}.plane",
                    rc[2],
                ) for rc in resolution_coefficient_list
            ]

        
        
        super().__init__(
            name=name,
            pos=pos,
            quat=quat,
            entity_type=self.__class__.__name__,
            **kwargs,
        )


        # Create child bodies: ball + plane
        self._create_bodies()
        self.friction_type = FrictionType.ROLLING

        self.coefficient_of_friction = coefficient_of_friction
        if self.coefficient_of_friction > 1e-2:
            self.friction_coefficient_list = [
                (
                    f"{self.name}.ball.geom",
                    f"{self.name}.plane.geom",
                    self.coefficient_of_friction,
                )
            ]
        
        self.trail_bodies = [(f"{self.name}.ball", 1600)]

    def _create_bodies(self):
        """
        Internal function: Create a ball (with initial velocity) and ground plane based on current parameters.
        This can also be called after randomize_parameters if rebuilding is needed.
        """
        # Clear old child bodies (if any)
        self.child_bodies.clear()
        # - If initial_angular_velocity is not 0, the ball will rotate and fall down. In this case, the initial_angle and initial_speed are ignored.
        # - If initial_angular_velocity is 0, the ball will move in a straight line.
        if self.initial_angular_velocity != 0:
            velocity_dict = {
                InitVelocityType.SPHERE: [0, 0, 0, 0, self.initial_angular_velocity, 0]
            }
        else:
            # 1) Calculate (vx, vz) components
            angle_radians = math.radians(self.initial_angle)
            vx = self.initial_speed * math.cos(angle_radians)
            vz = self.initial_speed * math.sin(angle_radians)
            vy = 0.0

            # Prepare an init_velocity for assigning initial velocity at the lower level
            velocity_dict = {
                InitVelocityType.SPHERE: [vx, vy, vz, 0, 0, 0]
            }

        # 2) Create the ball, pos=(0,0, initial_height)
        self.ball = Sphere(
            name=f"{self.name}.ball",
            pos=(0, 0, self.initial_height + DEFAULT_PLANE_THICKNESS + self.ball_radius),  # z=DEFAULT_PLANE_THICKNESS is the ground
            radius=self.ball_radius,
            mass=self.ball_mass,
            init_velocity=velocity_dict,
            joint_option=("free", (0, 0, 0)),  
        )

        # 3) Create the ground (z=0)
        self.ground = Plane(
            name=f"{self.name}.plane",
            pos=(0, 0, 0),
            size=(DEFAULT_PLANE_LENGTH, DEFAULT_PLANE_LENGTH, DEFAULT_PLANE_THICKNESS),   # Can be set arbitrarily
            quat=(1, 0, 0, 0),
            condim="1",
        )

        # 4) Add to the current Entity
        self.add_child_body(self.ball)
        self.add_child_body(self.ground)

    def to_xml(self) -> str:
        """
        (Optional) Output this entity and its child bodies as an XML string.
        """
        xml_str = f'<body name="{self.name}" pos="{self.pos[0]} {self.pos[1]} {self.pos[2]}" ' \
                  f'quat="{self.quat[0]} {self.quat[1]} {self.quat[2]} {self.quat[3]}">\n'
        xml_str += self.ball.to_xml() + "\n"
        xml_str += self.ground.to_xml() + "\n"
        xml_str += "</body>"
        return xml_str

    def randomize_parameters(
        self,
        degree_of_randomization: DegreeOfRandomization = DegreeOfRandomization.DEFAULT,
        reinitialize_instance: bool = False,
        **kwargs,
    ):
        """
        Randomize parameters like initial_height, initial_speed, initial_angle based on difficulty level.
        If reinitialize_instance=True, child bodies will be recreated after modifying the parameters.
        """
        # First define randomization strategies for different difficulty levels
        randomization_levels = self.randomization_levels

        # If degree_of_randomization=DEFAULT, randomly choose one
        if degree_of_randomization == DegreeOfRandomization.DEFAULT:
            degree_of_randomization = random.choice(
                [DegreeOfRandomization.EASY, DegreeOfRandomization.MEDIUM, DegreeOfRandomization.HARD]
            )

        if degree_of_randomization not in randomization_levels:
            # If a non-EASY/MEDIUM/HARD/DEFAULT value is passed, raise an error or handle it
            raise ValueError(f"Unsupported randomization level: {degree_of_randomization}")

        params = randomization_levels[degree_of_randomization]

        # Randomly generate projection angle
        if "angle_choices" in params:
            # EASY / MEDIUM: Randomly select from choices
            self.initial_angle = random.choice(params["angle_choices"])
        elif "angle_range" in params:
            # HARD: Random within [10,80] range
            a_min, a_max = params["angle_range"]
            self.initial_angle = round(random.uniform(a_min, a_max), 2)

        # Randomly generate initial angular velocity
        if "angular_velocity_range" in params:
            min_ang_vel, max_ang_vel = params["angular_velocity_range"]
            if min_ang_vel != 0 or max_ang_vel != 0:
                self.initial_angular_velocity = round(random.uniform(min_ang_vel, max_ang_vel), 2)
        
        # Randomly generate initial height
        h_min, h_max = params["height_range"]
        self.initial_height = round(random.uniform(h_min, h_max), 2)

        # Randomly generate initial speed
        v_min, v_max = params["speed_range"]
        self.initial_speed = round(random.uniform(v_min, v_max), 2)

        self.coefficient_of_friction = random.uniform(
            params["coefficient_of_friction"][0], params["coefficient_of_friction"][1]
        )

        coefficient_of_restitution = random.uniform(
            params["coefficient_of_restitution"][0], params["coefficient_of_restitution"][1]
        )
        self.resolution_coefficient_list = [
            (
                f"{self.name}.ball",
                f"{self.name}.plane",
                coefficient_of_restitution,
            )
        ]

        # If rebuilding child bodies is needed after parameter changes, call _create_bodies()
        if reinitialize_instance:
            self._create_bodies()

    def generate_entity_yaml(
        self,
        use_random_parameters: bool = False,
        degree_of_randomization: DegreeOfRandomization = DegreeOfRandomization.DEFAULT,
    ) -> dict:
        """
        Export the current entity's parameters for YAML/JSON serialization.
        
        If use_random_parameters=True, randomize_parameters will be called first before returning the result.
        """
        if use_random_parameters:
            self.randomize_parameters(degree_of_randomization, reinitialize_instance=False)

        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "position": list(self.pos),  # pos is a tuple, need to convert to list
            "parameters": {
                "initial_height": self.initial_height,
                "initial_speed": self.initial_speed,
                "initial_angle": self.initial_angle,
                "initial_angular_velocity": self.initial_angular_velocity,
                "ball_mass": self.ball_mass,
                "ball_radius": self.ball_radius,
                "coefficient_of_friction": self.coefficient_of_friction,
                "resolution_coefficient_list": self.resolution_coefficient_list,
            },
        }
    
    def get_nlq(self, symbolic = False):
        mass = "<mass>1"
        height = "<y>1"
        radius = "<radius>1"

        sym_dict = {
            mass: self.ball_mass,
            height: self.initial_height,
        }

        if self.initial_angular_velocity != 0:
            # If the ball is rotating, we don't need to consider angle and speed
            angular_velocity = "<vx>1"
            sym_dict[angular_velocity] = self.initial_angular_velocity

            description = (
                f"In a setup, a ball of mass {mass} kg and radius {radius} m is dropped from a height of {height} m from the ground. "
                f"The ball has an initial angular velocity of {angular_velocity} rad/s about the horizontal axis. "
            )

        else:
            angle = "<angle>1"
            v = "<vx>1"

            sym_dict.update(
                {
                    angle: abs(self.initial_angle),
                    v: self.initial_speed,
                }
            )
            
            angle_desc = ""
            if self.initial_angle == 0:
                angle_desc = "horizontally"
            elif self.initial_angle in [90, -90]:
                angle_desc = "vertically " + f"{['upward', 'downward'][self.initial_angle < 0]}"
            else:
                angle_desc = f"{['upward', 'downward'][self.initial_angle < 0]} at an angle of {angle} degrees with the horizontal"

            description = (
                f"In a setup, a ball of mass {mass} kg is thrown {angle_desc}. "
                f"The ball is launched from an initial height of {height} m from the ground and it has an initial speed of {v} m/s. "
                f"Assume that horizontal is the +x axis and vertical is +y upwards."
            )

        coef_restitution = "<restitution>1"
        sym_dict[coef_restitution] = self.resolution_coefficient_list[0][2]
        contact_description = (
            f" The coefficient of restitution between the ball and the ground is {coef_restitution}."
        )
        if self.coefficient_of_friction > 1e-2:
            coeff_fric = "<friction>1"
            sym_dict[coeff_fric] = self.coefficient_of_friction

            contact_description += (
                f" The coefficient of friction between the ball and the ground is {coeff_fric}."
            )

        description += contact_description

        if not symbolic:
            description = replace_all(description, sym_dict)
            return description
        
        return description, sym_dict
    
    def connecting_point_nl(self, cd, cp, csi):
        raise NotImplementedError("ThrowingMotionEntity is not supposed to have connections.")
    
    def get_question(self, sub_entity: str, quantity: str) -> str:
        """
        Get a question related to the entity
        
        Inputs:
            sub_entity: str
            quantity: str
            
        Returns:
            str
        """

        question = (
            f"What is the {quantity} of the ball"
        )
        
        return question
