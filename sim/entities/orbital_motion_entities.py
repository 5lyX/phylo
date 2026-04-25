from .base_entities import *
import pandas as pd
from tabulate import tabulate
from sim.utils import replace_all

class OrbitalMotionEntity(Entity):
    """
    OrbitalMotionEntity is used to create multiple spheres to simulate celestial or orbital motion.

    Args:
        name (str): Name of this entity.
        mass_list (List[float]): Masses of each celestial body.
                                 The i-th mass corresponds to the i-th sphere.
        radius_list (List[float]): Radii of each celestial body.
                                   The i-th radius corresponds to the i-th mass.
        reverse_influence (List[int]): A list of the same length as mass_list,
                                       indicating which smaller bodies exert reverse influence on larger bodies.
        init_velocities (List[float]): For each body, a single float that represents the velocity magnitude.
                                       We will compute the actual 3D velocity vector in `_create_sphere_bodies()`
                                       so that it is tangential to the radial offset from the previous sphere.
        relative_positions (List[Tuple[float, float]]): Each element is (r, theta_in_degrees),
                                                        representing how the i-th sphere is offset from the (i-1)-th sphere.
                                                        The first sphere is always placed at the origin.
                                                        Angles are in degrees; they will be converted to radians internally.
        pos (Tuple[float, float, float]): The world-base position of this entire entity (defaults to (0,0,0)).
        quat (Tuple[float, float, float, float]): Quaternion for entity's base orientation (defaults to identity).
    """

    randomization_levels = {
        DegreeOfRandomization.EASY: {
            "num_spheres_range": (1, 2),
            "mass_range": (0.5, 3.0),
            "radius_range": (0.1, 0.5),
            "velocity_float_range": (0.0, 1.0),   # single float for speed
            "distance_range": (0.0, 2.0),
            "theta_deg_range": (0.0, 360.0),      # degrees
        },
        DegreeOfRandomization.MEDIUM: {
            "num_spheres_range": (2, 4),
            "mass_range": (0.1, 5.0),
            "radius_range": (0.05, 1.0),
            "velocity_float_range": (0.0, 2.0),
            "distance_range": (0.0, 5.0),
            "theta_deg_range": (0.0, 360.0),
        },
        DegreeOfRandomization.HARD: {
            "num_spheres_range": (3, 6),
            "mass_range": (0.1, 10.0),
            "radius_range": (0.01, 2.0),
            "velocity_float_range": (0.0, 5.0),
            "distance_range": (0.0, 10.0),
            "theta_deg_range": (0.0, 360.0),
        },
    }

    G_constant = 1.0  # or 6.67430e-11, or any other constant

    def __init__(
        self,
        name: str,
        mass_list: List[float] = [1],
        radius_list: List[float] = [0.1],
        reverse_influence: List[int] = [0],
        init_velocities: List[float] = [0],
        relative_positions: List[Tuple[float, float]] = [(0, 0)],
        pos: Tuple[float, float, float] = (0, 0, 0),
        quat: Tuple[float, float, float, float] = (1, 0, 0, 0),
        **kwargs,
    ):

        # Basic parameter storage
        self.mass_list = mass_list
        self.radius_list = radius_list
        self.reverse_influence = reverse_influence
        self.init_velocities = init_velocities  # single float for each body
        self.relative_positions = relative_positions  # (r, theta_in_degrees)
        super().__init__(name, pos, quat, entity_type=self.__class__.__name__, **kwargs)

        # Validate lengths
        if not (
            len(self.mass_list)
            == len(self.radius_list)
            == len(self.reverse_influence)
            == len(self.init_velocities)
            == len(self.relative_positions)
        ):
            raise ValueError("All parameter lists must have the same length.")

        # Store created spheres
        self.created_bodies = []

        # Create spheres
        self._create_sphere_bodies()

        self.trail_bodies = [(f"{self.name}.sphere-{i}", 4000) for i in range(1, len(self.mass_list))]

    def _create_sphere_bodies(self):
        """
        1) Convert polar coordinates in degrees to absolute positions.
           - The first sphere is at origin (0,0,0).
           - For subsequent spheres, compute (r*cos(theta), r*sin(theta)) offset from the previous sphere.
        2) Convert each single-float velocity to a tangential 3D vector.
           - For sphere i > 0, the tangential direction is perpendicular to the radial offset from the (i-1)-th sphere.
             That direction is:
                 dx = -sin(theta_radians)
                 dy =  cos(theta_radians)
             multiplied by the velocity magnitude.
           - For the first sphere (i=0), we'll default to (0, 0, 0).
        """
        # Clear any existing children if reinitializing
        self.child_bodies.clear()
        self.created_bodies.clear()

        num_spheres = len(self.mass_list)
        # Prepare a list to store absolute positions in Cartesian
        absolute_positions = [(0.0, 0.0, 0.0)] * num_spheres

        # Compute absolute positions via polar offsets
        # The first sphere is placed at the origin by definition
        for i in range(1, num_spheres):
            r, theta_deg = self.relative_positions[i]
            theta_rad = math.radians(theta_deg)

            # Convert (r, theta_rad) to offset in Cartesian
            x_offset = r * math.cos(theta_rad)
            y_offset = r * math.sin(theta_rad)

            prev_x, prev_y, prev_z = absolute_positions[i - 1]
            # New absolute position for sphere i
            absolute_positions[i] = (prev_x + x_offset, prev_y + y_offset, prev_z)

        # Create each sphere with improved colors
        planet_colors = self._generate_planet_colors(num_spheres)
        
        for i in range(num_spheres):
            mass_value = self.mass_list[i]
            radius_value = self.radius_list[i]
            velocity_magnitude = self.init_velocities[i]
            abs_pos = absolute_positions[i]

            # Compute velocity vector in 2D plane (z=0), tangential to the radial offset
            if i == 0:
                # The first sphere is typically the reference (often at rest or center)
                velocity_3d = (0.0, 0.0, 0.0)
            else:
                # Convert degrees to radians
                _, theta_deg = self.relative_positions[i]
                theta_rad = math.radians(theta_deg)

                # A perpendicular (tangential) direction to (cos(theta), sin(theta)) is:
                #   dx = -sin(theta), dy = cos(theta)
                dx = -math.sin(theta_rad)
                dy = math.cos(theta_rad)
                velocity_3d = (velocity_magnitude * dx, velocity_magnitude * dy, 0.0)

            # Build an init_velocity dict for the sphere
            velocity_dict = {
                # If your code uses a different enum/string for sphere velocity, adjust accordingly
                InitVelocityType.SPHERE: velocity_3d
            }

            # Create the sphere body with dynamic colors
            sphere_body = Sphere(
                name=f"{self.name}.sphere-{i}",
                pos=abs_pos,
                radius=radius_value,
                mass=mass_value,
                init_velocity=velocity_dict,
                joint_option=("free", (0, 0, 0)),  # free joint for 3D motion
                rgba=planet_colors[i],  # Use dynamic colors
                material="reflectance",  # Use reflectance material for better color display
            )

            # Add to the entity hierarchy
            self.add_child_body(sphere_body)
            self.created_bodies.append(sphere_body)

    def _generate_planet_colors(self, num_spheres):
        """Generate elegant, scientifically-inspired colors for celestial bodies"""
        import random
        import math
        
        colors = []
        
        # Elegant celestial color palette inspired by real astronomical observations
        # More sophisticated and muted colors suitable for scientific papers
        planet_color_palette = [
            (0.9, 0.7, 0.5, 1.0),   # Warm golden (Sun-like, refined)
            (0.6, 0.7, 0.9, 1.0),   # Soft blue (Earth-like ocean)
            (0.8, 0.6, 0.4, 1.0),   # Muted rust (Mars-like, elegant)
            (0.9, 0.8, 0.6, 1.0),   # Pale gold (Jupiter-like, refined)
            (0.7, 0.6, 0.8, 1.0),   # Soft lavender (Ice world)
            (0.6, 0.8, 0.6, 1.0),   # Sage green (Forest world)
            (0.8, 0.7, 0.4, 1.0),   # Warm amber (Desert world)
            (0.6, 0.8, 0.8, 1.0),   # Soft cyan (Ice giant)
            (0.8, 0.6, 0.7, 1.0),   # Dusty rose (Exotic world)
            (0.7, 0.7, 0.6, 1.0),   # Warm grey (Rocky world)
            (0.6, 0.6, 0.8, 1.0),   # Soft periwinkle (Gas giant)
            (0.8, 0.8, 0.7, 1.0),   # Cream (Desert/Venus-like)
        ]
        
        for i in range(num_spheres):
            if i < len(planet_color_palette):
                base_color = planet_color_palette[i]
            else:
                # Generate sophisticated colors for additional planets
                hue = (i * 137.5) % 360
                # Use lower saturation and moderate value for scientific elegance
                saturation = 0.3 + 0.2 * random.random()  # More muted
                value = 0.6 + 0.2 * random.random()       # Not too bright
                h = hue / 60.0
                c = value * saturation
                x = c * (1 - abs((h % 2) - 1))
                m = value - c
                if h < 1:
                    r, g, b = c, x, 0
                elif h < 2:
                    r, g, b = x, c, 0
                elif h < 3:
                    r, g, b = 0, c, x
                elif h < 4:
                    r, g, b = 0, x, c
                elif h < 5:
                    r, g, b = x, 0, c
                else:
                    r, g, b = c, 0, x
                base_color = (r + m, g + m, b + m, 1.0)
            
            # Apply subtle distance-based gradient for orbital systems
            distance_factor = i / max(1, num_spheres - 1)  # 0 for first, 1 for last
            # Planets farther from star tend to be cooler (bluer/grayer)
            cooling_effect = distance_factor * 0.15
            gradient_color = (
                max(0.3, base_color[0] - cooling_effect * 0.5),  # Reduce red component
                max(0.3, base_color[1] - cooling_effect * 0.3),  # Slightly reduce green  
                min(0.9, base_color[2] + cooling_effect * 0.2),  # Slightly increase blue
                1.0
            )
            
            # Add very minimal variation for realism
            variation = 0.03  # Very small for elegance
            varied_color = (
                max(0.3, min(0.9, gradient_color[0] + (random.random() - 0.5) * variation)),
                max(0.3, min(0.9, gradient_color[1] + (random.random() - 0.5) * variation)),
                max(0.3, min(0.9, gradient_color[2] + (random.random() - 0.5) * variation)),
                1.0
            )
            colors.append(varied_color)
        
        return colors

    def get_attraction_forces(self) -> List[Tuple[str, str, str, float]]:
        """
        Return a list of tuples describing pairwise attraction forces in this entity,
        in the format: (body_A_name, body_B_name, force_type, gravitational_constant).
        Example: [("orbital_sphere_0", "orbital_sphere_1", "GRAVITY", 1.0), ...]

        If reverse_influence[i] == 1, we also add the reversed pair.
        By default, the gravitational constant is set to 1.0 for demonstration,
        but you can use e.g. 6.67430e-11 if simulating realistic physics.
        """
        G_constant = self.G_constant
        forces = []

        num_bodies = len(self.created_bodies)
        for i in range(num_bodies - 1):
            bodyA = self.created_bodies[i]
            bodyA_name = bodyA.name
            # By default, each body influences the next
            bodyB = self.created_bodies[i + 1]
            bodyB_name = bodyB.name

            forces.append((bodyA_name, bodyB_name, "GRAVITY", G_constant))

            # If reverse_influence[i+1] == 1, we also add B->A
            # i+1 is the index of bodyB in reverse_influence
            if self.reverse_influence[i + 1] == 1:
                forces.append((bodyB_name, bodyA_name, "GRAVITY", G_constant))

        return forces

    def get_parameters(self) -> List[dict]:
        """
        Returns a list of dictionaries describing the created spheres (names, masses, etc.).
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

    def generate_entity_yaml(
        self,
        use_random_parameters: bool = False,
        degree_of_randomization: DegreeOfRandomization = DegreeOfRandomization.NON_STRUCTURAL,
    ) -> dict:
        """
        Exports this entity's parameters to a dictionary for YAML/JSON serialization.
        """
        if use_random_parameters:
            self.randomize_parameters(degree_of_randomization)

        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "position": list(self.pos),
            "parameters": {
                "mass_list": self.mass_list,
                "radius_list": self.radius_list,
                "reverse_influence": self.reverse_influence,
                # init_velocities is now a list of floats
                "init_velocities": self.init_velocities,
                # We store (r, theta_in_degrees) as lists for each index
                "relative_positions": [list(p) for p in self.relative_positions],
            },
        }

    def to_xml(self) -> str:
        """
        Converts this entity (and its child bodies) to an XML representation suitable for a physics simulator.
        """
        body_xml = (
            f'<body name="{self.name}" '
            f'pos="{" ".join(map(str, self.pos))}" '
            f'quat="{" ".join(map(str, self.quat))}">'
        )

        # If the entity itself has joints, add them here
        for body in self.child_bodies:
            body_xml += body.to_xml() + "\n"

        body_xml += "</body>"
        return body_xml

    def randomize_parameters(
        self,
        degree_of_randomization: "DegreeOfRandomization" = None,
        reinitialize_instance: bool = False,
        **kwargs,
    ):
        """
        Randomizes core parameters (mass_list, radius_list, init_velocities, relative_positions),
        based on the selected difficulty: EASY, MEDIUM, HARD. Then optionally calls reinitialize().

        Note:
        - relative_positions angles are from 0 to 360 degrees (instead of 0 to 2*pi).
        - init_velocities is a single float per sphere, which will be converted to a 3D velocity
        vector in _create_sphere_bodies().
        - We do multiple retries to ensure all spheres do not overlap each other.
        """
        import math
        import random

        if degree_of_randomization is None:
            # Default to EASY if not specified
            degree_of_randomization = "EASY"

        randomization_levels = {
            DegreeOfRandomization.EASY: {
                "num_spheres_range": (1, 2),
                "mass_range": (0.5, 3.0),
                "radius_range": (0.1, 0.5),
                "velocity_float_range": (0.0, 1.0),   # single float for speed
                "distance_range": (0.0, 2.0),
                "theta_deg_range": (0.0, 360.0),      # degrees
            },
            DegreeOfRandomization.MEDIUM: {
                "num_spheres_range": (2, 4),
                "mass_range": (0.1, 5.0),
                "radius_range": (0.05, 1.0),
                "velocity_float_range": (0.0, 2.0),
                "distance_range": (0.0, 5.0),
                "theta_deg_range": (0.0, 360.0),
            },
            DegreeOfRandomization.HARD: {
                "num_spheres_range": (3, 6),
                "mass_range": (0.1, 10.0),
                "radius_range": (0.01, 2.0),
                "velocity_float_range": (0.0, 5.0),
                "distance_range": (0.0, 10.0),
                "theta_deg_range": (0.0, 360.0),
            },
        }

        # If "DEFAULT", randomly choose from [EASY, MEDIUM, HARD]
        if degree_of_randomization == "DEFAULT":
            degree_of_randomization = random.choice(["EASY", "MEDIUM", "HARD"])

        if degree_of_randomization not in randomization_levels:
            raise ValueError(f"Unsupported randomization level: {degree_of_randomization}")

        params = randomization_levels[degree_of_randomization]

        max_retries = 50  # maximum number of attempts to find a non-overlapping layout
        attempt = 0

        # We will store the "best" (or last) generated parameters below
        final_mass_list = None
        final_radius_list = None
        final_init_velocities = None
        final_relative_positions = None
        final_reverse_influence = None

        while attempt < max_retries:
            attempt += 1

            # 1. Generate a random number of spheres
            num_spheres = random.randint(*params["num_spheres_range"])

            # 2. Generate new mass_list
            new_mass_list = [
                round(random.uniform(*params["mass_range"]), 2)
                for _ in range(num_spheres)
            ]

            # 3. Generate new radius_list
            new_radius_list = [
                round(random.uniform(*params["radius_range"]), 3)
                for _ in range(num_spheres)
            ]

            # 4. Generate new init_velocities as a single float per sphere
            new_init_velocities = [
                round(random.uniform(*params["velocity_float_range"]), 2)
                for _ in range(num_spheres)
            ]

            # 5. Generate polar coordinates in degrees (relative positions)
            new_relative_positions = []
            for i in range(num_spheres):
                if i == 0:
                    # The first sphere at origin, r can be 0
                    r_min = 0.0
                else:
                    # Minimum distance from the previous sphere
                    r_min = new_radius_list[i - 1] + new_radius_list[i]

                r_max = params["distance_range"][1]
                if r_min > r_max:
                    # If the spheres are too large to fit the distance range
                    # We just clamp or adjust r_min
                    # Or you could break immediately
                    r_min = min(r_min, r_max * 0.9)

                r_val = round(random.uniform(r_min, r_max), 2)
                theta_deg = round(random.uniform(*params["theta_deg_range"]), 3)
                new_relative_positions.append((r_val, theta_deg))

            # 6. Generate reverse_influence
            new_reverse_influence = [
                random.choice([0, 1]) for _ in range(num_spheres)
            ]

            # -------------------------------------------------
            # Now check if all spheres overlap or not
            # We must convert the relative positions to absolute positions
            # to do a pairwise distance check
            # -------------------------------------------------
            absolute_positions = [(0.0, 0.0, 0.0)] * num_spheres
            for i in range(1, num_spheres):
                r, theta_deg = new_relative_positions[i]
                theta_rad = math.radians(theta_deg)
                x_offset = r * math.cos(theta_rad)
                y_offset = r * math.sin(theta_rad)
                prev_x, prev_y, prev_z = absolute_positions[i - 1]
                absolute_positions[i] = (prev_x + x_offset, prev_y + y_offset, prev_z)

            # Pairwise check
            any_overlap = False
            extra_gap = 0.0  # or 0.01, if you want a small safety margin
            for i in range(num_spheres):
                for j in range(i + 1, num_spheres):
                    x1, y1, z1 = absolute_positions[i]
                    x2, y2, z2 = absolute_positions[j]
                    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
                    if dist < (new_radius_list[i] + new_radius_list[j] + extra_gap):
                        # Overlap found, break and re-try
                        any_overlap = True
                        break
                if any_overlap:
                    break

            if not any_overlap:
                # Success: store these parameters and break
                final_mass_list = new_mass_list
                final_radius_list = new_radius_list
                final_init_velocities = new_init_velocities
                final_relative_positions = new_relative_positions
                final_reverse_influence = new_reverse_influence
                break
            else:
                # If overlap, continue to next attempt
                final_mass_list = new_mass_list
                final_radius_list = new_radius_list
                final_init_velocities = new_init_velocities
                final_relative_positions = new_relative_positions
                final_reverse_influence = new_reverse_influence

        if attempt >= max_retries:
            print("Warning: Could not find a non-overlapping configuration after max_retries, using the last generated set.")

        # Update the entity parameters
        self.mass_list = final_mass_list
        self.radius_list = final_radius_list
        self.init_velocities = final_init_velocities
        self.relative_positions = final_relative_positions
        self.reverse_influence = final_reverse_influence

        # Reinitialize if requested
        if reinitialize_instance:
            self.reinitialize()

    def get_description(self, simDSL2nlq=False):
        return super().get_description(simDSL2nlq)
    
    def get_nlq(self, symbolic = False):
        
        intro = (
            f"In a hypothetical celestial system, there are {len(self.created_bodies)} entities (spherical in shape), each orbiting around the previous one. "
        )

        frame = random.choice(["cartesian", "polar"])
        if frame == "cartesian":
            rel_pos = [(r * math.cos(math.radians(theta)), r * math.sin(math.radians(theta))) for r, theta in self.relative_positions]

            pos_list = [(0, 0)]
            for i, (x, y) in enumerate(rel_pos): pos_list.append((pos_list[-1][0] + x, pos_list[-1][1] + y))

            data = {
                "Entity": [i for i in range(1, len(self.created_bodies) + 1)],
                "Mass (M)": [f"<mass>{i}" for i in range(1, len(self.created_bodies) + 1)],
                "Radius (L)": [f"{r:.2f}" for r in self.radius_list],
                "Initial Position (L)": [f"({x:.2f}, {y:.2f})" for x, y in pos_list[1:]],
                "Initial Speed (L/T)": [f"{v:.2f}" for v in self.init_velocities],
            }

            try: 
                df = pd.DataFrame(data)
            except: st()
        else:
            data = {
                "Entity": [i for i in range(1, len(self.created_bodies) + 1)],
                "Mass (M)": [f"<mass>{i}" for i in range(1, len(self.created_bodies) + 1)],
                "Radius (L)": [f"{r:.2f}" for r in self.radius_list],
                "Relative Initial Position (L, degrees)": [f"({r:.2f}, {theta:.2f})" for r, theta in self.relative_positions],
                "Initial Speed (L/T)": [f"{v:.2f}" for v in self.init_velocities],
            }

            try:
                df = pd.DataFrame(data)
            except: st()

        properties_description = tabulate(df, headers='keys', tablefmt='github', showindex=False) + "\n"
        
        reverse_influence = [i+2 for i, ri in enumerate(self.reverse_influence) if ri == 1]
        reverse_influence_str = ""
        if len(reverse_influence) > 1: 
            reverse_influence_str = convert_list_to_natural_language(reverse_influence)

        general_info = (
            f"Each entity only applies gravitational force to the entity orbiting it. "
            f"" if len(reverse_influence) == 0 else f"Entities {reverse_influence_str} also apply gravitational force on the entity they are orbiting. "
            f"Each entities have initial velocity exactly along the tangential direction of their initial position vector. "
            f"The units M L T are custom units for mass, length and time, such that the gravitational constant G is {self.G_constant} in these units. "
            f"" if frame == "cartesian" else f"The relative positions of the entities are defined as position of the entity relative to the entity they are orbiting, and given in polar coordinates (r, theta). "
        )

        description = (
            f"{intro}\n"
            f"{properties_description}\n"
            f"{general_info}\n"
        )

        sym_dict = {
            f"<mass>{i+1}": m for i, m in enumerate(self.mass_list)
        }

        if not symbolic:
            description = replace_all(description, sym_dict)

            return description
        
        return description, sym_dict
    
    def connecting_point_nl(self, cd, cp, csi):
        raise NotImplementedError("OrbitalMotionEntity is not supposed to have connections.")
    
    def get_question(self, sub_entity: str, quantity: str) -> str:
        """
        Get a question related to the entity
        
        Inputs:
            sub_entity: str
            quantity: str
            
        Returns:
            str
        """

        descriptior = ""
        idx = int(sub_entity[7:]) # remove the "sphere-"
        descriptior = f"{(['1st', '2nd', '3rd'] + [f'{i + 4}th' for i in range(len(self.mass_list) - 3)])[idx]} entity"
        
        question = (
            f"What is the {quantity} of the {descriptior} in the celestial system"
        )
        
        return question

class GeneralCelestialEntity(Entity):
    """
    This entity class is used to simulate the orbital motion of multiple celestial bodies (binary star, triple star, etc.).
    Key differences:
      - Directly use absolute positions (positions) as the initial positions of the celestial bodies,
      - Directly use absolute velocities (init_velocities) as the initial velocity vectors of the celestial bodies.
      - Celestial bodies can influence each other in get_attraction_forces (typical n-body gravitational interaction).
      - In randomize_parameters, a simple, physically reasonable initialization (non-purely random) example is provided.
    """

    G_constant = 1.0

    def __init__(
            self,
            name: str,
            mass_list: List[float] = [1],
            radius_list: List[float] = [0.1],
            positions: List[Tuple[float, float, float]] = [(0, 0, 0)],
            init_velocities: List[Tuple[float, float, float]] = [(0, 0, 0)],
            pos: Tuple[float, float, float] = (0, 0, 0),
            quat: Tuple[float, float, float, float] = (1, 0, 0, 0),
            **kwargs,
    ):
        """
        Args:
            name (str): Entity name.
            mass_list (List[float]): List of masses for each celestial body.
            radius_list (List[float]): List of radii for each celestial body.
            positions (List[Tuple[float, float, float]]): Absolute initial positions for each celestial body.
            init_velocities (List[Tuple[float, float, float]]): Absolute initial velocity vectors for each celestial body.
            pos (Tuple[float, float, float]): Reference offset for the entire entity in the world coordinate system (usually kept as (0,0,0)).
            quat (Tuple[float, float, float, float]): Initial orientation of the entity (quaternion), default is unit orientation.
        """
        # Basic parameter storage
        self.mass_list = mass_list
        self.radius_list = radius_list
        self.positions = positions
        self.init_velocities = init_velocities

        super().__init__(name, pos, quat, entity_type=self.__class__.__name__, **kwargs)

        # Length verification
        if not (
                len(self.mass_list)
                == len(self.radius_list)
                == len(self.positions)
                == len(self.init_velocities)
        ):
            raise ValueError("The lengths of mass_list, radius_list, positions, init_velocities must be consistent.")

        # Storage for created bodies
        self.created_bodies = []

        # Create bodies
        self._create_sphere_bodies()

        self.trail_bodies = [(f"{self.name}.sphere-{i}", 4000) for i in range(len(self.mass_list))]

    def _create_sphere_bodies(self):
        """
        Create the corresponding number of spheres based on self.positions and self.init_velocities.
        """
        # First, clean up existing child bodies
        self.child_bodies.clear()
        self.created_bodies.clear()

        num_bodies = len(self.mass_list)
        celestial_colors = self._generate_celestial_colors(num_bodies)

        for i in range(num_bodies):
            mass_value = self.mass_list[i]
            radius_value = self.radius_list[i]
            pos_value = self.positions[i]
            velocity_3d = self.init_velocities[i]

            # Construct initial velocity dict
            velocity_dict = {
                InitVelocityType.SPHERE: velocity_3d
            }

            sphere_body = Sphere(
                name=f"{self.name}.sphere-{i}",
                pos=pos_value,
                radius=radius_value,
                mass=mass_value,
                init_velocity=velocity_dict,
                joint_option=("free", (0, 0, 0)),  # Can move freely in 3D space
                rgba=celestial_colors[i],  # Dynamic celestial colors
                material="reflectance",  # Use reflectance material for better color display
            )

            self.add_child_body(sphere_body)
            self.created_bodies.append(sphere_body)

    def _generate_celestial_colors(self, num_bodies):
        """Generate elegant colors for celestial bodies (stars, binary systems, etc.)"""
        import random
        
        colors = []
        
        # Elegant color palette for different types of stars and celestial bodies
        # Based on actual stellar classifications with sophisticated, muted tones
        star_color_palette = [
            (1.0, 0.9, 0.7, 1.0),   # G-type star (Sun-like) - warm, elegant
            (1.0, 0.8, 0.5, 1.0),   # K-type star (orange) - warm, sophisticated
            (0.9, 0.9, 1.0, 1.0),   # A-type star (blue-white) - cool, refined
            (1.0, 0.7, 0.4, 1.0),   # M-type red dwarf - warm, muted
            (0.8, 0.9, 1.0, 1.0),   # F-type star (yellow-white) - bright, elegant
            (0.7, 0.8, 1.0, 1.0),   # B-type star (blue) - cool, sophisticated
            (1.0, 0.85, 0.6, 1.0),  # Late K-type (orange-yellow) - warm
            (0.95, 0.8, 0.7, 1.0),  # Early M-type (orange-red) - subtle
        ]
        
        for i in range(num_bodies):
            if i < len(star_color_palette):
                base_color = star_color_palette[i]
            else:
                # Generate sophisticated colors based on stellar classification
                temperature_factor = random.random()
                if temperature_factor < 0.1:  # Very hot blue stars (rare)
                    base_color = (0.8, 0.85, 1.0, 1.0)  # Soft blue-white
                elif temperature_factor < 0.25:  # Hot white/blue-white
                    base_color = (0.9, 0.9, 1.0, 1.0)   # Elegant white
                elif temperature_factor < 0.5:  # Yellow stars (common)
                    base_color = (1.0, 0.9, 0.7, 1.0)   # Warm yellow
                elif temperature_factor < 0.75:  # Orange stars
                    base_color = (1.0, 0.8, 0.5, 1.0)   # Sophisticated orange
                else:  # Red stars (most common)
                    base_color = (1.0, 0.7, 0.4, 1.0)   # Elegant red-orange
            
            # Add sophisticated brightness variation based on stellar evolution
            brightness_factor = 0.85 + 0.3 * random.random()  # 0.85-1.15 range
            # Ensure we don't exceed realistic brightness
            brightness_factor = min(1.0, brightness_factor)
            
            varied_color = (
                min(1.0, base_color[0] * brightness_factor),
                min(1.0, base_color[1] * brightness_factor),
                min(1.0, base_color[2] * brightness_factor),
                1.0
            )
            
            # Add subtle stellar variability (very small for elegance)
            variation = 0.02
            final_color = (
                max(0.4, min(1.0, varied_color[0] + (random.random() - 0.5) * variation)),
                max(0.4, min(1.0, varied_color[1] + (random.random() - 0.5) * variation)),
                max(0.4, min(1.0, varied_color[2] + (random.random() - 0.5) * variation)),
                1.0
            )
            
            colors.append(final_color)
        
        return colors

    def get_attraction_forces(self) -> List[Tuple[str, str, str, float]]:
        """
        Returns the gravitational force pairs (multi-body interactions).
        Format: [(bodyA_name, bodyB_name, "GRAVITY", G), ...]

        This is the most general case: for all i<j, add both i->j and j->i gravitational forces;
        If you want unidirectional or exclude certain interactions, you can filter them yourself.
        """
        G_constant = 1.0  # Or 6.67430e-11, etc.
        forces = []

        num_bodies = len(self.created_bodies)
        # Double loop to add mutual interactions
        for i in range(num_bodies):
            for j in range(i + 1, num_bodies):
                bodyA = self.created_bodies[i]
                bodyB = self.created_bodies[j]
                bodyA_name = bodyA.name
                bodyB_name = bodyB.name

                # A -> B
                forces.append((bodyA_name, bodyB_name, "GRAVITY", G_constant))
                # B -> A
                forces.append((bodyB_name, bodyA_name, "GRAVITY", G_constant))

        return forces

    def get_parameters(self) -> List[dict]:
        """
        Returns the list of sphere parameters for this entity, for external viewing or logging.
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
                "mass_list": self.mass_list,
                "radius_list": self.radius_list,
                # positions and init_velocities are both 3D vectors
                "positions": [list(p) for p in self.positions],
                "init_velocities": [list(v) for v in self.init_velocities],
            },
        }

    def to_xml(self) -> str:
        """
        Outputs the XML representation of the current entity and its child bodies for loading into a physics simulator.
        """
        body_xml = (
            f'<body name="{self.name}" '
            f'pos="{" ".join(map(str, self.pos))}" '
            f'quat="{" ".join(map(str, self.quat))}">'
        )

        for body in self.child_bodies:
            body_xml += body.to_xml() + "\n"

        body_xml += "</body>"
        return body_xml

    def randomize_parameters(
            self,
            degree_of_randomization: "DegreeOfRandomization" = None,
            reinitialize_instance: bool = False,
            **kwargs,
    ):
        """
        Example: While maintaining the structure of randomization_levels,
        generate different numbers (2~4) and "rules" of celestial body distribution based on difficulty (EASY/MEDIUM/HARD).

        - EASY: [2~3 bodies], roughly symmetrical orbits, smaller randomization
        - MEDIUM: [2~4 bodies], moderate randomization in angles/radii
        - HARD: [3~4 bodies], more complicated randomization
        - We do multiple retries to ensure no overlap among all spheres.
        """
        import math
        import random

        if degree_of_randomization == DegreeOfRandomization.DEFAULT or degree_of_randomization == DegreeOfRandomization.NON_STRUCTURAL:
            options = [
                DegreeOfRandomization.EASY,
                DegreeOfRandomization.MEDIUM,
                DegreeOfRandomization.HARD,
            ]
            degree_of_randomization = random.choice(options)
            print(f"Selected randomization level: {degree_of_randomization.name}")

        if degree_of_randomization is None:
            degree_of_randomization = "EASY"

        # --------------------------------------------------------------------
        # 1) Read randomization_levels
        # --------------------------------------------------------------------
        randomization_levels = {
            DegreeOfRandomization.EASY: {
                "num_bodies_range": (2, 3),
                "mass_range": (0.5, 3.0),
                "radius_range": (0.1, 0.5),
                "system_radius_range": (1.0, 3.0),
            },
            DegreeOfRandomization.MEDIUM: {
                "num_bodies_range": (2, 4),
                "mass_range": (0.1, 5.0),
                "radius_range": (0.05, 1.0),
                "system_radius_range": (1.0, 5.0),
            },
            DegreeOfRandomization.HARD: {
                "num_bodies_range": (3, 4),  # Limited to 3~4, no more than 4
                "mass_range": (0.1, 10.0),
                "radius_range": (0.01, 2.0),
                "system_radius_range": (1.0, 8.0),
            },
        }

        if degree_of_randomization not in randomization_levels:
            raise ValueError(f"Unsupported randomization level: {degree_of_randomization}")

        params = randomization_levels[degree_of_randomization]

        num_min, num_max = params["num_bodies_range"]
        mass_rng = params["mass_range"]
        radius_rng = params["radius_range"]
        system_radius_rng = params["system_radius_range"]

        max_retries = 50  # maximum number of attempts to find a non-overlapping layout
        attempt = 0

        final_masses = None
        final_radii = None
        final_positions = None
        final_velocities = None

        # Gravitational constant (simplified scenario)
        G = 1.0
        self.G = 1.0

        # ---------------------------------------------
        # Define helper functions: generation of binary star / triple star / quadruple star
        # ---------------------------------------------

        def _generate_double_star(mode="easy"):
            """
            Generate initial parameters for a binary star (2 bodies).
            mode: "easy"/"medium"/"hard" represents different levels of rules/randomness.
            """
            m1 = random.uniform(*mass_rng)
            m2 = random.uniform(*mass_rng)
            r1 = random.uniform(*radius_rng)
            r2 = random.uniform(*radius_rng)

            # Distance d between the two bodies
            # Different difficulty levels can set different ranges
            if mode == "easy":
                # Stable range, ensures more regular circular orbits
                d = random.uniform(1.0, 2.0)
            elif mode == "medium":
                # Slightly larger, and may introduce elliptical/random offsets
                d = random.uniform(1.0, 3.0)
            else:  # "hard"
                d = random.uniform(1.0, 4.0)

            M = m1 + m2
            # Distance from the center of mass to m1: r1_c = (m2 / M) * d
            r1_c = (m2 / M) * d
            # Distance from the center of mass to m2: r2_c = (m1 / M) * d
            r2_c = (m1 / M) * d

            # EASY: symmetrically distributed on both sides of the x-axis
            # MEDIUM/HARD: can introduce slight disturbances or elliptical orbits
            # For example, in "medium"/"hard", a small angular offset is introduced
            angle_offset = 0.0
            if mode in ["medium", "hard"]:
                angle_offset = random.uniform(-math.pi / 12, math.pi / 12)  # Small perturbation of ±15 degrees

            # Star 1 at (-r1_c*cos(ang), -r1_c*sin(ang)), Star 2 at (+r2_c*cos(ang), +r2_c*sin(ang))
            # This is to demonstrate adding angle offset
            x1 = -r1_c * math.cos(angle_offset)
            y1 = -r1_c * math.sin(angle_offset)
            x2 = r2_c * math.cos(angle_offset)
            y2 = r2_c * math.sin(angle_offset)

            pos1 = (x1, y1, 0.0)
            pos2 = (x2, y2, 0.0)

            # Velocity magnitude: v1 = sqrt(G * m2 / d), v2 = sqrt(G * m1 / d)
            # Direction is perpendicular to the radius vector
            if d <= 0:
                v1 = v2 = 0.0
            else:
                v1 = math.sqrt(G * m2 / d)
                v2 = math.sqrt(G * m1 / d)

            # EASY => their velocities are mutually perpendicular (±90 degrees)
            # MEDIUM/HARD => add some random perturbation to this base
            # For pos1, pos2, the radial velocity should be perpendicular, taking a counter-clockwise direction

            def perp_vel(px, py, speed, random_factor=0.0):
                # Radial unit vector
                length = math.sqrt(px * px + py * py)
                if length < 1e-6:
                    return (0.0, 0.0, 0.0)
                nx, ny = px / length, py / length
                # Perpendicular vector to (nx, ny) is (-ny, nx)
                # Add some random rotation
                rot = random.uniform(-random_factor, random_factor) if random_factor > 0 else 0.0
                # baseline
                vx = -ny * speed
                vy = nx * speed
                # Apply a small rotation to vx, vy
                # Rotation formula: v'(x) = vx*cos(rot) - vy*sin(rot)
                #                   v'(y) = vx*sin(rot) + vy*cos(rot)
                vx_new = vx * math.cos(rot) - vy * math.sin(rot)
                vy_new = vx * math.sin(rot) + vy * math.cos(rot)
                return (vx_new, vy_new, 0.0)

            random_factor = 0.0
            if mode == "medium":
                random_factor = 0.1  # Small perturbation
            elif mode == "hard":
                random_factor = 0.3  # Larger perturbation

            vel1 = perp_vel(x1, y1, v1, random_factor)
            vel2 = perp_vel(x2, y2, v2, random_factor)

            # Return results
            return [m1, m2], [r1, r2], [pos1, pos2], [vel1, vel2]

        def _generate_triple_star(mode="easy"):
            """
            Generate initial parameters for a triple star (3 bodies).
            mode: "easy"/"medium"/"hard" => determines symmetry/randomness.
            """
            # Randomly generate masses and radii
            ms = [random.uniform(*mass_rng) for _ in range(3)]
            rs = [random.uniform(*radius_rng) for _ in range(3)]
            M_total = sum(ms)

            # Select a base radius within system_radius_rng
            base_R = random.uniform(*system_radius_rng)

            positions = []
            velocities = []

            # EASY => Equilateral triangle, all bodies on the (base_R) circle, angles = 0°, 120°, 240°
            # MEDIUM => Add some random variation (radius/angle) to the equilateral setup
            # HARD => Completely random angles/radii
            if mode == "easy":
                angles = [0, 120, 240]
                radius_variations = [0.0, 0.0, 0.0]
            elif mode == "medium":
                angles = [0 + random.uniform(-5, 5),
                          120 + random.uniform(-5, 5),
                          240 + random.uniform(-5, 5)]
                radius_variations = [
                    random.uniform(-0.1 * base_R, 0.1 * base_R) for _ in range(3)
                ]
            else:  # "hard"
                angles = [random.uniform(0, 360) for _ in range(3)]
                radius_variations = [
                    random.uniform(-0.2 * base_R, 0.2 * base_R) for _ in range(3)
                ]

            for i in range(3):
                theta_deg = angles[i]
                theta = math.radians(theta_deg)
                r_i = base_R + radius_variations[i]
                if r_i < 0:
                    r_i = abs(r_i)  # Avoid negative radius

                x = r_i * math.cos(theta)
                y = r_i * math.sin(theta)
                z = 0.0
                positions.append((x, y, z))

            # Calculate velocity: approximate uniform circular motion => v = sqrt(G * M_total / r_i), perpendicular direction
            # Can also add perturbations
            def perp_circular_speed(px, py, totalM, extra_angle=0.0):
                r = math.sqrt(px * px + py * py)
                if r < 1e-6:
                    return (0.0, 0.0, 0.0)
                v = math.sqrt(G * totalM / r)  # Simplified
                # Basic direction: (-py, px) counter-clockwise
                vx = -py / r * v
                vy = px / r * v
                # Add a small rotation
                rot = extra_angle
                vx_new = vx * math.cos(rot) - vy * math.sin(rot)
                vy_new = vx * math.sin(rot) + vy * math.cos(rot)
                return (vx_new, vy_new, 0.0)

            for i in range(3):
                (px, py, pz) = positions[i]
                if mode == "easy":
                    # Completely regular, no perturbation
                    velocities.append(perp_circular_speed(px, py, M_total, extra_angle=0.0))
                elif mode == "medium":
                    # Small perturbation
                    small_rot = random.uniform(-0.1, 0.1)
                    velocities.append(perp_circular_speed(px, py, M_total, extra_angle=small_rot))
                else:  # hard
                    big_rot = random.uniform(-0.3, 0.3)
                    velocities.append(perp_circular_speed(px, py, M_total, extra_angle=big_rot))

            return ms, rs, positions, velocities

        def _generate_quad_star(mode="easy"):
            """
            (Optional) Generate initial parameters for a quadruple star (4 bodies).
            This is only called in MEDIUM/HARD cases.
            You can decide its symmetry/randomness as needed.
            """
            ms = [random.uniform(*mass_rng) for _ in range(4)]
            rs = [random.uniform(*radius_rng) for _ in range(4)]
            M_total = sum(ms)

            base_R = random.uniform(*system_radius_rng)
            positions = []
            velocities = []

            # For 4 bodies, make a cross or square distribution (easy/medium) or completely random (hard)
            if mode == "medium":
                # Place in four quadrants with small random angles
                angles_deg = [45, 135, 225, 315]
                for i in range(4):
                    theta = math.radians(angles_deg[i] + random.uniform(-5, 5))
                    r_i = base_R + random.uniform(-0.1 * base_R, 0.1 * base_R)
                    x = r_i * math.cos(theta)
                    y = r_i * math.sin(theta)
                    z = 0.0
                    positions.append((x, y, z))
            else:
                # "hard" => completely random angles
                for i in range(4):
                    theta = random.uniform(0, 2 * math.pi)
                    r_i = random.uniform(0.8 * base_R, 1.2 * base_R)
                    x = r_i * math.cos(theta)
                    y = r_i * math.sin(theta)
                    z = 0.0
                    positions.append((x, y, z))

            # Velocity calculation similar to circular approximation
            def perp_speed(px, py, totalM, factor=1.0):
                r = math.sqrt(px * px + py * py)
                if r < 1e-6:
                    return (0.0, 0.0, 0.0)
                v = factor * math.sqrt(G * totalM / r)
                vx = -py / r * v
                vy = px / r * v
                return (vx, vy, 0.0)

            if mode == "medium":
                for (px, py, _) in positions:
                    velocities.append(perp_speed(px, py, M_total, factor=1.0))
            else:  # hard
                for (px, py, _) in positions:
                    # Increase random factor
                    factor = random.uniform(0.8, 1.2)
                    velocities.append(perp_speed(px, py, M_total, factor=factor))

            return ms, rs, positions, velocities

        while attempt < max_retries:
            attempt += 1

            # Randomly select the number of celestial bodies
            num_bodies = random.randint(num_min, num_max)

            # Based on difficulty + number, call different generation logic
            if degree_of_randomization == "EASY":
                # Only 2 or 3 bodies
                if num_bodies == 2:
                    new_m, new_r, new_pos, new_vel = _generate_double_star(mode="easy")
                else:  # 3
                    new_m, new_r, new_pos, new_vel = _generate_triple_star(mode="easy")

            elif degree_of_randomization == "MEDIUM":
                # 2~4 bodies
                if num_bodies == 2:
                    # "Complex binary star"
                    new_m, new_r, new_pos, new_vel = _generate_double_star(mode="medium")
                elif num_bodies == 3:
                    # "Moderately varied" triple star
                    new_m, new_r, new_pos, new_vel = _generate_triple_star(mode="medium")
                else:  # 4
                    new_m, new_r, new_pos, new_vel = _generate_quad_star(mode="medium")

            else:  # HARD
                # 3 or 4 bodies
                if num_bodies == 3:
                    new_m, new_r, new_pos, new_vel = _generate_triple_star(mode="hard")
                else:  # 4
                    new_m, new_r, new_pos, new_vel = _generate_quad_star(mode="hard")

            # Now check overlap among all spheres
            any_overlap = False
            extra_gap = 0.0  # additional gap if needed
            for i in range(len(new_m)):
                for j in range(i + 1, len(new_m)):
                    (x1, y1, z1) = new_pos[i]
                    (x2, y2, z2) = new_pos[j]
                    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
                    if dist < (new_r[i] + new_r[j] + extra_gap):
                        any_overlap = True
                        break
                if any_overlap:
                    break

            if not any_overlap:
                final_masses = new_m
                final_radii = new_r
                final_positions = new_pos
                final_velocities = new_vel
                break
            else:
                # If overlap, store them temporarily but continue to next attempt
                final_masses = new_m
                final_radii = new_r
                final_positions = new_pos
                final_velocities = new_vel

        if attempt >= max_retries:
            print("Warning: Could not find non-overlapping layout after max_retries, using the last generated set.")

        # Round and update entity parameters
        self.mass_list = [round(m, 3) for m in final_masses]
        self.radius_list = [round(r, 3) for r in final_radii]
        self.positions = [
            (round(px, 3), round(py, 3), round(pz, 3)) for (px, py, pz) in final_positions
        ]
        self.init_velocities = [
            (round(vx, 3), round(vy, 3), round(vz, 3)) for (vx, vy, vz) in final_velocities
        ]

        # Optionally reinitialize
        if reinitialize_instance:
            self.reinitialize()

    def get_nlq(self, symbolic=False):
        intro = (
            f"In a hypothetical celestial system, there are {len(self.created_bodies)} entities (spherical in shape), each orbiting around the previous one. "
        )

        init_speeds = [np.linalg.norm(v) for v in self.init_velocities] 
        data = {
            "Entity": [i for i in range(1, len(self.created_bodies) + 1)],
            "Mass (M)": [f"<mass>{i}" for i in range(1, len(self.created_bodies) + 1)],
            "Radius (L)": [f"{r:.2f}" for r in self.radius_list],
            "Initial Position (L)": [f"({x:.2f}, {y:.2f}, {z:.2f})" for x, y, z in self.positions],
            "Initial Speed (L/T)": [f"{v:.2f}" for v in init_speeds],
        }

        df = pd.DataFrame(data)
        properties_description = tabulate(df, headers='keys', tablefmt='github', showindex=False) + "\n"

        general_info = (
            f"Each entity only applies gravitational force to the entity orbiting it. "
            f"The units M L T are custom units for mass, length and time, such that the gravitational constant G is {self.G_constant} in these units. "
        )

        description = (
            f"{intro}\n"
            f"{properties_description}\n"
            f"{general_info}\n"
        )

        sym_dict = {
            f"<mass>{i+1}": m for i, m in enumerate(self.mass_list)
        }

        if not symbolic:
            description = replace_all(description, sym_dict)

            return description
        
        return description, sym_dict
    
    def get_connecting_point_nl(self, cd, cp, csi):
        """
        Get the connecting point description for the entity.
        
        Inputs:
            cd: str
            cp: str
            csi: str
            
        Returns:
            str
        """
        raise NotImplementedError("GeneralCelestialEntity is not supposed to have connections.")
    
    def get_question(self, sub_entity: str, quantity: str) -> str:
        """
        Get a question related to the entity
        
        Inputs:
            sub_entity: str
            quantity: str
            
        Returns:
            str
        """

        descriptior = ""
        idx = int(sub_entity[7:])
        descriptior = f"{(['1st', '2nd', '3rd'] + [f'{i + 4}th' for i in range(len(self.mass_list) - 3)])[idx]} entity"

        question = (
            f"What is the {quantity} of the {descriptior} in the celestial system"
        )
        return question

class SolarSystemEntity(Entity):
    """
    SolarSystemEntity is used to create a system where multiple planets revolve around a star.

    Args:
        name (str): Name of this entity.
        planets (List[Dict]): The i-th planet dict has the following properties:
                                - mass (float): Mass of the planet.
                                - radius (float): Radius of the planet.
                                - position (Tuple[float, float]): Position of the planet in 2D space.
                                - velocity (Tuple[float, float]): Velocity of the planet in 2D space.
        star_mass (float): Mass of the central star.
        star_radius (float): Radius of the central star.
        pos (Tuple[float, float, float]): The world-base position of this entire entity (defaults to (0,0,0)).
        quat (Tuple[float, float, float, float]): Quaternion for entity's base orientation (defaults to identity).
    """

    randomization_levels = {
        DegreeOfRandomization.EASY: {
            "num_planets_range": (1, 2),
            "mass_range": (3.0, 5.0),
            "planet_mass_range": (0.1, 0.5),
            "radius_range": (0.75, 1.25),
            "planet_radius_range": (0.05, 0.1),
            "semi_major_axis": (1.5, 2),
            "orbital_eccentricity": (0.0, 0.0),
            "theta_deg_range": (0.0, 360.0),      # degrees
        },
        DegreeOfRandomization.MEDIUM: {
            "num_planets_range": (2, 4),
            "mass_range": (3.0, 5.0),
            "planet_mass_range": (0.1, 0.5),
            "radius_range": (0.75, 1.25),
            "planet_radius_range": (0.05, 0.1),
            "semi_major_axis": (1.5, 2.5),
            "orbital_eccentricity": (0.0, 0.3),
            "theta_deg_range": (0.0, 360.0),
        },
        DegreeOfRandomization.HARD: {
            "num_planets_range": (3, 6),
            "mass_range": (3.0, 5.0),
            "planet_mass_range": (0.1, 0.5),
            "radius_range": (0.75, 1.25),
            "planet_radius_range": (0.05, 0.1),
            "semi_major_axis": (2.0, 3.0),
            "orbital_eccentricity": (0.0, 0.5),
            "theta_deg_range": (0.0, 360.0),
        },
    }
    
    G = 10.0
    
    def __init__(
        self,
        name: str,
        planets: List[Dict] = [],
        star_mass: float = 1.0,
        star_radius: float = 0.1,
        pos: Tuple[float, float, float] = (0, 0, 0),
        quat: Tuple[float, float, float, float] = (1, 0, 0, 0),
        **kwargs,
    ):
        # Basic parameter storage
        self.star_mass = star_mass
        self.star_radius = star_radius
        self.planets = planets
        super().__init__(name, pos, quat, entity_type=self.__class__.__name__, **kwargs)

        # Store created spheres
        self.created_bodies = []

        self._create_sphere_bodies()

        self.trail_bodies = [(f"{self.name}.planet-{i}", 400) for i in range(len(self.planets))]
    
    def _create_sphere_bodies(self):
        """
        1) Convert polar coordinates in degrees to absolute positions.
           - The first sphere is at origin (0,0,0).
           - For subsequent spheres, compute (r*cos(theta), r*sin(theta)) offset from the previous sphere.
        2) Convert each single-float velocity to a tangential 3D vector.
           - For sphere i > 0, the tangential direction is perpendicular to the radial offset from the (i-1)-th sphere.
             That direction is:
                 dx = -sin(theta_radians)
                 dy =  cos(theta_radians)
             multiplied by the velocity magnitude.
           - For the first sphere (i=0), we'll default to (0, 0, 0).
        """
        # Clear any existing children if reinitializing
        self.child_bodies.clear()
        self.created_bodies.clear()

        # Generate colors for planets and star
        planet_colors = self._generate_solar_system_colors(len(self.planets))
        star_color = self._generate_star_color()

        for i, planet in enumerate(self.planets):
            position = planet["position"]
            velocity = planet["velocity"]
            mass = planet["mass"]
            radius = planet["radius"]

            velocity_dict = {
                InitVelocityType.SPHERE: (*velocity, 0.0)
            }

            sphere_body = Sphere(
                name=f"{self.name}.planet-{i}",
                pos=(*position, 0.0),
                radius=radius,
                mass=mass,
                init_velocity=velocity_dict,
                joint_option=("free", (0, 0, 0)),  # free joint for 3D motion
                rgba=planet_colors[i],  # Dynamic planet colors
                material="reflectance",  # Use reflectance material for better color display
            )

            self.add_child_body(sphere_body)
            self.created_bodies.append(sphere_body)

        # Now create the central star
        star_body = Sphere(
            name=f"{self.name}.star",
            pos=(0.0, 0.0, 0.0),
            radius=self.star_radius,
            mass=self.star_mass,
            init_velocity={},
            joint_option=None, # Stationary
            rgba=star_color,  # Dynamic star color
            material="reflectance",  # Use reflectance material for better color display
        )

        self.add_child_body(star_body)
        self.created_bodies.append(star_body)

    def _generate_solar_system_colors(self, num_planets):
        """Generate elegant colors for planets in a solar system"""
        import random
        
        colors = []
        
        # Sophisticated planet color palette based on our solar system
        # with refined, muted tones suitable for scientific visualization
        realistic_planet_colors = [
            (0.7, 0.6, 0.4, 1.0),   # Mercury-like (warm grey-brown)
            (0.9, 0.8, 0.5, 1.0),   # Venus-like (elegant golden)
            (0.5, 0.6, 0.8, 1.0),   # Earth-like (sophisticated blue)
            (0.8, 0.5, 0.3, 1.0),   # Mars-like (refined rust)
            (0.9, 0.7, 0.5, 1.0),   # Jupiter-like (warm tan-orange)
            (0.9, 0.8, 0.6, 1.0),   # Saturn-like (soft pale gold)
            (0.6, 0.8, 0.9, 1.0),   # Uranus-like (soft cyan)
            (0.4, 0.5, 0.8, 1.0),   # Neptune-like (deep elegant blue)
        ]
        
        for i in range(num_planets):
            if i < len(realistic_planet_colors):
                base_color = realistic_planet_colors[i]
            else:
                # Generate sophisticated procedural colors for additional planets
                planet_type = random.choice(['rocky', 'gas', 'ice', 'exotic'])
                
                if planet_type == 'rocky':
                    base_color = (
                        0.6 + 0.2 * random.random(),  # Red: 0.6-0.8 (more muted)
                        0.5 + 0.2 * random.random(),  # Green: 0.5-0.7
                        0.3 + 0.2 * random.random(),  # Blue: 0.3-0.5
                        1.0
                    )
                elif planet_type == 'gas':
                    base_color = (
                        0.7 + 0.2 * random.random(),  # Red: 0.7-0.9
                        0.6 + 0.2 * random.random(),  # Green: 0.6-0.8
                        0.4 + 0.3 * random.random(),  # Blue: 0.4-0.7
                        1.0
                    )
                elif planet_type == 'ice':
                    base_color = (
                        0.6 + 0.2 * random.random(),  # Red: 0.6-0.8
                        0.7 + 0.2 * random.random(),  # Green: 0.7-0.9
                        0.8 + 0.1 * random.random(),  # Blue: 0.8-0.9
                        1.0
                    )
                else:  # exotic
                    base_color = (
                        0.6 + 0.2 * random.random(),  # Red: 0.6-0.8
                        0.4 + 0.3 * random.random(),  # Green: 0.4-0.7
                        0.5 + 0.3 * random.random(),  # Blue: 0.5-0.8
                        1.0
                    )
            
            # Add very subtle orbital distance effect (farther = cooler)
            distance_factor = i / max(1, num_planets - 1)
            cooling_effect = distance_factor * 0.1
            cooled_color = (
                max(0.3, base_color[0] - cooling_effect * 0.3),
                max(0.3, base_color[1] - cooling_effect * 0.2),
                min(0.9, base_color[2] + cooling_effect * 0.1),
                1.0
            )
            
            colors.append(cooled_color)
        
        return colors

    def _generate_star_color(self):
        """Generate elegant star color based on stellar classification"""
        import random
        
        # Sophisticated star types with their elegant colors
        # Based on actual stellar spectral classes
        star_types = [
            (1.0, 0.9, 0.7, 1.0),   # G-type (Sun-like) - warm, refined
            (1.0, 0.8, 0.5, 1.0),   # K-type (orange) - sophisticated
            (1.0, 0.7, 0.4, 1.0),   # M-type (red dwarf) - elegant red-orange
            (0.9, 0.9, 1.0, 1.0),   # A-type (white) - cool, elegant
            (0.85, 0.9, 1.0, 1.0),  # F-type (yellow-white) - refined
        ]
        
        # Weight towards more common and visually appealing star types
        weights = [0.4, 0.3, 0.2, 0.07, 0.03]  # G, K, M, A, F
        
        chosen_type = random.choices(star_types, weights=weights)[0]
        
        # Add very subtle variation for stellar realism
        variation = 0.03  # Very small for elegance
        varied_color = (
            max(0.5, min(1.0, chosen_type[0] + (random.random() - 0.5) * variation)),
            max(0.5, min(1.0, chosen_type[1] + (random.random() - 0.5) * variation)),
            max(0.5, min(1.0, chosen_type[2] + (random.random() - 0.5) * variation)),
            1.0
        )
        
        return varied_color

    def get_attraction_forces(self) -> List[Tuple[str, str, str, float]]:
        
        forces = [
            (
                self.created_bodies[i].name,
                self.created_bodies[-1].name,
                "GRAVITY",
                self.G,
            )
            for i in range(len(self.created_bodies) - 1)
        ]

        return forces

    def get_parameters(self) -> List[dict]:
        """
        Returns a list of dictionaries describing the created spheres (names, masses, etc.).
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

    def generate_entity_yaml(
        self,
        use_random_parameters: bool = False,
        degree_of_randomization: DegreeOfRandomization = DegreeOfRandomization.NON_STRUCTURAL,
    ) -> dict:
        """
        Exports this entity's parameters to a dictionary for YAML/JSON serialization.
        """
        if use_random_parameters:
            self.randomize_parameters(degree_of_randomization)

        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "position": list(self.pos),
            "parameters": {
                "star_mass": self.star_mass,
                "star_radius": self.star_radius,
                # planets is a list of dictionaries
                "planets": [
                    {
                        "mass": planet["mass"],
                        "radius": planet["radius"],
                        "position": list(planet["position"]),
                        "velocity": list(planet["velocity"]),
                    }
                    for planet in self.planets
                ],
            },
        }

    def to_xml(self) -> str:
        """
        Converts this entity (and its child bodies) to an XML representation suitable for a physics simulator.
        """
        body_xml = (
            f'<body name="{self.name}" '
            f'pos="{" ".join(map(str, self.pos))}" '
            f'quat="{" ".join(map(str, self.quat))}">'
        )

        # If the entity itself has joints, add them here
        for body in self.child_bodies:
            body_xml += body.to_xml() + "\n"

        body_xml += "</body>"
        return body_xml

    def randomize_parameters(
        self,
        degree_of_randomization: "DegreeOfRandomization" = None,
        reinitialize_instance: bool = False,
        **kwargs,
    ):
        """
        Initializes a planetary system:
        - Central star at origin
        - Planets placed on elliptical orbits
        - Velocities initialized for stable orbits (non-colliding)
        
        Parameters:
            num_planets: Number of orbiting bodies
            central_star_mass: Mass of the central body (sun)
            star_radius: Radius of the central body
            planet_mass_range: Range of planet masses
            planet_radius_range: Range of planet radii
            semi_major_axis_range: Range for semi-major axis of orbits
            orbital_eccentricity_range: Range for orbital eccentricity (0 = circle)
            G: Gravitational constant (can use normalized units)
        """
        
        randomization_levels = self.randomization_levels
    
        if degree_of_randomization in [
                None,
                DegreeOfRandomization.DEFAULT,
            ]:
            degree_of_randomization = random.choice(["EASY", "MEDIUM", "HARD"])

        if degree_of_randomization not in randomization_levels:
            raise ValueError(f"Unsupported randomization level: {degree_of_randomization}")

        params = randomization_levels[degree_of_randomization]

        (
            mass_list, 
            radius_list, 
            relative_position_list, 
            init_velocity_list,
        ) = [], [], [], []

        star_mass = round(random.uniform(*params["mass_range"]), 2)
        star_radius = round(random.uniform(*params["radius_range"]), 2)
        num_planets = random.randint(*params["num_planets_range"])

        a_min, a_max = params["semi_major_axis"]
        
        for i in range(num_planets):
            planet_mass = round(random.uniform(*params["planet_mass_range"]), 2)
            planet_radius = round(random.uniform(*params["planet_radius_range"]), 2)

            _r = random.uniform((a_max - a_min) / num_planets * 0.25, (a_max - a_min) / num_planets * 0.75)
            semi_major_axis = a_min + (a_max - a_min) * i / num_planets + _r

            e = random.uniform(*params["orbital_eccentricity"])
            r = semi_major_axis * (1 - e)  # Place planet at periapsis for max velocity

            theta = random.uniform(*params["theta_deg_range"])
            theta_rad = math.radians(theta)

            v_mag = math.sqrt(self.G * star_mass * (2 / r - 1 / semi_major_axis))

            # periapsis position
            x = round(r * math.cos(theta_rad), 2)
            y = round(r * math.sin(theta_rad), 2)
            
            # Velocity is perpendicular to radius vector at periapsis
            vx = round(-v_mag * math.sin(theta_rad), 2)
            vy = round( v_mag * math.cos(theta_rad), 2)

            mass_list.append(planet_mass)
            radius_list.append(planet_radius)
            relative_position_list.append((x, y))
            init_velocity_list.append((vx, vy))

        self.star_mass = star_mass
        self.star_radius = star_radius

        self.planets = [
            {
                "mass": mass_list[i],
                "radius": radius_list[i],
                "position": relative_position_list[i],
                "velocity": init_velocity_list[i],
            }
            for i in range(num_planets)
        ]

        # Optional reinitialization
        if reinitialize_instance:
            self.reinitialize()

    def get_description(self, simDSL2nlq=False):
        return super().get_description(simDSL2nlq)
    
    def get_nlq(self, symbolic = False):
        
        star_mass = "<mass>1"
        star_radius = "<radius>1"
        
        sym_dict = {
            star_mass: self.star_mass,
            star_radius: self.star_radius,
        }

        star_mass = sym_dict.pop(star_mass)
        star_radius = sym_dict.pop(star_radius)

        n_planet = len(self.planets)

        description = (
            f"In a planetary system, a star with mass {star_mass} M and radius {star_radius} R is at the center. "
            f"{n_planet} planets orbit around the star in coplanar orbits. "
            f" The planets have the following properties at a given time (lets say t=0): "
        )

        mass_list = [f"<mass>{i+2}" for i in range(n_planet)]
        radius_list = [f"<radius>{i+2}" for i in range(n_planet)]
        initial_position_list = [
            f"<x>{i+1} <y>{i+1}" for i in range(n_planet)
        ]
        initial_velocity_list = [
            f"<vx>{i+1} <vy>{i+1}" for i in range(n_planet)
        ]

        sym_dict.update({
            f"<mass>{i+2}": self.planets[i]["mass"] for i in range(n_planet)
        })

        sym_dict.update({
            f"<radius>{i+2}": self.planets[i]["radius"] for i in range(n_planet)
        })

        sym_dict.update({
            f"<x>{i+1}": self.planets[i]["position"][0] for i in range(n_planet)
        })

        sym_dict.update({
            f"<y>{i+1}": self.planets[i]["position"][1] for i in range(n_planet)
        })

        sym_dict.update({
            f"<vx>{i+1}": self.planets[i]["velocity"][0] for i in range(n_planet)
        })

        sym_dict.update({
            f"<vy>{i+1}": self.planets[i]["velocity"][1] for i in range(n_planet)
        })
        
        propoerties = {
            "Planet": [i + 1 for i in range(n_planet)],
            "Mass (M)": mass_list,
            "Radius (R)": radius_list,
            "Position (R)": initial_position_list,
            "Velocity (R/T)": initial_velocity_list,
        }

        df = pd.DataFrame(propoerties)
        properties_description = tabulate(df, headers='keys', tablefmt='github', showindex=False) + "\n"

        general_description = (
            f"Assume that the star remains fixed at the center of the system, and planets do not apply any gravitational force on each other. "
            f"The units (mass: M, length: R, time: T) are custom such that the gravitational constant G = {self.G}; "
            f"they are not SI units."
        )

        description = f"{description}\n{properties_description}\n{general_description}"

        if not symbolic:
            description = replace_all(description, sym_dict)

            return description
        
        return description, sym_dict
    
    def connecting_point_nl(self, cd, cp, csi):
        raise NotImplementedError("SolarSystemEntity is not supposed to have connections.")
    
    def get_question(self, sub_entity: str, quantity: str) -> str:
        """
        Get a question related to the entity
        
        Inputs:
            sub_entity: str
            quantity: str
            
        Returns:
            str
        """

        descriptior = ""
        if sub_entity == "star":
            descriptior = "star"
        else:
            idx = int(sub_entity[7:]) # remove the "planet-"
            descriptior = f"{(['1st', '2nd', '3rd'] + [f'{i + 4}th' for i in range(len(self.planets) - 3)])[idx]} planet"
        
        question = (
            f"What is the {quantity} of the {descriptior} in the planetary system"
        )
        
        return question
    
class RocketEntity(Entity):
    """
    RocketEntity simulates a vertically launching rocket with mass loss over time.
    It creates a static planet and a rocket that can ascend under thrust.
    """

    # ---------- 1) Randomization Configuration ----------
    randomization_levels = {
        DegreeOfRandomization.EASY: {
            "rocket_mass_range": (0.075, 0.125),
            # "rocket_radius_range": (0.008, 0.015),
            "v_exhaust_range": (25, 30),
            "dm_dt_range": (-0.01, -0.001),            # kg · s⁻¹  (negative == mass loss)
            "planet_density_range": (0.75, 1.25),           # mass = density * radius**3 * 1e5
            "planet_radius_range": (0.01, 0.06),
            "launch_angle_range": (0, 0),  # degrees
        },
        DegreeOfRandomization.MEDIUM: {
            "rocket_mass_range": (0.075, 0.125),
            # "rocket_radius_range": (0.008, 0.015),
            "v_exhaust_range": (25, 30),
            "dm_dt_range": (-0.04, -0.01),            # kg · s⁻¹  (negative == mass loss)
            "planet_density_range": (0.75, 1.25),           # mass = density * radius**3 * 1e5
            "planet_radius_range": (0.08, 0.12),
            "launch_angle_range": (0, 0),  # degrees
        },DegreeOfRandomization.HARD: {
            "rocket_mass_range": (0.075, 0.125),
            # "rocket_radius_range": (0.008, 0.015),
            "v_exhaust_range": (6, 7.25),
            "dm_dt_range": (-0.16, -0.04),            # kg · s⁻¹  (negative == mass loss)
            "planet_density_range": (0.12, 0.2),           # mass = density * radius**3 * 1e5
            "planet_radius_range": (0.5, 0.75),
            "launch_angle_range": (0, 90),  # degrees
        },
    }

    G = 1e-3  # Gravitational constant (normalized units)

    # ---------- 2) Constructor ----------
    def __init__(
        self,
        name: str,
        rocket_mass: float = 1.0,
        rocket_radius: float = 0.01,
        rocket_height: float = 0.05,                 # Compatible with old interface, can be ignored
        planet_mass: float = 10.0,
        planet_radius: float = 0.1,
        v_exhaust: float = 10.0,
        dm_dt: float = -0.1,
        min_mass: float = 0.1,
        launch_angle: float = 0,
        pos: Tuple[float, float, float] = (0, 0, 0),
        quat: Tuple[float, float, float, float] = (1, 0, 0, 0),
        **kwargs,
    ):
        # ======= Core Parameters =======
        self.rocket_mass = rocket_mass
        self.rocket_radius = rocket_radius
        self.rocket_height = rocket_height          # Still retained for backward compatibility
        self.planet_mass = planet_mass
        self.planet_radius = planet_radius
        self.v_exhaust = v_exhaust
        self.dm_dt = dm_dt
        self.min_mass = min_mass
        self.launch_angle = launch_angle

        super().__init__(name, pos, quat, entity_type=self.__class__.__name__, **kwargs)

        # ======= Create Child Bodies =======
        self._create_rocket()
        self._create_planet()

        self.trail_bodies = [(self.rocket_body.name, 600)]

    # ---------- 3) Create Child Bodies ----------
    def _create_rocket(self):
        """
        Add a rocket body with free joint and **sphere** geometry.
        """
        # rocket_body = Sphere( # Rocket(
        #     name=f"{self.name}.rocket",
        #     pos=(0, 0, self.planet_radius + self.rocket_radius),  # Slightly above ground
        #     radius=self.rocket_radius,
        #     mass=self.rocket_mass,
        #     # init_velocity={InitVelocityType.SPHERE: (0, 0, 0)},
        #     # joint_option=("free", (0, 0, 1)),
        #     rgba=(0.8, 0.1, 0.1, 1.0),
        # )

        quat = (
            np.cos(np.radians(self.launch_angle) / 2),
            0,
            np.sin(np.radians(self.launch_angle) / 2),
            0,
        )

        rocket_body = Rocket(
            name=f"{self.name}.rocket",
            pos=(0, 0, self.planet_radius),
            mass=self.rocket_mass,    
            quat=quat,
        )
        self.rocket_body = rocket_body
        self.add_child_body(rocket_body)

    def _create_planet(self):
        """
        Add a static sphere as the planet.
        """
        planet_color = self._generate_rocket_planet_color()
        
        planet_body = Sphere(
            name=f"{self.name}.planet",
            pos=(0, 0, 0),
            # Keep XML geometry exactly consistent with the declared parameter.
            radius=self.planet_radius,
            mass=self.planet_mass,
            init_velocity={InitVelocityType.SPHERE: (0, 0, 0)},
            joint_option=None,                       # Static
            rgba=planet_color,
            material="reflectance",  # Use reflectance material for better color display
        )
        self.add_child_body(planet_body)
        
    def _generate_rocket_planet_color(self):
        """Generate elegant planet colors for rocket launch scenarios"""
        import random
        
        # Sophisticated planet types with elegant colors for rocket launches
        # More muted and refined colors suitable for scientific visualization
        planet_types = [
            (0.6, 0.7, 0.9, 1.0),   # Earth-like (soft blue-green)
            (0.8, 0.6, 0.4, 1.0),   # Mars-like (elegant rust)
            (0.7, 0.5, 0.3, 1.0),   # Rocky desert (warm brown)
            (0.6, 0.7, 0.5, 1.0),   # Forest world (sage green)
            (0.8, 0.6, 0.3, 1.0),   # Volcanic world (refined amber)
            (0.6, 0.6, 0.7, 1.0),   # Moon-like (sophisticated grey)
            (0.5, 0.6, 0.8, 1.0),   # Ocean world (deep, elegant blue)
            (0.7, 0.7, 0.5, 1.0),   # Dusty world (warm tan)
            (0.6, 0.8, 0.7, 1.0),   # Tropical world (soft teal)
            (0.8, 0.7, 0.6, 1.0),   # Desert world (warm sand)
        ]
        
        # Select a base color with weighted preference for common planet types
        weights = [0.25, 0.2, 0.15, 0.1, 0.08, 0.08, 0.07, 0.05, 0.01, 0.01]
        base_color = random.choices(planet_types, weights=weights)[0]
        
        # Add very subtle variation for realism while maintaining elegance
        variation = 0.05  # Small variation for refinement
        varied_color = (
            max(0.3, min(0.9, base_color[0] + random.uniform(-variation, variation))),
            max(0.3, min(0.9, base_color[1] + random.uniform(-variation, variation))),
            max(0.3, min(0.9, base_color[2] + random.uniform(-variation, variation))),
            1.0
        )
        
        return varied_color

    # ---------- 4) Thrust and Mass Auxiliary Interfaces ----------
    def current_mass(self, t: float) -> float:
        """
        Return remaining mass (kg) at time *t* (seconds).
        Linear fuel consumption: m(t) = max(min_mass, rocket_mass + dm_dt · t)
        """
        return max(self.min_mass, self.rocket_mass + self.dm_dt * t)

    def get_thrust(self, t: float) -> float:
        """
        Return instantaneous thrust (Newton) at time *t*.
        Simplified model: F = -dm/dt · v_exhaust (if fuel is still being consumed, else 0)
        """
        if self.current_mass(t) > self.min_mass:
            return -self.dm_dt * self.v_exhaust
        return 0.0
    
    def get_rocket(self) -> Sphere:
        """
        Return the rocket body.
        """
        return self.rocket_body
    
    def get_attraction_forces(self) -> List[Tuple[str, str, str, float]]:
        
        forces = [
            (
                self.rocket_body.collision_geom.name,
                f"{self.name}.planet",
                "GRAVITY",
                self.G,
            )
        ]

        return forces

    # ---------- 5) Randomization ----------
    def randomize_parameters(
        self,
        degree_of_randomization: DegreeOfRandomization = DegreeOfRandomization.NON_STRUCTURAL,
        reinitialize_instance: bool = False,
        **kwargs,
    ):
        """
        Randomize the rocket & planet parameters.
        """
        if degree_of_randomization in (None, DegreeOfRandomization.DEFAULT):
            degree_of_randomization = random.choice(
                [DegreeOfRandomization.EASY,
                 DegreeOfRandomization.MEDIUM,
                 DegreeOfRandomization.HARD]
            )

        if degree_of_randomization not in self.randomization_levels:
            raise ValueError(f"Unsupported randomization level: {degree_of_randomization}")

        p = self.randomization_levels[degree_of_randomization]

        self.rocket_mass   = round(random.uniform(*p["rocket_mass_range"]), 3)
        # self.rocket_radius = round(random.uniform(*p["rocket_radius_range"]), 4)
        self.v_exhaust     = round(random.uniform(*p["v_exhaust_range"]), 2)
        self.dm_dt         = round(random.uniform(*p["dm_dt_range"]), 3)
        density = random.uniform(*p["planet_density_range"])
        self.planet_radius = round(random.uniform(*p["planet_radius_range"]), 3)
        self.planet_mass   = round(density * self.planet_radius**3, 2)
        self.launch_angle = round(random.uniform(*p["launch_angle_range"]), 2)
        
        # If the user has not specified, min_mass is 60% of the initial mass
        self.min_mass = round(self.rocket_mass * 0.6, 3)

        if reinitialize_instance:
            # Re-generate child bodies
            self.child_bodies.clear()
            self._create_rocket()
            self._create_planet()

    # ---------- 6) Export Parameters ----------
    def get_parameters(self) -> dict:
        """
        Export parameters for logging or serialization.
        """
        return {
            "rocket_mass": self.rocket_mass,
            # "rocket_radius": self.rocket_radius,
            "planet_mass": self.planet_mass,
            "planet_radius": self.planet_radius,
            "v_exhaust": self.v_exhaust,
            "dm_dt": self.dm_dt,
            "min_mass": self.min_mass,
            "launch_angle": self.launch_angle,
        }

    def generate_entity_yaml(
        self,
        use_random_parameters: bool = False,
        degree_of_randomization: DegreeOfRandomization = DegreeOfRandomization.NON_STRUCTURAL,
    ) -> dict:
        """
        Export this entity for YAML / JSON serialization.
        """
        if use_random_parameters:
            self.randomize_parameters(degree_of_randomization, reinitialize_instance=True)

        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "position": list(self.pos),
            "parameters": self.get_parameters(),
        }

    # ---------- 7) XML Export ----------
    def to_xml(self) -> str:
        """
        Convert this entity (and its child bodies) to an XML string for the simulator.
        """
        body_xml = (
            f'<body name="{self.name}" '
            f'pos="{" ".join(map(str, self.pos))}" '
            f'quat="{" ".join(map(str, self.quat))}">\n'
        )
        for child in self.child_bodies:
            body_xml += child.to_xml() + "\n"
        body_xml += "</body>"
        return body_xml
    
    # ---------- 8) Description and NLQ ----------
    def get_nlq(self, symbolic = False):
        rocket_mass = "<mass>1"
        planet_mass = "<mass>2"
        planet_radius = "<radius>1"
        v_exhaust = "<vx>1"
        dm_dt = "<mass>3"
        min_mass = "<mass>4"
        launch_angle = "<angle>1"

        sym_dict = {
            rocket_mass: self.rocket_mass,
            planet_mass: self.planet_mass,
            planet_radius: self.planet_radius,
            v_exhaust: self.v_exhaust,
            dm_dt: self.dm_dt,
            min_mass: self.min_mass,
            launch_angle: self.launch_angle,
        }

        description = (
            f"In a hypothetical setting, a planet with mass {planet_mass} M and radius {planet_radius} R is present free from any other influeneces. "
            f"A rocket of mass {rocket_mass} M is launched making an angle {launch_angle} with the vertical from the planet's surface. "
            f"The rocket expels gas at a constant velocity of {v_exhaust} R/T w.r.t to the nozzle, losing mass at a constant rate of {dm_dt} M/T. "
            f"The rocket's weight apart from the fuel is {min_mass} M, meaning when the fuel is exhausted, the rocket's mass is {min_mass} M. "
            f"The units (mass: M, length: R, time: T) are custom such that the gravitational constant G = {self.G}; "
            f"they are not SI units."
        )

        if symbolic: return description, sym_dict

        description = replace_all(description, sym_dict)
        return description
    
    def connecting_point_nl(self, cd, cp, csi):
        raise NotImplementedError("RocketEntity is not supposed to have connections.")
    
    def get_question(self, sub_entity: str, quantity: str) -> str:
        """
        Get a question related to the entity
        
        Inputs:
            sub_entity: str
            quantity: str
            
        Returns:
            str
        """

        descriptior = "rocket"
        
        question = (
            f"What is the {quantity} of the {descriptior} in this hypothetical system"
        )
        
        return question
