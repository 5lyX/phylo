from .mass_entities import *
from tabulate import tabulate
import pandas as pd
from sim.utils import replace_all, generate_collision_pair
import copy
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
    axis2 = (-math.sin(theta), 0, math.cos(theta))
    
    # Clear existing joints and add two slide joints
    body.joints = []
    body.add_joint(Joint("slide", axis1, f"{body.name}.joint-x"))
    body.add_joint(Joint("slide", axis2, f"{body.name}.joint-z"))

class TwoDCollisionPlane(Entity):

    randomization_levels = {
        DegreeOfRandomization.EASY: {
            "num_masses": {"min": 2, "max": 3},
            "mass_value": {"min": 1, "max": 5, "integer": True},
            "velocity": {"min": 0.5, "max": 1.5},
            "plane_slope": {"min": 0, "max": 0},
            "min_distance": 1.0,
        },
        DegreeOfRandomization.MEDIUM: {
            "num_masses": {"min": 2, "max": 4},
            "mass_value": {"min": 0.5, "max": 10.0, "integer": False},
            "velocity": {"min": 0.1, "max": 3.0},
            "plane_slope": {"min": 0, "max": 30},
            "min_distance": 0.75,
        },
        DegreeOfRandomization.HARD: {
            "num_masses": {"min": 3, "max": 5},
            "mass_value": {"min": 0.1, "max": 20.0, "integer": False},
            "velocity": {"min": 0.1, "max": 5.0},
            "plane_slope": {"min": 20, "max": 70},
            "min_distance": 0.5,
        },
    }

    def __init__(
        self,
        name: str,
        pos: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        plane_slope: float = 0.0,  # in degrees
        mass_values: List[float] = [1.0],
        mass_positions: List[Tuple[float, float]] = [(0.0, 0.0)],
        mass_initial_velocities: List[Tuple[float, float]] = [(0.0, 0.0)],
        radii: List[float] = None,
        resolution_coefficient_list = None,
        **kwargs,
    ):
        self.plane_slope = plane_slope
        self.mass_values = mass_values
        self.mass_positions = mass_positions
        self.mass_initial_velocities = mass_initial_velocities
        self.radii = radii if radii else [DEFAULT_SPHERE_RADIUS] * len(mass_values)
        self.spheres: List[Sphere] = []
        
        if resolution_coefficient_list is not None:  # save for description
            self.resolution_coefficient_list = [
                (
                    f"{name}.sphere-{rc[0]}",
                    f"{name}.sphere-{rc[1]}",
                    rc[2],
                ) for rc in resolution_coefficient_list
            ]

        super().__init__(name, pos, entity_type=self.__class__.__name__, **kwargs)
        
        # Create the spheres
        self.create_spheres()
        self.set_quat_with_angle(self.plane_slope)
        
    def create_spheres(self):
        plane_slope = math.radians(self.plane_slope)
        
        # Generate collision sphere colors
        collision_colors = self._generate_collision_colors(len(self.mass_values))
        
        for i, (mass, pos_xy, vel_xy, radius) in enumerate(
            zip(self.mass_values, self.mass_positions, self.mass_initial_velocities, self.radii)
        ):
            sphere_name = f"{self.name}.sphere-{i}"
            x, y, z = *pos_xy, 0      
            x, y, z = x, y * math.cos(plane_slope), y * math.sin(plane_slope) # 0.0  # Assuming the spheres are on the plane at z = 0
            pos = (x, y, z)

            # Initial velocity components
            vx, vy = vel_xy
            vz = 0 # ROTATION NOT TAKEN CARE OF BY SET_QUAT_WITH_ANGLE
            # Adjust velocity based on the plane slope
            vx, vy, vz = vx, vy * math.cos(plane_slope), vy * math.sin(plane_slope)
            
            # Create sphere with movement constrained to the X-Z plane
            sphere = Sphere(
                name=sphere_name,
                pos=pos,
                radius=radius,
                mass=mass,
                joint_option=None,  # We'll add custom planar joints instead
                init_velocity={InitVelocityType.SPHERE: [vx, vy, vz, 0.0, 0.0, 0.0]},
                rgba=collision_colors[i],  # Dynamic collision colors
                material="reflectance",  # Use reflectance material instead of pulley
            )
            sphere.add_planar_joint(
                self.plane_slope
            )
            self.spheres.append(sphere)
            # self.add_body(sphere)

    def _generate_collision_colors(self, num_spheres):
        """Generate elegant, scientific paper-style colors for collision spheres"""
        import random
        import math
        
        colors = []
        
        # Elegant color palette inspired by scientific papers and mass colors
        # More muted and sophisticated colors
        collision_palette = [
            (0.8, 0.4, 0.4, 1.0),   # Muted red
            (0.4, 0.6, 0.8, 1.0),   # Soft blue
            (0.6, 0.8, 0.4, 1.0),   # Soft green
            (0.8, 0.6, 0.4, 1.0),   # Warm orange
            (0.7, 0.4, 0.7, 1.0),   # Muted purple
            (0.5, 0.7, 0.7, 1.0),   # Soft teal
            (0.8, 0.7, 0.5, 1.0),   # Warm beige
            (0.6, 0.5, 0.8, 1.0),   # Soft lavender
            (0.7, 0.6, 0.5, 1.0),   # Warm brown
            (0.5, 0.6, 0.7, 1.0),   # Soft slate
            (0.7, 0.5, 0.6, 1.0),   # Dusty rose
            (0.5, 0.7, 0.6, 1.0),   # Sage green
        ]
        
        for i in range(num_spheres):
            if i < len(collision_palette):
                base_color = collision_palette[i]
            else:
                # Generate sophisticated colors for additional spheres
                hue = (i * 137.5) % 360
                # Lower saturation and higher value for elegance
                saturation = 0.4 + 0.2 * random.random()  # More muted
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
            
            # Apply gentle gradient effect for sophistication
            gradient_factor = i / max(1, num_spheres - 1)
            gradient_color = (
                base_color[0] * (1 - gradient_factor * 0.1) + 0.1 * gradient_factor,
                base_color[1] * (1 - gradient_factor * 0.1) + 0.1 * gradient_factor,
                base_color[2] * (1 - gradient_factor * 0.1) + 0.1 * gradient_factor,
                1.0
            )
            
            # Add minimal variation to avoid identical colors
            variation = 0.02  # Much smaller variation for elegance
            varied_color = (
                max(0.2, min(0.9, gradient_color[0] + (random.random() - 0.5) * variation)),
                max(0.2, min(0.9, gradient_color[1] + (random.random() - 0.5) * variation)),
                max(0.2, min(0.9, gradient_color[2] + (random.random() - 0.5) * variation)),
                1.0
            )
            colors.append(varied_color)
        
        return colors

    def randomize_parameters_old(
        self,
        degree_of_randomization: DegreeOfRandomization = DegreeOfRandomization.NON_STRUCTURAL,
        reinitialize_instance: bool = False,
        **kwargs,
    ):
        """
        - STRUCTURAL: Change the number of spheres, their positions, velocities, and plane_slope significantly.
        - NON_STRUCTURAL: Only adjust numerical values (e.g. masses, velocity magnitudes) without changing the number or directions.
        """
        randomization_levels = {
            DegreeOfRandomization.EASY: {
                "num_masses": {"min": 2, "max": 3},
                "mass_value": {"min": 1, "max": 5, "integer": True},
                "velocity": {"min": 0.5, "max": 1.5},
                "plane_slope": {"min": 0, "max": 0},
                "min_distance": 1.0,
            },
            DegreeOfRandomization.MEDIUM: {
                "num_masses": {"min": 2, "max": 4},
                "mass_value": {"min": 0.5, "max": 10.0, "integer": False},
                "velocity": {"min": 0.1, "max": 3.0},
                "plane_slope": {"min": 0, "max": 30},
                "min_distance": 0.75,
            },
            DegreeOfRandomization.HARD: {
                "num_masses": {"min": 3, "max": 5},
                "mass_value": {"min": 0.1, "max": 20.0, "integer": False},
                "velocity": {"min": 0.1, "max": 5.0},
                "plane_slope": {"min": 20, "max": 70},
                "min_distance": 0.5,
            },
        }

        self.randomization_levels = randomization_levels

        if degree_of_randomization == DegreeOfRandomization.DEFAULT or degree_of_randomization == DegreeOfRandomization.NON_STRUCTURAL:
            options = [
                DegreeOfRandomization.EASY,
                DegreeOfRandomization.MEDIUM,
                DegreeOfRandomization.HARD,
            ]
            degree_of_randomization = random.choice(options)
            # print(f"Selected randomization level: {degree_of_randomization.name}")

        if degree_of_randomization != DegreeOfRandomization.NON_STRUCTURAL:
            params = randomization_levels.get(
                degree_of_randomization,
                randomization_levels[DegreeOfRandomization.MEDIUM],
            )

            num_masses = random.randint(
                params["num_masses"]["min"], params["num_masses"]["max"]
            )
            # print(f"Number of masses: {num_masses}")

            if params["mass_value"]["integer"]:
                self.mass_values = [
                    random.randint(
                        params["mass_value"]["min"], params["mass_value"]["max"]
                    )
                    for _ in range(num_masses)
                ]
            else:
                self.mass_values = [
                    random.uniform(
                        params["mass_value"]["min"], params["mass_value"]["max"]
                    )
                    for _ in range(num_masses)
                ]
            # print(f"Mass values: {self.mass_values}")

            self.mass_positions = []
            min_distance = params["min_distance"]
            attempts_limit = 100

            for i in range(num_masses):
                attempts = 0
                while attempts < attempts_limit:
                    pos = (random.uniform(-5, 5), random.uniform(-5, 5))
                    if all(
                        math.hypot(pos[0] - existing[0], pos[1] - existing[1])
                        > min_distance
                        for existing in self.mass_positions
                    ):
                        self.mass_positions.append(pos)
                        # print(f"Mass {i+1} position: {pos}")
                        break
                    attempts += 1
                else:
                    raise ValueError(
                        f"Could not place mass {i+1} after {attempts_limit} attempts. "
                        "Try reducing the number of masses or the minimum distance."
                    )

            self.mass_initial_velocities = [
                (
                    random.uniform(
                        -params["velocity"]["max"], params["velocity"]["max"]
                    ),
                    random.uniform(
                        -params["velocity"]["max"], params["velocity"]["max"]
                    ),
                )
                for _ in range(num_masses)
            ]
            # print(f"Initial velocities: {self.mass_initial_velocities}")

            self.plane_slope = random.uniform(
                params["plane_slope"]["min"], params["plane_slope"]["max"]
            )
            # print(f"Plane slope: {self.plane_slope:.2f} degrees")
        else:
            # NON_STRUCTURAL
            self.mass_values = [
                mv * random.uniform(0.9, 1.1) for mv in self.mass_values
            ]
            new_vels = []
            for vx, vy in self.mass_initial_velocities:
                speed = math.sqrt(vx**2 + vy**2)
                new_speed = speed * random.uniform(0.9, 1.1)
                ratio = new_speed / speed if speed != 0 else 1
                new_vels.append((vx * ratio, vy * ratio))
            self.mass_initial_velocities = new_vels
            self.plane_slope = max(0, min(90, self.plane_slope + random.uniform(-5, 5)))

        # Re-initialize if needed
        if reinitialize_instance:
            self.reinitialize()

    def randomize_parameters(
        self,
        degree_of_randomization: DegreeOfRandomization = DegreeOfRandomization.NON_STRUCTURAL,
        reinitialize_instance: bool = False,
        **kwargs,
    ):
        """
        - STRUCTURAL: Change the number of spheres, their positions, velocities, and plane_slope significantly.
        - NON_STRUCTURAL: Only adjust numerical values (e.g. masses, velocity magnitudes) without changing the number or directions.
        """
        randomization_levels = {
            DegreeOfRandomization.EASY: {
                "num_masses": {"min": 2, "max": 3},
                "mass_value": {"min": 1, "max": 5, "integer": True},
                "speed": {"min": 0.5, "max": 1.5},
                "plane_slope": {"min": 0, "max": 0},
                "min_distance": 2.5 * DEFAULT_SPHERE_RADIUS,
                "time_to_impact": {"min": 0.2, "max": 1.5},
                "radius": {"min": DEFAULT_SPHERE_RADIUS, "max": 3 * DEFAULT_SPHERE_RADIUS},
                "restitution_coeff": {"min": 1, "max": 1},
            },
            DegreeOfRandomization.MEDIUM: {
                "num_masses": {"min": 2, "max": 4},
                "mass_value": {"min": 0.5, "max": 10.0, "integer": False},
                "speed": {"min": 0.1, "max": 3.0},
                "plane_slope": {"min": 0, "max": 0},
                "min_distance": 2.5 * DEFAULT_SPHERE_RADIUS,
                "time_to_impact": {"min": 0.2, "max": 1.5},
                "radius": {"min": 0.5 * DEFAULT_SPHERE_RADIUS, "max": 3 * DEFAULT_SPHERE_RADIUS},
                "restitution_coeff": {"min": 0, "max": 1},
            },
            DegreeOfRandomization.HARD: {
                "num_masses": {"min": 3, "max": 5},
                "mass_value": {"min": 0.1, "max": 20.0, "integer": False},
                "speed": {"min": 0.1, "max": 5.0},
                "plane_slope": {"min": 0, "max": 70},
                "min_distance": 2.5 * DEFAULT_SPHERE_RADIUS,
                "time_to_impact": {"min": 0.2, "max": 1.5},
                "radius": {"min": 0.5 * DEFAULT_SPHERE_RADIUS, "max": 4 * DEFAULT_SPHERE_RADIUS},
                "restitution_coeff": {"min": 0, "max": 1},
            },
        }

        self.randomization_levels = randomization_levels

        if degree_of_randomization == DegreeOfRandomization.DEFAULT or degree_of_randomization == DegreeOfRandomization.NON_STRUCTURAL:
            options = [
                DegreeOfRandomization.EASY,
                DegreeOfRandomization.MEDIUM,
                DegreeOfRandomization.HARD,
            ]
            degree_of_randomization = random.choice(options)
            # print(f"Selected randomization level: {degree_of_randomization.name}")

        if degree_of_randomization != DegreeOfRandomization.NON_STRUCTURAL:
            params = randomization_levels.get(
                degree_of_randomization,
                randomization_levels[DegreeOfRandomization.MEDIUM],
            )

            num_masses = random.randint(
                params["num_masses"]["min"], params["num_masses"]["max"]
            )
            
            if params["mass_value"]["integer"]:
                self.mass_values = [
                    random.randint(
                        params["mass_value"]["min"], params["mass_value"]["max"]
                    )
                    for _ in range(num_masses)
                ]
            else:
                self.mass_values = [
                    random.uniform(
                        params["mass_value"]["min"], params["mass_value"]["max"]
                    )
                    for _ in range(num_masses)
                ]

            self.radii = [
                random.uniform(
                    params["radius"]["min"], params["radius"]["max"]
                )
                for _ in range(num_masses)
            ]
            
            min_distance = params["min_distance"]

            speed_range = (params["speed"]["min"], params["speed"]["max"])
            t_impact_range = (params["time_to_impact"]["min"], params["time_to_impact"]["max"])

            # We'll generate at least num_masses // 2 collision pairs
            sphere_positions = []
            sphere_velocities = []
            coefficient_of_restitution = []

            assigned = set()
            sphere_id = 0
            
            num_collision_pairs = random.randint(1, num_masses // 2)
                
            while len(assigned) < num_collision_pairs * 2:
                # Choose a random impact point
                cx, cy = random.uniform(-4, 4), random.uniform(-4, 4)
                t_c = random.uniform(*t_impact_range)

                pair = generate_collision_pair((cx, cy), t_c, speed_range, min_distance, sphere_positions)
                if pair is None:
                    num_collision_pairs -= 1
                    continue
                
                if sphere_id + 2 > num_masses:
                    break  # leave room for extra random spheres if needed

                pair["ids"] = [sphere_id, sphere_id + 1]

                e = random.uniform(params["restitution_coeff"]["min"], params["restitution_coeff"]["max"])
                coefficient_of_restitution.append((sphere_id, sphere_id + 1, e))

                sphere_positions.extend(pair["positions"])
                sphere_velocities.extend(pair["velocities"])
                assigned.update(pair["ids"])
                sphere_id += 2
            
            for sphere_id in range(2 * num_collision_pairs, num_masses):
                # Generate random positions and velocities for the remaining spheres
                attempt = 0
                while attempt < 10:
                    pos = (random.uniform(-4, 4), random.uniform(-4, 4))
                    if all(
                        math.hypot(pos[0] - existing[0], pos[1] - existing[1])
                        > min_distance
                        for existing in sphere_positions
                    ):
                        sphere_positions.append(pos)
                        break
                    attempt += 1
                else:
                    continue

                # Generate random velocity
                vx = random.uniform(*speed_range)
                vy = random.uniform(*speed_range)
                sphere_velocities.append((vx, vy))

            self.mass_positions = sphere_positions
            self.mass_initial_velocities = sphere_velocities
            
            self.plane_slope = random.uniform(
                params["plane_slope"]["min"], params["plane_slope"]["max"]
            )

            # Shuffle the positions, masses and velocities so that RL doesn't learn to cheat
            combined = list(zip(range(num_masses), self.mass_positions, self.mass_values, self.mass_initial_velocities))
            random.shuffle(combined)
            shuffled_idx, self.mass_positions, self.mass_values, self.mass_initial_velocities = zip(*combined)

            coefficient_of_restitution = [
                (shuffled_idx[idx1], shuffled_idx[idx2], e)
                for idx1, idx2, e in coefficient_of_restitution
            ]

        else:
            # NON_STRUCTURAL
            raise NotImplementedError("This should be unreachable code, check for any errors.")
            pass
            # self.mass_values = [
            #     mv * random.uniform(0.9, 1.1) for mv in self.mass_values
            # ]
            # new_vels = []
            # for vx, vy in self.mass_initial_velocities:
            #     speed = math.sqrt(vx**2 + vy**2)
            #     new_speed = speed * random.uniform(0.9, 1.1)
            #     ratio = new_speed / speed if speed != 0 else 1
            #     new_vels.append((vx * ratio, vy * ratio))
            # self.mass_initial_velocities = new_vels
            # self.plane_slope = max(0, min(90, self.plane_slope + random.uniform(-5, 5)))

        # Round
        self.radii = [
            round(radius, 2) for radius in self.radii
        ]
        self.mass_values = [round(mv, 2) for mv in self.mass_values]
        self.mass_positions = [
            (round(pos[0], 2), round(pos[1], 2)) for pos in self.mass_positions
        ]
        self.mass_initial_velocities = [
            (round(vel[0], 2), round(vel[1], 2)) for vel in self.mass_initial_velocities
        ]
        self.plane_slope = round(self.plane_slope, 2)
        self.resolution_coefficient_list = [
            (f"{self.name}.sphere-{idx1}", f"{self.name}.sphere-{idx2}", round(e, 2)) for idx1, idx2, e in coefficient_of_restitution
        ]

        # Re-initialize if needed
        if reinitialize_instance:
            self.reinitialize()

    def generate_entity_yaml(
        self,
        use_random_parameters: bool = False,
        degree_of_randomization: DegreeOfRandomization = DegreeOfRandomization.NON_STRUCTURAL,
    ) -> dict:
        """
        Export entity parameters to a dict (YAML-friendly).
        """
        if use_random_parameters:
            self.randomize_parameters(
                degree_of_randomization
            )

        data = {
            "name": self.name,
            "type": self.__class__.__name__,
            "position": list(self.pos),
            "parameters": {
                "plane_slope": self.plane_slope,
                "mass_values": self.mass_values,
                "mass_positions": list(map(list, self.mass_positions)),
                "mass_initial_velocities": list(map(list, self.mass_initial_velocities)),
                "radii": self.radii,
                "resolution_coefficient_list": [
                    [n1.split('-')[-1], n2.split('-')[-1], coeff]
                    for n1, n2, coeff in self.resolution_coefficient_list
                ]
            },
        }

        return round_floats(data)

    def get_parameters(self) -> List[dict]:
        """
        Return details about all spheres (e.g. mass and names).
        """
        param_list = []
        for sphere in self.spheres:
            for geom in sphere.geoms:
                param_list.append(
                    {
                        "body_name": sphere.name,
                        "geom_name": geom.name,
                        "mass": float(geom.mass) if geom.mass else 0.0,
                    }
                )
        return param_list

    def to_xml(self) -> str:
        body_xml = (
            f"""<body name="{self.name}" pos="{' '.join(map(str, self.pos))}">\n"""
        )
        for sphere in self.spheres:
            body_xml += sphere.to_xml() + "\n"
        body_xml += "</body>"
        return body_xml

    def get_description(self, simDSL2nlq=False):
        if not simDSL2nlq:
            return super().get_description()

        descriptions = []

        for sphere in self.spheres:
            mass = sphere.geoms[0].mass
            x, y, *_ = tuple(sphere.pos)
            vx, vy, *_ = tuple(sphere.get_init_velocities()[sphere.name])

            body_name = sphere.name
            description = {
                "body_name": body_name,
                "mass": mass,
                "init_velocity": (vx, vy),
                "init_position": (x, y),
                "description": f"A sphere {body_name} has a mass of {mass} kg and radius {sphere.geoms[0].size} m, and is placed at ({x}, {y}) on a table. It is initially moving with a velocity of ({vx}, {vy}) m/s.",
            }

            descriptions.append(description)

        if hasattr(self, "resolution_coefficient_list"):
            for idx1, idx2, coeff in self.resolution_coefficient_list:
                descriptions[idx1][
                    "description"
                ] += f" The coefficient of restitution between {descriptions[idx1]['body_name']} and {descriptions[idx2]['body_name']} is {coeff}."
                descriptions[idx2][
                    "description"
                ] += f" The coefficient of restitution between {descriptions[idx2]['body_name']} and {descriptions[idx1]['body_name']} is {coeff}."

        return descriptions
    
    def get_nlq(self, symbolic = False):
        num_bodies = len(self.mass_values)

        sym_dict = {}

        plane_slope = "<angle>1" if symbolic else self.plane_slope

        table_description = f"In a system called '{self.name}', a smooth table (XY plane) has"
        sphere_description = "a sphere on it with the following properties:"
        if num_bodies > 1:
            sphere_description = f"{num_bodies} spheres on it with the following properties (position and velocity are relative to the table):"

        inclination_description = f"The table is now rotated, counterclockwise, along the x axis by an angle {plane_slope} degrees, therefore rotating the position and initial velocities too in global frame."

        propoerties = {
            "Sphere": [i + 1 for i in range(num_bodies)],
            "Mass (kg)": [f"{f'<mass>{i+1}' if symbolic else self.mass_values[i]}" for i in range(num_bodies)],
            "Initial Position (m)": [f"({self.mass_positions[i][0]}, {self.mass_positions[i][1]})" for i in range(num_bodies)],
            "Initial Velocity (m/s)": [f"({self.mass_initial_velocities[i][0]}, {self.mass_initial_velocities[i][1]})" for i in range(num_bodies)],
            "Radius (m)": list(map(str, self.radii)),
            # "Initial Position (m)": [f"({f'<x>{i+1}' if symbolic else self.mass_positions[i][0]}, {f'<y>{i+1}' if symbolic else self.mass_positions[i][1]})" for i in range(num_bodies)],
            # "Initial Velocity (m/s)": [f"({f'<vx>{i+1}' if symbolic else self.mass_initial_velocities[i][0]}, {f'<vy>{i+1}' if symbolic else self.mass_initial_velocities[i][1]})" for i in range(num_bodies)]
        }

        df = pd.DataFrame(propoerties)
        properties_description = tabulate(df, headers='keys', tablefmt='github', showindex=False) + "\n"

        if symbolic:
            sym_dict[plane_slope] = self.plane_slope

            sym_dict.update(
                {
                    f"<mass>{i+1}": self.mass_values[i] for i in range(num_bodies)

                }
            )

            # sym_dict.update(
            #     {
            #         f"<x>{i+1}": self.mass_positions[i][0] for i in range(num_bodies)

            #     }
            # )

            # sym_dict.update(
            #     {
            #         f"<y>{i+1}": self.mass_positions[i][1] for i in range(num_bodies)

            #     }
            # )

            # sym_dict.update(
            #     {
            #         f"<vx>{i+1}": self.mass_initial_velocities[i][0] for i in range(num_bodies)

            #     }
            # )

            # sym_dict.update(
            #     {
            #         f"<vy>{i+1}": self.mass_initial_velocities[i][1] for i in range(num_bodies)

            #     }
            # )

        restitution_description = f""
        if hasattr(self, "resolution_coefficient_list") and len(self.resolution_coefficient_list):
            restitution_description = "The coefficient of restitution between each pair of spheres is given as:\n"
            for idx1, idx2, coeff in self.resolution_coefficient_list:
                
                idx1, idx2 = int(idx1.split('-')[-1]), int(idx2.split('-')[-1])
                restitution_description += f"\\(e_{{{idx1+1},{idx2+1}}} = {f'<restitution>{idx1}{idx2}' if symbolic else coeff}\\), "
            
            restitution_description = restitution_description.rstrip(", ")

            if symbolic:
                sym_dict.update(
                    {
                        f"<restitution>{idx1.split('-')[-1]}{idx2.split('-')[-1]}": coeff for idx1, idx2, coeff in self.resolution_coefficient_list
                    }
                )

        # Add position, size and table slope info
        # Add coeff of restitution info
        
        description = (
            f"{table_description} {sphere_description}\n{properties_description}\n{inclination_description}\n{restitution_description}"
        )

        if symbolic: return description, sym_dict
        return description 

    def connecting_point_nl(self, cd, cp, csi):
        raise NotImplementedError("TwoDCollisionPlane is not supposed to have connections.")
    
    def get_question(self, sub_entity: str, quantity: str) -> str:
        """
        Get a question related to the entity
        
        Inputs:
            sub_entity: str
            quantity: str
            
        Returns:
            str
        """

        idx = int(sub_entity[7:]) # remove the "sphere-"
        
        question = (
            f"What is the {quantity} of the {(['1st', '2nd', '3rd'] + [f'{i + 4}th' for i in range(len(self.mass_values) - 3)])[idx]} sphere in the system '{self.name}'"
        )
        
        return question

class SpringBlockEntity(SpringBlock, Entity):

    # Define parameter ranges for each randomization level
    randomization_levels = {
        DegreeOfRandomization.EASY: {
            "num_springs": {"min": 1, "max": 3},
            "original_length": {"min": 0.5, "max": 1.0},
            "stiffness": {"min": 50, "max": 150},
            "connecting_angle": {"min": -30, "max": 30},
            "connecting_distance": {"min": 1.0, "max": 1.5},
        },
        DegreeOfRandomization.MEDIUM: {
            "num_springs": {"min": 2, "max": 4},
            "original_length": {"min": 0.3, "max": 1.5},
            "stiffness": {"min": 30, "max": 200},
            "connecting_angle": {"min": -45, "max": 45},
            "connecting_distance": {"min": 0.8, "max": 2.0},
        },
        DegreeOfRandomization.HARD: {
            "num_springs": {"min": 3, "max": 5},
            "original_length": {"min": 0.1, "max": 2.0},
            "stiffness": {"min": 20, "max": 250},
            "connecting_angle": {"min": -60, "max": 60},
            "connecting_distance": {"min": 0.5, "max": 2.5},
        },
    }

    def __init__(
        self,
        name: str,
        pos: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        original_lengths: List[float] = [0.5],
        stiffnesses: List[float] = [100.0],
        connecting_angles: List[float] = [0.0],
        connecting_distances: List[float] = [1.5],
        init_randomization_degree: DegreeOfRandomization = None,
        **kwargs,
    ):
        self.original_lengths = original_lengths
        self.stiffnesses = stiffnesses
        self.connecting_angles = connecting_angles
        self.connecting_distances = connecting_distances
        if (
            init_randomization_degree
        ):  # additional adjustment needed because of complex inherit issue
            self.randomize_parameters(
                init_randomization_degree, reinitialize_instance=False
            )
            init_randomization_degree = None

        super().__init__(
            name,
            pos,
            original_lengths=self.original_lengths,
            connecting_angles=self.connecting_angles,
            connecting_distances=self.connecting_distances,
            stiffnesses=self.stiffnesses,
            entity_type=self.__class__.__name__,
            init_randomization_degree=init_randomization_degree,
            **kwargs,
        )
        # this should happen after the super().__init__ call to
        # ensure that the connecting points reset
        self.initialize_connecting_points(
            connection_constraints={
                ConnectingPoint.DEFAULT: 1,
                ConnectingPoint.SURROUNDING: len(self.original_lengths),
            }
        )
        pass

    def get_connecting_tendon_sequence(
        self,
        direction: ConnectingDirection,
        connecting_point: ConnectingPoint = ConnectingPoint.DEFAULT,
        connecting_point_seq_id: Optional[ConnectingPointSeqId] = None,
        use_sidesite: bool = False,
    ) -> TendonSequence:
        """
        Get the specific tendon sequence for the given connecting point.
        """
        connecting_point_seq_id = (
            connecting_point_seq_id - 1
            if connecting_point_seq_id and connecting_point_seq_id > 0
            else 0
        )
        
        if connecting_point == ConnectingPoint.DEFAULT:
            sequence = self.get_connecting_tendon_sequences(direction)[
                0
            ]
        elif connecting_point == ConnectingPoint.SURROUNDING:
            sequence = self.get_connecting_tendon_sequences(
                direction, connecting_option=self.ConnectOption.SURROUNDING_SITES
            )[connecting_point_seq_id]
        else:
            raise ValueError(f"Unknown connecting_point: {connecting_point}")
        
        return TendonSequence(
            elements=sequence.get_elements(),
            description=f"Tendon sequence for connecting point {connecting_point}",
            name=f"{self.name}.connecting_tendon"
        )

    def randomize_parameters(
        self,
        degree_of_randomization: DegreeOfRandomization = DegreeOfRandomization.NON_STRUCTURAL,
        reinitialize_instance: bool = False,
        **kwargs,
    ):
        """
        Divides randomization into two types: STRUCTURAL and NON_STRUCTURAL
        """
        # Define parameter ranges for each randomization level
        randomization_levels = {
            DegreeOfRandomization.EASY: {
                "num_springs": {"min": 1, "max": 3},
                "original_length": {"min": 0.5, "max": 1.0},
                "stiffness": {"min": 5, "max": 10},
                "connecting_angle": {"min": -30, "max": 30},
                "initial_delta": {"min": -0.5, "max": 0.5},
            },
            DegreeOfRandomization.MEDIUM: {
                "num_springs": {"min": 2, "max": 4},
                "original_length": {"min": 0.3, "max": 1.5},
                "stiffness": {"min": 5, "max": 20},
                "connecting_angle": {"min": -45, "max": 45},
                "initial_delta": {"min": -0.5, "max": 1},
            },
            DegreeOfRandomization.HARD: {
                "num_springs": {"min": 3, "max": 5},
                "original_length": {"min": 0.1, "max": 2.0},
                "stiffness": {"min": 5, "max": 40},
                "connecting_angle": {"min": -60, "max": 60},
                "initial_delta": {"min": -1, "max": 2},
            },
        }

        self.randomization_levels = randomization_levels

        # If degree_of_randomization is DEFAULT, randomly choose EASY, MEDIUM, or HARD
        if degree_of_randomization == DegreeOfRandomization.DEFAULT or degree_of_randomization == DegreeOfRandomization.NON_STRUCTURAL:
            options = [
                DegreeOfRandomization.EASY,
                DegreeOfRandomization.MEDIUM,
                DegreeOfRandomization.HARD,
            ]
            degree_of_randomization = random.choice(options)
            # print(f"Selected randomization level: {degree_of_randomization.name}")

        if degree_of_randomization != DegreeOfRandomization.NON_STRUCTURAL:
            # Get the parameters for the selected randomization level
            params = randomization_levels.get(
                degree_of_randomization,
                randomization_levels[DegreeOfRandomization.MEDIUM],
            )
            # Randomize the number of springs based on the level
            num_springs = random.randint(
                params["num_springs"]["min"], params["num_springs"]["max"]
            )
            # Randomize original lengths
            self.original_lengths = [
                random.uniform(
                    params["original_length"]["min"], params["original_length"]["max"]
                )
                for _ in range(num_springs)
            ]
            # Randomize stiffnesses
            self.stiffnesses = [
                random.uniform(params["stiffness"]["min"], params["stiffness"]["max"])
                for _ in range(num_springs)
            ]
            # Randomize connecting angles
            self.connecting_angles = [
                random.uniform(
                    params["connecting_angle"]["min"], params["connecting_angle"]["max"]
                )
                for _ in range(num_springs)
            ]
            # Randomize connecting distances (initial lengths)
            self.connecting_distances = [
                self.original_lengths[_] + random.uniform(
                    params["initial_delta"]["min"],
                    params["initial_delta"]["max"],
                )
                for _ in range(num_springs)
            ]
        else:
            # Non-structural randomization: only modify existing spring parameters without changing the number of springs
            # Adjust original lengths
            self.original_lengths = [
                ol * random.uniform(0.9, 1.1) for ol in self.original_lengths
            ]
            # Adjust stiffnesses
            self.stiffnesses = [k * random.uniform(0.9, 1.1) for k in self.stiffnesses]
            # Adjust connecting angles
            self.connecting_angles = [
                ca + random.uniform(-5, 5) for ca in self.connecting_angles
            ]
            # Adjust connecting distances
            self.connecting_distances = [
                cd * random.uniform(0.9, 1.1) for cd in self.connecting_distances
            ]

        # Reinitialize the instance if required
        if reinitialize_instance:
            self.reinitialize()

    def generate_entity_yaml(
        self,
        use_random_parameters: bool = False,
        degree_of_randomization: DegreeOfRandomization = DegreeOfRandomization.NON_STRUCTURAL,
    ) -> dict:
        if use_random_parameters:
            self.randomize_parameters(degree_of_randomization)

        entity_dict = {
            "name": self.name,
            "type": self.__class__.__name__,
            "position": list(self.pos),
            "parameters": {
                "original_lengths": self.original_lengths,
                "stiffnesses": self.stiffnesses,
                "connecting_angles": self.connecting_angles,
                "connecting_distances": self.connecting_distances,
            },
        }
        return round_floats(entity_dict)

    def get_parameters(self) -> List[dict]:
        """
        If parameters for internal masses or other geometric entities are needed,
        iterate over self.child_bodies or other collections here.
        """
        param_list = []
        # Example: storing information about the number of springs and their details
        for i, (angle, ol, k) in enumerate(
            zip(self.connecting_angles, self.original_lengths, self.stiffnesses)
        ):
            param_list.append(
                {"spring_index": i, "angle": angle, "original_length": ol, "stiffness": k, "name": self.springs[i].name}
            )
        return param_list

    def get_description(self, simDSL2nlq=False):
        if not simDSL2nlq:
            return super().get_description()

        descriptions = []
        mass = self.tray_mass.mass_value

        params = {}
        params["angle"] = list(map(str, self.connecting_angles))
        params["connecting_distance"] = list(map(str, self.connecting_distances))
        params["stiffness"] = list(map(str, self.stiffnesses))
        params["natural_length"] = list(map(str, self.original_lengths))

        description = {
            "name": self.tray_mass.name,
            "angle": params["angle"],
            "natural_length": params["natural_length"],
            "stiffness": params["stiffness"],
            "connecting_distance": params["connecting_distance"],
            "mass": mass,
            "description": (
                f"A block named {self.tray_mass.name} is suspended in the air and attached to {len(self.springs)} springs, each with other end fixed on the wall."
                f"Mass of the block is {mass} kg."
                f'The springs have stiffness {", ".join(params["stiffness"])}, make angles {", ".join(params["angle"])} degrees with horizontal,'
                f'have natural lengths {", ".join(params["natural_length"])},'
                f'and initially streched / compressed to be of length {", ".join(params["connecting_distance"])}.'
            ),
        }

        descriptions.append(description)

        return descriptions

    def get_nlq(self, symbolic = False):
        mass = "<mass>1"
        sym_dict = {mass: self.tray_mass.mass_value}

        num_springs = len(self.connecting_angles)
        angles = self.connecting_angles
        natural_lengths = self.original_lengths 
        stiffnesses = [f"<k>{i}" for i in range(1, num_springs + 1)]
        sym_dict.update(
            {
                stiffnesses[i]: self.stiffnesses[i] for i in range(num_springs)
            }
        )
        current_lengths = self.connecting_distances

        angles = list(map(str, angles))
        natural_lengths = list(map(str, natural_lengths))
        current_lengths = list(map(str, current_lengths))

        descriptions = (
            f"A block of mass {mass} is connected to {num_springs} springs (in parallel) each fixed at one end and attached to the block at the other end. "
            f"The springs are initially inclined at angles of {convert_list_to_natural_language(angles)} degrees with the horizontal. "
            f"The springs have stiffness {convert_list_to_natural_language(stiffnesses)} N/m. "
            f" While their natural lengths are {convert_list_to_natural_language(natural_lengths)} m, they are initially streched / compressed "
            f"to be of lengths {convert_list_to_natural_language(current_lengths)} m."
        )

        if symbolic: return descriptions, sym_dict
        return replace_all(descriptions, sym_dict)

    def connecting_point_nl(self, cd, cp, csi):
        raise NotImplementedError("SpringBlockEntity is not supposed to have connections.")
    
    def get_question(self, sub_entity: str, quantity: str) -> str:
        """
        Get a question related to the entity
        
        Inputs:
            sub_entity: str
            quantity: str
            
        Returns:
            str
        """

        return f"What is the {quantity} of the block"

class ComplexCollisionPlane(Entity):

    # Define parameter ranges for each randomization level
    # Define parameter ranges for each randomization level
    randomization_levels = {
        DegreeOfRandomization.EASY: {
            "num_bodies": {"min": 2, "max": 3},
            "plane_slope": {"min": 0, "max": 0},  # No slope in EASY
            "position_spacing": 0.3,
            "body_types": ["mass", "fixed_spring"],
            "collision_body_params": {
                "mass": {
                    "mass_value": {"min": 0.5, "max": 3.0},
                    "position": {
                        "min": 0.5 * COLLISION_BODY_POSITION_RANGE["min"],
                        "max": 0.5 * COLLISION_BODY_POSITION_RANGE["max"],
                    },
                    "init_velocity": {"min": -1.0, "max": 1.0},
                },
                "fixed_spring": {
                    "original_length": {"min": 0.1, "max": 0.3},
                    "position_spacing": {"min": 0, "max": 0},  # No slope in EASY
                    "k": {"min": 50, "max": 150},
                },
            },
        },
        DegreeOfRandomization.MEDIUM: {
            "num_bodies": {"min": 2, "max": 4},
            "plane_slope": {"min": 0, "max": 30},
            "position_spacing": 0.3,
            "body_types": ["mass", "spring_mass", "fixed_mass"],
            "collision_body_params": {
                "mass": {
                    "mass_value": {"min": 0.5, "max": 5.0},
                    "position": {
                        "min": COLLISION_BODY_POSITION_RANGE["min"],
                        "max": COLLISION_BODY_POSITION_RANGE["max"],
                    },
                    "init_velocity": {"min": -2.0, "max": 2.0},
                },
                "fixed_mass": {
                    "mass_value": {"min": 0.5, "max": 5.0},
                    "position": {
                        "min": COLLISION_BODY_POSITION_RANGE["min"],
                        "max": COLLISION_BODY_POSITION_RANGE["max"],
                    },
                },
                "spring_mass": {
                    "mass_value": {"min": 0.5, "max": 3.0},
                    "num_mass": {"min": 2, "max": 4},
                    "position": {
                        "min": COLLISION_BODY_POSITION_RANGE["min"],
                        "max": COLLISION_BODY_POSITION_RANGE["max"],
                    },
                    "spring": {
                        "k": {"min": 50, "max": 200},
                        "original_length": {"min": 0.1, "max": 0.3},
                    },
                },
                "fixed_spring": {
                    "original_length": {"min": 0.1, "max": 0.3},
                    "slope": {"min": 0, "max": 0},
                    "k": {"min": 50, "max": 200},
                },
            },
        },
        DegreeOfRandomization.HARD: {
            "num_bodies": {"min": 3, "max": 5},
            "plane_slope": {"min": 15, "max": 45},
            "position_spacing": 0.3,
            "body_types": [
                "mass",
                "sphere",
                "fixed_mass",
                "spring_mass",
                "fixed_spring",
            ],
            "collision_body_params": {
                "mass": {
                    "mass_value": {"min": 0.5, "max": 10.0},
                    "position": {
                        "min": 0.5 * COLLISION_BODY_POSITION_RANGE["min"],
                        "max": 0.5 * COLLISION_BODY_POSITION_RANGE["max"],
                    },
                    "init_velocity": {"min": -3.0, "max": 3.0},
                },
                "sphere": {
                    "mass_value": {"min": 0.5, "max": 10.0},
                    "radius": {"min": 0.5 * DEFAULT_SPHERE_RADIUS, "max": 1.5 * DEFAULT_SPHERE_RADIUS},
                    "position": {
                        "min": 0.5 * COLLISION_BODY_POSITION_RANGE["min"],
                        "max": 0.5 * COLLISION_BODY_POSITION_RANGE["max"],
                    },
                    "init_velocity": {"min": -3.0, "max": 3.0},
                },
                "fixed_mass": {
                    "mass_value": {"min": 0.5, "max": 10.0},
                    "position": {
                        "min": 0.5 * COLLISION_BODY_POSITION_RANGE["min"],
                        "max": 0.5 * COLLISION_BODY_POSITION_RANGE["max"],
                    },
                },
                "spring_mass": {
                    "mass_value": {"min": 0.5, "max": 4.0},
                    "num_mass": {"min": 2, "max": 5},
                    "position": {
                        "min": 0.5 * COLLISION_BODY_POSITION_RANGE["min"],
                        "max": 0.5 * COLLISION_BODY_POSITION_RANGE["max"],
                    },
                    "spring": {
                        "k": {"min": 50, "max": 250},
                        "original_length": {"min": 0.1, "max": 0.2},
                    },
                },
                "fixed_spring": {
                    "original_length": {"min": 0.1, "max": 0.3},
                    "k": {"min": 50, "max": 250},
                    "position": {
                        "min": COLLISION_BODY_POSITION_RANGE["min"],
                        "max": COLLISION_BODY_POSITION_RANGE["max"],
                    },
                },
            },
        },
    }

    def __init__(
        self,
        name: str,
        pos: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        plane_slope: float = 0.0,  # in degrees
        collision_bodies: List[dict] = None,
        resolution_coefficients: List[
            float
        ] = None,  # The collision coefficients between each collision body
        **kwargs,
    ):
        # Create a quaternion to rotate the plane
        theta = math.radians(-plane_slope)
        qx = 0.0
        qy = math.sin(theta / 2)
        qz = 0.0
        qw = math.cos(theta / 2)
        quat = (qw, qx, qy, qz)
        self.resolution_coefficients = resolution_coefficients or []
        self.plane_slope = plane_slope
        self.collision_bodies = collision_bodies or []
        self.bodies = []

        super().__init__(
            name, pos, quat=quat, entity_type=self.__class__.__name__, **kwargs
        )

        # Create the plane
        self.plane = Plane(
            name=f"{self.name}.plane",
            size=(DEFAULT_COLLISION_PLANE_LENGTH, DEFAULT_PLANE_WIDTH, DEFAULT_PLANE_THICKNESS),
            quat=(1, 0, 0, 0),  # No rotation
        )

        # Create the collision bodies
        for i, body_spec in enumerate(self.collision_bodies):
            body_type = body_spec.get("body_type")
            if body_type == "mass":
                self.create_mass_body(i, body_spec)
            elif body_type == "sphere":
                self.create_sphere_body(i, body_spec)
            elif body_type == "fixed_mass":
                self.create_fixed_mass_body(i, body_spec)
            elif body_type == "spring_mass":
                self.create_spring_mass_body(i, body_spec)
            elif body_type == "fixed_spring":
                self.create_fixed_spring_body(i, body_spec)
            else:
                raise ValueError(f"Unknown body_type: {body_type}")
            if i > 0:  # add collision coefficients between the all collision bodies
                self.resolution_coefficient_list.extend(
                    [
                        (
                            g1.name[:-5],
                            g2.name[:-5],
                            (
                                1
                                if self.resolution_coefficients == []
                                else self.resolution_coefficients[i - 1]
                            )
                        ) for g1 in get_all_geoms_in_body(self.bodies[-1]) for g2 in get_all_geoms_in_body(self.bodies[-2])
                    ]
                )
            # # Add restitution with plane
            # self.resolution_coefficient_list.extend(
            #     [
            #         (
            #             g.name[:-5],
            #             self.plane.name,
            #             (
            #                 0
            #             ),
            #         )
            #         for g in get_all_geoms_in_body(self.bodies[-1])
            #     ]
            # )

        # Remove duplicate tuples in resolution_coefficient_list
        unique_pairs = {}
        for x, y, coeff in self.resolution_coefficient_list:
            pair = tuple(sorted((x, y)))  # Ensure (x, y) and (y, x) are treated the same
            unique_pairs[pair] = coeff
        self.resolution_coefficient_list = list((x, y, coeff) for (x, y), coeff in unique_pairs.items())

    def create_mass_body(self, index: int, body_spec: dict):
        mass_value = body_spec.get("mass_value", 1.0)
        position_along_x = body_spec.get("position", 0.0)
        init_velocity_along_x = body_spec.get("init_velocity", 0.0)
        pos_xyz = (position_along_x, 0.0, DEFAULT_MASS_SIZE + self.plane.size[2])
        init_velocity = [init_velocity_along_x, 0.0, 0.0, 0.0, 0.0, 0.0]
        constant_force = body_spec.get("constant_force", {})

        mass_body = Mass(
            name=f"{self.name}.mass-{index}",
            positions=[(0, 0, 0)],
            mass_value=mass_value,
            constant_force=constant_force,
            joint_option=None,  # We'll add custom planar joints instead
        )
        mass_body.set_pose(pos_xyz)
        # Add X-Z planar joint for the mass
        add_xz_planar_joint(mass_body, self.plane_slope)
        mass_body.init_velocity_dict[mass_body.name] = init_velocity
        self.bodies.append(mass_body)

    def create_sphere_body(self, index: int, body_spec: dict):
        mass_value = body_spec.get("mass_value", 1.0)
        position_along_x = body_spec.get("position", 0.0)
        init_velocity_along_x = body_spec.get("init_velocity", 0.0)
        radius = body_spec.get("radius", DEFAULT_SPHERE_RADIUS)
        pos_xyz = (position_along_x, 0.0, radius + self.plane.size[2])
        init_velocity = [init_velocity_along_x, 0.0, 0.0, 0.0, 0.0, 0.0]
        constant_force = body_spec.get("constant_force", {})

        sphere_body = Sphere(
            name=f"{self.name}.sphere-{index}",
            pos=pos_xyz,
            radius=radius,
            mass=mass_value,
            joint_option=None,  # We'll add custom planar joints instead
            init_velocity={InitVelocityType.SPHERE: init_velocity},
            constant_force=constant_force,
        )
        # Add X-Z planar joint for the sphere
        add_xz_planar_joint(sphere_body, self.plane_slope)
        self.bodies.append(sphere_body)

    def create_fixed_mass_body(self, index: int, body_spec: dict):
        position_along_x = body_spec.get("position", 0.0)
        mass_size = body_spec.get("mass_size", DEFAULT_MASS_SIZE)
        pos_xyz = (position_along_x, 0.0, self.plane.size[2] + DEFAULT_MASS_SIZE)

        fixed_mass_body = FixedMass(
            name=f"{self.name}.fixed_mass-{index}",
            positions=[(0, 0, 0)],
            mass_value=body_spec.get("mass_value", 1.0),
        )
        fixed_mass_body.set_pose(pos_xyz)
        self.bodies.append(fixed_mass_body)

    def create_fixed_spring_body(self, index: int, body_spec: dict):
        """
        create a fixed spring body with the given parameters
        """
        position_along_x = body_spec.get("position", 0.0)
        slope = body_spec.get("slope", 0.0)
        k = body_spec.get("k", 100.0)
        damping = body_spec.get("damping", 0.0)
        original_length = body_spec.get("original_length", 1.0)
        mass_value = body_spec.get("mass_value", 1.0)
        mass_rgba = body_spec.get("mass_rgba", (0, 1, 0, 1))
        constant_force = body_spec.get("constant_force", {})

        # set the z-coordinate of the fixed spring body to be above the plane
        pos_xyz = (
            position_along_x,
            0.0,
            self.plane.size[2] + DEFAULT_MASS_SIZE,
        )

        fixed_spring_body = FixedSpring(
            name=f"{self.name}.fixed_spring-{index}",
            pos=pos_xyz,
            slope=slope,
            k=k,
            original_length=original_length,
            mass_value=mass_value,
            mass_rgba=mass_rgba,
            constant_force=constant_force,
            damping=damping,
        )

        self.bodies.append(fixed_spring_body)

    def create_spring_mass_body(self, index: int, body_spec: dict):
        mass_values = body_spec.get("mass_values", [])
        mass_positions = body_spec.get("mass_positions", [])
        springs = body_spec.get("springs", [])
        damping = body_spec.get("damping", 0.0)
        constant_force = body_spec.get("constant_force", {})
        position_along_x = body_spec.get("position", 0)

        # Set the z-coordinate of the masses to be above the plane
        mass_positions_xyz = [
            (pos + position_along_x, 0.0, DEFAULT_MASS_SIZE + self.plane.size[2]) for pos in mass_positions
        ]

        spring_mass_body = SpringMass(
            name=f"{self.name}.spring_mass-{index}",
            mass_values=mass_values,
            mass_positions=mass_positions,
            spring_configs=springs,
            constant_force=constant_force,
            damping=damping,
        )
        # Set the positions of the masses
        for mass_body, pos in zip(spring_mass_body.masses, mass_positions_xyz):
            mass_body.set_pose(pos)

        self.bodies.append(spring_mass_body)

    def get_connecting_tendon_sequence(
        self,
        direction: ConnectingDirection,
        connecting_point: ConnectingPoint = ConnectingPoint.DEFAULT,
        connecting_point_seq_id: Optional[ConnectingPointSeqId] = None,
        use_sidesite: bool = False,
    ) -> TendonSequence:

        # Get the first and last sequences default with outer_to_inner direction
        first_sequence = [self.plane.left_site.create_spatial_site()]
        # print(f"self.bodies[0]: {self.bodies[0]}")

        first_sequence.extend(
            self.bodies[0].get_connecting_tendon_sequences(
                ConnectingDirection.LEFT_TO_RIGHT
            )[0].get_elements()
        )
        last_sequence = [self.plane.right_site.create_spatial_site()]
        # print(f"self.bodies[-1]: {self.bodies[-1]}")
        last_sequence.extend(
            self.bodies[-1].get_connecting_tendon_sequences(
                ConnectingDirection.LEFT_TO_RIGHT
            )[-1].get_elements()
        )
        sequence = []
        # Get the connecting tendon sequence
        if connecting_point == ConnectingPoint.LEFT:
            sequence = first_sequence
        elif connecting_point == ConnectingPoint.RIGHT:
            sequence = last_sequence
        if direction == ConnectingDirection.INNER_TO_OUTER:
            sequence.reverse()
        return TendonSequence(
            elements=sequence,
            description=f"Tendon sequence for connecting point {connecting_point}",
            name=f"{self.name}.connecting_tendon"
        )
    
    def generate_evenly_distributed_positions(self, min_val, max_val, num_bodies, border_buffer=0.1):
        if num_bodies < 1:
            return []

        total_range = max_val - min_val
        if total_range <= 2 * border_buffer:
            raise ValueError("Range too small for border buffer.")

        usable_range = total_range - 2 * border_buffer
        segment_width = usable_range / num_bodies
        positions = []

        for i in range(num_bodies):
            segment_start = min_val + border_buffer + i * segment_width
            segment_end = segment_start + segment_width
            pos = random.uniform(segment_start, segment_end)
            positions.append(pos)

        return positions

    def randomize_parameters(
        self,
        degree_of_randomization: DegreeOfRandomization = DegreeOfRandomization.NON_STRUCTURAL,
        reinitialize_instance: bool = False,
        **kwargs,
    ):
        """
        - STRUCTURAL: Re-generate a new list of collision_bodies (including quantity, type, etc.),
          and also re-generate resolution_coefficients.
        - NON_STRUCTURAL: Only do small in-place modifications on the existing collision_bodies,
          without changing the number or type. Also adjust resolution_coefficients in a minor way.
        """

        # Define parameter ranges for each randomization level
        randomization_levels = {
            DegreeOfRandomization.EASY: {
                "num_bodies": {"min": 2, "max": 3},
                "plane_slope": {"min": 0, "max": 0},  # No slope in EASY
                "position_spacing": 0.3,
                "body_types": ["mass", "fixed_spring"],
                "body_type_limits": {"fixed_spring": 1},
                "collision_body_params": {
                    "mass": {
                        "mass_value": {"min": 0.5, "max": 3.0},
                        "position": {
                            "min": 0.5 * COLLISION_BODY_POSITION_RANGE["min"],
                            "max": 0.5 * COLLISION_BODY_POSITION_RANGE["max"],
                        },
                        "init_velocity": {"min": -1.0, "max": 1.0},
                    },
                    "fixed_spring": {
                        "original_length": {"min": 0.1, "max": 0.3},
                        "position_spacing": {"min": 0, "max": 0},  # No slope in EASY
                        "k": {"min": 50, "max": 150},
                    },
                },
            },
            DegreeOfRandomization.MEDIUM: {
                "num_bodies": {"min": 2, "max": 4},
                "plane_slope": {"min": 0, "max": 30},
                "position_spacing": 0.3,
                "body_types": ["mass", "spring_mass", "fixed_mass"],
                "body_type_limits": {"fixed_mass": 1},
                "collision_body_params": {
                    "mass": {
                        "mass_value": {"min": 0.5, "max": 5.0},
                        "position": {
                            "min": 1.0 * COLLISION_BODY_POSITION_RANGE["min"],
                            "max": 1.0 * COLLISION_BODY_POSITION_RANGE["max"],
                        },
                        "init_velocity": {"min": -2.0, "max": 2.0},
                    },
                    "fixed_mass": {
                        "mass_value": {"min": 0.5, "max": 5.0},
                        "position": {
                            "min": 1.0 * COLLISION_BODY_POSITION_RANGE["min"],
                            "max": 1.0 * COLLISION_BODY_POSITION_RANGE["max"],
                        },
                    },
                    "spring_mass": {
                        "mass_value": {"min": 0.5, "max": 3.0},
                        "num_mass": {"min": 2, "max": 4},
                        "position": {
                            "min": 1.0 * COLLISION_BODY_POSITION_RANGE["min"],
                            "max": 1.0 * COLLISION_BODY_POSITION_RANGE["max"],
                        },
                        "mass_positions": {
                            "min": MASS_POSITION_SCALE * COLLISION_BODY_POSITION_RANGE["min"],
                            "max": MASS_POSITION_SCALE * COLLISION_BODY_POSITION_RANGE["max"],
                        },
                        "spring": {
                            "k": {"min": 50, "max": 200},
                            "initial_delta": {"min": -0.1, "max": 0.3},
                        },
                    },
                    "fixed_spring": {
                        "original_length": {"min": 0.1, "max": 0.3},
                        "slope": {"min": 0, "max": 0},
                        "k": {"min": 50, "max": 200},
                    },
                },
            },
            DegreeOfRandomization.HARD: {
                "num_bodies": {"min": 3, "max": 5},
                "plane_slope": {"min": 15, "max": 45},
                "position_spacing": 0.3,
                "body_types": [
                    "mass",
                    "sphere",
                    "fixed_mass",
                    "spring_mass",
                    "fixed_spring",
                ],
                "body_type_limits": {"fixed_spring": 1, "fixed_mass": 1},
                "collision_body_params": {
                    "mass": {
                        "mass_value": {"min": 0.5, "max": 10.0},
                        "position": {
                            "min": 0.5 * COLLISION_BODY_POSITION_RANGE["min"],
                            "max": 0.5 * COLLISION_BODY_POSITION_RANGE["max"],
                        },
                        "init_velocity": {"min": -3.0, "max": 3.0},
                    },
                    "sphere": {
                        "mass_value": {"min": 0.5, "max": 10},
                        "radius": {"min": 0.75 * DEFAULT_MASS_SIZE, "max": 1.25 * DEFAULT_MASS_SIZE},
                        "position": {
                            "min": 0.5 * COLLISION_BODY_POSITION_RANGE["min"],
                            "max": 0.5 * COLLISION_BODY_POSITION_RANGE["max"],
                        },
                        "init_velocity": {"min": -3.0, "max": 3.0},
                    },
                    "fixed_mass": {
                        "mass_value": {"min": 0.5, "max": 10.0},
                        "position": {
                            "min": 0.5 * COLLISION_BODY_POSITION_RANGE["min"],
                            "max": 0.5 * COLLISION_BODY_POSITION_RANGE["max"],
                        },
                    },
                    "spring_mass": {
                        "mass_value": {"min": 0.5, "max": 4.0},
                        "num_mass": {"min": 2, "max": 5},
                        "position": {
                            "min": MASS_POSITION_SCALE * COLLISION_BODY_POSITION_RANGE["min"],
                            "max": MASS_POSITION_SCALE * COLLISION_BODY_POSITION_RANGE["max"],
                        },
                        "mass_positions": {
                            "min": MASS_POSITION_SCALE * COLLISION_BODY_POSITION_RANGE["min"],
                            "max": MASS_POSITION_SCALE * COLLISION_BODY_POSITION_RANGE["max"],
                        },
                        "spring": {
                            "k": {"min": 50, "max": 250},
                            "initial_delta": {"min": -0.5, "max": 0.5},
                        },
                    },
                    "fixed_spring": {
                        "original_length": {"min": 0.1, "max": 0.3},
                        "k": {"min": 50, "max": 250},
                        "position": {
                            "min": 1.0 * COLLISION_BODY_POSITION_RANGE["min"],
                            "max": 1.0 * COLLISION_BODY_POSITION_RANGE["max"],
                        },
                    },
                },
            },
        }

        self.randomization_levels = randomization_levels

        # If degree_of_randomization is DEFAULT, randomly choose EASY, MEDIUM, or HARD
        if degree_of_randomization == DegreeOfRandomization.DEFAULT or degree_of_randomization == DegreeOfRandomization.NON_STRUCTURAL:
            options = [
                DegreeOfRandomization.EASY,
                DegreeOfRandomization.MEDIUM,
                DegreeOfRandomization.HARD,
            ]
            degree_of_randomization = random.choice(options)

        if degree_of_randomization != DegreeOfRandomization.NON_STRUCTURAL:
            # Randomly create a new list of collision_bodies
            new_num = random.randint(
                randomization_levels[degree_of_randomization]["num_bodies"]["min"],
                randomization_levels[degree_of_randomization]["num_bodies"]["max"],
            )
            new_bodies = []
            positions = []  # record existing collision_bodies 的 position

            total_range = (COLLISION_BODY_POSITION_RANGE["min"], COLLISION_BODY_POSITION_RANGE["max"])
            
            new_num = min(new_num, math.floor((total_range[1] - total_range[0]) / (DEFAULT_MASS_SIZE + 0.1)))  # limit the number of bodies to position range
            min_x, max_x = total_range

            positions = self.generate_evenly_distributed_positions(
                min_x, max_x, new_num, border_buffer=0.1
            )

            new_bodies = []
            body_type_limits = randomization_levels[degree_of_randomization].get("body_type_limits", {})
            type_counts = {k: 0 for k in body_type_limits}

            for i in range(new_num):
                selectable_types = [
                    t for t in randomization_levels[degree_of_randomization]["body_types"]
                    if t not in body_type_limits or type_counts[t] < body_type_limits[t]
                ]

                if not selectable_types:
                    break

                body_type = random.choice(selectable_types)

                if body_type in body_type_limits:
                    type_counts[body_type] += 1

                spec = self._create_random_collision_body_spec(
                    body_type, positions[i], randomization_levels[degree_of_randomization]
                )
                new_bodies.append(spec)


            self.collision_bodies = new_bodies

            # Randomly set plane_slope in a broad range
            self.plane_slope = random.uniform(0, 45)

            # Re-generate resolution_coefficients with a length of (new_num - 1)
            self.resolution_coefficients.clear()
            if len(self.collision_bodies) > 1:
                for _ in range(len(self.collision_bodies) - 1):
                    self.resolution_coefficients.append(random.uniform(0.1, 1.0))

        else:  # NON_STRUCTURAL
            # In-place modification of existing collision_bodies
            for body_spec in self.collision_bodies:
                self._non_structural_update_body(body_spec)

            # Slightly adjust plane_slope in a smaller range
            self.plane_slope = max(0, min(90, self.plane_slope + random.uniform(-5, 5)))

            # Slightly tweak each resolution_coefficient
            for i, coeff in enumerate(self.resolution_coefficients):
                new_val = coeff * random.uniform(0.95, 1.05)
                new_val = max(0.0, min(1.0, new_val))  # clamp to [0,1] if you want
                self.resolution_coefficients[i] = new_val

        # If needed, reinitialize to rebuild bodies based on updated collision_bodies
        if reinitialize_instance:
            self.reinitialize()

    def generate_entity_yaml(
        self,
        use_random_parameters: bool = False,
        degree_of_randomization: DegreeOfRandomization = DegreeOfRandomization.NON_STRUCTURAL,
    ) -> dict:
        """
        Export entity parameters to a dict (YAML-friendly).
        """
        if use_random_parameters:
            self.randomize_parameters(
                degree_of_randomization
            )

        data = {
            "name": self.name,
            "type": self.__class__.__name__,
            "position": list(self.pos),
            "parameters": {
                "plane_slope": self.plane_slope,
                "collision_bodies": self.collision_bodies,
                "resolution_coefficients": self.resolution_coefficients,
            },
        }
        return round_floats(data)

    def get_parameters(self) -> List[dict]:
        """
        Return detailed parameters of all collision bodies (e.g. mass, velocity, etc.).
        We can also iterate over self.bodies to get actual Body objects' params.
        """
        param_list = []
        for body in self.bodies:
            # Each body could be Mass, Sphere, etc., providing get_masses_quality() or similar
            param_list.extend(body.get_masses_quality())
        return param_list

    # ---------------- Helper functions for collision_bodies ----------------

    def _create_random_collision_body_spec(
        self, body_type: str, position_along_x: float, randomization_level: dict
    ) -> dict:
        """
        Generate a random body spec (dict) according to the given body_type and randomization level.
        This does NOT create the actual Body object; creation is handled in __init__.
        """
        spec = {"body_type": body_type}
        params = randomization_level["collision_body_params"][body_type]
        spec["position"] = position_along_x

        if body_type == "mass":
            spec["mass_value"] = random.uniform(
                params["mass_value"]["min"], params["mass_value"]["max"]
            )
            spec["init_velocity"] = random.uniform(
                params["init_velocity"]["min"], params["init_velocity"]["max"]
            )

        elif body_type == "sphere":
            spec["mass_value"] = random.uniform(
                params["mass_value"]["min"], params["mass_value"]["max"]
            )
            spec["radius"] = random.uniform(
                params["radius"]["min"], params["radius"]["max"]
            )
            spec["init_velocity"] = random.uniform(
                params["init_velocity"]["min"], params["init_velocity"]["max"]
            )

        elif body_type == "fixed_mass":
            spec["mass_value"] = random.uniform(
                params["mass_value"]["min"], params["mass_value"]["max"]
            )

        elif body_type == "spring_mass":
            num_mass = random.randint(
                params["num_mass"]["min"], params["num_mass"]["max"]
            )
            total_range = (params["mass_positions"]["min"], params["mass_positions"]["max"])
            num_mass = min(num_mass, math.floor((total_range[1] - total_range[0]) / (DEFAULT_MASS_SIZE + 0.1)))  # limit the number of bodies to position range
            spec["mass_values"] = [
                random.uniform(params["mass_value"]["min"], params["mass_value"]["max"])
                for _ in range(num_mass)
            ]
            spec["mass_positions"] = self.generate_evenly_distributed_positions(
                params["mass_positions"]["min"],
                params["mass_positions"]["max"],
                num_mass,
                border_buffer=0.1
            )
            spring_params = params["spring"]
            spec["springs"] = [
                {
                    "k": random.uniform(
                        spring_params["k"]["min"], spring_params["k"]["max"]
                    ),
                    "original_length": abs(spec["mass_positions"][_ + 1] - spec["mass_positions"][_]) - random.uniform(
                        spring_params["initial_delta"]["min"],
                        spring_params["initial_delta"]["max"],
                    ),
                }
                for _ in range(num_mass - 1)
            ]

        elif body_type == "fixed_spring":
            spec["original_length"] = random.uniform(
                params["original_length"]["min"], params["original_length"]["max"]
            )
            spec["k"] = random.uniform(params["k"]["min"], params["k"]["max"])

        return spec

    def _non_structural_update_body(self, body_spec: dict):
        """
        Update an existing collision_body spec in a minor (non-structural) way.
        """
        body_type = body_spec.get("body_type", "")
        if body_type == "mass":
            if "mass_value" in body_spec:
                body_spec["mass_value"] *= random.uniform(0.9, 1.1)
            if "init_velocity" in body_spec:
                body_spec["init_velocity"] *= random.uniform(0.9, 1.1)
        elif body_type == "sphere":
            if "mass_value" in body_spec:
                body_spec["mass_value"] *= random.uniform(0.9, 1.1)
            if "init_velocity" in body_spec:
                body_spec["init_velocity"] *= random.uniform(0.9, 1.1)
            if "radius" in body_spec:
                body_spec["radius"] *= random.uniform(0.95, 1.05)
        elif body_type == "fixed_mass":
            if "mass_value" in body_spec:
                body_spec["mass_value"] *= random.uniform(0.9, 1.1)
        elif body_type == "spring_mass":
            if "mass_values" in body_spec:
                body_spec["mass_values"] = [
                    mv * random.uniform(0.9, 1.1) for mv in body_spec["mass_values"]
                ]
            if "springs" in body_spec:
                for s in body_spec["springs"]:
                    s["k"] = s["k"] * random.uniform(0.9, 1.1)
                    if "original_length" in s:
                        s["original_length"] = s["original_length"] * random.uniform(
                            0.95, 1.05
                        )
        elif body_type == "fixed_spring":
            if "mass_value" in body_spec:
                body_spec["mass_value"] *= random.uniform(0.9, 1.1)
            if "k" in body_spec:
                body_spec["k"] *= random.uniform(0.9, 1.1)

    def to_xml(self) -> str:
        body_xml = f"""<body name="{self.name}" pos="{' '.join(map(str, self.pos))}" quat="{' '.join(map(str, self.quat))}">\n"""
        body_xml += self.plane.to_xml() + "\n"
        for body in self.bodies:
            body_xml += body.to_xml() + "\n"
        body_xml += "</body>"
        return body_xml

    def get_description(self, simDSL2nlq=False):
        if not simDSL2nlq:
            return super().get_description(simDSL2nlq)

        descriptions = []

        for idx, body in enumerate(self.collision_bodies):
            body_type = body["body_type"]

            if body_type in ["mass", "sphere", "fixed_mass"]:
                mass = body.get("mass_value", 1.0)
                x = body.get("position", 0.0)
                body_name = f"{self.name}.{body_type}-{idx}"

                name_mapping = {
                    "mass": "block",
                    "sphere": "sphere",
                    "fixed_mass": "fixed wall",
                }

                description = {
                    "name": body_name,
                    "mass": mass,
                    "init_position": (x),
                    "description": (
                        f"A {name_mapping[body_type]} named {body_name} {'' if body_type == 'fixed_mass' else f'with a mass {mass} kg and '}is placed at {x} m on a table called {self.name}."
                    ),
                }

                if body_type != "fixed_mass":
                    description["init_velocity"] = body.get("init_velocity", 0.0)
                    description[
                        "description"
                    ] += f" It is initially moving with a velocity of {description['init_velocity']} m/s."

                descriptions.append(description)

            elif body_type == "spring_mass":
                mass_values = body.get("mass_values", [])
                mass_positions = body.get("mass_positions", [])
                springs = body.get("springs", [])
                stiffness = [spring["k"] for spring in springs]
                natural_length = [spring["original_length"] for spring in springs]
                damping = body.get("damping", 0.0)
                constant_force = body.get("constant_force", {}).get(
                    ConstantForceType.MASS, []
                )
                body_name = f"{self.name}.spring_mass-{idx}"

                stiffness_str = (
                    ", ".join(map(str, stiffness[:-1]))
                    + (" and" if len(stiffness) > 1 else "")
                    + str(stiffness[-1])
                    + " N/m"
                )

                natural_len_str = (
                    ", ".join(map(str, natural_length[:-1]))
                    + (" and" if len(natural_length) > 1 else "")
                    + str(natural_length[-1])
                    + " m"
                )

                constant_force_str = ""
                if len(constant_force):
                    constant_force_str = (
                        ", ".join(map(str, constant_force[:-1]))
                        + (" and" if len(constant_force) > 1 else "")
                        + str(constant_force[-1])
                        + " N"
                    )

                description = {
                    "name": body_name,
                    "mass_values": mass_values,
                    "mass_positions": mass_positions,
                    "springs": springs,
                    "damping": damping,
                    "description": (
                        f"{len(mass_values)} blocks are connected in series by springs of stiffness {stiffness_str} respectively, and have damping {damping} N/m."
                        f" The springs have natural lengths of {natural_len_str} respectively."
                        f" The blocks are placed at {', '.join(map(str, mass_positions))} m on a table called {self.name}, and have masses {', '.join(map(str, mass_values))} Kg respectively."
                        f" The blocks are also subjected to constant forces {constant_force_str} respectively."
                    ),
                }

                descriptions.append(description)

            else:  # its a fixed spring
                name = f"{self.name}.fixed_spring-{idx}"
                slope = body.get("slope", 0.0)
                k = body.get("k", 100.0)
                damping = body.get("damping", 0.0)
                original_length = body.get("original_length", 1.0)
                position_x = body.get("position", 0.0)

                description = {
                    "name": name,
                    "body_type": "spring",
                    "init_position": (position_x),
                    "description": (
                        f"A spring named {name} is fixed at {position_x} m and makes an angle {slope} degrees with horizontal."
                        f" It has a stiffness of {k} N/m and damping of {damping} Kg/s, and has a natural length of {original_length} m."
                    ),
                }

                descriptions.append(description)

        return descriptions

    def get_nlq(self, symbolic = False):
        
        sym_dict = {}
        mass_count, pos_count, vel_count, k_count, b_count = 0, 0, 0, 0, 0

        plane_slope = "<angle>1" if symbolic else self.plane_slope
        sym_dict["<angle>1"] = self.plane_slope

        table_description = f"A smooth table is inclined at an angle {plane_slope} degrees with the horizontal. There are multiple systems on this table."

        body_descriptions = []
        for idx, body in enumerate(self.collision_bodies):
            body_type = body["body_type"]

            if body_type == "fixed_mass":
                body_description = f"A wall is fixed at {body.get('position', 0)} m on the table."
                body_descriptions.append(body_description)
            
            if body_type == "spring_mass":
                name = self.bodies[idx].name
                mass_values = copy.deepcopy(body.get("mass_values", []))
                mass_positions = copy.deepcopy(body.get("mass_positions", []))
                springs = copy.deepcopy(body.get("springs", []))
                stiffness = [spring["k"] for spring in springs]
                damping = copy.deepcopy(body.get("damping", 0.0))
                natural_length = [spring["original_length"] for spring in springs]
                num_blocks = len(mass_values)

                for i in range(num_blocks):
                    sym_dict[f"<mass>{mass_count}"] = mass_values[i]
                    mass_values[i] = f"<mass>{mass_count}"
                    mass_count += 1
                    
                    if i < num_blocks - 1: 
                        sym_dict[f"<k>{k_count}"] = stiffness[i]
                        stiffness[i] = f"<k>{k_count}"
                        k_count += 1

                if damping > 1e-2: 
                    sym_dict[f"<b>{b_count}"] = damping
                    damping = f"<b>{b_count}"
                    b_count += 1
                        
                mass_positions = list(map(str, mass_positions))
                natural_length = list(map(str, natural_length))

                body_description = (
                    f"In a system called '{name}', {num_blocks} blocks are placed at {convert_list_to_natural_language(mass_positions)} m on the table."
                    f" They have masses {convert_list_to_natural_language(mass_values)} Kg respectively."
                    f" These blocks are connected in series by springs (which might be initially stretched or compressed)"
                    f" of stiffness {convert_list_to_natural_language(stiffness)} N/m respectively."
                    f" The natural length of these springs are {convert_list_to_natural_language(natural_length)} respectively."
                )

                if damping:
                    body_description += (
                        f" All springs have a damping of {damping} N/m."
                    )

                if not symbolic: body_description = replace_all(body_description, sym_dict)

                body_descriptions.append(body_description)

            if body_type == "fixed_spring":
                stiffness = body.get("k", 100.0)
                damping = body.get("damping", 0.0)
                natural_length = body.get("original_length", 1.0)
                mass_value = body.get("mass_value", 1.0)

                sym_dict[f"<k>{k_count}"] = stiffness
                stiffness = f"<k>{k_count}"
                k_count += 1

                damping_desc = ""
                if damping > 1e-2:
                    sym_dict[f"<b>{b_count}"] = damping
                    damping = f"<b>{b_count}"
                    b_count += 1
                    damping_desc = f", damping of {damping} Kg/s"
                
                _cond = False
                if not isinstance(mass_value, str) and mass_value > 1e-2:
                    _cond = True
                    sym_dict[f"<mass>{mass_count}"] = mass_value
                    mass_value = f"<mass>{mass_count}"
                    mass_count += 1

                body_description = (
                    f"A spring (initially relaxed) fixed on its right end, extends to the left to {body.get('position', 0)} m on the table."
                    f" It has a stiffness of {stiffness} N/m{damping_desc}, and a natural length of {natural_length} m."
                )
                if _cond:
                    body_description += (
                        f" A block of mass {mass_value} kg is attached to the left end of the spring."
                    )

                if not symbolic:
                    body_description = replace_all(body_description, sym_dict)
                
                body_descriptions.append(body_description)
            
            if body_type == "mass":
                name = self.bodies[idx].name
                mass = body.get("mass_value", 1.0)
                position = body.get("position", 0.0)
                init_velocity = body.get("init_velocity", 0.0)

                sym_dict[f"<mass>{mass_count}"] = mass
                mass = f"<mass>{mass_count}"
                mass_count += 1

                body_description = (
                    f"In a system called '{name}', a block of mass {mass} kg is placed at {position} m on the table."
                    f" It is initially moving with a velocity of {init_velocity} m/s."
                )

                if not symbolic: body_description = replace_all(body_description, sym_dict)

                body_descriptions.append(body_description)

            if body_type == "sphere":
                name = self.bodies[idx].name
                mass_value = body.get("mass_value", 1.0)
                position_along_x = body.get("position", 0.0)
                init_velocity_along_x = body.get("init_velocity", 0.0)
                radius = body.get("radius", DEFAULT_SPHERE_RADIUS)

                sym_dict[f"<mass>{mass_count}"] = mass_value
                mass_value = f"<mass>{mass_count}"
                mass_count += 1

                body_description = (
                    f"In a system called '{name}', sphere of mass {mass_value} kg is placed at {position_along_x} m on the table."
                    f" It is initially moving with a velocity of {init_velocity_along_x} m/s."
                )

                if not symbolic: body_description = replace_all(body_description, sym_dict)

                body_descriptions.append(body_description)

        body_descriptions = '\n'.join(body_descriptions)
        description = (
            f"{table_description}\n{body_descriptions}"
            f"\n All collisions are perfectly elastic. All blocks and walls have dimensions of {DEFAULT_MASS_SIZE} x {DEFAULT_MASS_SIZE} x {DEFAULT_MASS_SIZE} and all spheres have radius of {DEFAULT_SPHERE_RADIUS} m."
        )
        
        if symbolic:
            return description, sym_dict
        
        return description

    def connecting_point_nl(self, cd, cp, csi):
        raise NotImplementedError("ComplexCollisionPlane is not supposed to have connections.")
    
    def get_question(self, sub_entity: str, quantity: str) -> str:
        """
        Get a question related to the entity
        
        Inputs:
            sub_entity: str
            quantity: str
            
        Returns:
            str
        """

        sub_entity = sub_entity.split('.')
        if len(sub_entity) == 1:
            body_name = sub_entity[0]

            if 'mass' in body_name:
                return f"What is the {quantity} of the block in the system '{body_name}'"
            elif 'sphere' in body_name:
                return f"What is the {quantity} of the sphere in the system '{body_name}'"
            else:
                raise ValueError(f"Unknown sub-entity '{body_name}'")
        
        body_name, instance_name = sub_entity[0], sub_entity[1]

        if "spring_mass" in body_name:
            instance_idx = int(instance_name.split('-')[-1])
            return f"What is the {quantity} of the {(['1st', '2nd', '3rd'] + [f'{i}th' for i in range(4, instance_idx + 2)])[instance_idx]} block in the system '{body_name}'"
        
        return f"What is the {quantity} of the block in the system '{body_name}'"

class SliderWithArchPlaneSpheres(Entity):
    """
    An entity that combines a Plane, a SliderWithArch body, and spheres placed on the slider.
    """

    def __init__(
        self,
        name: str,
        pos: Tuple[float, float, float] = (0, 0, 0),
        quat: Tuple[float, float, float, float] = (1, 0, 0, 0),
        mass_values: List[float] = None,
        mass_positions: List[float] = None,
        sphere_radius: float = 0.1,  # Default small sphere
        slide_length: float = 4.0,
        arch_radius: float = 0.5,
        arch_stl_path: str = f"{GEOM_FIXED_SOURCES_PATH}/round_arch.stl",
        **kwargs,
    ) -> None:
        super().__init__(
            name,
            pos=pos,
            quat=quat,
            entity_type=self.__class__.__name__,
            **kwargs,
        )

        # Validate inputs
        if mass_values is None or mass_positions is None:
            raise ValueError(
                "Both 'mass_values' and 'mass_positions' must be provided."
            )
        if len(mass_values) != len(mass_positions):
            raise ValueError(
                "'mass_values' and 'mass_positions' must be of the same length."
            )

        # Initialize the plane (plane_slope is always zero)
        self.plane = Plane(
            name=name + "_plane",
            size=(DEFAULT_COLLISION_PLANE_LENGTH, 2, DEFAULT_PLANE_THICKNESS),
            quat=(1, 0, 0, 0),  # No rotation
        )

        # Initialize the SliderWithArch
        self.slider_with_arch = SliderWithArch(
            name=name + ".slider_with_arch",
            slide_length=slide_length,
            arch_radius=arch_radius,
            arch_stl_path=arch_stl_path,
        )

        # Store references to geoms in SliderWithArch for computing sizes
        self.slider_geom = self.slider_with_arch.geoms[0]
        self.arch_geom = self.slider_with_arch.geoms[1]

        # Initialize the spheres
        self.spheres = []
        for i in range(len(mass_values)):
            sphere = Sphere(
                name=f"{name}.sphere{i}",
                pos=(0, 0, 0),
                joint_option=None,  # We'll add custom planar joints instead
                radius=sphere_radius,
                mass=mass_values[i],
            )
            # Add X-Z planar joint for movement along the slider (X-axis primarily)
            add_xz_planar_joint(sphere, 0.0)  # No slope for slider
            self.spheres.append(sphere)

        # Position the SliderWithArch on top of the plane
        # Since plane slope is zero, positions are straightforward

        # Compute total height of the slider
        slider_height = self.slider_geom.size[2] * 2  # Geom sizes are half-sizes

        # Position the slider above the plane
        slider_z_pos = (
            slider_height / 2 + self.plane.size[2]
        )  # Centered on top of the plane

        self.slider_with_arch.set_pose(
            (0, 0, slider_z_pos),
            self.plane.quat,
        )

        # Compute the right end of the arch (zero point for mass_positions)
        # arch_pos_x = self.slider_with_arch.pos[0] + self.arch_geom.pos[0]
        # arch_size_x = self.arch_geom.size[0] * 2  # Full size
        # arch_right_end_x = arch_pos_x + arch_size_x / 2  # Right edge of the arch
        arch_right_end_x = -slide_length / 2

        # Position the spheres along the slider
        for i, sphere in enumerate(self.spheres):
            sphere_x = arch_right_end_x + mass_positions[i]
            # Ensure the sphere is on the slider and not beyond its length
            max_x = self.slider_with_arch.pos[0] + self.slider_geom.size[0]
            if sphere_x > max_x:
                raise ValueError(f"Sphere {i} position exceeds slider length.")

            # Sphere's z position (on top of the slider)
            sphere_z_pos = slider_z_pos + slider_height / 2 + sphere_radius

            sphere.set_pose(
                (sphere_x, 0, sphere_z_pos),
                self.plane.quat,
            )

    def to_xml(self) -> str:
        """
        Convert the entity to an XML string.
        """
        xml = f"""<body name="{self.name}" pos="{' '.join(map(str, self.pos))}" quat="{' '.join(map(str, self.quat))}">\n"""
        xml += self.plane.to_xml() + "\n"
        xml += self.slider_with_arch.to_xml() + "\n"
        for sphere in self.spheres:
            xml += sphere.to_xml() + "\n"
        xml += "</body>"
        return xml
    
class SpringMassPlaneEntity(SpringMass, Entity):
    """
    An Entity that internally uses the SpringMass body definition,
    and additionally connects the first (leftmost) and the last (rightmost) mass with a spring.
    """

    # Define parameter ranges for different randomization difficulty levels
    randomization_levels = {
        DegreeOfRandomization.EASY: {
            "mass_value": {"min": 0.5, "max": 3.0},
            "num_mass": {"min": 2, "max": 4},
            "mass_positions": {
                "min": MASS_POSITION_SCALE * COLLISION_BODY_POSITION_RANGE["min"],
                "max": MASS_POSITION_SCALE * COLLISION_BODY_POSITION_RANGE["max"],
            },
            "spring": {
                "k": {"min": 50, "max": 200},
                "initial_delta": {"min": -0.1, "max": 0.3},
            },
            "end_spring_probability": 0,
            "damping": {"min": 10, "max": 20},
        },
        DegreeOfRandomization.MEDIUM: {
            "mass_value": {"min": 0.5, "max": 3.0},
            "num_mass": {"min": 3, "max": 4},
            "mass_positions": {
                "min": MASS_POSITION_SCALE * COLLISION_BODY_POSITION_RANGE["min"],
                "max": MASS_POSITION_SCALE * COLLISION_BODY_POSITION_RANGE["max"],
            },
            "spring": {
                "k": {"min": 50, "max": 200},
                "initial_delta": {"min": -0.1, "max": 0.3},
            },
            "end_spring_probability": 0.3,
            "damping": {"min": 10, "max": 20},
        },
        DegreeOfRandomization.HARD: {
            "mass_value": {"min": 0.5, "max": 3.0},
            "num_mass": {"min": 3, "max": 6},
            "mass_positions": {
                "min": MASS_POSITION_SCALE * COLLISION_BODY_POSITION_RANGE["min"],
                "max": MASS_POSITION_SCALE * COLLISION_BODY_POSITION_RANGE["max"],
            },
            "spring": {
                "k": {"min": 50, "max": 200},
                "initial_delta": {"min": -0.1, "max": 0.3},
            },
            "end_spring_probability": 0.9,
            "damping": {"min": 10, "max": 20},
        },
    }

    def __init__(
        self,
        name: str,
        mass_values: List[float] = [1.0, 1.0],
        mass_positions: List[float] = [0.0, 1.0],
        spring_configs: List[Dict[str, float]] = [{'k': 100, 'original_length': 1.0}],
        end_spring_config: Optional[Dict[str, float]] = None,
        pos: Tuple[float, float, float] = (0, 0, 0),
        constant_force: Optional[Dict[str, List[Union[List, float]]]] = None,
        damping: float = 0.0,
        **kwargs,
    ) -> None:
        self.name = name
        self.mass_values = mass_values
        self.mass_positions = mass_positions
        self.spring_configs = spring_configs
        self.end_spring_config = end_spring_config
        self.pos = pos
        self.constant_force = constant_force
        self.damping = damping

        # Call the SpringMass constructor
        super().__init__(
            name=self.name,
            mass_values=self.mass_values,
            mass_positions=self.mass_positions,
            spring_configs=self.spring_configs,
            end_spring_config=self.end_spring_config,
            constant_force=self.constant_force,
            damping=self.damping,
            pos=self.pos,
            entity_type=self.__class__.__name__,
            **kwargs
        )

        # If end_spring is set, connect the first and last masses
        if self.end_spring_config is not None and len(self.mass_values) > 1:
            mass_left = self.masses[0]
            mass_right = self.masses[-1]
            self.springs.append(
                Spring(
                    name=f"{self.name}.spring-{len(self.spring_configs)}",
                    left_site=mass_left.center_site,
                    right_site=mass_right.center_site,
                    stiffness=self.end_spring_config["k"],
                    springlength=self.end_spring_config["original_length"],
                    damping=self.damping,
                )
            )

        # Place these masses above the plane
        self.plane = Plane(
            name=f"{self.name}.plane",
            size=(DEFAULT_COLLISION_PLANE_LENGTH, 2, DEFAULT_PLANE_THICKNESS),
            quat=(1, 0, 0, 0),  # No rotation
            pos=(0, 0, -DEFAULT_PLANE_THICKNESS - DEFAULT_MASS_SIZE),
        )
        self.add_child_body(self.plane)

    def randomize_parameters(
        self,
        degree_of_randomization: DegreeOfRandomization = DegreeOfRandomization.NON_STRUCTURAL,
        reinitialize_instance: bool = False,
        **kwargs,
    ):
        """
        Randomizes (or slightly perturbs) the current parameters of SpringMassPlaneEntity 
        (e.g., mass_values, mass_positions, springs, end_spring, etc.).
        
        - When degree_of_randomization is STRUCTURAL / EASY / MEDIUM / HARD: Regenerate (structural randomization);
        - When NON_STRUCTURAL: Only small random perturbations on existing parameters;
        - When DEFAULT, a simple random choice between EASY/MEDIUM/HARD.
        - If reinitialize_instance=True, calls self.reinitialize() to rebuild with new parameters.
        """
        # 1) If DEFAULT, randomly choose between EASY/MEDIUM/HARD; if NON_STRUCTURAL, follow "small perturbation" path
        if degree_of_randomization == DegreeOfRandomization.DEFAULT:
            degree_of_randomization = random.choice([
                DegreeOfRandomization.EASY,
                DegreeOfRandomization.MEDIUM,
                DegreeOfRandomization.HARD,
            ])
        if degree_of_randomization == DegreeOfRandomization.NON_STRUCTURAL:
            # == Small perturbation ==
            self._non_structural_update()
        else:
            # == Structural re-generation ==
            config = self.randomization_levels.get(degree_of_randomization)
            if not config:
                # If STRUCTURAL is passed but no corresponding config, default to EASY
                config = self.randomization_levels[DegreeOfRandomization.EASY]

            self._structural_update(config)

        # If reinitialization is needed
        if reinitialize_instance:
            self.reinitialize()

    def generate_evenly_distributed_positions(self, min_val, max_val, num_bodies, border_buffer=0.1):
        if num_bodies < 1:
            return []

        total_range = max_val - min_val
        if total_range <= 2 * border_buffer:
            raise ValueError("Range too small for border buffer.")

        usable_range = total_range - 2 * border_buffer
        segment_width = usable_range / num_bodies
        positions = []

        for i in range(num_bodies):
            segment_start = min_val + border_buffer + i * segment_width
            segment_end = segment_start + segment_width
            pos = random.uniform(segment_start, segment_end)
            positions.append(pos)

        return positions

    def _structural_update(self, config: dict):
        """
        "Reconstruct" mass_values, mass_positions, springs, end_spring, etc. based on random ranges.
        """
        spec = {}
        
        num_mass = random.randint(
            config["num_mass"]["min"], config["num_mass"]["max"]
        )
        total_range = (config["mass_positions"]["min"], config["mass_positions"]["max"])
        num_mass = min(num_mass, math.floor((total_range[1] - total_range[0]) / (DEFAULT_MASS_SIZE + 0.1)))  # limit the number of bodies to position range
        spec["mass_values"] = [
            random.uniform(config["mass_value"]["min"], config["mass_value"]["max"])
            for _ in range(num_mass)
        ]
        spec["mass_positions"] = self.generate_evenly_distributed_positions(
            config["mass_positions"]["min"],
            config["mass_positions"]["max"],
            num_mass,
            border_buffer=0.1
        )
        spring_params = config["spring"]
        spec["spring_configs"] = [
            {
                "k": random.uniform(
                    spring_params["k"]["min"], spring_params["k"]["max"]
                ),
                "original_length": abs(spec["mass_positions"][i + 1] - spec["mass_positions"][i]) - random.uniform(
                    spring_params["initial_delta"]["min"],
                    spring_params["initial_delta"]["max"],
                ),
            }
            for i in range(num_mass - 1)
        ]

        r = random.random() 
        if r < config["end_spring_probability"]:
            spec["spring_configs"].append(
                {
                    "k": random.uniform(
                        spring_params["k"]["min"], spring_params["k"]["max"]
                    ),
                    "original_length": abs(spec["mass_positions"][0] - spec["mass_positions"][-1]) - random.uniform(
                        spring_params["initial_delta"]["min"],
                        spring_params["initial_delta"]["max"],
                    ),
                }
            )
        spec["damping"] = random.uniform(
            config["damping"]["min"], config["damping"]["max"]
        )

        self.mass_values = spec["mass_values"]
        self.mass_positions = spec["mass_positions"]
        self.spring_configs = spec["spring_configs"][:len(self.mass_values) - 1]
        
        if len(self.spring_configs) < len(spec["spring_configs"]):
            self.end_spring_config = spec["spring_configs"][-1]
        
        self.damping = spec["damping"]
        
    def _non_structural_update(self):
        """
        Make small random perturbations to the current mass_values, mass_positions, springs, end_spring, damping, etc.,
        without changing their structural count or types.
        """
        # 1) Perturb mass_values by multiplying by a random factor
        for i in range(len(self.mass_values)):
            factor = random.uniform(0.9, 1.1)
            self.mass_values[i] *= factor
            # Prevent negative or too small values
            self.mass_values[i] = max(self.mass_values[i], 0.01)

        # 2) Slightly adjust mass_positions
        for i in range(len(self.mass_positions)):
            shift = random.uniform(-0.2, 0.2)
            self.mass_positions[i] += shift

        # 3) Slightly perturb the spring k and original_length by random factors
        for s in self.spring_configs:
            s["k"] *= random.uniform(0.9, 1.1)
            s["original_length"] *= random.uniform(0.95, 1.05)
            # Prevent negative values
            s["k"] = max(s["k"], 1e-3)
            s["original_length"] = max(s["original_length"], 1e-3)

        # 4) If end_spring exists, apply similar perturbations
        if self.end_spring_config is not None:
            self.end_spring_config["k"] *= random.uniform(0.9, 1.1)
            self.end_spring_config["original_length"] *= random.uniform(0.95, 1.05)
            self.end_spring_config["k"] = max(self.end_spring_config["k"], 1e-3)
            self.end_spring_config["original_length"] = max(self.end_spring_config["original_length"], 1e-3)

        # 5) Randomly vary damping
        self.damping *= random.uniform(0.8, 1.2)
        self.damping = max(self.damping, 0.0)

    def generate_entity_yaml(
        self,
        use_random_parameters: bool = False,
        degree_of_randomization: DegreeOfRandomization = DegreeOfRandomization.NON_STRUCTURAL,
    ) -> dict:
        """
        Return a dictionary for later serialization into YAML.
        If use_random_parameters=True, randomization is applied first.
        """
        if use_random_parameters:
            self.randomize_parameters(degree_of_randomization)

        param_dict = {
            "mass_values": self.mass_values,
            "mass_positions": self.mass_positions,
            "spring_configs": self._serialize_spring_configs_info(self.spring_configs),
            "end_spring_config": self._serialize_end_spring_config_info(self.end_spring_config),
            "damping": self.damping,
        }

        data = {
            "name": self.name,
            "type": self.__class__.__name__,
            "entity_type": self.entity_type,
            "position": list(self.pos),
            "parameters": param_dict,
        }
        return round_floats(data)

    def _serialize_spring_configs_info(self, spring_configs_list):
        """
        Convert the internal spring information into a serializable structure.
        Note that the spring_configs_list contains Spring objects, not dictionaries.
        """
        serialized = []
        for idx, sp in enumerate(spring_configs_list):
            serialized.append({
                "index": idx,
                "k": sp['k'],
                "original_length": sp['original_length'],
                "damping": sp.get('damping', self.damping),
            })
        return serialized

    def _serialize_end_spring_config_info(self, end_spring_config):
        """
        Similarly, make the end spring information serializable.
        """
        if end_spring_config is None:
            return None
        return {
            "k": end_spring_config["k"],
            "original_length": end_spring_config["original_length"],
        }

    def get_description(self, simDSL2nlq=False):
        description = (
            f"Entity '{self.name}' consists of multiple masses connected by springs. "
            f"The leftmost and rightmost masses are also connected by an additional spring."
        )
        return [{
            "name": self.name,
            "body_type": "SpringMassPlaneEntity",
            "description": description,
        }]

    def get_nlq(self, symbolic = False):
        sym_dict = {}
        
        mass_values = copy.deepcopy(self.mass_values)
        mass_positions = copy.deepcopy(self.mass_positions)
        spring_configs = copy.deepcopy(self.spring_configs[:len(self.mass_values) - 1])
        end_spring_config = copy.deepcopy(self.end_spring_config)       
        
        num_blocks = len(mass_values)

        stiffness = [spring_config["k"] for spring_config in spring_configs]
        natural_length = [spring_config["original_length"] for spring_config in spring_configs]
        damping = copy.deepcopy(self.damping)

        if end_spring_config is not None:
            end_stiffness = end_spring_config["k"]
            end_natural_length = end_spring_config["original_length"]

        mass_count, k_count, b_count = (
            1,
            1,
            1,
        )

        # Symbolize masses and stiffnesses (solvable algebraically)
        for i in range(num_blocks):
            sym_dict[f"<mass>{mass_count}"] = self.mass_values[i]
            mass_values[i] = f"<mass>{mass_count}"
            mass_count += 1

            if i < num_blocks - 1: 
                sym_dict[f"<k>{k_count}"] = stiffness[i]
                stiffness[i] = f"<k>{k_count}"
                k_count += 1
        if end_spring_config is not None:
            sym_dict[f"<k>{k_count}"] = end_stiffness
            end_stiffness = f"<k>{k_count}"
            k_count += 1

        if damping > 1e-2: 
            sym_dict[f"<b>{b_count}"] = damping
            damping = f"<b>{b_count}"
            b_count += 1
                
        # Keep positions and natural lengths as numbers (geometric constraints)
        mass_positions_str = list(map(str, mass_positions))
        natural_length_str = list(map(str, natural_length))
        if end_spring_config is not None:
            end_natural_length_str = str(end_natural_length)

        # Build block labels: "block 1", "block 2", ...
        block_labels = [f"block {i+1}" for i in range(num_blocks)]

        # Introduction: blocks on a flat table, positions, masses, and dimensions
        body_description = (
            f"In a SpringMass system, {num_blocks} blocks rest on a flat, horizontal, frictionless table."
            f" The blocks are cubes of side length {DEFAULT_MASS_SIZE} m."
            f" Their centers are positioned along a horizontal line at"
            f" x = {convert_list_to_natural_language(mass_positions_str)} m respectively."
            f" {convert_list_to_natural_language(block_labels).capitalize()}"
            f" have masses {convert_list_to_natural_language(mass_values)} kg respectively."
        )

        # Spring connections: explicitly state which blocks each spring connects
        body_description += (
            f" Adjacent blocks are connected by springs:"
        )
        for i in range(num_blocks - 1):
            body_description += (
                f" a spring of stiffness {stiffness[i]} N/m and natural length"
                f" {natural_length_str[i]} m connects block {i+1} to block {i+2};"
            )

        # End spring
        if end_spring_config is not None:
            body_description += (
                f" Additionally, block 1 and block {num_blocks} are connected"
                f" by a spring of stiffness {end_stiffness} N/m and natural length"
                f" {end_natural_length_str} m, forming a closed loop."
            )

        # Damping
        if damping:
            body_description += (
                f" All springs have a damping coefficient of {damping} Ns/m."
            )

        if not symbolic: 
            body_description = replace_all(body_description, sym_dict)
            return body_description

        return body_description, sym_dict

    def connecting_point_nl(
        self, 
        cd: ConnectingDirection, 
        cp: ConnectingPoint, 
        csi: int,
        first: bool = False
    ) -> str:
        raise NotImplementedError("SpringMassPlaneEntity is not supposed to have connections.")

    def get_question(self, sub_entity: str, quantity: str) -> str:
        """
        Get a question related to the entity
        
        Inputs:
            sub_entity: str
            quantity: str
            
        Returns:
            str
        """

        sub_entity = sub_entity.split('-')
        if len(sub_entity) == 1:
            body_name = sub_entity[0]

            if 'mass' in body_name:
                return f"What is the {quantity} of the block in the system '{body_name}'"
            else:
                raise ValueError(f"Unknown sub-entity '{body_name}'")
        
        body_name, instance_name = self.name, sub_entity[1]

        instance_idx = int(instance_name.split('-')[-1])
        return f"What is the {quantity} of the {(['1st', '2nd', '3rd'] + [f'{i}th' for i in range(4, instance_idx + 2)])[instance_idx]} block in the system '{self.name}'"