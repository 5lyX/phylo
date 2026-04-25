from .base_entities import *
import pandas as pd
from tabulate import tabulate
from sim.utils import replace_all
from copy import deepcopy


class MagneticElectricEntity(Entity):
    """
    MagneticElectricEntity is used to create a magnetic or electric field
    acting on a single charged particle.

    -------------------------------------------------------------------------
    Detailed description of field_configs
    -------------------------------------------------------------------------
    * Type: List[Dict[str, Any]]
    * Each dict in the list represents an **independent field** (electric or magnetic).
    * Supports three strength modes, corresponding to three field configurations:
        1) **Uniform Linear Field**               →  field_strength: float
        2) **Polynomial Linear Field**            →  field_strength: "field_polynomial"
        3) **Polynomial Circular Field**          →  field_strength: "field_polynomial_from_distance"

    ----------------------------
    Common keys (required for all modes)
    ----------------------------
    - field_type      : "electric" | "magnetic"
    - field_shape     : "linear" | "circular"
                       · linear    → direction given by field_angle, range by field_range  
                       · circular  → centered at field_position, only valid for magnetic fields
    - (see each mode for additional keys)

    ----------------------------
    1) Uniform Linear Field
    ----------------------------
    {
        "field_type"        : "electric",               # or "magnetic"
        "field_strength"    : 3.5,                      # uniform strength (float)
        "field_shape"       : "linear",
        "field_angle"       : 45.0,                     # degrees, 0° = +x direction, CCW positive
        "field_range"       : (0.0, 2.0),               # effective distance (min_r, max_r)
        "field_position"    : (0, 0, 0),               # placeholder, ignored for linear fields
    }

    ----------------------------
    2) Polynomial Linear Field
       -- strength varies with x/y displacement
    ----------------------------
    {
        "field_type"                   : "electric",
        "field_strength"               : "field_polynomial",
        "field_polynomial"             : {
            "x": ( 2, 1.2),             # (power, coeff) →  f_x = coeff * (Δx)^power
            "y": (-1, 0.8),             # supports powers 0, 1, 2, -1, -2
            "z": ( 0, 0),               # z component remains zero
        },
        "field_shape"                  : "linear",
        "field_angle"                  : 90.0,
        "field_range"                  : (0.0, 1.5),
        "field_position"               : (0, 0, 0),           # origin for polynomial
    }

    ----------------------------
    3) Polynomial Circular Field (magnetic only)
       -- strength varies with radius r, direction along ±z axis
    ----------------------------
    {
        "field_type"                              : "magnetic",
        "field_strength"                          : "field_polynomial_from_distance",
        "field_polynomial_from_distance"          : {
            "r": (-2, 2.0),                        # (power, coeff) →  f_z = coeff * (r)^power
        },
        "field_shape"                             : "circular",
        "field_position"                          : (0.5, -0.5, 0),    # center of the field
        # keys for linear fields can be omitted
    }

    ----------------------------
    Combination / Superposition Rules
    ----------------------------
    • **Circular** mode can only exist alone and must be magnetic  
    • To superpose multiple fields, they must all be "linear"  
    • Linear combinations can mix electric / magnetic and uniform / polynomial

    ----------------------------
    Difficulty level examples
    ----------------------------
    EASY   : Single uniform electric field  
    MEDIUM : Uniform electric + uniform magnetic, or single polynomial field  
    HARD   : Single circular polynomial magnetic field or 2–3 linear fields superposed

    -------------------------------------------------------------------------
    Other class details omitted…
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(
        self,
        name: str,
        pos: Tuple[float, float, float] = (0, 0, 0),
        quat: Tuple[float, float, float, float] = (1, 0, 0, 0),
        mass: float = 1.0,
        q: float = 1.0,
        init_velocity: Tuple[float, float, float, float, float, float] = (1, 0, 0, 0, 0, 0),
        field_configs: List[Dict[str, Any]] = None,
        **kwargs,
    ):
        self.mass = mass
        self.q = q
        self.init_velocity = init_velocity
        self.field_configs = field_configs or []

        super().__init__(name, pos, quat, entity_type=self.__class__.__name__, **kwargs)

        # create the charged particle
        self.particle = Mass(
            name=f"{self.name}.particle",
            positions=[pos],
            quat=quat,
            mass_value=self.mass,
            init_velocity={InitVelocityType.MASS: list(self.init_velocity)},
        )
        self.add_child_body(self.particle)

    # ------------------------------------------------------------------------------------------------------------------
    # Compute fields acting on the particle at a given position
    # ------------------------------------------------------------------------------------------------------------------
    def get_fields(self, pos: np.ndarray) -> List[Dict[str, Any]]:
        """
        Returns a list of dicts of the form:
        {
            # "body_name"    : <str>,
            "field_type"   : "electric" | "magnetic",
            "field_strength": (fx, fy, fz),
            # "q"            : <charge>
        }
        """
        fields = []
        pos = tuple(pos)  # ensure pos is a tuple

        for field_config in self.field_configs:
            spec = field_config["field_strength"]

            # if field field_range is not none and pos is outside the range, skip the field
            box_cond = field_config.get("field_range") is None or (
                (abs(pos[0] - field_config["field_position"][0]) <= field_config["field_range"][0] / 2) and 
                (abs(pos[1] - field_config["field_position"][1]) <= field_config["field_range"][1] / 2)
            )

            circ_cond = field_config.get("field_range") is None or (
                field_config["field_range"][0] <= math.hypot(pos[0] - field_config["field_position"][0], pos[1] - field_config["field_position"][1]) <= field_config["field_range"][1]
            )

            if (
                (field_config.get("field_shape") == "linear" and not box_cond) or 
                (field_config.get("field_shape") == "circular" and not circ_cond)
            ):
                continue

            # 1. Uniform linear field: numeric strength + angle
            if isinstance(spec, (int, float)):
                angle_deg = field_config.get("field_angle", 0.0)
                angle = math.radians(angle_deg)
                fx = spec * math.cos(angle)
                fy = spec * math.sin(angle)
                field_strength = (fx, fy, 0.0)

            # 2. Polynomial linear field (x/y directions)
            elif spec == "field_polynomial":
                poly = field_config["field_polynomial"]
                dx = pos[0] - field_config["field_position"][0]
                dy = pos[1] - field_config["field_position"][1]
                fx = (dx ** poly["x"][0]) * poly["x"][1]
                fy = (dy ** poly["y"][0]) * poly["y"][1]
                field_strength = (fx, fy, 0.0)

            # 3. Polynomial circular field (z direction only)
            elif spec == "field_polynomial_from_distance":
                r_power, coeff = field_config["field_polynomial_from_distance"]["r"]
                r = math.hypot(pos[0] - field_config["field_position"][0],
                               pos[1] - field_config["field_position"][1])
                fz = (r ** r_power) * coeff
                field_strength = (0.0, 0.0, fz)

            else:
                raise ValueError(f"Unknown field_strength spec: {spec}")

            fields.append({
                # "body_name"      : self.particle.name,
                "field_type"     : field_config["field_type"],
                "field_strength" : field_strength,
                # "q"              : self.q,
            })

        return fields

    # ------------------------------------------------------------------------------------------------------------------
    # Generate YAML for this entity
    # ------------------------------------------------------------------------------------------------------------------
    def generate_entity_yaml(
        self,
        use_random_parameters: bool = False,
        degree_of_randomization: "DegreeOfRandomization" = "DegreeOfRandomization.NON_STRUCTURAL",
    ) -> Dict[str, Any]:
        if use_random_parameters:
            self.randomize_parameters(degree_of_randomization)

        data = {
            "name"      : self.name,
            "type"      : self.__class__.__name__,
            "position"  : list(self.pos),
            "parameters": {
                "mass"         : self.mass,
                "q"            : self.q,
                "init_velocity": list(self.init_velocity),
                "field_configs": self.field_configs,
            },
        }
        # Remove or implement round_floats as needed
        return round_floats(data)  # type: ignore

    # ------------------------------------------------------------------------------------------------------------------
    # Randomize parameters according to difficulty level
    # ------------------------------------------------------------------------------------------------------------------
    def randomize_parameters(
        self,
        degree_of_randomization: "DegreeOfRandomization" = "DegreeOfRandomization.NON_STRUCTURAL",
        reinitialize_instance: bool = False,
        **kwargs,
    ):
        """
        Randomization rules:

        EASY   —— single uniform electric or magnetic field;
        MEDIUM —— either (1) superposition of uniform electric + uniform magnetic, or
                   (2) single polynomial field (linear or circular);
        HARD   —— either (1) single circular polynomial magnetic field, or
                   (2) 2–3 linear fields (uniform or polynomial) superposed.

        DEFAULT / NON_STRUCTURAL will pick one difficulty at random or just tweak existing values.
        """

        randomization_levels = {
            DegreeOfRandomization.EASY: {
                "mass"  : (0.5, 1.5),
                "charge": (0.2, 2.0),
                "speed" : (0.5, 3.0),
            },
            DegreeOfRandomization.MEDIUM: {
                "mass"  : (1, 2.0),
                "charge": (0.1, 2.0),
                "speed" : (1.0, 5.0),
            },
            DegreeOfRandomization.HARD: {
                "mass"  : (1, 2.0),
                "charge": (0.5, 2.0),
                "speed" : (1.0, 5.0),
            },
        }
        self.randomization_levels = randomization_levels  # for debugging

        # If DEFAULT/NON_STRUCTURAL, choose a random difficulty
        if degree_of_randomization in (
            DegreeOfRandomization.DEFAULT,
            DegreeOfRandomization.NON_STRUCTURAL,
        ):
            degree_of_randomization = random.choice([
                DegreeOfRandomization.EASY,
                DegreeOfRandomization.MEDIUM,
                DegreeOfRandomization.HARD,
            ])

        # NON_STRUCTURAL: only tweak existing values slightly
        if degree_of_randomization == DegreeOfRandomization.NON_STRUCTURAL:
            # tweak mass and charge
            self.mass *= random.uniform(0.9, 1.1)
            self.q *= random.uniform(0.9, 1.1)

            # tweak initial velocity magnitude in xy-plane
            vx, vy, vz, wx, wy, wz = self.init_velocity
            speed_xy = math.hypot(vx, vy)
            if speed_xy > 0:
                new_speed = speed_xy * random.uniform(0.9, 1.1)
                ratio = new_speed / speed_xy
                vx, vy = vx * ratio, vy * ratio
            self.init_velocity = (vx, vy, vz, wx, wy, wz)

            # tweak field strengths or coefficients
            for fc in self.field_configs:
                if isinstance(fc["field_strength"], (int, float)):
                    fc["field_strength"] *= random.uniform(0.9, 1.1)
                elif fc["field_strength"] == "field_polynomial":
                    for axis in ("x", "y"):
                        power, coeff = fc["field_polynomial"][axis]
                        fc["field_polynomial"][axis] = (power, coeff * random.uniform(0.9, 1.1))
                elif fc["field_strength"] == "field_polynomial_from_distance":
                    p, coeff = fc["field_polynomial_from_distance"]["r"]
                    fc["field_polynomial_from_distance"]["r"] = (p, coeff * random.uniform(0.9, 1.1))
            return  # done for non-structural

        # STRUCTURAL randomization
        params = randomization_levels[degree_of_randomization]

        # 1) mass & charge
        self.mass = round(random.uniform(*params["mass"]), 3)
        self.q = round(random.choice([-1, 1]) * random.uniform(*params["charge"]), 3)

        # 2) initial velocity (angle within ±90° of +x)
        v_mag = random.uniform(*params["speed"])
        theta = random.uniform(-math.pi / 2, math.pi / 2)
        vx, vy = v_mag * math.cos(theta), v_mag * math.sin(theta)
        self.init_velocity = (vx, vy, 0.0, 0.0, 0.0, 0.0)

        # 3) generate field configurations
        self.field_configs = []

        def _uniform_field(field_type: str) -> Dict[str, Any]:
            return {
                "field_type"      : field_type,
                "field_strength"  : round(random.uniform(0.1, 2), 2) * (5 if field_type == "electric" else 1),
                "field_shape"     : random.choice(["linear", "circular"]),
                "field_angle"     : round(random.uniform(0, 360), 2),
                "field_range"     : (round(random.uniform(0.5, 1.5), 2), round(random.uniform(2.0, 3.0), 2)),
                "field_position"                         : (
                    round(random.uniform(-1.0, 1.0), 2),
                    round(random.uniform(-1.0, 1.0), 2),
                    0.0,
                ),
            }

        def _poly_field_linear(field_type: str) -> Dict[str, Any]:
            poly = {
                axis: (random.choice([1, 2, ]), round(random.uniform(0.1, 2), 2)) # temporarily remove negative powers
                for axis in ("x", "y")
            }

            if field_type == "electric":
                poly = {axis: (power, coeff * 5) for axis, (power, coeff) in poly.items()}

            poly["z"] = (0, 0)
            return {
                "field_type"                  : field_type,
                "field_strength"              : "field_polynomial",
                "field_polynomial"            : poly,
                "field_shape"                 : random.choice(["linear", "circular"]),
                "field_angle"                 : round(random.uniform(0, 360), 2),
                "field_range"                 : (round(random.uniform(0.5, 1.5), 2), round(random.uniform(2.0, 3.0), 2)),
                "field_position"                         : (
                    round(random.uniform(-1.0, 1.0), 2),
                    round(random.uniform(-1.0, 1.0), 2),
                    0.0,
                ),
            }

        def _poly_field_circular() -> Dict[str, Any]:
            p = random.choice([1, 2]) #, -1, -2]) temporarily remove negative powers
            coeff = round(random.uniform(0.1, 2), 2)

            return {
                "field_type"                             : "magnetic",
                "field_strength"                         : "field_polynomial_from_distance",
                "field_polynomial_from_distance"         : {"r": (p, coeff)},
                "field_shape"                            : "circular",
                "field_position"                         : (
                    round(random.uniform(-1.0, 1.0), 2),
                    round(random.uniform(-1.0, 1.0), 2),
                    0.0,
                ),
                "field_range":                            (round(random.uniform(0.5, 1.5), 2), round(random.uniform(2.0, 3.0), 2)),
            }

        if degree_of_randomization == DegreeOfRandomization.EASY:
            self.field_configs.append(_uniform_field(random.choice(["electric", "magnetic"])))
        elif degree_of_randomization == DegreeOfRandomization.MEDIUM:
            mode = random.choice(["superposition", "single_poly"])
            if mode == "superposition":
                self.field_configs.extend([
                    _uniform_field("electric"),
                    _uniform_field("magnetic"),
                ])
            else:
                if random.choice([True, False]):
                    self.field_configs.append(_poly_field_linear(random.choice(["electric", "magnetic"])))
                else:
                    self.field_configs.append(_poly_field_circular())
        else:  # HARD
            mode = random.choice(["single_circular", "multi_mix"])
            if mode == "single_circular":
                self.field_configs.append(_poly_field_circular())
            else:
                for _ in range(random.randint(2, 3)):
                    ftype = random.choice(["electric", "magnetic"])
                    self.field_configs.append(
                        _uniform_field(ftype) if random.choice([True, False]) else _poly_field_linear(ftype)
                    )

        # deep copy to avoid accidental references
        self.field_configs = deepcopy(self.field_configs)

        # 4) reinitialize if requested
        if reinitialize_instance and hasattr(self, "reinitialize"):
            self.reinitialize()

    def get_nlq(self, symbolic = False):
        charge = "<charge>1"
        mass = "<mass>1"
        vx = "<vx>1"
        vy = "<vy>1"

        sym_dict = {
            charge: self.q,
            mass: self.mass,
            vx: self.init_velocity[0],
            vy: self.init_velocity[1],
        }

        particle_description = (
            f"A charged particle with charge {charge} C and mass {mass} kg is initially launched in the space. "
            f"It has an initial velocity of {vx} m/s in the x-direction and {vy} m/s in the y-direction. "
        )

        field_description = ""
        # Add field descriptions later
        for field_config in self.field_configs:
            field_type = field_config["field_type"]
            field_strength = field_config["field_strength"]
            range_desc = (
                    f"The field is effective in an annular (ring-shaped) region (XY plane) with radii from {field_config['field_range'][0]} m to {field_config['field_range'][1]} m, "
                    f"centered at {field_config['field_position'][:-1]}."
            )
            if field_config["field_shape"] == "linear":
                range_desc = (
                    f"The field is effective in a rectangular region (XY plane) of width {field_config['field_range'][0]} m and height {field_config['field_range'][1]} m, "
                    f"centered at {field_config['field_position'][:-1]}."
                )
            if isinstance(field_strength, (int, float)):
                field_description += (
                    f"There is a uniform {field_type} field of strength {field_strength} along the XY plane, making an "
                    f"angle of {field_config['field_angle']} degrees with the x-axis. "
                    f"{range_desc}"
                )
            elif field_strength == "field_polynomial":
                field_description += (
                    f"There is a {field_type} field whose strength varies with the position as "
                    f"f_x = {field_config['field_polynomial']['x'][1]} * (Δx)^{field_config['field_polynomial']['x'][0]} and "
                    f"f_y = {field_config['field_polynomial']['y'][1]} * (Δy)^{field_config['field_polynomial']['y'][0]}. "
                    f"{range_desc} "
                    f"Δx and Δy refer to the position vector of the point wrt the circular region's center."
                )
            elif field_strength == "field_polynomial_from_distance":
                field_description += (
                    f"There is a {field_type} field whose strength varies with the position as "
                    f"f_z = {field_config['field_polynomial_from_distance']['r'][1]} * (r)^{field_config['field_polynomial_from_distance']['r'][0]}. "
                    f"{range_desc} "
                    f"r refers to the distance from the center of the circular region to the point."
                )
            
        gi = (
            "The charged particle is only affected by the fields mentioned. "
            "Forces by other charged particles or gravity are negligible. "
            "The particle is not affected by any other forces."
        )

        description = particle_description + '\n' + field_description + '\n' + gi

        if not symbolic:
            description = replace_all(description, sym_dict)
            return description
        
        return description, sym_dict

    def connecting_point_nl(self, cd, cp, csi):
        raise NotImplementedError("MagneticElectricEntity is not supposed to have connections.")
    
    def get_question(self, sub_entity: str, quantity: str) -> str:
        """
        Get a question related to the entity
        
        Inputs:
            sub_entity: str
            quantity: str
            
        Returns:
            str
        """

        return f"What is the {quantity} of the charged particle"

class ElectroMagneticEntity(Entity):
    """
    ElectroMagenticEntity is used to create a magnetic or electric field
    acting on a single charged particle.

    This class is an alternate version for EM.
    """

    randomization_levels = {
        DegreeOfRandomization.EASY: {
            "mass"  : (0.5, 1.5),
            "charge_magnitude": (1.0, 3.0),
            "charge_sign": [-1, 1],
            "speed" : (0.5, 3.0),
            "modes":{
                "uniform_electric": {
                    "field_strength"  : (0.1, 2),
                    "field_shape"     : ["box", "circle"],
                    "field_angle"     : (0, 360),
                    "field_range"     : (0.5, 1.5),
                    "field_range_center"  : (-1.0, 1.0),
                },
                "uniform_magnetic": {
                    "field_strength"  : (0.1, 2),
                    "field_shape"     : ["box", "circle"],
                    "field_range"     : (0.5, 1.5),
                    "field_range_center"  : (-1.0, 1.0),
                },
            },
        },
        DegreeOfRandomization.MEDIUM: {
            "mass"  : (0.5, 1.5),
            "charge_magnitude": (1.0, 3.0),
            "charge_sign": [-1, 1],
            "speed" : (0.5, 3.0),
            "modes":{
                "electrostatic": {
                    "polynomial_coeff": (0.1, 1.5),
                    "polynomial_power": [0, 1, -1, -2], # -1 means sin, -2 means cos
                    "field_shape"     : ["box", "circle"],
                    "field_range"     : (0.5, 1.5),
                    "field_range_center"  : (-1.0, 1.0),
                },
                "radial_electric": {
                    "field_strength"  : (0.1, 2),
                    "field_center"    : (1.0, 2.0),
                },
                "radial_magnetic": {
                    "field_strength"  : (0.1, 2),
                    "wire_point": (0.5, 1.0),
                    "wire_angle": (-90, 180),
                },
            },
        },
        DegreeOfRandomization.HARD: {
            "mass"  : (0.5, 1.5),
            "charge_magnitude": (1.0, 3.0),
            "charge_sign": [-1, 1],
            "speed" : (0.5, 3.0),
            "modes": {
                "drift": {
                    "electrostatic_strength"  : (0.1, 2),
                    "electrostatic_shape"     : ["box", "circle"],
                    "electrostatic_angle"     : (0, 360),
                    "electrostatic_range"     : (0.5, 1.5),
                    "electrostatic_range_center"  : (-1.0, 1.0),
                    "magnetic_strength"  : (0.1, 1.5),
                },
            },
        },
    }

    def __init__(
        self,
        name: str,
        pos: Tuple[float, float, float] = (0, 0, 0),
        quat: Tuple[float, float, float, float] = (1, 0, 0, 0),
        mass: float = 1.0,
        q: float = 1.0,
        init_velocity: Tuple[float, float, float, float, float, float] = (1, 0, 0, 0, 0, 0),
        field_configs: List[Dict[str, Any]] = None,
        **kwargs,
    ):
        self.mass = mass
        self.q = q
        self.init_velocity = init_velocity
        self.field_configs = field_configs or []

        super().__init__(name, pos, quat, entity_type=self.__class__.__name__, **kwargs)

        # create the charged particle
        self.particle = Mass(
            name=f"{self.name}.particle",
            positions=[pos],
            quat=quat,
            mass_value=self.mass,
            init_velocity={InitVelocityType.MASS: list(self.init_velocity)},
        )
        self.add_child_body(self.particle)

        self.trail_bodies = [(f"{self.name}.particle", 4000)]

    def get_fields(self, pos: np.ndarray) -> List[Dict[str, Any]]:
        """
        Returns a list of dicts of the form:
        {
            "field_type"   : "electric" | "magnetic",
            "field_strength": (fx, fy, fz),
        }
        """
        fields = []
        pos = tuple(pos)

        for field_config in self.field_configs:
            mode = field_config["mode"]

            if mode in ["uniform_electric", "uniform_magnetic", "electrostatic"] or (mode == "drift" and field_config["field_type"] == "electric"):
                # if field field_range is not none and pos is outside the range, skip the field
                box_cond = field_config.get("field_range") is None or (
                    (abs(pos[0] - field_config["field_position"][0]) <= field_config["field_range"][0] / 2) and 
                    (abs(pos[1] - field_config["field_position"][1]) <= field_config["field_range"][1] / 2)
                )

                circ_cond = field_config.get("field_range") is None or (
                    field_config["field_range"][0] <= math.hypot(pos[0] - field_config["field_position"][0], pos[1] - field_config["field_position"][1]) <= field_config["field_range"][1]
                )

                if (
                    (field_config.get("field_shape") == "box" and not box_cond) or 
                    (field_config.get("field_shape") == "circle" and not circ_cond)
                ):
                    continue
            
            if mode in ["uniform_electric", "uniform_magnetic", "drift"]:
                strength = field_config["field_strength"]
                angle_deg = field_config.get("field_angle", 0.0)
                if angle_deg is None: angle_deg = 0.0
                angle = math.radians(angle_deg)
                fx = strength * math.cos(angle)
                fy = strength * math.sin(angle)
                field_strength = (fx, fy, 0.0)
                if field_config["field_type"] == "magnetic":
                    field_strength = (0.0, 0.0, fx)
                fields.append({
                    "field_type"     : field_config["field_type"],
                    "field_strength" : field_strength,
                })
            if mode == "electrostatic":
                poly = field_config["field_polynomial"]
                dx = pos[0] - field_config["field_position"][0]
                dy = pos[1] - field_config["field_position"][1]
                
                if poly["x"][0] == -1:
                    fx = poly["x"][1] * math.sin(dx)
                elif poly["x"][0] == -2:
                    fx = poly["x"][1] * math.cos(dx)
                else: 
                    fx = (dx ** poly["x"][0]) * poly["x"][1]
                
                if poly["y"][0] == -1:
                    fy = poly["y"][1] * math.sin(dy)
                elif poly["y"][0] == -2:
                    fy = poly["y"][1] * math.cos(dy)
                else:
                    fy = (dy ** poly["y"][0]) * poly["y"][1]

                field_strength = (fx, fy, 0.0)
                fields.append({
                    "field_type"     : field_config["field_type"],
                    "field_strength" : field_strength,
                })
            if mode == "radial_electric":
                strength = field_config["field_strength"]
                field_center = field_config["field_position"]
                dx = pos[0] - field_center[0]
                dy = pos[1] - field_center[1]
                r = math.hypot(dx, dy)
                fx = strength * (dx / (r + 1e-2)**3)
                fy = strength * (dy / (r + 1e-2)**3)
                field_strength = (fx, fy, 0.0)
                fields.append({
                    "field_type"     : field_config["field_type"],
                    "field_strength" : field_strength,
                })
            if mode == "radial_magnetic":
                """Field direction is tangential to the radial vector from wire.
                wire point is a point on the wire, and wire angle is the angle of the wire with respect to the x-axis. This defines the 2D axis of the wire. 
                field is mu I / 2pi r, where I is the current, and r is the distance from the wire to the point of interest.
                I assume mu I / 2pi is = field_strength"""
                strength = field_config["field_strength"]
                wire_point = field_config["wire_position"]
                wire_angle = field_config["wire_angle"]
                dx = pos[0] - wire_point[0]
                dy = pos[1] - wire_point[1]
                r = math.hypot(dx, dy) + abs(pos[2]) + 1e-2 # To avoid division by zero

                def tangential_direction(P, r0, theta):
                    # Inputs
                    x, y, z = P
                    x0, y0 = r0
                    d = np.array([np.cos(theta), np.sin(theta), 0])         # axis vector
                    w = np.array([x - x0, y - y0, 0])                       # 2D displacement
                    s = np.dot(w, d)                                        # scalar projection
                    C = np.array([x0, y0, 0]) + s * d                       # closest point on wire
                    R = np.array([x, y, z]) - C                             # radial vector
                    v = np.cross(d, R)                                     # tangential direction
                    return v

                # Calculate the field direction
                field_direction = tangential_direction(pos, wire_point, wire_angle)
                field_direction /= np.linalg.norm(field_direction)  # Normalize the direction vector
                
                field_strength = (strength / r) * field_direction 
                field_strength = (field_strength[0], field_strength[1], field_strength[2])

                fields.append({
                    "field_type"     : field_config["field_type"],
                    "field_strength" : field_strength,
                })
        
        return fields

    def randomize_parameters(self, degree_of_randomization = DegreeOfRandomization.DEFAULT, reinitialize_instance=False, **kwargs):
        if degree_of_randomization not in  self.randomization_levels:
            # Choose one randomly
            degree_of_randomization = random.choice(list(self.randomization_levels.keys()))
        
        params = self.randomization_levels[degree_of_randomization]

        # 1) mass & charge
        self.mass = round(random.uniform(*params["mass"]), 2)
        self.q = round(random.choice(params["charge_sign"]) * random.uniform(*params["charge_magnitude"]), 2)

        # 2) initial velocity 
        v_mag = random.uniform(*params["speed"])
        theta = random.uniform(0, 2 * math.pi)
        vx, vy = v_mag * math.cos(theta), v_mag * math.sin(theta)
        self.init_velocity = (round(vx, 2), round(vy, 2), 0.0, 0.0, 0.0, 0.0)

        # 3) generate field configurations
        self.field_configs = []
        # Randomly choose a mode
        mode = random.choice(list(params["modes"].keys()))
        mode_params = params["modes"][mode]

        if mode == "uniform_electric":
            field_strength = round(random.uniform(*mode_params["field_strength"]), 2)
            field_shape = random.choice(mode_params["field_shape"])
            field_angle = round(random.uniform(*mode_params["field_angle"]), 2)
            if field_shape == "circle":
                field_range = (round(random.uniform(0, mode_params["field_range"][0]), 2), round(random.uniform(*mode_params["field_range"]), 2))
            else:
                field_range = (round(random.uniform(*mode_params["field_range"]), 2), round(random.uniform(*mode_params["field_range"]), 2))
            field_range_center = (round(random.uniform(*mode_params["field_range_center"]), 2), round(random.uniform(*mode_params["field_range_center"]), 2))
            self.field_configs.append({
                "field_type"      : "electric",
                "field_strength"  : field_strength,
                "field_shape"     : field_shape,
                "field_angle"     : field_angle,
                "field_range"     : field_range,
                "field_position"  : field_range_center,
            })
        elif mode == "uniform_magnetic":
            field_strength = round(random.uniform(*mode_params["field_strength"]), 2)
            field_shape = random.choice(mode_params["field_shape"])
            if field_shape == "circle":
                field_range = (round(random.uniform(0, mode_params["field_range"][0]), 2), round(random.uniform(*mode_params["field_range"]), 2))
            else:
                field_range = (round(random.uniform(*mode_params["field_range"]), 2), round(random.uniform(*mode_params["field_range"]), 2))
            field_range_center = (round(random.uniform(*mode_params["field_range_center"]), 2), round(random.uniform(*mode_params["field_range_center"]), 2))
            self.field_configs.append({
                "field_type"      : "magnetic",
                "field_strength"  : field_strength,
                "field_shape"     : field_shape,
                "field_angle"     : None,
                "field_range"     : field_range,
                "field_position"  : field_range_center,
            })
        elif mode == "electrostatic":
            field_shape = random.choice(mode_params["field_shape"])
            if field_shape == "circle":
                field_range = (round(random.uniform(0, mode_params["field_range"][0]), 2), round(random.uniform(*mode_params["field_range"]), 2))
            else:
                field_range = (round(random.uniform(*mode_params["field_range"]), 2), round(random.uniform(*mode_params["field_range"]), 2))
            field_range_center = (round(random.uniform(*mode_params["field_range_center"]), 2), round(random.uniform(*mode_params["field_range_center"]), 2))
            
            polynomial_coeff = [round(random.uniform(*mode_params["polynomial_coeff"]), 2) for i in range(2)]
            polynomial_power = [random.choice(mode_params["polynomial_power"]) for i in range(2)]
            
            self.field_configs.append({
                "field_type"      : "electric",
                "field_strength"  : "field_polynomial",
                "field_polynomial": {
                    "x": (polynomial_power[0], polynomial_coeff[0]),
                    "y": (polynomial_power[1], polynomial_coeff[1]),
                    "z": (0, 0),
                },
                "field_shape"     : field_shape,
                "field_range"     : field_range,
                "field_position"  : field_range_center,
            })
        elif mode == "radial_electric":
            field_strength = round(random.uniform(*mode_params["field_strength"]), 2)
            field_center = (round(random.uniform(*mode_params["field_center"]), 2), round(random.uniform(*mode_params["field_center"]), 2))
            self.field_configs.append({
                "field_type"      : "electric",
                "field_strength"  : field_strength,
                "field_shape"     : "circular",
                "field_position"  : field_center,
            })
        elif mode == "radial_magnetic":
            field_strength = round(random.uniform(*mode_params["field_strength"]), 2)
            wire_point = (round(random.uniform(*mode_params["wire_point"]), 2), round(random.uniform(*mode_params["wire_point"]), 2))
            wire_angle = round(random.uniform(*mode_params["wire_angle"]), 2)
            self.field_configs.append({
                "field_type"      : "magnetic",
                "field_strength"  : field_strength,
                "field_shape"     : "circular",
                "wire_angle"     : wire_angle,
                "wire_position"  : wire_point,
            })
        elif mode == "drift":
            electrostatic_strength = round(random.uniform(*mode_params["electrostatic_strength"]), 2)
            electrostatic_shape = random.choice(mode_params["electrostatic_shape"])
            electrostatic_angle = round(random.uniform(*mode_params["electrostatic_angle"]), 2)
            if electrostatic_shape == "circle":
                electrostatic_range = (round(random.uniform(0, mode_params["electrostatic_range"][0]), 2), round(random.uniform(*mode_params["electrostatic_range"]), 2))
            else:
                electrostatic_range = (round(random.uniform(*mode_params["electrostatic_range"]), 2), round(random.uniform(*mode_params["electrostatic_range"]), 2))
            electrostatic_range_center = (round(random.uniform(*mode_params["electrostatic_range_center"]), 2), round(random.uniform(*mode_params["electrostatic_range_center"]), 2))
            magnetic_strength = round(random.uniform(*mode_params["magnetic_strength"]), 2)
            self.field_configs.append({
                "field_type"      : "electric",
                "field_strength"  : electrostatic_strength,
                "field_shape"     : electrostatic_shape,
                "field_angle"     : electrostatic_angle,
                "field_range"     : electrostatic_range,
                "field_position"  : electrostatic_range_center,
            })
            self.field_configs.append({
                "field_type"      : "magnetic",
                "field_strength"  : magnetic_strength,
                "field_shape"     : None,
            }
            )
        [fc.update({"mode": mode}) for fc in self.field_configs]

        # deep copy to avoid accidental references
        self.field_configs = deepcopy(self.field_configs)

        # 4) reinitialize if requested
        if reinitialize_instance and hasattr(self, "reinitialize"):
            self.reinitialize()

    def generate_entity_yaml(
        self,
        use_random_parameters: bool = False,
        degree_of_randomization: "DegreeOfRandomization" = "DegreeOfRandomization.NON_STRUCTURAL",
    ) -> Dict[str, Any]:
        if use_random_parameters:
            self.randomize_parameters(degree_of_randomization)

        data = {
            "name"      : self.name,
            "type"      : self.__class__.__name__,
            "position"  : list(self.pos),
            "parameters": {
                "mass"         : self.mass,
                "q"            : self.q,
                "init_velocity": list(self.init_velocity),
                "field_configs": self.field_configs,
            },
        }
        # Remove or implement round_floats as needed
        return round_floats(data)  # type: ignore

    def get_nlq(self, symbolic = False):
        charge = "<charge>1"
        mass = "<mass>1"
        vx = "<vx>1"
        vy = "<vy>1"

        sym_dict = {
            charge: self.q,
            mass: self.mass,
            vx: self.init_velocity[0],
            vy: self.init_velocity[1],
        }

        particle_description = (
            f"A charged particle with charge {charge} C and mass {mass} kg is initially launched from the origin in the space. "
            f"It has an initial velocity of {vx} m/s in the x-direction and {vy} m/s in the y-direction. "
        )

        field_description = ""
        # Add field descriptions
        for field_config in self.field_configs:
            field_type = field_config["field_type"]
            field_strength = field_config["field_strength"]
            
            range_desc = ""
            if field_config["field_shape"] == "circle":
                range_desc = (
                        f"The field is effective in an annular (ring-shaped) region (XY plane) with radii from {field_config['field_range'][0]} m to {field_config['field_range'][1]} m, "
                        f"centered at {field_config['field_position']}."
                )
            elif field_config["field_shape"] == "box":
                range_desc = (
                    f"The field is effective in a rectangular region (XY plane) of width {field_config['field_range'][0]} m and height {field_config['field_range'][1]} m, "
                    f"centered at {field_config['field_position']}."
                )
            
            if field_config["mode"] in ["uniform_electric", "uniform_magnetic", "drift"]:
                axis_desc = f"+Z axis. " if field_config["field_type"] == "magnetic" else f"XY plane, making an angle of {field_config['field_angle']} degrees with the x-axis. "
                field_description += (
                    f"There is a uniform {field_type} field of strength {field_strength} along the "
                    f"{axis_desc}"                    
                    f"{range_desc}"
                )
            elif field_config["mode"] == "electrostatic":

                def poly_desc(power, coeff, axis = "x"):
                    if power == -1:
                        return f"{coeff} * sin(Δ{axis})"
                    elif power == -2:
                        return f"{coeff} * cos(Δ{axis})"
                    else:
                        return f"({coeff}) * (Δ{axis})^{power}"

                field_description += (
                    f"There is a {field_type} field whose strength varies with the position as "
                    f"f_x = {poly_desc(*field_config['field_polynomial']['x'], 'x')} and "
                    f"f_y = {poly_desc(*field_config['field_polynomial']['y'], 'y')}. "
                    f"{range_desc} "
                    f"Δx and Δy refer to the position vector of the point wrt the region's center."
                )
            elif field_config["mode"] == "radial_electric":
                field_description += (
                    f"There is a {field_type} field whose strength varies with the position as "
                    f"f_x = {field_strength} / Δr^2 and "
                    f"direction is along the Δr vector. "
                    f"Δr refers to the position vector from ({field_config['field_position'][0]}, {field_config['field_position'][1]}) m to the point."
                )        
            elif field_config["mode"] == "radial_magnetic":
                field_description += (
                    f"There is an infinitely long wire (creating a magnetic field around it) in the Z=0 plane, that passes through the point "
                    f"({field_config['wire_position'][0]}, {field_config['wire_position'][1]}) m and "
                    f"makes an angle of {field_config['wire_angle']} degrees with the +X-axis. "
                    f"The value of μ I / 2 pi is equal to {field_strength}."
                )
            
        gi = (
            "The charged particle is only affected by the fields mentioned. "
            "Forces by other charged particles or gravity are negligible. "
            "The particle is not affected by any other forces."
        )

        description = particle_description + '\n' + field_description + '\n' + gi

        if not symbolic:
            description = replace_all(description, sym_dict)
            return description
        
        return description, sym_dict

    def connecting_point_nl(self, cd, cp, csi):
        raise NotImplementedError("ElectroMagneticEntity is not supposed to have connections.")
    
    def get_question(self, sub_entity: str, quantity: str) -> str:
        """
        Get a question related to the entity
        
        Inputs:
            sub_entity: str
            quantity: str
            
        Returns:
            str
        """

        return f"What is the {quantity} of the charged particle"

    def get_shortcut(self):
        self.field_configs = []
        return True
