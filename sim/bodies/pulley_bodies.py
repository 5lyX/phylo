from .base_bodies import *


class FixedPulley(Body):
    """
    A specialized Body class to represent a fixed pulley system.
    """

    def __init__(
        self,
        name: str,
        pos: Tuple[float, float, float],
        offset: float = DEFAULT_PULLEY_RADIUS,
        **kwargs,
    ) -> None:
        super().__init__(name, pos, **kwargs)
        self.left_site = None
        self.right_site = None
        self.site = None
        self.body_type = "fixed pulley"
        if offset > 0:  # FixedPulley can be just one
            self.left_site = Site(
                f"{name}.left_site", (-offset, 0, 0.0), body_name=name
            )
            self.add_site(self.left_site)
            self.right_site = Site(
                f"{name}.right_site", (offset, 0, 0.0), body_name=name
            )
            self.add_site(self.right_site)
        else:
            self.site = Site(f"{name}.site", (0, 0, 0.0), body_name=name)
            self.add_site(self.site)

class MovablePulley(Body):
    """
    A specialized Body class to represent a movable pulley system.
    """

    def __init__(
        self,
        name: str,
        pos: Tuple[float, float, float],
        size_x: float = DEFAULT_PULLEY_RADIUS,
        size_y: float = DEFAULT_PULLEY_HEIGHT,
        material: str = DEFAULT_MATERIAL,
        rgba: Tuple[float, float, float, float] = DEFAULT_RGBA,
        mass: float = 0.0,
        quat: Tuple[float, float, float, float] = (0, 0, 0.7071068, 0.7071068),
        constant_force: Optional[Dict[str, List[float]]] = None,
        winding_direction: str = "down",
    ) -> None:
        super().__init__(name, pos)
        self.body_type = "movable pulley"
        self.winding_direction = winding_direction

        self.mass_value = mass

        # Add a geom and sites specific to a movable pulley
        self.add_geom(
            Geom(
                f"{name}.geom",
                "cylinder",
                (0, 0, 0),
                (size_x, size_y),
                material,
                rgba,
                mass,
                quat,
            )
        )
        # this site is for the mass to attach
        if winding_direction == "down":
            self.mass_site = Site(
                f"{name}.site", (0.0, 0, -DEFAULT_PULLEY_RADIUS), body_name=name
            )
        elif winding_direction == "up":
            self.mass_site = Site(
                f"{name}.site", (0.0, 0, DEFAULT_PULLEY_RADIUS), body_name=name
            )
        else:
            raise ValueError("Invalid winding direction. Please choose 'up' or 'down'.")

        self.sensor_site = Site(f"{name}.sensor", (0, 0, 0), body_name=name)
        self.add_site(self.mass_site)
        self.add_site(self.sensor_site)
        # this site is for the tendon to attach
        self.tendon_site = Site(
            f"{name}.tendon_site",
            (0.0, 0, -DEFAULT_PULLEY_RADIUS - DEFAULT_MASS_SIZE / 2),
            body_name=name,
        )
        self.left_tendon_site = Site(
            f"{name}.left_tendon_site",
            (-DEFAULT_PULLEY_RADIUS, 0, 0),
            body_name=name,
        )
        self.right_tendon_site = Site(
            f"{name}.right_tendon_site",
            (DEFAULT_PULLEY_RADIUS, 0, 0),
            body_name=name,
        )
        self.add_site(self.tendon_site)
        self.add_site(self.left_tendon_site)
        self.add_site(self.right_tendon_site)
        # Add joints specific to a movable pulley
        self.add_joint(Joint("slide", (1, 0, 0)))
        self.add_joint(Joint("slide", (0, 1, 0)))
        self.add_joint(Joint("slide", (0, 0, 1)))

        if constant_force and ConstantForceType.PULLEY in constant_force:
            self.constant_force_dict[self.name] = constant_force[
                ConstantForceType.PULLEY
            ]

    def generate_spatial_elements(
        self, use_sidesite: bool = False
    ) -> Union[List[Geom], List[Site]]:
        """
        Create a simplified geom element for use in tendon attachments that only references the original geom and its site.
        """
        if not use_sidesite:
            return [
                self.left_tendon_site.create_spatial_site(),
                self.right_tendon_site.create_spatial_site(),
            ]
        return [
            Geom(
                geom=self.geoms[
                    0
                ].name,  # Reference to the original geom name for tendon usage
                sidesite=self.tendon_site.name,  # Reference to the tendon attachment site
                quat=None,
            )
        ]
    
