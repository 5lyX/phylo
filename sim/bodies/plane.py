from .base_bodies import *


class Plane(Body):
    """
    Represents an inclined plane, typically fixed and non-movable.
    """

    def __init__(
        self,
        name: str,
        pos: Tuple[float, float, float] = (0, 0, 0),
        rgba: Tuple[float, float, float, float] = (
            0.82745098,
            0.75686275,
            0.76470588,
            1,
        ),  # D3C1C3
        mass: float = 1.0,
        quat: Tuple[float, float, float, float] = (1, 0, 0, 0),
        size: Tuple[float, float, float] = (1, 2, 0.01),
        site_padding: float = DEFAULT_MASS_SIZE,  # padding in the z direction (upwards) to make site parallel to the mass center
        condim: str = "1",
    ) -> None:
        super().__init__(name, pos, quat)
        self.body_type = "plane"
        self.size = size
        self.left_site = Site(
            f"{self.name}.left",
            (-self.size[0] - DEFAULT_MASS_SIZE / 2, 0, self.size[2] + site_padding),
            (1, 0, 0, 0),
            body_name=name,
        )
        self.right_site = Site(
            f"{self.name}.right",
            (self.size[0] + DEFAULT_MASS_SIZE / 2, 0, self.size[2] + site_padding),
            (1, 0, 0, 0),
            body_name=name,
        )

        self.add_geom(
            Geom(
                name=f"{name}.geom",
                geom_type="box",
                pos=(0, 0, 0),
                size=size,
                rgba=rgba,
                mass=mass,
                condim=condim,
                material="reflectance",
            )
        )
        self.add_sites()

    def add_sites(self) -> None:
        """
        Add sites at specific positions and a sensor site at the center.
        """
        self.add_site(
            Site(f"{self.name}.sensor", (0, 0, 0), (1, 0, 0, 0), body_name=self.name)
        )
        self.add_site(self.left_site)
        self.add_site(self.right_site)

    def pos_on_top(
        self,
        x: float,  # [-1,1], -1 is the left edge, 1 is the right edge
        y: float,  # [-1,1], -1 is the inner edge, 1 is the outer edge
        z_padding: float = 0,  # padding in the z direction (upwards) to make sure the mass is not in the ground
    ) -> Tuple[float, float, float]:
        """
        Calculate the position on the plane. (global coordinates)
        """
        x_pos = x * self.size[0]
        y_pos = y * self.size[1]
        z_pos = (
            self.size[2] + z_padding
        )  # upward movement by z_padding to avoid the object being in the ground
        local_pos = (x_pos, y_pos, z_pos)

        # local to global
        frame = Frame(origin=np.array(self.pos), quat=np.array(self.quat))
        global_pos = frame.rel2global(local_pos)
        return local_pos, tuple(global_pos), tuple(self.quat)
