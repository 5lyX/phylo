import os, shutil
import random
import hydra
from omegaconf import DictConfig
import yaml
import numpy as np
import datetime
from collections import deque
from typing import Dict, Any, List

from sim.scene import (
    Entity,
    MassWithFixedPulley,
    MassWithMovablePulley,
    DirectedMass,
    TwoSideMassPlane,
    MassPrismPlaneEntity,
    FixedPulleyEntity,
    StackedMassPlane,
    MassWithReversedMovablePulley,
    ConstantForceFixedPulley,
    TwoDCollisionPlane,
    ComplexCollisionPlane,
    SpringBlockEntity,
    RigidRotationEntity,
    BarPlaneSupport,
    PendulumEntity,
    DiskRackWithSphereEntity,
    ConnectingDirection,
    ConnectingPoint,
    DegreeOfRandomization,
    GeneralCelestialEntity,
    OrbitalMotionEntity,
    RollingPlaneEntity,
    MassBoxPlaneEntity,
    MassPrismPulleyPlane,
    ConstantVelocityPuller,
    SpringMassPlaneEntity,
    SolarSystemEntity,
    RocketEntity,
    ThrowingMotionEntity, 
    MagneticElectricEntity,
    ElectroMagneticEntity,
    parse_scene,
)
import ipdb

st = ipdb.set_trace

import subprocess

# dir_path = "/Users/yangminl/Documents/PHO/batch_generation_out"

# subprocess.run(["rm", "-rf", dir_path], check=True)

# ==============================
# 1) Global Scene Configuration: SCENE_CONFIGS
# ==============================
# Here we list all the mentioned subcategories into a large dictionary according to the problem description.
# Each subcategory includes:
#   - entity_range: (minimum number, maximum number) — used to randomly decide how many regular entities the BFS main body needs to generate
#   - closing_entities: entities that can be optionally added at the end (e.g., MassWithFixedPulley, ConstantForceFixedPulley, etc.). If not needed, leave empty []
#   - possible_entities: a list listing available entities and their corresponding difficulty (mapped to generate_entity_yaml parameters)

SCENE_CONFIGS = {
    # 1. Pulley & Pulley Systems
    "BasicPulley": {
        "entity_range": (1, 2),  # 1 to 2 regular entities
        "closing_entities": ["MassWithFixedPulley"],  # + MassWithFixedPulley
        "possible_entities": [
            {"class_name": "MassWithFixedPulley", "difficulty": "EASY"},
            {"class_name": "MassWithMovablePulley", "difficulty": "EASY"},
            {"class_name": "MassWithReversedMovablePulley", "difficulty": "EASY"},
        ],
    },
    "IntermediatePulley": {
        "entity_range": (2, 3),  # 2 to 3 regular entities
        "closing_entities": ["MassWithFixedPulley", "ConstantForceFixedPulley"],
        "possible_entities": [
            {"class_name": "MassWithFixedPulley", "difficulty": "EASY"},
            {"class_name": "MassWithMovablePulley", "difficulty": "EASY"},
            {"class_name": "MassWithReversedMovablePulley", "difficulty": "EASY"},
            {"class_name": "DirectedMass", "difficulty": "EASY"},
        ],
    },
    # 2. Inclined Plane & Friction
    "BasicInclinedPlaneFriction": {
        "entity_range": (1, 1),  # 1 regular entity
        "closing_entities": ["MassWithFixedPulley"],  # + MassWithFixedPulley
        "possible_entities": [
            {"class_name": "TwoSideMassPlane", "difficulty": "EASY"},
            {"class_name": "MassPrismPlaneEntity", "difficulty": "EASY"},
        ],
    },
    "IntermediateInclinedPlaneFriction": {
        "entity_range": (2, 3),  # 2 to 3 regular entities
        "closing_entities": ["MassWithFixedPulley"],  # + MassWithFixedPulley
        "possible_entities": [
            {"class_name": "TwoSideMassPlane", "difficulty": "MEDIUM"},
            {"class_name": "MassPrismPlaneEntity", "difficulty": "MEDIUM"},
            {"class_name": "StackedMassPlane", "difficulty": "EASY"},
        ],
    },
    "AdvancedInclinedPlaneFriction": {
        "entity_range": (3, 3),  # 3 regular entities
        "closing_entities": ["MassWithFixedPulley", "ConstantForceFixedPulley"],
        "possible_entities": [
            {"class_name": "TwoSideMassPlane", "difficulty": "HARD"},
            {"class_name": "MassPrismPlaneEntity", "difficulty": "MEDIUM"},
            {"class_name": "StackedMassPlane", "difficulty": "MEDIUM"},
        ],
    },
    # 3. Pulley + Inclined Plane Hybrid
    "IntermediateHybrid": {
        "entity_range": (2, 3),
        "closing_entities": ["MassWithFixedPulley", "ConstantForceFixedPulley"],
        "possible_entities": [
            {"class_name": "MassWithFixedPulley", "difficulty": "MEDIUM"},
            {"class_name": "MassWithMovablePulley", "difficulty": "MEDIUM"},
            {"class_name": "MassWithReversedMovablePulley", "difficulty": "MEDIUM"},
            {"class_name": "TwoSideMassPlane", "difficulty": "MEDIUM"},
            {"class_name": "MassPrismPlaneEntity", "difficulty": "MEDIUM"},
            {"class_name": "StackedMassPlane", "difficulty": "EASY"},
        ],
    },
    "AdvancedHybrid": {
        "entity_range": (3, 4),
        "closing_entities": ["MassWithFixedPulley", "ConstantForceFixedPulley"],
        "possible_entities": [
            {"class_name": "MassWithFixedPulley", "difficulty": "HARD"},
            {"class_name": "MassWithMovablePulley", "difficulty": "HARD"},
            {"class_name": "MassWithReversedMovablePulley", "difficulty": "HARD"},
            {"class_name": "TwoSideMassPlane", "difficulty": "HARD"},
            {"class_name": "MassPrismPlaneEntity", "difficulty": "MEDIUM"},
            {"class_name": "StackedMassPlane", "difficulty": "HARD"},
            {"class_name": "DirectedMass", "difficulty": "HARD"},
        ],
    },
    # 4. Collision
    "BasicCollision": {
        "entity_range": (1, 1),
        "closing_entities": [],
        "possible_entities": [
            {"class_name": "TwoDCollisionPlane", "difficulty": "EASY"},
            {"class_name": "ComplexCollisionPlane", "difficulty": "EASY"},
        ],
    },
    "IntermediateCollision": {
        "entity_range": (1, 1),
        "closing_entities": [],
        "possible_entities": [
            {"class_name": "TwoDCollisionPlane", "difficulty": "MEDIUM"},
            {"class_name": "ComplexCollisionPlane", "difficulty": "MEDIUM"},
        ],
    },
    "AdvancedCollision": {
        "entity_range": (1, 1),
        "closing_entities": [],
        "possible_entities": [
            {"class_name": "TwoDCollisionPlane", "difficulty": "HARD"},
            {"class_name": "ComplexCollisionPlane", "difficulty": "HARD"},
        ],
    },
    # 5. Rotation (single subtype case)
    "Rotation": {
        "entity_range": (1, 1),
        "closing_entities": [],
        "possible_entities": [
            # {"class_name": "BarPlaneSupport", "difficulty": "EASY"},
            {"class_name": "PendulumEntity", "difficulty": "DEFAULT"},
            # {"class_name": "DiskRackWithSphereEntity", "difficulty": "EASY"},
        ],
    },
    # 6. Spring-Block Systems
    "SpringBlockSystems": {
        "entity_range": (1, 1),
        "closing_entities": [],
        # The problem states "SpringBlockEntity (Easy/Medium/Hard)",
        # here we can make 3 lines and randomly select
        "possible_entities": [
            {"class_name": "SpringBlockEntity", "difficulty": "EASY"},
            {"class_name": "SpringBlockEntity", "difficulty": "MEDIUM"},
            {"class_name": "SpringBlockEntity", "difficulty": "HARD"},
        ],
    },
    # 7. Rigid Body Rotation
    "RigidBodyRotation": {
        "entity_range": (1, 1),
        "closing_entities": [],
        "possible_entities": [
            {"class_name": "RigidRotationEntity", "difficulty": "EASY"},
            {"class_name": "RigidRotationEntity", "difficulty": "MEDIUM"},
            {"class_name": "RigidRotationEntity", "difficulty": "HARD"},
        ],
    },
    # 8. Difficult Scenes
    "DifficultPulley": {
        "entity_range": (1, 1),
        "closing_entities": ["ConstantForceFixedPulley"],
        "possible_entities": [
            {"class_name": "MassBoxPlaneEntity", "difficulty": "EASY"},
            {"class_name": "MassBoxPlaneEntity", "difficulty": "MEDIUM"},
            {"class_name": "MassBoxPlaneEntity", "difficulty": "HARD"},
            {"class_name": "MassPrismPulleyPlane", "difficulty": "EASY"},
            {"class_name": "MassPrismPulleyPlane", "difficulty": "MEDIUM"},
            {"class_name": "MassPrismPulleyPlane", "difficulty": "HARD"},
        ],
    },
    "DifficultSpringMass": {
        "entity_range": (1, 1),
        "closing_entities": [],
        "possible_entities": [
            {"class_name": "SpringMassPlaneEntity", "difficulty": "EASY"},
            {"class_name": "SpringMassPlaneEntity", "difficulty": "MEDIUM"},
            {"class_name": "SpringMassPlaneEntity", "difficulty": "HARD"},
        ],
    },
    "DifficultOrbitalMotion": {
        "entity_range": (1, 1),
        "closing_entities": [],
        "possible_entities": [
            {"class_name": "SolarSystemEntity", "difficulty": "EASY"},
            {"class_name": "SolarSystemEntity", "difficulty": "MEDIUM"},
            {"class_name": "SolarSystemEntity", "difficulty": "HARD"},
        ],
    },
    "DifficultRocket": {
        "entity_range": (1, 1),
        "closing_entities": [],
        "possible_entities": [
            {"class_name": "RocketEntity", "difficulty": "EASY"},
            {"class_name": "RocketEntity", "difficulty": "MEDIUM"},
            {"class_name": "RocketEntity", "difficulty": "HARD"},
        ],
    },
    "DifficultProjectile": {
        "entity_range": (1, 1),
        "closing_entities": [],
        "possible_entities": [
            {"class_name": "ProjectileEntity", "difficulty": "EASY"},
            {"class_name": "ProjectileEntity", "difficulty": "MEDIUM"},
            {"class_name": "ProjectileEntity", "difficulty": "HARD"},
        ],
    },
    "DifficultElectroMagnetic": {
        "entity_range": (1, 1),
        "closing_entities": [],
        "possible_entities": [
            {"class_name": "ElectroMagneticEntity", "difficulty": "EASY"},
            {"class_name": "ElectroMagneticEntity", "difficulty": "MEDIUM"},
            {"class_name": "ElectroMagneticEntity", "difficulty": "HARD"},
        ],
    },
}


# Used to map string difficulty to DegreeOfRandomization
def map_difficulty_to_degree(difficulty_str: str) -> DegreeOfRandomization:
    difficulty_str = difficulty_str.upper()
    if difficulty_str == "EASY":
        return DegreeOfRandomization.EASY
    elif difficulty_str == "MEDIUM":
        return DegreeOfRandomization.MEDIUM
    elif difficulty_str == "HARD":
        return DegreeOfRandomization.HARD
    elif difficulty_str == "DEFAULT":
        return DegreeOfRandomization.DEFAULT
    else:
        return DegreeOfRandomization.NON_STRUCTURAL


# ==============================
# 2) Other Global Constants or Configurations
# ==============================
POSITION_INCREMENT =10.0  # Increment for y-axis position
MAX_TENDON_DEPTH = 2  # Maximum allowed depth in BFS
MAX_TENDON_WIDTH = 2  # Maximum allowed number of tendons in BFS
PROBABILITY_WEIGHTS = 0.1

# Here we retain the original mapping of entity class names to class objects
# Note that if you want to use TwoDCollisionPlane / SpringBlockEntity etc., corresponding definitions need to exist in sim.scene
GENERATABLE_ENTITY_CLASSES = {
    "MassWithFixedPulley": MassWithFixedPulley,
    "MassWithMovablePulley": MassWithMovablePulley,
    "DirectedMass": DirectedMass,
    "TwoSideMassPlane": TwoSideMassPlane,
    "MassPrismPlaneEntity": MassPrismPlaneEntity,
    "FixedPulleyEntity": FixedPulleyEntity,
    "StackedMassPlane": StackedMassPlane,
    "MassWithReversedMovablePulley": MassWithReversedMovablePulley,
    "ConstantForceFixedPulley": ConstantForceFixedPulley,
    "TwoDCollisionPlane": TwoDCollisionPlane,
    "ComplexCollisionPlane": ComplexCollisionPlane,
    "SpringBlockEntity": SpringBlockEntity,
    "RigidRotationEntity": RigidRotationEntity,
    "BarPlaneSupport": BarPlaneSupport,
    "PendulumEntity": PendulumEntity,
    "DiskRackWithSphereEntity": DiskRackWithSphereEntity,
    "GeneralCelestialEntity": GeneralCelestialEntity,
    "OrbitalMotionEntity": OrbitalMotionEntity,
    "RollingPlaneEntity": RollingPlaneEntity,
    "MassBoxPlaneEntity": MassBoxPlaneEntity,
    "MassPrismPulleyPlane": MassPrismPulleyPlane,
    "ConstantVelocityPuller": ConstantVelocityPuller,
    "SpringMassPlaneEntity": SpringMassPlaneEntity,
    "SolarSystemEntity": SolarSystemEntity,
    "RocketEntity": RocketEntity,
    "ProjectileEntity": ThrowingMotionEntity,
    "MagneticElectricEntity": MagneticElectricEntity,
    "ElectroMagneticEntity": ElectroMagneticEntity,
}


class SceneGenerator:
    def __init__(self, subtype: str, max_tendon_depth=2, max_tendon_width=2, seed=42, test=False):
        """
        :param subtype: Corresponding key in SCENE_CONFIGS, e.g., "BasicPulley", "AdvancedInclinedPlaneFriction", etc.
        :param max_tendon_depth: Maximum depth of tendons in BFS process
        :param max_tendon_width: Maximum number of tendons in BFS process
        :param seed: Random seed
        """
        if subtype not in SCENE_CONFIGS:
            raise ValueError(f"Unknown subtype: {subtype}")

        # Extract the configuration of the subcategory from the configuration dictionary
        self.subtype = subtype
        self.subtype_config = SCENE_CONFIGS[subtype]
        # Decide the value of N based on the configuration, e.g., (1,2) randomly selects between 1 and 2
        min_n, max_n = self.subtype_config["entity_range"]
        self.N = random.randint(
            min_n, max_n
        )  # Total number of entities to be generated by the main body

        # BFS related initialization
        self.max_tendon_depth = max_tendon_depth
        self.max_tendon_width = max_tendon_width

        # Counters / storage
        self.entity_count = 0
        self.generated_entities = {}
        self.connections = []
        self.entity_positions = {}
        self.current_y_position = 0.0
        self.tendon_count = 0

        # Set random seed
        self.seed_everything(seed)

        self.seed = seed
        self.test = test

    def seed_everything(self, seed=42):
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)

    def generate_scene_yaml(self) -> Dict[str, Any]:
        """
        Core BFS process, similar to the original logic, but in create_random_entity and other steps, entities are selected from self.subtype_config.
        """
        queue = deque()
        # Initially, put an empty tendon with a placeholder
        queue.append([{"placeholder": True}])

        def check_depth_and_proceed(tendon):
            # Check if the current tendon depth exceeds MAX_TENDON_DEPTH
            if self.get_tendon_depth(tendon) < MAX_TENDON_DEPTH:
                # If both ends have placeholders, put back into the queue and wait for generation
                if "placeholder" in tendon[0] or "placeholder" in tendon[-1]:
                    queue.append(tendon)
                else:
                    self.connections.append({"tendon": tendon})
            else:
                # If depth exceeded, close placeholders
                closed_tendon = self.close_tendon_placeholders(tendon)
                self.connections.append({"tendon": closed_tendon})

        while (
                queue
                and self.entity_count < self.N
                and self.tendon_count < self.max_tendon_width
        ):
            tendon = queue.popleft()
            if self.entity_count >= self.N:
                break

            if len(tendon) == 1 and "placeholder" in tendon[0]:
                # Initial (or empty) tendon, generate a new random entity
                new_entity_name, new_entity_instance = self.create_random_entity()
                self.tendon_count += 1

                # Generate new tendons based on the connection points of the entity
                new_tendons = self.generate_tendons_for_entity(
                    entity_name=new_entity_name,
                    entity_instance=new_entity_instance,
                )
                for new_tendon in new_tendons:
                    queue.append(new_tendon)

            else:
                # Check left end placeholder
                if "placeholder" in tendon[0]:
                    if self.entity_count >= self.N:
                        break
                    (
                        new_entity_name,
                        new_entity_instance,
                        cp_instance,
                    ) = self.create_entity_for_tendon_end(side="left")

                    tendon = self.replace_placeholder(
                        tendon,
                        new_entity_name,
                        new_entity_instance,
                        cp_instance,
                        "left",
                    )
                    # Generate additional tendons
                    additional_tendons = self.generate_additional_tendons(
                        entity_name=new_entity_name,
                        entity_instance=new_entity_instance,
                    )
                    queue.extend(additional_tendons)
                    check_depth_and_proceed(tendon)

                elif "placeholder" in tendon[-1]:
                    if self.entity_count >= self.N:
                        break
                    (
                        new_entity_name,
                        new_entity_instance,
                        cp_instance,
                    ) = self.create_entity_for_tendon_end(side="right")

                    tendon = self.replace_placeholder(
                        tendon,
                        new_entity_name,
                        new_entity_instance,
                        cp_instance,
                        "right",
                    )
                    additional_tendons = self.generate_additional_tendons(
                        entity_name=new_entity_name,
                        entity_instance=new_entity_instance,
                    )
                    queue.extend(additional_tendons)
                    check_depth_and_proceed(tendon)

                else:
                    # No placeholder, indicating the tendon is completed
                    self.connections.append({"tendon": tendon})

        # After BFS exits, if there are remaining tendons in the queue, close all placeholders
        while queue:
            tendon = queue.popleft()
            tendon = self.close_tendon_placeholders(tendon)
            if len(tendon) > 1:
                self.connections.append({"tendon": tendon})

        # Construct the final scene_yaml
        scene_yaml = {
            "scene": {
                "name": "Generated Scene",
                "tag": self.subtype,
                "entities": list(self.generated_entities.values()),
                "connections": self.connections,
            }
        }

        # Clean up redundant connecting_point_seq_id = 1
        for entity_tendon in scene_yaml["scene"]["connections"]:
            for connect_point_data in entity_tendon["tendon"]:
                if "connecting_point_seq_id" in connect_point_data:
                    if connect_point_data["connecting_point_seq_id"] == 1:
                        del connect_point_data["connecting_point_seq_id"]
                    else:
                        # In the example, keep it; if you want to delete all, handle it yourself
                        pass

        return scene_yaml

    # ==============================
    # 3) Key Function to Generate Entities
    # ==============================

    def get_random_entity_info(self):
        """
        Randomly select an entity and its difficulty information from subtype_config["possible_entities"]
        """
        possible_list = self.subtype_config["possible_entities"]
        if self.test: return possible_list[self.seed]
        return random.choice(possible_list)

    def create_random_entity(self):
        """
        The very beginning place where the entire system generates entities (and may be called later in BFS).
        Takes a random entity information from the subcategory configuration and generates it.
        """
        entity_info = self.get_random_entity_info()
        class_name = entity_info["class_name"]
        difficulty_str = entity_info["difficulty"]

        # Find the actual class object based on class_name
        entity_class = GENERATABLE_ENTITY_CLASSES[class_name]
        entity_name = f"{class_name.lower()}_{self.entity_count}"

        # Position
        entity_pos = (0.0, self.current_y_position, 0.0)
        self.current_y_position += POSITION_INCREMENT

        # Initialize the entity
        entity_instance = self.create_entity(entity_class, entity_name, entity_pos, difficulty_str)
        self.entity_count += 1

        # Generate corresponding YAML and store
        self.create_entity_yaml(
            entity_instance=entity_instance,
            entity_name=entity_name,
            entity_pos=entity_pos,
            difficulty_str=difficulty_str,
        )
        return entity_name, entity_instance

    def create_entity_for_tendon_end(self, side):
        """
        In BFS, for each round targeting a tendon endpoint (left/right), if it is a placeholder, an entity needs to be placed.
        Here, also filter based on subtype_config's possible_entities, but also check if their connecting_point meets the requirements.
        """
        if side == "left":
            required_directions = [
                ConnectingDirection.INNER_TO_OUTER,
                ConnectingDirection.RIGHT_TO_LEFT,
            ]
        else:
            required_directions = [
                ConnectingDirection.OUTER_TO_INNER,
                ConnectingDirection.LEFT_TO_RIGHT,
            ]

        # Find entity types from possible_entities that have at least the required connecting point
        available_classes = []
        for entity_info in self.subtype_config["possible_entities"]:
            class_name = entity_info["class_name"]
            e_class = GENERATABLE_ENTITY_CLASSES[class_name]
            # First create a temporary object to check if it can connect
            temp_obj = self.create_entity(e_class, "temp_check", (0.0, 0.0, 0.0))
            if temp_obj.check_connecting_point_availability(required_directions):
                available_classes.append(entity_info)

        if len(available_classes) == 0:
            # If none are available, fallback to default strategy (e.g., FixedPulleyEntity)
            # You can also modify to other behaviors
            fallback_class_name = "FixedPulleyEntity"
            fallback_info = {
                "class_name": fallback_class_name,
                "difficulty": "NON_STRUCTURAL",
            }
            available_classes = [fallback_info]

        # Randomly select one
        chosen_info = random.choice(available_classes)
        class_name = chosen_info["class_name"]
        difficulty_str = chosen_info["difficulty"]

        # Actually create the object
        entity_class = GENERATABLE_ENTITY_CLASSES[class_name]
        entity_name = f"{class_name.lower()}_{self.entity_count}"
        entity_pos = (0.0, self.current_y_position, 0.0)
        self.current_y_position += POSITION_INCREMENT

        entity_instance = self.create_entity(entity_class, entity_name, entity_pos, difficulty_str)
        self.entity_count += 1
        if class_name == "MassWithMovablePulley":
            pass
        # Get a corresponding connecting_point in the required direction
        connect_point_instance = entity_instance.get_next_connecting_point(
            required_directions
        )
        if class_name == "MassWithMovablePulley":
            pass
        if connect_point_instance is None:
            raise ValueError(
                f"No connecting points available for {class_name} on side={side}."
            )

        self.create_entity_yaml(
            entity_instance=entity_instance,
            entity_name=entity_name,
            entity_pos=entity_pos,
            difficulty_str=difficulty_str,
        )

        return entity_name, entity_instance, connect_point_instance

    def create_entity_yaml(
            self,
            entity_instance,
            entity_name,
            entity_pos,
            difficulty_str="NON_STRUCTURAL",
            use_random_parameters=False,
    ):
        """
        Here, map difficulty_str to the corresponding DegreeOfRandomization, and finally pass it to generate_entity_yaml
        """
        degree_of_randomization = map_difficulty_to_degree(difficulty_str)
        entity_yaml = entity_instance.generate_entity_yaml(
            use_random_parameters=use_random_parameters,
            degree_of_randomization=degree_of_randomization,
        )

        # Convert tuples to lists to ensure serializability
        def convert_tuples_to_lists(obj):
            if isinstance(obj, tuple):
                return list(obj)
            elif isinstance(obj, list):
                return [convert_tuples_to_lists(item) for item in obj]
            elif isinstance(obj, dict):
                return {
                    key: convert_tuples_to_lists(value) for key, value in obj.items()
                }
            return obj

        entity_yaml["position"] = list(entity_pos)

        # If parameters contain Enum, convert to their value
        if "parameters" in entity_yaml:
            for param_key, param_value in list(entity_yaml["parameters"].items()):
                if isinstance(param_value, str):
                    continue
                if hasattr(param_value, "value"):
                    entity_yaml["parameters"][param_key] = param_value.value
            entity_yaml["parameters"] = convert_tuples_to_lists(
                entity_yaml["parameters"]
            )

        if entity_name in self.generated_entities:
            raise ValueError(f"Entity with name {entity_name} already exists.")
        self.generated_entities[entity_name] = entity_yaml
        self.entity_positions[entity_name] = entity_pos

    def create_entity(self, entity_class, entity_name, entity_pos, difficulty_str="DEFAULT") -> Entity:
        degree_of_randomization = map_difficulty_to_degree(difficulty_str)
        return entity_class(
            name=entity_name, pos=entity_pos, init_randomization_degree=degree_of_randomization
        )

    # ==============================
    # 4) Helper Functions to Generate Tendons in BFS Process
    # ==============================
    def generate_tendons_for_entity(
            self,
            entity_name,
            entity_instance,
            required_directions=None,
    ):
        """
        Similar to original, generate new tendons based on the entity's connecting points.
        """
        tendons = []
        available_points_num = entity_instance.get_available_connecting_points_num(
            required_directions
        )
        if available_points_num == 0:
            return tendons

        # Randomly decide how many potential tendons to generate based on available points
        possible_values = list(range(1, available_points_num + 1))
        weights = [PROBABILITY_WEIGHTS if i == 1 else 1 for i in possible_values]
        num_tendons = random.choices(possible_values, weights=weights, k=1)[0]
        # num_tendons = available_points_num
        # print(f"available_points_num: {available_points_num}, num_tendons: {num_tendons}")

        for _ in range(num_tendons):
            cp_instance = entity_instance.get_next_connecting_point(required_directions)
            if cp_instance is None:
                raise ValueError(f"cp_instance is None")

            entity_tendon = {
                "entity": entity_name,
                "direction": cp_instance.direction.value,
                "connecting_point": cp_instance.connecting_point.value,
                "connecting_point_seq_id": cp_instance.connecting_point_seq_id,
            }

            d = cp_instance.direction
            if d in [
                ConnectingDirection.LEFT_TO_RIGHT,
                ConnectingDirection.RIGHT_TO_LEFT,
            ]:
                tendon = [
                    {"placeholder": True},
                    entity_tendon,
                    {"placeholder": True},
                ]
            elif d == ConnectingDirection.INNER_TO_OUTER:
                tendon = [
                    entity_tendon,
                    {"placeholder": True},
                ]
            elif d == ConnectingDirection.OUTER_TO_INNER:
                tendon = [
                    {"placeholder": True},
                    entity_tendon,
                ]
            else:
                raise ValueError(f"Unknown cp_instance.direction: {cp_instance.direction}")

            tendons.append(tendon)

        return tendons

    def generate_additional_tendons(self, entity_name, entity_instance):
        """
        After each tendon endpoint replacement in BFS, can generate some additional tendons
        """
        tendons = self.generate_tendons_for_entity(entity_name, entity_instance)
        possible_values = list(range(len(tendons) + 1))
        weights = [PROBABILITY_WEIGHTS if i == 0 else 1 for i in possible_values]
        num_additional = random.choices(possible_values, weights=weights, k=1)[0]
        num_additional = len(tendons)

        return tendons[:num_additional]

    def replace_placeholder(
            self, tendon, entity_name, entity_instance, cp_instance, side
    ):
        replace_tendon = {
            "entity": entity_name,
            "direction": cp_instance.direction.value,
            "connecting_point": cp_instance.connecting_point.value,
        }
        if side == "left":
            tendon[0] = replace_tendon
            if cp_instance.direction == ConnectingDirection.RIGHT_TO_LEFT:
                tendon = [{"placeholder": True}] + tendon
        else:
            tendon[-1] = replace_tendon
            if cp_instance.direction == ConnectingDirection.LEFT_TO_RIGHT:
                tendon = tendon + [{"placeholder": True}]
        return tendon

    def get_tendon_depth(self, tendon):
        return sum(1 for d in tendon if "entity" in d)

    def close_tendon_placeholders(self, tendon):
        """
        When BFS reaches the end or exceeds depth, placeholders need to be closed using 'closing_entities'.
        For example, "Number of Entities: 1–2 + MassWithFixedPulley" indicates that a MassWithFixedPulley needs to be added at the end.
        If there are multiple types in closing_entities, randomly select one.
        """
        closing_list = self.subtype_config.get("closing_entities", [])

        if "placeholder" in tendon[0]:
            result = self.create_closing_entity(side="left", closing_list=closing_list)
            if result[0] is not None:
                entity_name, _, direction, connecting_point = result
                tendon[0] = {
                    "entity": entity_name,
                    "direction": direction.value,
                    "connecting_point": connecting_point.value,
                }
            else:
                del tendon[0]

        if "placeholder" in tendon[-1]:
            result = self.create_closing_entity(side="right", closing_list=closing_list)
            if result[0] is not None:
                entity_name, _, direction, connecting_point = result
                tendon[-1] = {
                    "entity": entity_name,
                    "direction": direction.value,
                    "connecting_point": connecting_point.value,
                }
            else:
                del tendon[-1]
        return tendon

    def create_closing_entity(self, side: str, closing_list: List[str]):
        """
        When closing, select an entity based on subtype_config['closing_entities'].
        If empty, fallback to the original random choice among [MassWithFixedPulley, FixedPulleyEntity, ConstantForceFixedPulley]
        """
        if not closing_list:
            return None, None, None, None

        chosen_entity_type = random.choice(closing_list)
        # Here you can map based on previous possible_entities difficulty,
        # or simply assign NON_STRUCTURAL or EASY
        difficulty_str = "EASY"  # Or "NON_STRUCTURAL", etc.

        entity_class = GENERATABLE_ENTITY_CLASSES[chosen_entity_type]
        entity_name = f"{chosen_entity_type.lower()}_{self.entity_count}"
        entity_pos = (0.0, self.current_y_position, 0.0)
        self.current_y_position += POSITION_INCREMENT

        entity_instance = self.create_entity(entity_class, entity_name, entity_pos)
        self.entity_count += 1

        # Closing entities generally do not need highly random parameters; set to False if needed
        self.create_entity_yaml(
            entity_instance,
            entity_name,
            entity_pos,
            difficulty_str=difficulty_str,
            use_random_parameters=False,
        )
        direction = (
            ConnectingDirection.INNER_TO_OUTER
            if side == "left"
            else ConnectingDirection.OUTER_TO_INNER
        )
        connecting_point = ConnectingPoint.DEFAULT
        return entity_name, entity_class, direction, connecting_point


# ==============================
# 5) Generate Multiple Scenes as Examples
# ==============================
def generate_X_scenes(subtype: str, X: int = 10, cfg: DictConfig = None, offset = 0):
    """
    Simple example: specify a subtype, such as "BasicPulley",
    continuously generate X scenes and save to dsl_output/scene_{i}/scene_output.yaml
    """
    seed = datetime.datetime.now().microsecond

    for i in range(X):
        if cfg is None:
            folder_name = f"./batch_generation_output/{subtype}_{i + offset}"
        else:
            folder_name = f"{cfg.root_dir}/{subtype}/scene_{i + offset}"
        os.makedirs(folder_name, exist_ok=True)

        # Initialize a SceneGenerator
        scene_generator = SceneGenerator(subtype=subtype, seed=seed)
        seed += 1

        # BFS to generate the scene
        scene_yaml = scene_generator.generate_scene_yaml()

        # Save
        yaml_path = os.path.join(folder_name, "scene_output.yaml")
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(scene_yaml, f, sort_keys=False)
        xml_path = os.path.join(folder_name, "scene_output.xml")
        xml = parse_scene(yaml_path).to_xml()
        with open(xml_path, "w", encoding="utf-8") as f:
            f.write(xml)

        # Also save the current seed used
        seed_txt = os.path.join(folder_name, "seed.txt")
        with open(seed_txt, "w", encoding="utf-8") as f:
            f.write(str(seed))

def generate_test_scenes(cfg: DictConfig = None):
    """
    Generate test scenes with single entity per scene.
    """
    
    test_cfg = SCENE_CONFIGS["Test"]["possible_entities"]

    for seed in range(len(test_cfg)):
        folder_name = f"sim/DSLs/test/{test_cfg[seed]['class_name']}"
        
        os.makedirs(folder_name, exist_ok=True)

        # Initialize a SceneGenerator
        scene_generator = SceneGenerator(subtype="Test", seed=seed, test=True)

        # BFS to generate the scene
        scene_yaml = scene_generator.generate_scene_yaml()

        # Save
        yaml_path = os.path.join(folder_name, "scene_output.yaml")
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(scene_yaml, f, sort_keys=False)

        # Also save the current seed used
        seed_txt = os.path.join(folder_name, "seed.txt")
        with open(seed_txt, "w", encoding="utf-8") as f:
            f.write(str(seed))

        print("Generated test scene: ", folder_name)

def generate_scenes_by_config(config_dict: Dict[str, int], cfg: DictConfig = None):
    """
    Generate multiple scenes based on the provided configuration dictionary.

    Parameters:
    - config_dict (Dict[str, int]): Keys are subcategory names in SCENE_CONFIGS,
      values are the number of scenes to generate for that subcategory. For example:
      {
          "BasicPulley": 10,
          "IntermediateCollision": 5,
          "AdvancedInclinedPlaneFriction": 3
      }
    """
    if cfg and os.path.exists(cfg.root_dir) and not cfg.scene_generation.exist_ok:
        print(f"WARNING: Root directory {cfg.root_dir} already exists.")
        assert False, f"WARNING: Root directory {cfg.root_dir} already exists."

    for subtype, count in config_dict.items():
        
        _count = count
        if os.path.exists(os.path.join(cfg.root_dir, subtype)):
            # How many folders in this directory
            num_folders = len([name for name in os.listdir(os.path.join(cfg.root_dir, subtype)) if os.path.isdir(os.path.join(cfg.root_dir, subtype, name))])
            _count = count - num_folders
            
            if _count > 0 and _count < count:
                print(f"Skipping {num_folders} generations for {subtype} as they already exist.")
            elif _count <= 0:
                print("Skipping existing scene type: ", subtype)
                continue

        print(f"Generating {_count} '{subtype}' scenes...")
        generate_X_scenes(subtype=subtype, X=_count, cfg=cfg, offset=count - _count)
        print(
            f"Successfully generated {_count} '{subtype}' scenes. Saved to {cfg.root_dir if cfg else './batch_generation_output'}/{subtype}")
        # try:
        #     generate_X_scenes(subtype=subtype, X=count, cfg=cfg)
        #     print(f"Successfully generated {count} '{subtype}' scenes. Saved to {cfg.root_dir if cfg else './batch_generation_output'}/{subtype}")
        # except ValueError as e:
        #     print(f"Error: {e}")
        # except Exception as e:
        #     print(
        #         f"An unexpected error occurred while generating '{subtype}' scenes: {e}"
        #     )


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    # Get scene generation config from hydra config
    scene_types = cfg.scene_generation.scene_types
    num_scenes = cfg.scene_generation.num_scenes

    # Create config dict mapping scene types to number of scenes
    scene_generation_config = {
        k: num_scenes for k in SCENE_CONFIGS if k in scene_types
    }
    # Generate scenes based on config
    generate_scenes_by_config(scene_generation_config, cfg)


def manual_test():
    scene_generation_config = {
        "RollingPlane": 1,
        # Add more subcategories and counts
    }
    # Call the function to generate scenes
    generate_scenes_by_config(scene_generation_config)

def test_gen():
    generate_test_scenes()

if __name__ == "__main__":
    # manual_test()
    main()