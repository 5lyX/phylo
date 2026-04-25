import copy
from typing import get_type_hints, get_origin, get_args, Any
import mujoco
from sim.entities import *
from sim.bodies import *
from sim.xml_body_unpacker import XMLBodyUnpacker
from sim.logger_manager import LoggerManager, LoggerType
import json
import yaml
import argparse
from collections import defaultdict

from sim.utils import replace_all, create_mappings, parse_mtl_to_mujoco, find_values

st = ipdb.set_trace

import re
import os

SCENE_TYPE_TO_CATEGORY_MAP = {
    "pulley": ["BasicPulley", "IntermediatePulley", "BasicInclinedPlaneFriction", "IntermediateInclinedPlaneFriction", "AdvancedInclinedPlaneFriction", "IntermediateHybrid", "AdvancedHybrid", "DifficultPulley"],
    "collision": ["BasicCollision", "IntermediateCollision", "AdvancedCollision"],
    "spring": ["SpringBlockSystems", "DifficultSpringMass"],
    "rotation": ["Rotation", "RigidBodyRotation"],
    "orbital": ["OrbitalMotion", "DifficultOrbitalMotion"],
    "em": ["DifficultElectroMagnetic"],
}

def simplify_names(text: str) -> str:
    string_map, pulley_map, mass_map = create_mappings(text)
    
    # Replace in order (longest matches first to avoid partial replacements)
    for old, new in sorted(string_map.items(), key=lambda x: len(x[0]), reverse=True):
        text = text.replace(old, new)
    
    for old, new in sorted(pulley_map.items(), key=lambda x: len(x[0]), reverse=True):
        text = text.replace(old, new)
        
    for old, new in sorted(mass_map.items(), key=lambda x: len(x[0]), reverse=True):
        text = text.replace(old, new)
        
    return text

# # Example usage
# simplified = simplify_names(text)
# print(simplified)

logger = LoggerManager()

ENTITY_CLASSES = {
    "MassWithFixedPulley": MassWithFixedPulley,
    "MassWithMovablePulley": MassWithMovablePulley,
    "DirectedMass": DirectedMass,
    "TwoSideMassPlane": TwoSideMassPlane,
    "ComplexMovablePulley": ComplexMovablePulley,
    "FixedPulleyEntity": FixedPulleyEntity,
    "PulleyGroupEntity": PulleyGroupEntity,
    "MassPrismPlaneEntity": MassPrismPlaneEntity,
    # TODO: can we do "MassPrismPlane" instead of ""MassPrismPlaneEntity"?
    "StackedMassPlane": StackedMassPlane,
    "MassWithReversedMovablePulley": MassWithReversedMovablePulley,
    "ConstantForceFixedPulley": ConstantForceFixedPulley,
    "TwoDCollisionPlane": TwoDCollisionPlane,
    "SliderWithArchPlaneSpheres": SliderWithArchPlaneSpheres,
    "ComplexCollisionPlane": ComplexCollisionPlane,
    "SpringBlockEntity": SpringBlockEntity,
    "RigidRotationEntity": RigidRotationEntity,
    "BarPlaneSupport": BarPlaneSupport,
    "PendulumEntity": PendulumEntity,
    "DiskRackWithSphereEntity": DiskRackWithSphereEntity,
    "OrbitalMotionEntity": OrbitalMotionEntity,
    "GeneralCelestialEntity": GeneralCelestialEntity,
    "RollingPlaneEntity": RollingPlaneEntity,
    "ThrowingMotionEntity": ThrowingMotionEntity,
    "MassBoxPlaneEntity": MassBoxPlaneEntity,
    "MassPrismPulleyPlane": MassPrismPulleyPlane,
    "ConstantVelocityPuller": ConstantVelocityPuller,
    "SpringMassPlaneEntity": SpringMassPlaneEntity,
    "SolarSystemEntity": SolarSystemEntity,
    "RocketEntity": RocketEntity,
    "ThrowingMotionEntity": ThrowingMotionEntity,
    "MagneticElectricEntity": MagneticElectricEntity,
    "ElectroMagneticEntity": ElectroMagneticEntity,
}

ENTITY_SIMPLIFIED_NAMES = {
    "MassWithFixedPulley": "Fixed",
    "MassWithMovablePulley": "Movable",
    "DirectedMass": "Directed",
    "TwoSideMassPlane": "Plane",
    "ComplexMovablePulley": "Complex",
    "FixedPulleyEntity": "Fix",
    "PulleyGroupEntity": "Group",
    "MassPrismPlaneEntity": "Prism",
    "StackedMassPlane": "Stacked",
    "MassWithReversedMovablePulley": "Reversed",
    "ConstantForceFixedPulley": "Force",
    "MassPrismPlane": "Prism",
    "MassPlane": "Plane",
    "Mass": "Mass",
    "TwoDCollisionPlane": "Collision2D",
    "SliderWithArchPlaneSpheres": "Slider",
    "ComplexCollisionPlane": "Collision1D",
    "SpringBlockEntity": "SpringBlock",
    "RigidRotationEntity": "Rotation",
    "BarPlaneSupport": "BarSupport",
    "PendulumEntity": "Pendulum",
    "DiskRackWithSphereEntity": "DiskRackSphere",
    "OrbitalMotionEntity": "OrbitalMotion",
    "GeneralCelestialEntity": "GeneralCelestial",
    "RollingPlaneEntity": "RollingPlane",
    "ThrowingMotionEntity": "ThrowingMotion",
    "MassBoxPlaneEntity": "MassBoxPlane",
    "MassPrismPulleyPlane": "MassPrismPulleyPlane",
    "ConstantVelocityPuller": "ConstantVelocityPuller",
    "SpringMassPlaneEntity": "SpringMass",
    "SolarSystemEntity": "SolarSystem",
    "RocketEntity": "Rocket",
    "ThrowingMotionEntity": "Projectile",
    "MagneticElectricEntity": "MagneticElectric",
    "ElectroMagneticEntity": "ElectroMagnetic",
}


def generate_final_xml(
    entities=[],
    tendons=[],
    springs=[],
    gravity=-9.81,
    use_equality=False,
    sensors=[],
    actuators=[],
    custom_sensors=[],
    tag = None,
):
    """
    Generate the XML string for the entire scene, including bodies and tendons.
    """

    # Add all bodies to the scene
    worldbody_xml = ""
    for body in entities:
        worldbody_xml += body.to_xml() + "\n"

    mesh_names = find_values(worldbody_xml, value="mesh")
    mat_names = find_values(worldbody_xml, value="material")

    import_names = set(mesh_names + mat_names)
    
    # Add all tendons to the scene
    tendon_xml = ""
    for tendon in tendons:
        tendon_xml += tendon.to_xml() + "\n"
    if use_equality:
        for tendon in tendons:
            tendon_xml += tendon.generate_equality().to_xml() + "\n"

    # Add all springs to the scene
    spring_xml = ""
    for spring in springs:
        tendon = Tendon(name="tendon_" + spring.name, spring=True)
        tendon.add_spatial(spring)
        spring_xml += tendon.to_xml() + "\n"

    # Add all sensor to the scene (for mass)
    sensor_xml = ""
    # TODO: Currently Sensor is not used, sensor shall raise issues and should be checked if time available
    # for sensor in sensors:
    #     sensor_xml += sensor.to_xml() + "\n"

    # Add all custom sensor to the scene (for tendon)
    custom_xml = ""
    # for sensor in custom_sensors:
    #     custom_xml += sensor.to_xml() + "\n"
    contact_xml = ""

    resolution_coefficient_list = []
    for entity in entities:
        resolution_coefficient_list.extend(entity.get_resolution_coefficients())

    friction_coefficient_list = []
    for entity in entities:
        L = entity.get_friction_coefficients()
        friction_coefficient_list.extend([(*t, entity.friction_type.value) for t in L])

    if len(resolution_coefficient_list):
        resolution_coefficients_content = "-".join(map(str, resolution_coefficient_list)).replace("'", "")
        custom_xml += f'<text name="coefficient_restitution" data="{resolution_coefficients_content}" />\n'
        for coeff in resolution_coefficient_list:
            b1, b2, c = coeff
            b1 += ".geom"
            b2 += ".geom"

            contact_xml += f"""<pair geom1="{b1}" geom2="{b2}" solref="-2500 0"/>\n"""
    if len(friction_coefficient_list):
        friction_coefficient_list = [(g1, g2, c, c, r[0]) for g1, g2, c, *r in friction_coefficient_list if c > 1e-2]
        friction_coefficients_content = "-".join(map(str, friction_coefficient_list)).replace("'", "")
        custom_xml += f'<text name="coefficient_friction" data="{friction_coefficients_content}" />\n'
        
        # for coeff in friction_coefficient_list:
        #     b1, b2, c = coeff
        #     b1 += ".geom"
        #     b2 += ".geom"
        #     contact_xml += f"""<pair geom1="{b1}" geom2="{b2}" solref="-2500 0"/>\n"""

    # Add all actuators to the scene
    actuator_xml = ""
    for actuator in actuators:
        actuator_xml += actuator.to_xml() + "\n"

    # Directory containing the mesh files
    mesh_directory = os.path.join(os.path.dirname(__file__), "geom_sources")

    # Generate the mesh XML dynamically
    use_plugin = "sdf" in worldbody_xml
    plugin_content = '\n<plugin instance="sdf"/>' if use_plugin else ""
    
    mesh_entries = []
    for root, _, files in os.walk(mesh_directory):
        for file_name in files:
            if ".".join(file_name.split(".")[:-1]) not in import_names: continue
            if file_name.endswith(".stl") or file_name.endswith(".obj"):
                rel_path = os.path.relpath(os.path.join(root, file_name), mesh_directory)
                scale = """scale="0.001 0.001 0.001" """ if file_name.endswith(".obj") else ""
                mesh_entries.append(
                    f'<mesh name="{os.path.splitext(file_name)[0]}" {scale}file="{os.path.join(mesh_directory, rel_path)}"> {plugin_content}\n</mesh>'
                )
            elif file_name.endswith(".mtl"):
                rel_path = os.path.relpath(os.path.join(root, file_name), mesh_directory)
                
                mat_entry = parse_mtl_to_mujoco(os.path.join(mesh_directory, rel_path))
                if len(mat_entry) > 1: continue
                mat_entry = mat_entry[0]
                mat_entry = re.sub('name="[^"]*"', f'name="{os.path.splitext(file_name)[0]}"', mat_entry)
                mesh_entries.append(mat_entry)
                
    mesh_xml = "\n".join(mesh_entries)
    skybox_xml = ""
    # skybox_xml = f"""<texture name="skybox_tex" type="skybox" builtin="none"
    #        fileright="{mesh_directory}/fixed_sources/rocket/rocket.white-texture-background_sqr.png"  
    #        fileup="{mesh_directory}/fixed_sources/rocket/rocket.white-texture-background_sqr.png"  
    #        fileleft="{mesh_directory}/fixed_sources/rocket/rocket.white-texture-background_sqr.png"  
    #        filedown="{mesh_directory}/fixed_sources/rocket/rocket.white-texture-background_sqr.png"  
    #        filefront="{mesh_directory}/fixed_sources/rocket/rocket.white-texture-background_sqr.png"  
    #        fileback="{mesh_directory}/fixed_sources/rocket/rocket.white-texture-background_sqr.png"/>
    # """

    if tag is None: category = "pulley"
    else: 
        category = [k for k, v in SCENE_TYPE_TO_CATEGORY_MAP.items() if tag in v]
        if len(category) == 1: category = category[0]
        else: category = "pulley"

    default_xml = """<default class="main">
            <geom friction="0 0 0" solimp="0.9 1 0.001 0.01 20" solref="0.02 1" condim="1"/>
        </default>
        """
    
    plugin_xml = ""
    if use_plugin:
        plugin_xml = """<extension>
            <plugin plugin="mujoco.sdf.sdflib">
                <instance name="sdf">
                    <config key="aabb" value="0" />
                </instance>
            </plugin>
        </extension>
        <option sdf_iterations="20" sdf_initpoints="40" />"""

    rgb1, rgb2 = "0 0.5 0.7", "0 0 0"
    # rgb1, rgb2 = "1 1 1", "0.75 0.75 0.75"

    xml_structure = f"""
        <mujoco>
            <compiler angle="radian" autolimits="true"/>
            <compiler meshdir="asset" texturedir="asset" assetdir="asset"/>
            <option gravity="0 0 {gravity}"/>
            {default_xml}
        <visual>
            <headlight diffuse="0.7 0.7 0.7" ambient="0.5 0.5 0.5"/>
            <global offwidth="1920" offheight="1080"/>
        </visual>
        {plugin_xml}
        <asset>
            <texture type="skybox" builtin="gradient" rgb1="{rgb1}" rgb2="{rgb2}" width="512" height="512"/>
            <texture type="2d" name="checks1" builtin="checker" rgb1="1 1 1" rgb2="0 0 0" width="256" height="256"/>
            <material name="test_color" texture="checks1" texuniform="true" texrepeat="2 2"/>
            <material name="reflectance" reflectance=".4"/>
            <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1="1.0 0.6235 0.1098" rgb2="1.0 0.749 0.4118"/>
            <texture name="white-texture-background" type="2d" file="{mesh_directory}/fixed_sources/rocket/rocket.white-texture-background.png" content_type="image/png"/>
            <material name="pulley" texture="grid" texrepeat="2 2" texuniform="true"/>
            {mesh_xml}
            {skybox_xml}
        </asset>
        <worldbody>
            {worldbody_xml}
        </worldbody>
        {tendon_xml}
        {spring_xml}
        <sensor>
            {sensor_xml}
        </sensor>
        <custom>
            {custom_xml}
        </custom>
        <contact>
            {contact_xml}
        </contact>
        <actuator>
            {actuator_xml}
        </actuator>
    """
    xml_structure += "</mujoco>"
    processor = XMLBodyUnpacker()
    tree = processor.load_xml_from_str(xml_structure)
    # print(xml_structure)
    processor.parse_xml(tree, update_tendon_lengths=(not use_equality))
    # self.name_mapping = processor.simplify_names(tree)
    xml_structure = processor.save_xml_to_str(tree)
    return xml_structure


class Scene:
    """
    A class to represent a collection of bodies and generate a Mujoco XML scene.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.entities = []
        self.tendons = []
        self.sensors = []
        self.custom_sensors = []
        self.name_mapping = {}
        self.connections = (
            []
        )  # Now it will store a list of tendons, each tendon is a list of connections
        # logger.debug(f"Created scene with name: {name}")
        self.gravity = -9.81
        self.use_equality = False
        self.constant_force_dict = {}
        self.init_velocity_dict = {}
        self.actuators = []
        self.springs = []

    def add_entity(self, entity: Entity) -> None:
        """
        Add a body to the scene.
        """
        self.entities.append(entity)

    def add_connection(
        self,
        tendon_connections: List[
            Tuple[str, ConnectingDirection, ConnectingPoint, ConnectingPointSeqId]
        ],
    ) -> None:
        """
        Add a tendon connection to the scene.
        Each tendon connection is a list of tuples (entity_name, direction, connecting_point)
        """
        self.connections.append(tendon_connections)

    def add_tendon(self, tendon: Tendon) -> None:
        """
        Add a tendon to the scene.
        """
        self.tendons.append(tendon)

    def connect_entities(self):
        # Process each tendon separately
        for tendon_idx, tendon_connections in enumerate(self.connections):
            # Collect entities and their connection info for this tendon
            entity_list = []
            direction_list = []
            connecting_point_list = []
            connecting_point_seq_id_list = []
            has_force_actuator = False
            for connection in tendon_connections:
                entity_name, direction, connecting_point, connecting_point_seq_id = (
                    connection
                )
                # Find the entity with the given name
                entity = next((e for e in self.entities if e.name == entity_name), None)
                if entity:
                    entity_list.append(entity)
                    direction_list.append(direction)
                    connecting_point_list.append(connecting_point)
                    connecting_point_seq_id_list.append(connecting_point_seq_id)
                    if entity.__class__.__name__ == "ConstantForceFixedPulley":
                        has_force_actuator = True
                        entity.set_actuator_tendon_spatial(f"tendon_force_{tendon_idx}")
                else:
                    raise ValueError(f"Entity '{entity_name}' not found in the scene.")

            # Connect the entities for this tendon
            full_sequence = []
            for idx, entity in enumerate(entity_list):
                dir = direction_list[idx]
                connecting_point = connecting_point_list[idx]
                seq_id = connecting_point_seq_id_list[idx]
                sequence = entity.get_connecting_tendon_sequence(
                    dir, connecting_point, seq_id
                )
                full_sequence.extend(sequence.get_elements())

            # Create tendon and spatial for the full sequence
            tendon_name = f"tendon_{tendon_idx}"
            if has_force_actuator:
                spatial_name = f"tendon_force_{tendon_idx}"
            else:
                spatial_name = f"spatial_{tendon_idx}"

            tendon = Tendon(name=tendon_name)
            spatial = Spatial(name=spatial_name)

            for element in full_sequence:
                spatial.add_element(element)

            tendon.add_spatial(spatial)
            self.tendons.append(tendon)

    def add_ready_tendons(self):
        # dispose all ready tendons
        for entity in self.entities:
            if hasattr(entity, "get_ready_tendon_sequences"):
                ready_sequences: List[TendonSequence] = entity.get_ready_tendon_sequences("left_to_right")
                for idx, sequence in enumerate(ready_sequences):
                    tendon_name = f"tendon.{entity.name}.{sequence.name}-{idx}"
                    spatial_name = f"{entity.name}.{sequence.name}-{idx}"
                    tendon = Tendon(name=tendon_name)
                    spatial = Spatial(name=spatial_name)
                    for element in sequence.get_elements():
                        spatial.add_element(element)
                    # print("Tendon Description:")
                    tendon_description = sequence.get_description()
                    # print(tendon_description)

                    tendon.add_spatial(spatial)
                    self.tendons.append(tendon)

    def add_sensors(self):
        """
        Traverse through all entities and collect their sensors, appending them to the scene's sensor list.
        """
        for entity in self.entities:
            self.sensors.extend(entity.get_sensor_list())
        # if not self.use_equality:
        for tendon in self.tendons:
            self.custom_sensors.extend(tendon.get_custom_sensor_list())
        # else:
        #     for tendon in self.tendons:
        #         self.custom_sensors.extend(tendon.get_custom_sensor_list())

    def add_constant_forces(self):
        for entity in self.entities:
            self.constant_force_dict.update(entity.get_constant_forces())

    def add_actuators(self):
        for entity in self.entities:
            actuator = entity.get_actuator()
            if actuator:
                self.actuators.append(actuator)

    def add_springs(self):
        for entity in self.entities:
            self.springs.extend(entity.get_springs())

    def add_init_velocities(self):
        for entity in self.entities:
            self.init_velocity_dict.update(entity.get_init_velocities())

    def set_gravity_force(self):
        for entity in self.entities:
            if entity.__class__.__name__ in ["OrbitalMotionEntity", "GeneralCelestialEntity", "SolarSystemEntity", "RocketEntity", "MagneticElectricEntity", "ElectroMagneticEntity"]:
                self.gravity = 0
                break

    def get_rockets(self):
        rockets = {}

        for entity in self.entities:
            if entity.__class__.__name__ == "RocketEntity":
                rockets[entity.get_rocket().name] = {
                    "v_exhaust": entity.v_exhaust,
                    "dm_dt": entity.dm_dt,
                    "min_mass": entity.min_mass
                }
        
        return rockets

    def get_trail_bodies(self):
        """
        Gather all trail bodies from all entities in the scene.
        """
        trail_bodies = []
        for entity in self.entities:
            if hasattr(entity, "trail_bodies"):
                trail_bodies.extend(entity.trail_bodies)
        
        return trail_bodies

    def get_constant_force_dict(self) -> Dict:
        return self.constant_force_dict

    def get_init_velocity_dict(self) -> Dict:
        return self.init_velocity_dict

    def get_attraction_forces(self) -> List[Tuple[str, str, str, float]]:
        """
        Gather all attraction force definitions from all entities in the scene.
        """ 
        all_forces = []
        for entity in self.entities:
            all_forces.extend(entity.get_attraction_forces())
        return all_forces
    
    def get_center(self)-> np.array:
        """
        Calculate the center of all entities in the scene.
        """
        center_of_scene = np.zeros(3)

        for entity in self.entities:
            entity_center = entity.pos
            center_of_scene += entity_center

        center_of_scene = center_of_scene / len(self.entities) 

        return center_of_scene
    
    def get_charged_particles(self) -> List[Tuple[str, float]]:
        """
        Gather all charged particles from all entities in the scene.
        """
        all_charged_particles = []
        for entity in self.entities:
            if isinstance(entity, MagneticElectricEntity) or isinstance(entity, ElectroMagneticEntity):
                all_charged_particles.append(
                    (entity.name + ".particle", entity.q)
                )

        return all_charged_particles

    def get_EM_fields(self, position: np.array):
        """
        Gather all EM fields from all entities in the scene.
        """
        all_EM_fields = []
        for entity in self.entities:
            if isinstance(entity, MagneticElectricEntity) or isinstance(entity, ElectroMagneticEntity):
                raw_out = entity.get_fields(position)
                E_net = sum([np.array(e["field_strength"]) for e in raw_out if e["field_type"] == "electric"], 0)
                M_net = sum([np.array(e["field_strength"]) for e in raw_out if e["field_type"] == "magnetic"], 0)
                all_EM_fields.append(
                    (E_net, M_net)    
                )

        return all_EM_fields
    
    def get_EM_configs(self) -> List[dict]:
        """
        Gather all EM configurations from all entities in the scene.
        """
        all_EM_configs = []
        for entity in self.entities:
            if isinstance(entity, MagneticElectricEntity) or isinstance(entity, ElectroMagneticEntity):
                all_EM_configs.extend(entity.field_configs)

        return all_EM_configs

    def set_attributes_from_entities(self):
        # Connect all entities after all entity connections are added
        self.connect_entities()

        # Add ready tendons for all entities
        self.add_ready_tendons()

        # Add sensors for all entities
        self.add_sensors()

        # Add constant forces for all entities
        self.add_constant_forces()

        # Add springs for all entities
        self.add_springs()

        self.add_actuators()

        # Add initial velocities for all entities
        self.add_init_velocities()

        self.set_gravity_force()

    def to_xml(self) -> str:
        """
        Generate the XML string for the entire scene, including bodies and tendons.
        """
        return generate_final_xml(
            entities=self.entities,
            tendons=self.tendons,
            springs=self.springs,
            gravity=self.gravity,
            use_equality=self.use_equality,
            sensors=self.sensors,
            actuators=self.actuators,
            custom_sensors=self.custom_sensors,
            tag=self.tag,
        )

    def randomize_entities(self, vary_idx=None):
        """
        Randomize the parameters of all entities in the scene and reinitialize them.
        After randomization, regenerate connections and tendons.
        """
        # Randomize gravity value
        self.gravity = -9.81 + random.uniform(-2, 2)
        # st()
        # Randomize each entity
        for entity in self.entities:
            entity.randomize_parameters(
                degree_of_randomization=DegreeOfRandomization.NON_STRUCTURAL,
                reinitialize_instance=True,
            )

        # Clear existing tendons, but should not be needed if there is no structural change
        self.tendons = []

        # Reconnect entities based on existing connections
        self.connect_entities()

        # Add ready tendons for all entities
        self.add_ready_tendons()

        # Add sensors for all entities
        self.sensors = []  # Clear existing sensors before readding ones
        self.custom_sensors = []  # Clear custom sensors
        self.add_sensors()

    @staticmethod
    def replace_substrings_recursive(obj: Any, name_mapping: dict) -> Any:
        if isinstance(obj, dict):
            new_dict = {}
            for key, value in obj.items():
                new_key = (
                    Scene.replace_substrings_recursive(key, name_mapping)
                    if isinstance(key, str)
                    else key
                )
                new_value = Scene.replace_substrings_recursive(value, name_mapping)
                new_dict[new_key] = new_value
            return new_dict
        elif isinstance(obj, list):
            return [
                Scene.replace_substrings_recursive(item, name_mapping)
                for item in obj
            ]
        elif isinstance(obj, str):
            for map_key, map_value in name_mapping.items():
                if map_key in obj:
                    obj = obj.replace(map_key, map_value)
            return obj
        else:
            return obj
        
    def get_parameters(self) -> List[dict]:
        """
        Get the parameters of all entities in the scene.
        """
        # if not self.name_mapping:
        #     self.to_xml()
        parameters = []
        entity_params = []
        tendon_params = []

        parameters.append({"gravity": self.gravity})
        for entity in self.entities:
            entity_params.extend(entity.get_parameters())
        for tendon in self.tendons:
            tendon_params.extend(tendon.get_parameters())

        # Replace substrings in entity_params and tendon_params
        entity_params = [
            Scene.replace_substrings_recursive(param, self.name_mapping) 
            for param in entity_params
        ]

        tendon_params = [
            Scene.replace_substrings_recursive(param, self.name_mapping) 
            for param in tendon_params
        ]

        return parameters, entity_params, tendon_params
    
    def update_tendon_info(self) -> dict:
        """
        Get connection information of tendons in the scene.
        {
            body_name1: [tendon_name1, tendon_name2],
            body_name2: [tendon_name3, tendon_name4]
        }
        """
        xml = self.to_xml() 
        try: model = mujoco.MjModel.from_xml_string(xml)
        except: 
            print(xml)
            raise NotImplementedError("XML is not valid")
        data = mujoco.MjData(model)

        mjWRAP_CONSTANTS = {
            0: "mjWRAP_NONE",                   # null object
            1: "mjWRAP_JOINT",                   # constant moment arm
            2: "mjWRAP_PULLEY",                  # pulley used to split tendon
            3: "mjWRAP_SITE",                    # pass through site
            4: "mjWRAP_SPHERE",                  # wrap around sphere
            5: "mjWRAP_CYLINDER"                 
        }
        
        def get_info_from_wrap(wrap_id, wrap_type):
            if wrap_type == "mjWRAP_SITE":
                site_name = model.names[model.name_siteadr[wrap_id]:].split(b'\x00', 1)[0].decode("utf-8")
                body_id = model.site_bodyid[wrap_id]
                xpos = data.site_xpos[wrap_id]
                body_name = model.names[model.name_bodyadr[body_id]:].split(b'\x00', 1)[0].decode("utf-8")
                return site_name, body_name, xpos
            else:
                assert False, f"wrap_type {wrap_type} not implemented"
                return None, None, None
            
        
        metainfo = defaultdict(list) # metainfo is a dictionary with body name as key and a list of tendons as value
        # get tendon data
        for tendon_idx in range(model.ntendon):
            start_index = model.tendon_adr[tendon_idx]
            # Number of path elements for this tendon
            num_elements = model.tendon_num[tendon_idx]

            tendon_name = model.names[model.name_tendonadr[tendon_idx]:].split(b'\x00', 1)[0].decode("utf-8")

            for i in range(start_index, start_index + num_elements - 1):
                wrap_id = model.wrap_objid[i]
                wrap_id_next = model.wrap_objid[i+1]
                wrap_type = mjWRAP_CONSTANTS[model.wrap_type[i]]
                wrap_type_next = mjWRAP_CONSTANTS[model.wrap_type[i+1]]
                wrap_name, body_name, position = get_info_from_wrap(wrap_id, wrap_type)

                wrap_name_next, body_name_next, position_next = get_info_from_wrap(wrap_id_next, wrap_type_next)
                if i == start_index:
                    if tendon_name not in metainfo[body_name]:
                        metainfo[body_name].append(tendon_name)
                if i+1 == start_index + num_elements - 1:
                    if tendon_name not in metainfo[body_name_next]:
                        metainfo[body_name_next].append(tendon_name)

        return metainfo

    def get_description(self, simDSL2nlq = False) -> List[dict]:
        """
        Get the parameters of all entities in the scene.
        """
        bodies_description = []

        for entity in self.entities:
            bodies_description.extend(entity.get_description(simDSL2nlq=simDSL2nlq))
        
        # Replace substrings in entity_params and tendon_params
        bodies_description = [
            Scene.replace_substrings_recursive(param, self.name_mapping) 
            for param in bodies_description
        ]
        
        metainfo = self.update_tendon_info()
        
        # update the tendon connection information in the bodies_description
        # for body in bodies_description:
        #     body_name = body["name"]
        #     if body_name in metainfo:
        #         body["tendon"] = metainfo[body_name]
        #         if len(metainfo[body_name]) == 0:
        #             continue
        #         elif len(metainfo[body_name]) == 1:
        #             tendon_name = metainfo[body_name][0]
        #             body["description"] += f" The string named {tendon_name} is attached to {body_name}."
        #         else:   # name1, name2, and name3 are connected to body_name
        #             tendon_names = ", ".join(metainfo[body_name])
        #             # add "and" before the last tendon name
        #             last_comma = tendon_names.rfind(",")
        #             tendon_names = tendon_names[:last_comma] + " and" + tendon_names[last_comma+1:]
        #             body["description"] += f" The strings named {tendon_names} are attached to {body_name}."
        
        for body in bodies_description:
            tendon = []
            # get tendon that is connected to the body
            for idx in range(len(self.tendons)):
                connected_sites = [self.tendons[idx].spatials[0].elements[i].get_body_name() for i in range(len(self.tendons[idx].spatials[0].elements))]
                if body["name"] in connected_sites:
                    _idx = connected_sites.index(body["name"])
                    tendon.append((_idx in [0, len(connected_sites) - 1], self.tendons[idx].spatials[0].name))
                    continue
            
            body['tendon'] = [t[1] for t in tendon]

            end_tendons = [t[1] for t in tendon if t[0]]
            mid_tendons = [t[1] for t in tendon if not t[0]]

            if len(end_tendons) == 0:
                continue
            elif len(end_tendons) == 1:
                body["description"] += f" The light string named {end_tendons[0]} is attached to {body['name']}."
            else:   # name1, name2, and name3 are connected to body_name
                tendon_names = ", ".join(end_tendons)
                # add "and" before the last tendon name
                last_comma = tendon_names.rfind(",")
                tendon_names = tendon_names[:last_comma] + " and" + tendon_names[last_comma+1:]
                body["description"] += f" The light strings named {tendon_names} are attached to {body['name']}."

            if len(mid_tendons) == 0:
                continue
            elif len(mid_tendons) == 1:
                body["description"] += f" The light string named {mid_tendons[0]} winds over {body['name']}."
            else:   # name1, name2, and name3 are connected to body_name
                tendon_names = ", ".join(mid_tendons)
                # add "and" before the last tendon name
                last_comma = tendon_names.rfind(",")
                tendon_names = tendon_names[:last_comma] + " and" + tendon_names[last_comma+1:]
                body["description"] += f" The light strings named {tendon_names} wind over {body['name']}."

        tendon_descriptions = []

        for tendon in self.tendons:
            tendon_descriptions.extend(tendon.get_description(simDSL2nlq=simDSL2nlq))
        
        # Replace substrings in entity_params and tendon_params
        tendon_descriptions = [
            Scene.replace_substrings_recursive(param, self.name_mapping) 
            for param in tendon_descriptions
        ]

        if simDSL2nlq: return bodies_description, tendon_descriptions
        return bodies_description
    
    def get_nlq_new(self):
        """
        Generates a natural language description of the physics scene.
        
        Parameters:
        scene
                        
        Returns:
        str: A natural language description of the scene.
        """
        # Build a dictionary mapping entity names to their data for easy lookup.
        entities_dict = {entity.name: entity for entity in self.entities}
        
        # Build an adjacency list for the graph and record connection details.
        graph = {name: [] for name in entities_dict}
        # Use a dictionary with frozenset keys (order-insensitive) to capture edge details.
        connection_details = {}  # key: frozenset({entity1, entity2}), value: list of connection details
        
        # Process each connection block (assumed to be a tendon with a list of segments).
        for conn in self.connections:
            # For consecutive segments in the tendon, build edges.
            for i in range(len(conn) - 1):
                # conn: (entity_name, connecting_direction, connecting_point, connecting_seq_idx)

                name1 = conn[i][0]
                name2 = conn[i+1][0]
                
                # Add the edge in both entities (undirected graph).
                if name1 in graph and name2 in graph:
                    graph[name1].append(name2)
                    graph[name2].append(name1)
                
                # Record connecting point details. We use frozenset as the key to avoid duplicate edges.
                edge_key = frozenset({name1, name2})
                if edge_key not in connection_details:
                    connection_details[edge_key] = []

                connection_details[edge_key].append({
                    "from": name1,
                    "to": name2,
                    "from_point": conn[i][1:],
                    "to_point": conn[i + 1][1:]
                })
        
        # Traverse the graph (handling possible multiple connected components).
        visited = set()
        description_lines = []
        
        def bfs(start):
            queue = [start]
            visited.add(start)
            while queue:
                current = queue.pop(0)
                # Append the natural language description for the current entity.
                if hasattr(entities_dict[current], 'get_nlq'):
                    entity_desc = entities_dict[current].get_nlq()
                else:
                    entity_desc = f"Description not implemented for entity {entities_dict[current].__class__.__module__ + '.' + entities_dict[current].__class__.__name__}"
                    
                description_lines.append(entity_desc)
                # Explore each adjacent neighbor.
                for neighbor in graph[current]:
                    # Create a unique key for the edge.
                    edge_key = frozenset({current, neighbor})
                    if edge_key in connection_details:
                        for detail in connection_details[edge_key]:
                            # To avoid duplicating descriptions for the same connection,
                            # only output the connection when the current entity matches the 'from' field.
                            if detail["from"] == current:
                                if hasattr(entities_dict[current], 'connecting_point_nl'):
                                    point1 = entities_dict[current].connecting_point_nl(*detail['from_point'])
                                else:
                                    point1 = detail['from_point'][1].value
                                if hasattr(entities_dict[neighbor], 'connecting_point_nl'):
                                    point2 = entities_dict[neighbor].connecting_point_nl(*detail['to_point'])
                                else:
                                    point2 = detail['to_point'][1].value

                                # Sort the points based on the connecting direction.
                                if detail['from_point'][0] == ConnectingDirection.OUTER_TO_INNER:
                                    point1, point2 = point2, point1
                                elif detail['to_point'][0] == ConnectingDirection.INNER_TO_OUTER:
                                    point1, point2 = point2, point1

                                # conn_sentence = (
                                #     f"A string extends from '{current}' ({point1}) "
                                #     f"to '{neighbor}' ({point2})."
                                # )
                                conn_sentence = (
                                    f"{point1}"
                                    f" {point2}"
                                )
                                description_lines.append(conn_sentence)
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
        
        # Handle all connected components in the scene.
        for entity_name in graph:
            if entity_name not in visited:
                bfs(entity_name)
        
        # Combine all sentences into a coherent multi-line description.
        return "\n".join(description_lines)

    def get_nlq(self, symbolic = False):
        """
        Generates a natural language description of the physics scene.

        Returns:
        str: A natural language description of the scene.
        """
        # Build a dictionary mapping entity names to their data for easy lookup.
        entities_dict = {entity.name: entity for entity in self.entities}

        # Build a directed adjacency list
        graph = {name: [] for name in entities_dict}

        # Dictionary to store connection details with ordered edges
        connection_details = {}  # key: tuple (entity1, entity2), value: list of connection details

        global_sym_dict, var_counters = {}, {}

        name_simplification = {
            "mass": "m_",
            "charge": "q_",
            "angle": "θ_",
            "force": "F_",
            "x": "x_",
            "y": "y_",
            "z": "z_",
            "vx": "vx_",
            "vy": "vy_",
            "vz": "vz_",
            "restitution": "e_",
            "friction": "μ_",
            "k": "k_",
            "b": "b_",
            "radius": "r_",
            "height": "h_",
            "length": "l_",
            "width": "w_",
        }

        # Process each connection in the order they appear
        for conn in self.connections:
            for i in range(len(conn) - 1):
                name1 = conn[i][0]
                name2 = conn[i + 1][0]

                # Treat connections as directed: name1 -> name2
                if name1 in graph:
                    graph[name1].append(name2)

                # Store connection details
                edge_key = (name1, name2)  # Preserve direction
                if edge_key not in connection_details:
                    connection_details[edge_key] = []

                connection_details[edge_key].append({
                    "from": name1,
                    "to": name2,
                    "from_point": conn[i][1:],  # (connecting_direction, connecting_point, connecting_seq_idx)
                    "to_point": conn[i + 1][1:]
                })

        # Traverse the graph following connection order (BFS-based)
        visited = set()
        description_lines = []

        def bfs(start):
            queue = [start]
            visited.add(start)
            while queue:
                current = queue.pop(0)
                
                # Append entity description
                if hasattr(entities_dict[current], 'get_nlq'):
                    entity_desc = entities_dict[current].get_nlq(symbolic = symbolic)
                else:
                    entity_desc = f"Description not implemented for entity {entities_dict[current].__class__.__name__}"

                if not isinstance(entity_desc, str):
                    entity_desc, sym_dict = entity_desc

                    # Process symbolic variables for this entity
                    for key, value in sym_dict.items():
                        # Extract type from key format "<type>number"
                        match = re.match(r"<(\w+)>\d+", key)
                        if match:
                            sym_type = match.group(1)  # e.g., 'mass', 'angle', 'force', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'restitution'
                            
                            # Initialize counter for this type if not exists
                            if sym_type not in var_counters:
                                var_counters[sym_type] = 1
                            
                            # Create new key with sequential numbering
                            new_key = f"{name_simplification[sym_type]}{var_counters[sym_type]}"
                            
                            # Update global mapping and counter
                            global_sym_dict[new_key] = value
                            var_counters[sym_type] += 1
                            
                            # Replace old key with new key in description
                            entity_desc = entity_desc.replace(key, new_key)

                description_lines.append(entity_desc)

                # Explore each directed neighbor in order
                for neighbor in graph[current]:
                    edge_key = (current, neighbor)
                    if edge_key in connection_details:
                        for detail in connection_details[edge_key]:
                            # Get readable connection points
                            if hasattr(entities_dict[current], 'connecting_point_nl'):
                                point1 = entities_dict[current].connecting_point_nl(*detail['from_point'], first = True)
                            else:
                                point1 = detail['from_point'][1].value
                            
                            if hasattr(entities_dict[neighbor], 'connecting_point_nl'):
                                point2 = entities_dict[neighbor].connecting_point_nl(*detail['to_point'])
                            else:
                                point2 = detail['to_point'][1].value

                            # Format connection description
                            conn_sentence = f"{point1} {point2}"
                            # conn_sentence = f"A string extends from '{current}' ({point1}) to '{neighbor}' ({point2})."
                            description_lines.append(conn_sentence)

                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

        # Start BFS from entities in the order they appear in `self.connections`
        for conn in self.connections:
            start_entity = conn[0][0]  # First entity in the first connection
            if start_entity not in visited:
                bfs(start_entity)

        if len(self.connections) == 0: # No connections, only one entity
            bfs(list(entities_dict.keys())[0])

        if symbolic: return "\n".join(description_lines), global_sym_dict
        return "\n".join(description_lines)
    
    def get_question(self, time_to_ask: float, attribute_to_ask: tuple[str], mode = "masses", keys = None):
        """
        Generates a question about the scene.
        
        Parameters:
            time_to_ask: float
            attribute_to_ask: tuple[str]
        
        Returns:
            str: A question about the scene.
        """
        
        if mode == "masses":
            entity, sub_entity, quantity = attribute_to_ask
            assert entity in keys and sub_entity in keys[entity]

            entity = [e for e in self.entities if e.name == entity]
            if not entity: raise ValueError(f"Entity '{entity}' not found in the scene.")
            entity = entity[0]

            q_description = entity.get_question(sub_entity, quantity)

        elif mode == "strings":
            string, quantity = attribute_to_ask
            quantity = "tension" # Assume only tension is asked. Change when spring questions are introduced
            assert string in keys

            # Robust lookup: Find the tendon object that matches the name in the string
            tendon = None
            for t in self.tendons:
                if t.name in string:
                    tendon = t
                    break
            
            if not tendon:
                # Fallback for older naming conventions or indexed tendons
                try: 
                    import re
                    matches = re.findall(r'\d+', string)
                    if matches:
                        tendon_idx = int(matches[-1])
                        if 0 <= tendon_idx < len(self.tendons):
                            tendon = self.tendons[tendon_idx]
                except (ValueError, IndexError):
                    pass
            
            if not tendon:
                logger.warning(f"Could not find tendon object for string '{string}'. Skipping question.")
                return ""

            seq = []

            for e in tendon.spatials[0].elements:
                site_name = e.site
                parts = site_name.split('.')
                entity_name = parts[0]
                sub_entity_name = None 

                for i in range(1, len(parts)):
                    part = '.'.join(parts[1:i+1])
                    if entity_name in keys and part in keys[entity_name]:
                        seq.append((entity_name, part))
                        sub_entity_name = part
                        break

            part1 = [e for e in self.entities if e.name == seq[0][0]][0].get_question(seq[0][1], quantity)

            if len(seq) > 1:
                part2 = [e for e in self.entities if e.name == seq[1][0]][0].get_question(seq[1][1], quantity)
            else:
                part2 = "the other end"

            opening = f"What is the {quantity} of the" 
            part1 = part1[len(opening) + 1:]
            if len(seq) > 1: part2 = part2[len(opening) + 1:]

            q_description = f"{opening} string that connects {part1} to {part2}"


        question = (
            f"{q_description} after {time_to_ask:.2f} seconds?"
        )

        return question
    
    def get_entity_and_body(self):
        """
        return entities and bodies name in a hierarchical dictionary
        """
        entity_dict = {}
        for entity in self.entities:
            bodies = entity.get_bodies()
            if bodies:
                entity_dict[entity.name] = bodies

        return entity_dict

    def get_shortcut(self):
        condition = False
        for entity in self.entities:
            condition = condition or entity.get_shortcut()
        return condition

def parse_scene(dsl_path, scene_data_dict=None) -> Scene:
    def process_parameters(parameters, entity_class):
        processed_params = {}
        type_hints = get_type_hints(entity_class.__init__)
        for param_name, param_value in parameters.items():
            param_type = type_hints.get(param_name, None)
            if isinstance(param_type, type) and issubclass(param_type, Enum):
                if isinstance(param_value, str):
                    # Convert string to Enum
                    processed_params[param_name] = param_type[param_value.upper()]
                else:
                    processed_params[param_name] = param_value
            elif param_type == Optional[PulleyParam]:
                if isinstance(param_value, dict):
                    # Instantiate PulleyParam from dict
                    processed_params[param_name] = PulleyParam(**param_value)
                else:
                    processed_params[param_name] = param_value
            else:
                processed_params[param_name] = param_value
        return processed_params

    if scene_data_dict is None:
        with open(dsl_path, "r") as file:
            scene_data = yaml.safe_load(file)
    else:
        scene_data = scene_data_dict

    scene = Scene(name=scene_data["scene"]["name"])

    entities = {}
    for entity_data in scene_data["scene"]["entities"]:
        entity_type = entity_data["type"]
        entity_class = ENTITY_CLASSES[entity_type]
        parameters = entity_data.get("parameters", {})
        processed_parameters = process_parameters(parameters, entity_class)
        entity = entity_class(
            name=entity_data["name"],
            pos=tuple(entity_data["position"]),
            **processed_parameters,
        )
        entities[entity_data["name"]] = entity
        scene.add_entity(entity)

    # Process connections
    proceed_connections(scene, scene_data)

    scene.set_attributes_from_entities()

    scene.tag = scene_data["scene"].get("tag", None)

    return scene


def proceed_connections(scene, scene_data):
    # Process connections
    if "connections" in scene_data["scene"]:
        for connection in scene_data["scene"]["connections"]:
            tendon_connections = []
            for conn in connection["tendon"]:
                entity_name = conn["entity"]
                connecting_point_str = conn.get("connecting_point", "default")
                connecting_point = ConnectingPoint(connecting_point_str)
                connecting_point_seq_id = conn.get("connecting_point_seq_id", None)
                direction_str = conn.get("direction", "left_to_right")
                direction = ConnectingDirection(direction_str)
                tendon_connections.append(
                    (entity_name, direction, connecting_point, connecting_point_seq_id)
                )
            scene.add_connection(tendon_connections)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene_num",
        type=int,
        default=None,
        help="Scene number to parse, eg: 0 for scene_0",
    )
    parser.add_argument(
        "--textbook_question",
        type=str,
        default=None,
        help="Textbook question to parse",
    )
    parser.add_argument(
        "--DSL2DSL",
        type=str,
        default=None,
        help="DSL2DSL question to parse",
    )
    parser.add_argument(
        "--custom_path",
        type=str,
        default=None,
        help="custom path to yaml to parse",
    )

    parser.add_argument(
        '--simDSL2nlq',
        action='store_true',
        help='Retrieve simDSL2nlq description'
    )
    args = parser.parse_args()

    cur_path = os.path.dirname(os.path.realpath(__file__))
    # scene_yaml_path = os.path.join(cur_path, "./DSLs/mass_prism_plane_entity.yaml")
    scene_yaml_path = os.path.join(
        cur_path,
        "./DSLs/debug.yaml",
    )

    if args.DSL2DSL is not None:
        scene_yaml_path = os.path.join(
            cur_path,
            f"../DSL2DSL/textbook_questions/{args.DSL2DSL}/simDSL.yaml",
        )
    elif args.textbook_question is not None:
        scene_yaml_path = os.path.join(
            cur_path,
            f"./DSLs/textbook_questions/{args.textbook_question}/question.yaml",
        )
    elif args.scene_num is not None:
        scene_yaml_path = os.path.join(
            cur_path, f"./dsl_output/scene_{args.scene_num}/scene_output.yaml"
        )
    elif args.custom_path is not None:
        scene_yaml_path = os.path.join(cur_path, f"../{args.custom_path}")

    logger.debug(f"Using scene YAML file: {scene_yaml_path}")

    scene = parse_scene(scene_yaml_path)
    xml_output = scene.to_xml()
    
    with open(os.path.join(cur_path, "xml_output/scene_output.xml"), "w") as file:
        file.write(xml_output)

    new_nlq = scene.get_nlq()
    try:
        new_nlq = scene.get_nlq()
        print(new_nlq)
    except Exception as e:
        print('Error in generating new NLQ')
        print(e)
        new_nlq = "Error in generating new NLQ"

    print('***********DESCRIPTION BEGIN***********')
    description = scene.get_description(args.simDSL2nlq)

    if args.simDSL2nlq:
        dsl = 'Body description:\n' + '\n'.join([_dsl['description'] for _dsl in description[0]])
        dsl += '\nConnection description:\n' + '\n'.join([_dsl['description'] for _dsl in description[1]])

        mappings = create_mappings(dsl)
        mappings = {k:v for mapping in mappings for k, v in mapping.items()}
        description = replace_all(dsl, mappings)

    print(description)
    print('************DESCRIPTION END************')

    if args.simDSL2nlq: 
        mappings = create_mappings(dsl)
        print(mappings)

    with open(os.path.join(cur_path, './xml_output/description_old.txt'), 'w') as f:
        f.write(str(description))
    with open(os.path.join(cur_path, './xml_output/mapping.json'), 'w') as f:
        json.dump(mappings, f, indent=4)
    
    with open(os.path.join(cur_path, './xml_output/description_new.txt'), 'w') as f:
        f.write(new_nlq)

    constant_force_dict = scene.get_constant_force_dict()
    if len(constant_force_dict) > 0:
        with open(
            os.path.join(cur_path, "./xml_output/scene_output_constant_force.json"), "w"
        ) as file:
            json.dump(constant_force_dict, file, indent=4)

    init_velocity_dict = scene.get_init_velocity_dict()
    if len(init_velocity_dict) > 0:
        with open(
            os.path.join(cur_path, "./xml_output/scene_output_init_velocity.json"), "w"
        ) as file:
            json.dump(init_velocity_dict, file, indent=4)

    logger.debug("Scene parameters: " + str(scene.get_parameters()))

    # Randomize the scene and save the randomized XML
    scene.randomize_entities()
    xml_output = scene.to_xml()

    with open(
        os.path.join(cur_path, "./xml_output/scene_output_randomized.xml"), "w"
    ) as file:
        file.write(xml_output)

    logger.debug("Randomized scene parameters: " + str(scene.get_parameters()))
