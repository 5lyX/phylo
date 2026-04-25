import copy
import xml.etree.ElementTree as ET
import numpy as np
import ipdb
import re

st = ipdb.set_trace


class XMLBodyUnpacker:
    def __init__(self):
        self.sites = {}
        self.geoms = {}
        self.bodys = {}
        self.name_mapping = {}
        self.name_counters = {}

    @staticmethod
    def load_xml_from_file(file_path):
        tree = ET.parse(file_path)
        return tree

    @staticmethod
    def load_xml_from_str(xml_str):
        root = ET.fromstring(xml_str)
        tree = ET.ElementTree(root)
        return tree

    @staticmethod
    def save_xml_to_str(tree):
        return ET.tostring(tree.getroot(), encoding="unicode", method="xml")

    @staticmethod
    def save_xml_file(tree, file_path):
        tree.write(file_path)

    @staticmethod
    def quat_multiply(q1, q2):
        # Quaternion multiplication implementation
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,  # W
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,  # X
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,  # Y
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,  # Z
        ]

    @staticmethod
    def rotate_vector(vec, quat):
        # Rotate the vector vec using the quaternion quat
        q_vec = [0] + vec
        quat_conj = [quat[0], -quat[1], -quat[2], -quat[3]]
        q_res = XMLBodyUnpacker.quat_multiply(
            XMLBodyUnpacker.quat_multiply(quat, q_vec), quat_conj
        )
        return q_res[1:]

    @staticmethod
    def quat_inverse(q):
        w, x, y, z = q
        # Calculate the squared norm of the quaternion
        q_norm_sq = w * w + x * x + y * y + z * z
        # Calculate the conjugate of the quaternion
        conjugate = [w, -x, -y, -z]
        # Calculate the inverse of the quaternion
        inverse_q = [conjugate[i] / q_norm_sq for i in range(4)]
        return inverse_q

    def parse_xml(self, tree, update_tendon_lengths=True):
        root = tree.getroot()
        # Process the content inside <worldbody>
        worldbody = root.find("worldbody")
        if worldbody is not None:
            new_bodies = self.process_body(worldbody)
            worldbody.clear()  # Clear existing bodies
            worldbody.extend(new_bodies)  # Add the processed bodies
        if update_tendon_lengths:
            self.update_tendon_lengths(tree)
        # self.update_constant_force(tree)

    def process_body(self, parent_body, parent_pos=[0, 0, 0], parent_quat=[1, 0, 0, 0]):
        new_bodies = []
        for body in parent_body.findall("body"):
            pos = body.get("pos", "0 0 0")
            quat = body.get("quat", "1 0 0 0")
            # Compute updated position and quaternion
            updated_pos, updated_quat = self.compute_updated_transform(pos, quat, parent_pos,
                                                                       parent_quat)

            has_joint = body.find("joint") is not None
            has_nested_body = len(body.findall("body")) > 0

            # Process the current body's sites and geoms
            self.process_sites(body, updated_pos, updated_quat)
            self.process_geoms(body, updated_pos, updated_quat)

            if not has_joint and has_nested_body:
                # Recursively process nested bodies and get a list of modified elements
                nested_bodies = self.process_body(body, updated_pos, updated_quat)
                new_bodies.extend(nested_bodies)
                # delete all nested bodies in the current body
                for nested_body in body.findall("body"):
                    body.remove(nested_body)
            else:
                if has_nested_body:
                    # just record the nested bodies sites and geoms pos quat, without modifying them
                    self.process_body(copy.deepcopy(body), updated_pos, updated_quat)
            # Update the current body with new pos and quat
            body.set("pos", " ".join(map(str, updated_pos)))
            body.set("quat", " ".join(map(str, updated_quat)))
            # if body has child
            if len(body.findall("body")) > 0 or len(body.findall("geom")) > 0 or len(body.findall("site")) > 0:
                new_bodies.append(body)
                self.bodys[body.get("name")] = {
                    "pos": updated_pos,
                    "quat": updated_quat,
                }
        return new_bodies

    def compute_updated_transform(self, pos_str, quat_str, parent_pos, parent_quat):
        # Convert position and quaternion strings to lists of floats
        current_pos = list(map(float, pos_str.strip().split()))
        current_quat = list(map(float, quat_str.strip().split()))
        # Rotate and translate the current position
        rotated_pos = self.rotate_vector(current_pos, parent_quat)
        updated_pos = [p1 + p2 for p1, p2 in zip(parent_pos, rotated_pos)]
        # Update the current quaternion
        updated_quat = self.quat_multiply(parent_quat, current_quat)
        return updated_pos, updated_quat

    def process_sites(self, body, updated_pos, updated_quat):
        for site in body.findall("site"):
            site_name = site.get("name")
            site_pos = site.get("pos", "0 0 0")
            site_pos = list(map(float, site_pos.strip().split()))
            # Compute global position of site
            rotated_site_pos = self.rotate_vector(site_pos, updated_quat)
            global_site_pos = [p1 + p2 for p1, p2 in zip(updated_pos, rotated_site_pos)]
            # Update site's quaternion to inverse of updated_quat
            updated_site_quat = self.quat_inverse(updated_quat)
            site.set("quat", " ".join(map(str, updated_site_quat)))
            self.sites[site_name] = {
                "pos": global_site_pos,
                "quat": updated_site_quat,
            }

    def process_geoms(self, body, updated_pos, updated_quat):
        for geom in body.findall("geom"):
            geom_name = geom.get("name")
            geom_pos = geom.get("pos", "0 0 0")
            geom_quat = geom.get("quat", "1 0 0 0")
            geom_type = geom.get("type")
            geom_size = geom.get("size")
            geom_pos = list(map(float, geom_pos.strip().split()))
            geom_quat = list(map(float, geom_quat.strip().split()))
            geom_size = list(map(float, geom_size.strip().split()))
            # Compute global position of geom
            rotated_geom_pos = self.rotate_vector(geom_pos, updated_quat)
            global_geom_pos = [p1 + p2 for p1, p2 in zip(updated_pos, rotated_geom_pos)]
            self.geoms[geom_name] = {
                "pos": global_geom_pos,
                "quat": geom_quat,
                "type": geom_type,
                "size": geom_size,
            }

    def update_constant_force(self, tree):
        # Get the root element of the XML tree
        root = tree.getroot()

        # Iterate through each actuator found in the XML structure
        for actuator in root.findall("actuator/general"):
            # Retrieve the name of the body associated with the actuator
            body_name = actuator.get("body")

            # Obtain the original bias parameters as a numpy array of floats
            original_biasprm = np.array(
                list(map(float, actuator.get("biasprm").split()))
            )

            # Extract the original force and torque from the bias parameters
            original_force = original_biasprm[
                :3
            ]  # First three elements represent force
            original_torque = original_biasprm[
                3:
            ]  # Remaining elements represent torque

            # Check if the body exists in the geometry mappings
            if body_name in self.bodys:
                body_pos = self.bodys[body_name]["pos"]  # Get the position of the body
            else:
                body_pos = [0, 0, 0]  # Default to origin if body not found

            # Calculate the magnitude of the original force vector
            force_magnitude = np.linalg.norm(original_force)
            if force_magnitude == 0:
                new_force = [0, 0, 0]  # Set new force to zero if original force is zero
            else:
                # Normalize the original force vector to obtain the unit vector
                unit_force = original_force / force_magnitude
                # Calculate the new direction by translating the unit force vector
                new_direction = (
                    np.array(body_pos) + unit_force
                )  # Adjust the calculation as needed
                new_direction_magnitude = np.linalg.norm(
                    new_direction
                )  # Get the magnitude of the new direction

                # Calculate the new force based on the magnitude and direction
                new_force = (
                    (new_direction / new_direction_magnitude) * force_magnitude
                    if new_direction_magnitude != 0
                    else [0, 0, 0]
                )

            # Update the bias parameters with the new force and original torque
            new_biasprm = np.concatenate((new_force, original_torque))
            actuator.set(
                "biasprm", " ".join(map(str, new_biasprm))
            )  # Convert to string and update XML

    def update_tendon_lengths(self, tree):
        # Ensure that self.sites and self.geoms are populated
        # if not self.sites or not self.geoms:
        #     self.parse_xml(tree)
        root = tree.getroot()
        for tendon in root.findall("tendon"):
            for spatial in tendon.findall("spatial"):
                # Get the elements in the spatial path
                elements = []
                for elem in spatial:
                    if elem.tag == "site":
                        elements.append({"type": "site", "name": elem.get("site")})
                    elif elem.tag == "geom":
                        elements.append({"type": "geom", "name": elem.get("geom")})
                # Now compute the total length
                total_length = 0
                set_range = True
                for i in range(len(elements) - 1):
                    elem1 = elements[i]
                    elem2 = elements[i + 1]
                    if elem1["type"] == "site" and elem2["type"] == "site":
                        pos1 = self.sites[elem1["name"]]["pos"]
                        pos2 = self.sites[elem2["name"]]["pos"]
                        dist = self.compute_distance(pos1, pos2)
                    elif elem1["type"] == "site" and elem2["type"] == "geom":
                        # set_range = False
                        # break  # break for now as we don't have a good way to compute geom
                        pos1 = self.sites[elem1["name"]]["pos"]
                        geom2 = self.geoms[elem2["name"]]
                        if geom2["type"] == "cylinder":
                            dist = self.compute_site_geom_distance(pos1, geom2)
                        else:
                            pos2 = geom2["pos"]
                            dist = self.compute_distance(pos1, pos2)
                    elif elem1["type"] == "geom" and elem2["type"] == "site":
                        # set_range = False
                        # break  # break for now as we don't have a good way to compute geom
                        geom1 = self.geoms[elem1["name"]]
                        pos2 = self.sites[elem2["name"]]["pos"]
                        if geom1["type"] == "cylinder":
                            dist = self.compute_site_geom_distance(pos2, geom1)
                        else:
                            pos1 = geom1["pos"]
                            dist = self.compute_distance(pos1, pos2)
                    else:
                        # set_range = False
                        # break  # break for now as we don't have a good way to compute geom
                        # For geom to geom, use direct distance
                        pos1 = self.geoms[elem1["name"]]["pos"]
                        pos2 = self.geoms[elem2["name"]]["pos"]
                        dist = self.compute_distance(pos1, pos2)
                    total_length += dist
                # Update the spatial's range parameter
                if set_range:
                    spatial.set("range", f"0 {total_length}")
                    spatial.set("limited", "true")
                    spatial.set("solimplimit", "0.99 1 0.001 0.01 20")
                    spatial.set("solreflimit", "0.02 1")
                if spatial.get("name").startswith("tendon_force_"):
                    # this will be used as a constant force, set limited to false and remove range
                    spatial.set("limited", "false")
                    if "range" in spatial.attrib:
                        del spatial.attrib["range"]
                if spatial.get("name").startswith("spring-") \
                        or spatial.get("name").endswith("spring")\
                        or spatial.get("name").endswith("spring")\
                        or re.match(r".*spring(-\d+)?$", spatial.get("name")):
                    # for spring, we don't want to limit the range
                    spatial.set("limited", "false")
                    if "range" in spatial.attrib:
                        del spatial.attrib["range"]

    @staticmethod
    def compute_distance(pos1, pos2):
        # Euclidean distance between two positions
        return np.linalg.norm(np.array(pos1) - np.array(pos2))

    @staticmethod
    def compute_site_geom_distance(site_pos, geom):
        # Calculate the distance from a site to a cylinder geom
        geom_pos = geom["pos"]
        geom_size = geom["size"]
        radius = geom_size[0]
        height = geom_size[1]
        # Vector from geom to site
        vec = np.array(site_pos) - np.array(geom_pos)
        # Project vec onto the plane perpendicular to the cylinder's axis (assuming y-axis)
        vec_xz = vec.copy()
        vec_xz[1] = 0  # Zero out the y-component
        distance_xz = np.linalg.norm(vec_xz)
        if distance_xz < radius:
            # The site is inside the cylinder's projection, set distance to zero
            l1 = 0
            arc_length = 0
        else:
            # Distance from site to tangent point
            l1 = np.sqrt(distance_xz**2 - radius**2)
            # Angle between vec_xz and the tangent point
            theta = np.arccos(radius / distance_xz)
            # Arc length along the cylinder surface
            arc_length = radius * theta
        # Vertical distance (along y-axis)
        vertical_distance = abs(vec[1])
        total_dist = l1 + arc_length + vertical_distance
        return total_dist

    def simplify_names(self, tree) -> dict:
        self.name_mapping = {}
        self.name_counters = {}
        root = tree.getroot()

        def process_element(element):
            # Attributes to be processed (including 'name', 'site', 'sidesite', 'data')
            attributes_to_process = [
                "name",
                "site",
                "sidesite",
                "data",
                "tendon",
                "geom",
            ]

            for attr in attributes_to_process:
                # Process the specified attributes if they exist in the element
                if attr in element.attrib:
                    original_name = element.get(attr)
                    # Check if the name is already in the mapping
                    if original_name in self.name_mapping:
                        new_name = self.name_mapping[original_name]
                    else:
                        # Generate new name based on element type
                        if element.tag == "body" and attr == "name":
                            # Process as per body rules
                            parts = original_name.split("_")
                            # Iterate from end to find the last non-numeric string
                            for part in reversed(parts):
                                if not part.isdigit():
                                    base_name = part
                                    break
                            else:
                                base_name = (
                                    "body"  # Default base name if all parts are numeric
                                )
                        else:
                            # For other elements or attributes, use the tag or attribute name as the base
                            base_name = element.tag

                        # Initialize counter for base_name if not present
                        if base_name not in self.name_counters:
                            self.name_counters[base_name] = 0
                        else:
                            self.name_counters[base_name] += 1

                        # Generate new name
                        new_name = f"{base_name}_{self.name_counters[base_name]}"
                        # Save mapping
                        self.name_mapping[original_name] = new_name

                    # Set the new name for the attribute
                    element.set(attr, new_name)

            # Recurse on child elements
            for child in element:
                process_element(child)

        # Start processing from <worldbody> and its following elements
        elements_to_process = []
        found_worldbody = False
        for child in root:
            if child.tag == "worldbody":
                found_worldbody = True
                elements_to_process.append(child)
            elif found_worldbody:
                elements_to_process.append(child)

        for elem in elements_to_process:
            process_element(elem)

        return self.name_mapping
