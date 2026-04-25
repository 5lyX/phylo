from typing import Dict
import re, os
from collections import defaultdict

from ipdb import set_trace as st

def replace_all(text: str, mapping: dict) -> str:
    mapping = {k.encode('unicode_escape').decode():str(v) for k, v in mapping.items()}
    text = text.encode('unicode_escape').decode()
    
    if len(mapping) == 0: return text 

    # Create pattern of all keys, longest first
    pattern = '|'.join(map(re.escape, sorted(mapping.keys(), key=len, reverse=True)))
    
    # Replace function
    def replace(match):
        return mapping[match.group(0)]
    
    # Do all replacements at once
    text = re.sub(pattern, replace, text)
    text = text.encode().decode('unicode_escape')  # 还原
    return text

def find_closest_value(sorted_list: list, target: float) -> float:
    """
    Find closest value to target in a sorted list using binary search.
    
    Args:
        sorted_list: List of sorted numbers
        target: Value to find closest match for
        
    Returns:
        Idx of closest value from the list
    """
    if not sorted_list:
        raise ValueError("List is empty")
        
    # Handle edge cases
    if target <= sorted_list[0]:
        return sorted_list[0]
    if target >= sorted_list[-1]:
        return sorted_list[-1]
        
    # Binary search
    left, right = 0, len(sorted_list) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if sorted_list[mid] == target:
            return mid
            
        if target < sorted_list[mid]:
            right = mid - 1
        else:
            left = mid + 1
            
    # Compare the two closest values
    if abs(sorted_list[left] - target) < abs(sorted_list[right] - target):
        return left
    return right

def create_mappings(text: str) -> tuple[Dict, Dict, Dict]:
    # Create mappings for different patterns
    string_pattern = r'spatial_\d+'
    string_pattern_alt = r'\bspatial_ready\w+'
    string_pattern_alt_alt = r'tendon_force_\d+'
    pulley_pattern = r'\w+pulley\b'
    pulley_pattern_alt = r'\bconstant_force_fixed_pulley_entity_\d+'
    pulley_pattern_alt_alt = r'\bfixed_pulley_entity_\d+'
    pulley_pattern_alt_alt_alt = r'\w+pulley_\d+'
    pulley_pattern_alt_alt_alt_alt = r'\w+plane\b'
    mass_pattern = r'\w+mass\d*\b'
    prism_pattern = r'\w+prism\d*\b'
    
    # Find all unique matches
    strings = sorted(
        set(re.findall(string_pattern, text)) |
        set(re.findall(string_pattern_alt, text)) |
        set(re.findall(string_pattern_alt_alt, text))
    )
    pulleys = sorted(
        set(re.findall(pulley_pattern, text)) |
        set(re.findall(pulley_pattern_alt, text)) |
        set(re.findall(pulley_pattern_alt_alt, text)) |
        set(re.findall(pulley_pattern_alt_alt_alt, text)) |
        set(re.findall(pulley_pattern_alt_alt_alt_alt, text))
    )
    masses = sorted(set(re.findall(mass_pattern, text)))
    prisms = sorted(set(re.findall(prism_pattern, text)))
    
    # Create mapping dictionaries
    string_map = {s: f'string{i+1}' for i, s in enumerate(strings)}
    pulley_map = {p: f'pulley{i+1}' for i, p in enumerate(pulleys)}
    mass_map = {m: f'mass{i+1}' for i, m in enumerate(masses)}
    prism_map = {m: f'prism{i+1}' for i, m in enumerate(prisms)}
    
    return string_map, pulley_map, mass_map, prism_map

def find_tags(tag: str, text: str, return_one = True) -> str:
    # text = text.encode('unicode_escape').decode() # Remove this line it causes \\n to be added
    # Remove <tag>...</tag> blocks
    pattern = rf'<{tag}>.*?</{tag}>'
    # Find all matches, handle multiline
    matches = re.findall(pattern, text, flags=re.DOTALL)
    if return_one:
        match = matches[0]
        # Remove leading/trailing newlines
        cleaned = match.strip().replace(f'<{tag}>', '').replace(f'</{tag}>', '').strip()
        return cleaned
    else:
        return list(map(lambda x: x.strip().replace(f'<{tag}>', '').replace(f'</{tag}>', '').strip(), matches))
    
def replace_tag(tag: str, text: str, replacement: str) -> str:
    # text = text.encode('unicode_escape').decode()
    # replacement = replacement.encode('unicode_escape').decode()
    # Remove <tag>...</tag> blocks
    pattern = rf'<{tag}>.*?</{tag}>'
    # Replace all matches
    return re.sub(pattern, rf'<{tag}>{replacement}</{tag}>', text, flags=re.DOTALL)

def replace_tag_new(tag, text, new_content):
    pattern = re.compile(rf'<{tag}>(.*?)</{tag}>', re.DOTALL)
    def _replacement(match):
        # Return exactly <tag> + new_content + </tag> with no special handling of backslashes
        return f"<{tag}>{new_content}</{tag}>"
    return pattern.sub(_replacement, text)

def restructure_data(data: defaultdict):
    restructured_data = {}
    for k in data:
        if k in [
            "global",
            "contact",
        ]: 
            restructured_data[k] = data[k]
            continue

        if 'spatial' in k: # tendon connections
            restructured_data[k] = data[k]
            continue

        if 'tendon' in k: # Ready tendon
            restructured_data[k] = data[k]
            continue

        if 'velocity_actuator' in k: # velocity actuator
            restructured_data[k] = data[k]
        
        dot_separated = k.split('.')

        entity_name = dot_separated[0]

        if dot_separated[-1][:6] == 'spring': # Spring
            restructured_data[k] = data[k]
            continue    

        restructured_data[entity_name] = restructured_data.get(entity_name, {})
        restructured_data[entity_name]['.'.join(dot_separated[1:])] = data[k]

    return restructured_data

from scipy.spatial.transform import Rotation as R
import numpy as np

def rotated_axes_from_quaternion(quat: list | tuple):
    """
    Given a quaternion [x, y, z, w], return the rotated coordinate axes.
    Returns a 3x3 matrix where each row is the rotated X, Y, Z axis.
    """
    r = R.from_quat(quat)  # Note: scipy uses [x, y, z, w] order
    basis = np.eye(3)      # Standard X, Y, Z axes
    rotated = r.apply(basis)
    return rotated

import math, random

def unit_vector_from_angle(theta):
    return (math.cos(theta), math.sin(theta))

def add_vectors(a, b):
    return (a[0] + b[0], a[1] + b[1])

def scale_vector(v, scalar):
    return (v[0] * scalar, v[1] * scalar)

def generate_collision_pair(collision_point, t_c, speed_range, min_distance, existing_positions):
    
    attempt = 0
    while attempt < 10:
        # Pick incoming directions (randomized angle with ~pi difference)
        theta_i = random.uniform(0, 2 * math.pi)
        theta_j = random.uniform(0, 2 * math.pi)  # slightly oblique
        
        d_i = unit_vector_from_angle(theta_i)
        d_j = unit_vector_from_angle(theta_j)
        
        v_i_mag = random.uniform(*speed_range)
        v_j_mag = random.uniform(*speed_range)

        v_i = scale_vector(d_i, -v_i_mag * t_c)
        v_j = scale_vector(d_j, -v_j_mag * t_c)
        
        # Ensure minimum distance between the two objects
        new_t_c = max(min_distance / ((v_i[0] + v_j[0]) ** 2 + (v_i[1] + v_j[1]) ** 2) ** 0.5, t_c)

        v_i = scale_vector(d_i, -v_i_mag * new_t_c / t_c)
        v_j = scale_vector(d_j, -v_j_mag * new_t_c / t_c)

        pos_i = add_vectors(collision_point, v_i)
        pos_j = add_vectors(collision_point, v_j)

        vel_i = scale_vector(d_i, v_i_mag)
        vel_j = scale_vector(d_j, v_j_mag)

        if all(
                math.hypot(pos_i[0] - existing[0], pos_i[1] - existing[1])
                > min_distance
                for existing in existing_positions
            ) and all(
                math.hypot(pos_j[0] - existing[0], pos_j[1] - existing[1])
                > min_distance
                for existing in existing_positions
            ): break

        attempt += 1
    
    else:
        return None

    return {
        "positions": (pos_i, pos_j),
        "velocities": (vel_i, vel_j),
        "ids": None,  # to be assigned in main loop
    }

def parse_mtl_to_mujoco(mtl_filepath):
    """
    Parses a .mtl file and converts material definitions to MuJoCo XML strings.

    Args:
        mtl_filepath (str): Path to the .mtl file.

    Returns:
        list: A list of strings, each containing a MuJoCo <material> XML tag.
              Returns an empty list if the file cannot be read or no materials are found.
    """
    if not os.path.exists(mtl_filepath):
        # logging.error(f"MTL file not found: {mtl_filepath}")
        return []

    materials_xml = []
    current_material_props = None
    material_name = None

    try:
        with open(mtl_filepath, 'r') as mtl_file:
            for line_num, line in enumerate(mtl_file):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue  # Skip empty lines and comments

                parts = line.split()
                keyword = parts[0].lower() # Use lowercase for case-insensitivity

                # --- Start of a new material definition ---
                if keyword == 'newmtl':
                    # If we were processing a previous material, finalize and add it
                    if current_material_props is not None and material_name is not None:
                        xml_string = _convert_props_to_mujoco_xml(material_name, current_material_props)
                        if xml_string:
                            materials_xml.append(xml_string)

                    # Start a new material
                    if len(parts) > 1:
                        material_name = parts[1]
                        # Reset properties with defaults (MTL defaults, not MuJoCo yet)
                        current_material_props = {
                            'kd': [0.8, 0.8, 0.8],  # Default diffuse grey
                            'ks': [0.0, 0.0, 0.0],  # Default specular black
                            'ke': [0.0, 0.0, 0.0],  # Default emission black
                            'ns': 10.0,             # Default specular exponent (low)
                            'd': 1.0,               # Default opaque
                            'illum': 2,             # Default illum model (diffuse+specular)
                            'map_kd': None          # Default no diffuse map
                            # Add other properties like Ka, Ni, map_bump etc. if needed
                        }
                    else:
                        # logging.warning(f"Line {line_num+1}: 'newmtl' found without a name in {mtl_filepath}. Skipping.")
                        material_name = None
                        current_material_props = None
                    continue

                # --- Parse properties for the current material ---
                if current_material_props is None:
                    # Skip lines before the first 'newmtl'
                    continue

                try:
                    values = parts[1:]
                    if keyword == 'kd': # Diffuse color
                        if len(values) == 3:
                            current_material_props['kd'] = [float(v) for v in values]
                    elif keyword == 'ks': # Specular color
                         if len(values) == 3:
                            current_material_props['ks'] = [float(v) for v in values]
                         # else:
                             # logging.warning(f"Line {line_num+1}: Invalid Ks format in '{material_name}'. Expected 3 values, got {len(values)}. Using default.")
                    elif keyword == 'ke': # Emission color
                         if len(values) == 3:
                            current_material_props['ke'] = [float(v) for v in values]
                         # else:
                             # logging.warning(f"Line {line_num+1}: Invalid Ke format in '{material_name}'. Expected 3 values, got {len(values)}. Using default.")
                    elif keyword == 'ns': # Specular exponent (Shininess)
                        if len(values) == 1:
                            current_material_props['ns'] = float(values[0])
                        # else:
                             # logging.warning(f"Line {line_num+1}: Invalid Ns format in '{material_name}'. Expected 1 value, got {len(values)}. Using default.")
                    elif keyword == 'd': # Dissolve (Opacity)
                        if len(values) == 1:
                            current_material_props['d'] = float(values[0])
                        # else:
                             # logging.warning(f"Line {line_num+1}: Invalid d format in '{material_name}'. Expected 1 value, got {len(values)}. Using default.")
                    elif keyword == 'tr': # Transparency (alternative to d)
                        if len(values) == 1:
                            # Tr = 1 - d
                            current_material_props['d'] = 1.0 - float(values[0])
                        # else:
                             # logging.warning(f"Line {line_num+1}: Invalid Tr format in '{material_name}'. Expected 1 value, got {len(values)}. Using default d=1.0.")
                    elif keyword == 'illum': # Illumination model
                         if len(values) == 1:
                            current_material_props['illum'] = int(values[0])
                         # else:
                             # logging.warning(f"Line {line_num+1}: Invalid illum format in '{material_name}'. Expected 1 value, got {len(values)}. Using default.")
                    elif keyword == 'map_kd': # Diffuse texture map
                        if len(values) >= 1:
                            # Basic handling: just take the filename part
                            # More robust parsing might handle options like -s, -o etc.
                            filename = os.path.basename(values[-1]) # Get filename from path
                            # Store base name without extension as potential MuJoCo texture asset name
                            current_material_props['map_kd'] = os.path.splitext(filename)[0]
                        # else:
                            # logging.warning(f"Line {line_num+1}: Invalid map_Kd format in '{material_name}'. Expected filename.")
                    # Add parsing for other keywords like Ka, Ni, map_Ks, map_bump if needed

                except (ValueError, IndexError) as e:
                    pass
                    # logging.warning(f"Line {line_num+1}: Error parsing '{line}' in '{material_name}': {e}. Skipping property.")


            # --- Process the last material found in the file ---
            if current_material_props is not None and material_name is not None:
                xml_string = _convert_props_to_mujoco_xml(material_name, current_material_props)
                if xml_string:
                    materials_xml.append(xml_string)

    except IOError as e:
        # logging.error(f"Error reading MTL file {mtl_filepath}: {e}")
        return []

    return materials_xml


def _convert_props_to_mujoco_xml(name, props):
    """Helper function to convert parsed MTL props to a MuJoCo XML string."""
    try:
        # --- Convert MTL props to MuJoCo attributes ---
        # RGBA: Combine Kd (diffuse) and d (alpha/dissolve)
        kd = props.get('kd', [0.8, 0.8, 0.8])
        d = np.clip(props.get('d', 1.0), 0.0, 1.0) # Ensure alpha is in [0, 1]
        rgba_str = f"{kd[0]:.4f} {kd[1]:.4f} {kd[2]:.4f} {d:.4f}"

        # Specular: Average the Ks components
        ks_rgb = props.get('ks', [0.0, 0.0, 0.0])
        specular_val = np.clip(sum(ks_rgb) / 3.0, 0.0, 1.0) # Average and clamp

        # Shininess: Convert Ns (exponent) to MuJoCo's [0, 1] range
        ns_val = props.get('ns', 10.0)
        shininess_val = np.clip(ns_val / 128.0, 0.0, 1.0) # Convert Ns to [0,1] range and clamp

        # Emission: Use the maximum component of Ke as the scalar emission factor
        ke_rgb = props.get('ke', [0.0, 0.0, 0.0])
        emission_val = np.clip(max(ke_rgb), 0.0, 1.0) # Max component and clamp

        # Apply Illumination Model Effects
        illum = props.get('illum', 2)
        if illum < 2: # Illum 0 (color only) or 1 (ambient + diffuse) typically lack specular highlights
            specular_val = 0.0
            shininess_val = 0.0 # Less shiny if no specular component defined by illum

        # Texture
        texture_name = props.get('map_kd', None)
        texture_attr = f' texture="{texture_name}"' if texture_name else ""
        # Note: Using texture attribute means rgba acts as a modulator.
        # If using PBR textures (metallic, roughness), you'd omit this
        # basic 'texture' attribute and use <layer> elements instead.

        # --- Construct XML String ---
        # Use {:.4f} for consistent float formatting
        xml_string = (
            f'<material name="{name}"{texture_attr}'
            f' rgba="{rgba_str}"'
            f' specular="{specular_val:.4f}"'
            f' shininess="{shininess_val:.4f}"'
            f' emission="{emission_val:.4f}"'
            # Add reflectance="...", metallic="...", roughness="..." if needed/parsed
            '/>'
        )
        return xml_string

    except Exception as e:
        # logging.error(f"Error converting material '{name}' properties to XML: {e}")
        return None
    
def find_values(text, value = "mesh"):
    """
    Finds all occurrences of mesh="value" in a string and returns the values.

    Args:
    text: The input string to search within.

    Returns:
    A list of strings, where each string is the value inside the quotes
    following 'mesh='. Returns an empty list if no matches are found.
    """
    # Regex breakdown:
    # mesh="   : Matches the literal string 'mesh="'
    # ([^"]+) : Matches one or more characters that are NOT a double quote (").
    #           The parentheses () create a capturing group for this part.
    # "        : Matches the closing double quote.
    pattern = rf'{value}="([^"]+)"'

    # re.findall returns a list of all non-overlapping matches.
    # If the pattern has capturing groups, it returns a list of strings
    # corresponding to the captured group(s). In our case, it's group 1.
    matches = re.findall(pattern, text)

    return matches

def convert_list_to_natural_language(arr: list) -> str:
    """
    Converts a list of strings into a natural language sentence.
    
    Args:
        arr: List of strings to convert.
        
    Returns:
        A natural language sentence.
    """

    arr = list(map(str, arr))

    if not arr:
        return ""
    
    if len(arr) == 1:
        return arr[0]
    
    return ', '.join(arr[:-1]) + ' and ' + arr[-1]

def remove_empty_keys(data):
    """
    Recursively removes keys with empty values (empty lists or dictionaries) 
    from a nested dictionary. Scalar values (int, float, str) are kept.
    """
    if isinstance(data, dict):
        new_data = {}
        for key, value in data.items():
            cleaned = remove_empty_keys(value)
            # Keep if it's a non-empty container OR it's not a container at all (scalar)
            if isinstance(cleaned, (dict, list)):
                if cleaned:
                    new_data[key] = cleaned
            elif cleaned is not None:
                new_data[key] = cleaned
        return new_data
    elif isinstance(data, list):
        # Recursively clean list items
        return [remove_empty_keys(item) for item in data if item is not None]
    return data

import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return json.JSONEncoder.default(self, obj)