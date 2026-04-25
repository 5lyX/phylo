import numpy as np
import ipdb
import mujoco

st = ipdb.set_trace

def prune_spikes(time_series, std_threshold=1):
    """
    Identifies and removes spikes in acceleration using a moving standard deviation.
    
    Args:
        time_series (numpy.ndarray): A 1D array of acceleration values over time.
        window_size (int): Number of past values to compute the rolling mean and std.
        std_threshold (float): Number of standard deviations beyond which a point is considered a spike.

    Returns:
        numpy.ndarray: Boolean mask where True means the data point is valid.
    """

    # Ensure time_series is a NumPy array
    time_series = np.asarray(time_series)

    window_size = estimate_window_size(time_series)
    
    # Compute rolling mean and standard deviation using convolution
    kernel = np.ones(window_size) / window_size
    rolling_mean = np.convolve(time_series, kernel, mode='valid')
    rolling_std = np.sqrt(np.convolve((time_series - np.mean(time_series))**2, kernel, mode='valid'))

    # Pad to match original array length
    pad_size = window_size - 1
    rolling_mean = np.pad(rolling_mean, (pad_size, 0), mode='edge')
    rolling_std = np.pad(rolling_std, (pad_size, 0), mode='edge')

    # Apply minimum standard deviation to avoid over-pruning
    rolling_std = np.maximum(rolling_std, 1e-4)

    # Identify spikes
    valid_mask = np.abs(time_series - rolling_mean) <= std_threshold * rolling_std

    # Transience period:
    valid_mask[:window_size] = True
    
    return valid_mask

def estimate_window_size(time_series, threshold=0.1, max_lag=None):
    """
    Estimates an appropriate window size for smoothing based on the decorrelation time
    of the time series.
    
    Args:
        time_series (numpy.ndarray): 1D array of time_series values.
        threshold (float): The autocorrelation value at which we define the timescale.
        max_lag (int, optional): Maximum lag to consider. Defaults to len(time_series)//2.

    Returns:
        int: Estimated window size.
    """
    time_series = np.asarray(time_series)
    n = len(time_series)
    max_lag = max_lag or n // 2  # Limit max lag to half the dataset

    # Compute autocorrelation function (ACF)
    mean_val = np.mean(time_series)
    var_acc = np.var(time_series)
    autocorr = np.correlate(time_series - mean_val, time_series - mean_val, mode='full') / (var_acc * n)
    autocorr = autocorr[n-1:]  # Take the positive lags

    # Find first lag where autocorrelation drops below threshold
    decorrelation_lag = np.where(autocorr < threshold)[0]
    window_size = decorrelation_lag[0] if len(decorrelation_lag) > 0 else max_lag // 10  # Fallback

    return max(100, min(window_size, max_lag // 5))  # Clamp within reasonable limits

def get_geom_speed(model, data, geom_name):
    """Returns the speed of a geom."""
    geom_vel = np.zeros(6)
    geom_type = mujoco.mjtObj.mjOBJ_GEOM
    geom_id = data.geom(geom_name).id
    mujoco.mj_objectVelocity(model, data, geom_type, geom_id, geom_vel, 0)
    return np.linalg.norm(geom_vel)

def add_visual_capsule(scene, point1, point2, radius, rgba):
    """Adds one capsule to an mjvScene."""
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1  # increment ngeom
    # initialise a new capsule, add it to the scene using mjv_connector
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom-1],
                    mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                    np.zeros(3), np.zeros(9), rgba.astype(np.float32))
    mujoco.mjv_connector(scene.geoms[scene.ngeom-1],
                    mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                    point1, point2)

def estimate_trail_radius_from_geom(model, geom_name):
    """
    Fast, heuristic radius estimate for drawing trail capsules from geom size.
    
    Args:
        model: MjModel
        geom_name: str

    Returns:
        radius (float): approximate trail radius
    """
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    geom_size = model.geom_size[geom_id]
    geom_type = model.geom_type[geom_id]

    # Typical heuristics per shape
    if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        r = geom_size[0]
        return r
    elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
        r, h = geom_size
        return r
    elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        sx, sy, sz = geom_size
        return np.mean([sy, sz])  # cross-section orthogonal to x (forward)
    elif geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        rx, ry, rz = geom_size
        return np.mean([ry, rz])
    else:
        # fallback: average of all size dimensions
        return np.mean(geom_size)

def add_visual_region(scene, center, shape, size, rgba):
    """Add a translucent field region geom."""
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1
    geom = scene.geoms[scene.ngeom-1]

    if shape == "cylinder":
        # size = [radius, height]
        mujoco.mjv_initGeom(geom,
                            mujoco.mjtGeom.mjGEOM_CYLINDER,
                            np.array([size[0], size[1]/2, size[0]]),  # scale
                            center,
                            np.eye(3).flatten(),  # identity rotation
                            rgba.astype(np.float32))
    elif shape == "box":
        # size = [half_width, half_height, half_depth]
        mujoco.mjv_initGeom(geom,
                            mujoco.mjtGeom.mjGEOM_BOX,
                            np.array(size),
                            center,
                            np.eye(3).flatten(),
                            rgba.astype(np.float32))

def draw_regions(model, data, scn, field_configs):
    """
    Draws regions in the scene based on field_config.
    
    Args:
        model: MjModel
        data: MjData
        scn: MjvScene
        field_configs: list with EM region info
    """
    alpha = 0.2
    for idx, field_config in enumerate(field_configs):
        field_type = field_config.get("field_type", "electric")
        rgba = np.array([1, 0.5, 0.5, alpha] if field_type == "electric" else [0.5, 0.5, 1, alpha], dtype=np.float32) 
        
        field_shape = field_config.get("field_shape", None)
        if field_shape is None or "field_size" not in field_config:
            field_shape = "box"
            field_size = [100, 100]
            field_position = [0, 0]
        else:
            field_size = field_config.get("field_range")
            field_position = field_config.get("field_position")

        field_position = np.array([*field_position, 0])
        field_size = np.array([*field_size, 8])
        if field_shape == "circle": field_shape = "cylinder"

        add_visual_region(scn, field_position, field_shape, field_size, rgba)
         
def draw_trails(scn, positions, vels, times, forward, idx = None, max_len = 4000, r_circ = 0.0015, body_color = None, body_name = None):
    """Draw position trace with vivid colors that fade to transparent. All trails are consistently thin regardless of body size."""
    MAX_LEN = max_len
    if len(positions) > 1:
        start = -2 if idx is None else (-1 - len(positions) + idx)
        indices = list(range(start, -min(len(positions), scn.maxgeom, MAX_LEN), -1))

        # FORCE vivid colors - ignore body_color completely for now to ensure colors work
        vivid_colors = [
            np.array([1.0, 0.1, 0.3]),  # Vivid red/pink
            np.array([0.1, 0.8, 1.0]),  # Vivid cyan
            np.array([1.0, 0.7, 0.1]),  # Vivid gold
            np.array([0.2, 1.0, 0.1]),  # Vivid green
            np.array([1.0, 0.3, 0.1]),  # Vivid orange
            np.array([0.7, 0.1, 1.0]),  # Vivid purple
            np.array([1.0, 0.1, 0.8]),  # Vivid magenta
            np.array([0.1, 1.0, 0.6]),  # Vivid emerald
        ]
        
        # Choose color based on body_name to ensure consistent colors per trail
        if body_name is not None:
            # Use body_name for consistent color selection
            import hashlib
            hash_val = int(hashlib.md5(body_name.encode()).hexdigest(), 16)
            color_index = hash_val % len(vivid_colors)
            trail_color = vivid_colors[color_index]
            
            # Optional debug logging (uncomment to enable)
            # import logging
            # logging.basicConfig(filename='trail_debug.log', level=logging.DEBUG)
            # logging.debug(f"Body: {body_name}, Color index: {color_index}, Color: {trail_color}")
        else:
            # Fallback: use first color if no body_name provided
            trail_color = vivid_colors[0]

        for i in indices:
            speed = vels[i]
            speed = np.linalg.norm(speed)
            
            # Calculate fade ratio: 0 (oldest) to 1 (newest)
            fade_ratio = (MAX_LEN + 1 + i) / MAX_LEN
            
            # NEW: Start semi-transparent immediately and fade faster
            # Use very steep exponential decay that starts from lower transparency
            transparency_curve = fade_ratio ** 5.0  # Even steeper curve
            
            # Start from 0.5 (semi-transparent) and fade to 0.002 (almost invisible)
            alpha = 0.002 + transparency_curve * 0.498
            
            # Use the chosen trail color - ensure it's bright and vivid
            rgba = np.array([
                trail_color[0],  # Vivid red component
                trail_color[1],  # Vivid green component
                trail_color[2],  # Vivid blue component
                alpha            # Fast fading transparency
            ], dtype=np.float32)

            if speed < 5e-2: continue 
            
            offset = - vels[i]/(speed+1e-4) * 0.015
            
            # Back to original radius calculation with adjustment for large bodies
            base_radius = r_circ * 0.3  # Thinner base radius
            speed_factor = (1 + np.clip(speed/2, 0, 1)) * 0.5  # Reduced speed influence
            # Age factor for thickness variation
            age_factor = 0.6 + transparency_curve * 0.4  # Age factor from 0.6 to 1.0
            
            radius = base_radius * speed_factor * age_factor
            
            point1 = positions[i] + offset
            point2 = positions[i+1] + offset
            add_visual_capsule(scn, point1, point2, radius, rgba)

def get_body_color(model: mujoco.MjModel, body_name: str) -> np.ndarray:
    """
    Get the visual color of a body from its geom.
    
    Args:
        model: MjModel
        body_name: name of the body
        
    Returns:
        RGBA color array, or default bright color if not found
    """
    try:
        # Try to find the geom associated with the body
        geom_name = body_name + ".geom"
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id >= 0:
            # Get the material ID for this geom
            mat_id = model.geom_matid[geom_id]
            if mat_id >= 0:
                # Get RGBA from material
                rgba = model.mat_rgba[mat_id]
                return rgba.copy()
            else:
                # Use geom rgba if no material
                rgba = model.geom_rgba[geom_id]
                return rgba.copy()
    except:
        pass
    
    # Default saturated celestial colors for different body types
    if "sphere" in body_name.lower():
        # Rich, saturated celestial colors for planets/spheres
        colors = [
            np.array([0.9, 0.6, 0.2, 1.0]),  # Deep golden
            np.array([0.2, 0.6, 0.9, 1.0]),  # Deep blue  
            np.array([0.9, 0.2, 0.4, 1.0]),  # Deep pink/red
            np.array([0.3, 0.8, 0.2, 1.0]),  # Deep green
            np.array([0.9, 0.4, 0.1, 1.0]),  # Deep orange
            np.array([0.6, 0.2, 0.8, 1.0]),  # Deep purple
        ]
        import hashlib
        hash_val = int(hashlib.md5(body_name.encode()).hexdigest(), 16)
        return colors[hash_val % len(colors)]
    elif "particle" in body_name.lower():
        return np.array([0.2, 0.8, 0.9, 1.0])  # Deep cyan for particles
    elif "rocket" in body_name.lower():
        return np.array([0.9, 0.1, 0.1, 1.0])  # Deep red for rockets
    elif "ball" in body_name.lower():
        return np.array([0.9, 0.8, 0.1, 1.0])  # Deep yellow for balls
    else:
        return np.array([0.7, 0.7, 0.9, 1.0])  # Light blue-white default

def get_body_forward_vector(model: mujoco.MjModel, data: mujoco.MjData, body_name_or_id) -> np.ndarray:
    """
    Calculates the forward direction vector (local +x axis) of a body
    expressed in the global world frame.

    Args:
        model: The MuJoCo MjModel object.
        data: The MuJoCo MjData object containing the current simulation state.
              (Ensure mj_step or mj_forward has been called to update data).
        body_name_or_id: The name (string) or ID (integer) of the body.

    Returns:
        A NumPy array of shape (3,) representing the body's forward vector
        (its local +x axis) in global coordinates.

    Raises:
        ValueError: If the body name or ID is invalid.
        TypeError: If body_name_or_id is not a string or integer.
    """
    if isinstance(body_name_or_id, str):
        # Get body ID from name
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name_or_id)
        if body_id == -1:
            raise ValueError(f"Body with name '{body_name_or_id}' not found in the model.")
    elif isinstance(body_name_or_id, int):
        # Use body ID directly
        body_id = body_name_or_id
        if not (0 <= body_id < model.nbody):
             raise ValueError(f"Invalid body ID: {body_id}. Model has {model.nbody} bodies.")
    else:
        raise TypeError("body_name_or_id must be a string or an integer.")

    # --- Access the Body's Rotation Matrix ---
    # data.xmat contains the 3x3 rotation matrices for all bodies, stored flat (row-major).
    # It transforms vectors from the global frame to the body's local frame.
    # However, its *columns* represent the body's local axes (x, y, z) in global coordinates.

    # Extract the 9 elements for the specific body
    body_rotation_flat = data.xmat[body_id]

    # Reshape into a 3x3 matrix
    rotation_matrix = body_rotation_flat.reshape(3, 3)

    # --- Get the Forward Vector ---
    # The first column of the rotation matrix represents the body's local +x axis
    # (which we define as 'forward') expressed in the global coordinate system.
    forward_vector_global = rotation_matrix[:, -1] # Select the first column

    # Return a copy to prevent accidental modification of the underlying data array
    return np.copy(forward_vector_global)

def unit_cos(t):
    return 0.5 - np.cos(np.pi * np.clip(t, 0, 1)) / 2