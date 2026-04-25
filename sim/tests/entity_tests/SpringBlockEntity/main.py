from sim.scene import parse_scene, Scene
import os, pathlib, hydra
import math

from omegaconf import DictConfig, OmegaConf
import ipdb
import numpy as np

from recorder.recorder import Recorder, SCENE_TYPE_TO_CATEGORY_MAP

@hydra.main(config_path="../../../../config", config_name="config")
def main(cfg: DictConfig):
    script_path = pathlib.Path(__file__).parent
    scene = parse_scene(os.path.join(script_path, 'main.yaml'))
    
    xml_output = scene.to_xml()

    description = scene.get_nlq()

    # print(xml_output)
    # print(description)

    with open(os.path.join(script_path, 'main.xml'), 'w') as f:
        f.write(xml_output)
    with open(os.path.join(script_path, 'main.txt'), 'w') as f:
        f.write(description)

    cfg = cfg.recorder
    
    scene_type = scene.tag
    scene_folder = str(script_path)

    category = [k for k in SCENE_TYPE_TO_CATEGORY_MAP if scene_type in SCENE_TYPE_TO_CATEGORY_MAP[k]]
    if len(category) == 1:
        category = category[0]
    else: category = None

    recorder = Recorder(scene, cfg, scene_folder, category=category)
    print(f"[Recorder Init] Scene folder set to: {scene_folder}")
    data, metadata, instability = recorder.simulate()


    ##### INFO #####
    # data keys:
    #   'global'  — time, gravity
    #   '<body_name>' — position, velocity_linear, kinetic_energy_linear,
    #                    potential_energy, mass, acceleration_linear, net_force_linear, ...
    #   '<tendon_name>' — length, velocity, force, stiffness
    # The single mass body name: "springblockentity_0.tray_mass"
    # Tendon names: "springblockentity_0.spring-{i}.spring"


    ## ASSERTION CODE (to validate if the physics is correct)

    # ---- Setup ----
    entity = scene.entities[0]
    body_name = f"{entity.name}.tray_mass"
    num_springs = len(entity.connecting_angles)

    N = len(data["global"]["time"])
    times = np.array(data["global"]["time"])
    g_mag = abs(data["global"]["gravity"][0])  # magnitude of gravity

    mass = data[body_name]["mass"][0]
    position = np.array(data[body_name]["position"])        # (N, 3)
    velocity = np.array(data[body_name]["velocity_linear"]) # (N, 3)
    KE       = np.array(data[body_name]["kinetic_energy_linear"])  # (N,)

    # Spring / tendon info
    spring_data = {}
    for i in range(num_springs):
        tendon_name = f"{entity.name}.spring-{i}.spring"
        spring_data[i] = {
            "length":    np.array(data[tendon_name]["length"]),     # (N,)
            "velocity":  np.array(data[tendon_name]["velocity"]),   # (N,)
            "force":     np.array(data[tendon_name]["force"]),      # (N,)
            "stiffness": np.array(data[tendon_name]["stiffness"]),  # (N,)
            "natural_length": entity.original_lengths[i],
            "k":         entity.stiffnesses[i],
            "angle_deg": entity.connecting_angles[i],
        }

    print(f"\n{'='*60}")
    print(f"PHYSICS ASSERTIONS — {N} timesteps recorded")
    print(f"  Mass: {mass} kg, Gravity: {g_mag} m/s²")
    print(f"  Number of springs: {num_springs}")
    for i in range(num_springs):
        sd = spring_data[i]
        print(f"    Spring {i}: angle={sd['angle_deg']}°, L0={sd['natural_length']}m, "
              f"k_yaml={sd['k']} N/m, k_model={sd['stiffness'][0]} N/m")
    print(f"{'='*60}\n")


    # ================================================================
    # 1. TOTAL MECHANICAL ENERGY CONSERVATION
    #    E_total = KE + gravitational PE + elastic PE (all springs)
    #    With no damping, E_total should be constant throughout.
    #    Gravitational PE = m * g * z  (z is height, position index 2)
    #    Elastic PE = 0.5 * k * (L - L0)^2  for each spring
    # ================================================================
    print("--- [1] Checking total mechanical energy conservation ---")

    # Gravitational PE
    grav_PE = mass * g_mag * position[:, 2]  # z-coordinate

    # Elastic PE from each spring
    elastic_PE = np.zeros(N)
    for i in range(num_springs):
        sd = spring_data[i]
        k_actual = sd["stiffness"]     # stiffness from model (may differ from yaml!)
        L = sd["length"]
        L0 = sd["natural_length"]
        extension = L - L0
        elastic_PE += 0.5 * k_actual * extension**2

    total_E = KE + grav_PE + elastic_PE

    E_ref = total_E[0]
    E_drift = np.max(np.abs(total_E - E_ref))
    print(f"  Initial: KE={KE[0]:.6f}, grav_PE={grav_PE[0]:.6f}, elastic_PE={elastic_PE[0]:.6f}")
    print(f"  Initial total energy: {E_ref:.6f}")
    print(f"  Max energy drift: {E_drift:.6e}")
    print(f"  Relative drift: {E_drift / max(abs(E_ref), 1e-12):.6e}")
    assert E_drift < 1e-1, (
        f"Total mechanical energy not conserved: max drift = {E_drift:.6e}"
    )
    print("  ✓ Total mechanical energy conserved.\n")


    # ================================================================
    # 2. SPRING FORCE CONSISTENCY
    #    The recorded tendon force should match k * (L - L0).
    #    MuJoCo's tendon force = stiffness * (length - springlength)
    #    + damping * velocity.
    #    With damping=0, force should equal stiffness * extension.
    # ================================================================
    print("--- [2] Checking spring force consistency ---")
    for i in range(num_springs):
        sd = spring_data[i]
        k_actual = sd["stiffness"]
        L = sd["length"]
        L0 = sd["natural_length"]
        recorded_force = sd["force"]
        velocity_tendon = sd["velocity"]

        # Expected force: k * (L - L0) + damping * v
        # With damping = 0: k * (L - L0)
        expected_force = k_actual * (L - L0)

        force_error = np.max(np.abs(recorded_force - expected_force))
        force_scale = np.max(np.abs(expected_force)) + 1e-12

        print(f"  Spring {i} (angle={sd['angle_deg']}°):")
        print(f"    Max |recorded - expected| force: {force_error:.6e}")
        print(f"    Relative error: {force_error / force_scale:.6e}")
        assert force_error / force_scale < 5e-2, (
            f"Spring {i} force inconsistency: max error = {force_error:.6e}"
        )
    print("  ✓ Spring forces consistent with Hooke's law.\n")


    # ================================================================
    # 3. NEWTON'S SECOND LAW (F = m·a)
    #    The net force on the mass should equal m * acceleration.
    #    Net force includes:
    #      - Gravity: (0, 0, -m*g) in world frame
    #      - Spring forces: along each spring direction
    #    We check that recorded net_force ≈ m * recorded acceleration.
    # ================================================================
    print("--- [3] Checking Newton's second law (F = m·a) ---")
    acc = np.array(data[body_name]["acceleration_linear"])   # (N, 3)
    net_force = np.array(data[body_name]["net_force_linear"])  # (N, 3)

    # Skip first few and last few timesteps due to initialization artifacts
    check_start = 10
    check_end = N - 10
    if check_end <= check_start:
        check_start, check_end = 0, N

    ma = mass * acc[check_start:check_end]
    F_net = net_force[check_start:check_end]

    F_ma_error = np.max(np.abs(F_net - ma))
    F_scale = np.max(np.abs(F_net)) + 1e-12

    print(f"  Checking timesteps {check_start} to {check_end}")
    print(f"  Max |F_net - m*a|: {F_ma_error:.6e}")
    print(f"  Relative error: {F_ma_error / F_scale:.6e}")
    assert F_ma_error / F_scale < 5e-2, (
        f"Newton's second law violation: max error = {F_ma_error:.6e}"
    )
    print("  ✓ Newton's second law verified.\n")


    # ================================================================
    # 4. SPRING LENGTHS vs MASS POSITION CONSISTENCY
    #    Each spring connects a fixed endpoint to the tray mass.
    #    The tendon length should equal |pos_mass - pos_fixed_endpoint|.
    #    Fixed endpoints are computed from slope and natural length:
    #      endpoint = R(-slope) · (-L0 - padding, 0, 0)
    #    where R rotates around Y-axis by -slope degrees.
    # ================================================================
    print("--- [4] Checking spring length vs mass position consistency ---")

    # The tray mass position in body-local frame is (0,0,0), but the body
    # is placed at the entity position. The mass body pos relative to entity:
    entity_pos = np.array(entity.pos)

    # The body position from data is in the world frame
    for i in range(num_springs):
        sd = spring_data[i]
        slope_rad = math.radians(sd["angle_deg"])

        # Fixed endpoint position (in entity frame, then shifted to world)
        # From create_springs: axis_angles = (0, -slope, 0), then
        # endpoint = Frame.rel2global((-L0 - DEFAULT_MASS_SIZE, 0, 0))
        # DEFAULT_MASS_SIZE = 0.1 (typically)
        # The rotation is around Y-axis by -slope degrees
        L0 = sd["natural_length"]
        distance = entity.connecting_distances[i]

        # Rotation of (-L0 - padding, 0, 0) by -slope about Y:
        # R_y(theta) * v = (v_x*cos(theta) + v_z*sin(theta), v_y, -v_x*sin(theta) + v_z*cos(theta))
        # Here theta = -slope_rad, v = (-L0 - padding, 0, 0)
        cos_s = math.cos(-slope_rad)
        sin_s = math.sin(-slope_rad)
        v_x = -distance
        fixed_x = v_x * cos_s + entity_pos[0]
        fixed_y = 0 + entity_pos[1]
        fixed_z = -v_x * sin_s + entity_pos[2]
        fixed_endpoint = np.array([fixed_x, fixed_y, fixed_z])

        # Compute expected tendon lengths from mass position
        mass_pos_world = position  # (N, 3) — world frame positions of tray_mass
        diff = mass_pos_world - fixed_endpoint[None, :]
        computed_lengths = np.linalg.norm(diff, axis=1)

        recorded_lengths = sd["length"]
        length_error = np.max(np.abs(computed_lengths - recorded_lengths))
        length_scale = np.max(recorded_lengths) + 1e-12

        print(f"  Spring {i} (angle={sd['angle_deg']}°):")
        print(f"    Fixed endpoint (world): ({fixed_x:.4f}, {fixed_y:.4f}, {fixed_z:.4f})")
        print(f"    Max |computed_L - recorded_L|: {length_error:.6e}")
        print(f"    Relative error: {length_error / length_scale:.6e}")
        # Generous tolerance — the padding/offset and site positions may introduce
        # small discrepancies vs this simplified calculation
        assert length_error / length_scale < 0.1, (
            f"Spring {i} length vs position inconsistency: max error = {length_error:.6e}"
        )
    print("  ✓ Spring lengths match mass position.\n")


    # ================================================================
    # 5. VELOCITY-POSITION CONSISTENCY
    #    Numerically, v(t) ≈ (pos(t+1) - pos(t)) / dt.
    #    Check that recorded velocity is consistent with position data.
    # ================================================================
    print("--- [5] Checking velocity-position consistency ---")
    dt = times[1] - times[0]  # assume uniform timestep
    # Finite-difference velocity from positions
    vel_fd = np.diff(position, axis=0) / dt  # (N-1, 3)
    # Compare with recorded velocity (use midpoint: average of vel[t] and vel[t+1])
    vel_mid = 0.5 * (velocity[:-1] + velocity[1:])

    # Skip edges
    cs = 10
    ce = min(N - 1, N - 10)
    if ce <= cs:
        cs, ce = 0, N - 1

    vel_error = np.max(np.abs(vel_fd[cs:ce] - vel_mid[cs:ce]))
    vel_scale = np.max(np.abs(vel_mid[cs:ce])) + 1e-12

    print(f"  dt = {dt:.6f}")
    print(f"  Max |v_fd - v_recorded_mid|: {vel_error:.6e}")
    print(f"  Relative error: {vel_error / vel_scale:.6e}")
    assert vel_error / vel_scale < 5e-2, (
        f"Velocity-position inconsistency: max error = {vel_error:.6e}"
    )
    print("  ✓ Velocity consistent with position.\n")


    print(f"{'='*60}")
    print("ALL PHYSICS ASSERTIONS PASSED ✓")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()

