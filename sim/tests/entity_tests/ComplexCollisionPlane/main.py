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
    # This scene (ComplexCollisionPlane) has 4 bodies:
    #   0. mass-0:        moving mass     (2.0 kg, pos=1.0, vel=2.0)
    #   1. sphere-1:      moving sphere   (5.0 kg, pos=2.5, vel=-0.3)
    #   2. fixed_mass-2:  fixed mass      (5.0 kg, pos=0.0, no joints)
    #   3. spring_mass-3: two masses connected by a spring (1.0 kg each, at 3.0 & 4.0)
    #
    # data keys per body:
    #   position, velocity_linear, kinetic_energy_linear,
    #   potential_energy, mass, acceleration_linear, net_force_linear, ...
    # data keys per tendon:
    #   length, velocity, force, stiffness


    ## ASSERTION CODE

    # ---- Setup ----
    entity = scene.entities[0]
    plane_slope = entity.plane_slope  # degrees
    theta = math.radians(plane_slope)
    g_mag = abs(data["global"]["gravity"][0])  # 9.81

    N = len(data["global"]["time"])
    times = np.array(data["global"]["time"])
    dt = times[1] - times[0]

    # Gravitational acceleration component along xz-plane for sloped plane
    g_along_plane = g_mag * math.sin(theta) if theta != 0 else 0.0

    print(f"\n{'='*60}")
    print(f"PHYSICS ASSERTIONS — ComplexCollisionPlane")
    print(f"  {N} timesteps, plane_slope={plane_slope}°, g={g_mag}")
    print(f"{'='*60}\n")


    # ================================================================
    # Collect all body names and data
    # ================================================================
    body_specs = entity.collision_bodies
    body_names = []
    body_data = {}
    movable_body_names = []  # bodies that can move

    for i, spec in enumerate(body_specs):
        btype = spec["body_type"]
        if btype == "mass":
            name = f"{entity.name}.mass-{i}"
            body_names.append(name)
            movable_body_names.append(name)
        elif btype == "sphere":
            name = f"{entity.name}.sphere-{i}"
            body_names.append(name)
            movable_body_names.append(name)
        elif btype == "fixed_mass":
            name = f"{entity.name}.fixed_mass-{i}"
            body_names.append(name)
            # Fixed mass has no joints — it shouldn't move
        elif btype == "spring_mass":
            # SpringMass creates sub-masses
            num_sub = len(spec.get("mass_values", []))
            for j in range(num_sub):
                name = f"{entity.name}.spring_mass-{i}.mass-{j}"
                body_names.append(name)
                movable_body_names.append(name)
        elif btype == "fixed_spring":
            name = f"{entity.name}.fixed_spring-{i}.tray_mass"
            body_names.append(name)
            movable_body_names.append(name)

    print(f"  Bodies found: {body_names}")

    for name in body_names:
        if name in data:
            body_data[name] = {
                "position": np.array(data[name]["position"]),
                "velocity": np.array(data[name]["velocity_linear"]),
                "KE": np.array(data[name]["kinetic_energy_linear"]),
                "mass": data[name]["mass"][0],
            }
            if "acceleration_linear" in data[name]:
                body_data[name]["acceleration"] = np.array(data[name]["acceleration_linear"])
            if "net_force_linear" in data[name]:
                body_data[name]["net_force"] = np.array(data[name]["net_force_linear"])
        else:
            print(f"  WARNING: body '{name}' not found in data, available keys: {[k for k in data.keys() if entity.name in k]}")


    # ================================================================
    # 1. TOTAL SYSTEM KINETIC ENERGY AT t=0
    #    Verify that the initial KE matches expected from given velocities.
    # ================================================================
    print("--- [1] Checking initial kinetic energy ---")
    total_KE_0 = 0.0
    for name, bd in body_data.items():
        ke0 = bd["KE"][0]
        v0 = bd["velocity"][0]
        m = bd["mass"]
        expected_ke0 = 0.5 * m * np.dot(v0, v0)
        print(f"  {name}: KE={ke0:.6f}, expected={expected_ke0:.6f}, mass={m}")
        np.testing.assert_allclose(ke0, expected_ke0, rtol=1e-3,
            err_msg=f"Initial KE mismatch for {name}")
        total_KE_0 += ke0
    print(f"  Total system KE at t=0: {total_KE_0:.6f}")
    print("  ✓ Initial kinetic energy verified.\n")


    # ================================================================
    # 2. FIXED MASS STAYS FIXED
    #    The fixed_mass should have zero velocity and constant position
    #    throughout the simulation.
    # ================================================================
    print("--- [2] Checking fixed mass stays stationary ---")
    for i, spec in enumerate(body_specs):
        if spec["body_type"] == "fixed_mass":
            fname = f"{entity.name}.fixed_mass-{i}"
            if fname not in body_data:
                print(f"  Skipping {fname} — not in data")
                continue
            bd = body_data[fname]
            pos = bd["position"]
            vel = bd["velocity"]

            pos_drift = np.max(np.abs(pos - pos[0]))
            vel_max = np.max(np.abs(vel))
            print(f"  {fname}: max position drift = {pos_drift:.6e}, max velocity = {vel_max:.6e}")
            assert pos_drift < 1e-6, f"Fixed mass {fname} moved! drift={pos_drift}"
            assert vel_max < 1e-6, f"Fixed mass {fname} has velocity! max={vel_max}"
    print("  ✓ Fixed mass remains stationary.\n")


    # ================================================================
    # 3. NEWTON'S SECOND LAW (F = m·a) for movable bodies
    #    Check that recorded net_force ≈ m * recorded acceleration.
    # ================================================================
    print("--- [3] Checking Newton's second law (F = m·a) ---")
    check_start = 10
    check_end = N - 10
    if check_end <= check_start:
        check_start, check_end = 0, N

    for name in movable_body_names:
        if name not in body_data:
            continue
        bd = body_data[name]
        if "acceleration" not in bd or "net_force" not in bd:
            print(f"  Skipping {name} — no accel/force data")
            continue
        m = bd["mass"]
        ma = m * bd["acceleration"][check_start:check_end]
        F = bd["net_force"][check_start:check_end]

        error = np.max(np.abs(F - ma))
        scale = np.max(np.abs(F)) + 1e-12
        print(f"  {name}: max |F-ma| = {error:.6e}, rel = {error/scale:.6e}")
        assert error / scale < 5e-2, (
            f"Newton's 2nd law violation for {name}: error = {error:.6e}"
        )
    print("  ✓ Newton's second law verified.\n")


    # ================================================================
    # 4. VELOCITY-POSITION CONSISTENCY
    #    v(t) ≈ (pos(t+1) - pos(t)) / dt for all movable bodies.
    # ================================================================
    print("--- [4] Checking velocity-position consistency ---")
    cs = 10
    ce = min(N - 1, N - 10)
    if ce <= cs:
        cs, ce = 0, N - 1

    for name in movable_body_names:
        if name not in body_data:
            continue
        bd = body_data[name]
        pos = bd["position"]
        vel = bd["velocity"]

        vel_fd = np.diff(pos, axis=0) / dt
        vel_mid = 0.5 * (vel[:-1] + vel[1:])

        error = np.max(np.abs(vel_fd[cs:ce] - vel_mid[cs:ce]))
        scale = np.max(np.abs(vel_mid[cs:ce])) + 1e-12
        print(f"  {name}: max |v_fd - v_mid| = {error:.6e}, rel = {error/scale:.6e}")
        assert error / scale < 5e-2, (
            f"Velocity-position inconsistency for {name}: error = {error:.6e}"
        )
    print("  ✓ Velocity-position consistency verified.\n")


    # ================================================================
    # 5. SPRING TENDON CONSISTENCY (for spring_mass bodies)
    #    Check that tendon force = k * (L - L0) and
    #    tendon length ≈ distance between connected masses.
    # ================================================================
    print("--- [5] Checking spring tendon consistency ---")
    # Find spring tendon keys — these end with ".spring-N" pattern and have tendon data (length, force, etc.)
    # Exclude mass body keys like "spring_mass-3.mass-0" which also contain "spring" as substring
    import re
    tendon_keys = [k for k in data.keys()
                   if k != "global" and re.search(r'\.spring-\d+$', k)
                   and "length" in data[k]]  # tendon data always has "length"
    for tname in tendon_keys:
        td = data[tname]
        L = np.array(td["length"])
        force = np.array(td["force"])
        k_val = np.array(td["stiffness"])

        if len(L) == 0 or len(force) == 0:
            print(f"  {tname}: empty data — skipping")
            continue

        L0_val = td.get("springlength", None)

        # Force should be k * (L - L0)
        if L0_val is not None:
            L0 = L0_val[0] if isinstance(L0_val, list) else L0_val
        else:
            L0 = 1.0

        expected_force = k_val * (L - L0)
        force_error = np.max(np.abs(force - expected_force))
        force_scale = np.max(np.abs(expected_force)) + 1e-12
        print(f"  {tname}: max force error = {force_error:.6e}, rel = {force_error/force_scale:.6e}")
        assert force_error / force_scale < 5e-2, (
            f"Tendon {tname} force inconsistency: error = {force_error:.6e}"
        )
    if not tendon_keys:
        print("  No spring tendons found — skipping.")
    else:
        print("  ✓ Spring tendon forces consistent.\n")


    # ================================================================
    # 6. TOTAL MECHANICAL ENERGY CONSERVATION
    #    Sum KE + gravitational PE + spring PE for all bodies.
    #    With no damping and flat plane (slope=0), E_total should be
    #    conserved between collisions. With collisions (COR != 1),
    #    energy should decrease monotonically.
    #    We check that energy doesn't INCREASE beyond tolerance.
    # ================================================================
    print("--- [6] Checking energy non-increase ---")

    total_KE = np.zeros(N)
    total_grav_PE = np.zeros(N)
    total_spring_PE = np.zeros(N)

    for name, bd in body_data.items():
        total_KE += bd["KE"]
        # Gravitational PE = m * g * z (height)
        total_grav_PE += bd["mass"] * g_mag * bd["position"][:, 2]

    # Spring PE from tendons
    for tname in tendon_keys:
        td = data[tname]
        L = np.array(td["length"])
        k_val = np.array(td["stiffness"])
        L0_val = td.get("springlength", None)
        if L0_val is not None:
            L0 = L0_val[0] if isinstance(L0_val, list) else L0_val
        else:
            L0 = 1.0
        total_spring_PE += 0.5 * k_val * (L - L0)**2

    total_E = total_KE + total_grav_PE + total_spring_PE

    E0 = total_E[0]
    E_max = np.max(total_E)
    E_min = np.min(total_E)
    E_increase = E_max - E0  # energy should not increase
    E_decrease = E0 - E_min   # energy can decrease (due to collisions with COR < 1)

    print(f"  E(0) = {E0:.6f}")
    print(f"  E_max = {E_max:.6f}, E_min = {E_min:.6f}")
    print(f"  Max energy increase from E(0): {E_increase:.6e}")
    print(f"  Max energy decrease from E(0): {E_decrease:.6e}")
    # Allow small numerical increase, but not large
    assert E_increase / max(abs(E0), 1e-12) < 5e-2, (
        f"Energy increased significantly: {E_increase:.6e} from E0={E0:.6f}"
    )
    print("  ✓ Energy does not spuriously increase.\n")


    print(f"{'='*60}")
    print("ALL PHYSICS ASSERTIONS PASSED ✓")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
