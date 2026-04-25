from sim.scene import parse_scene, Scene
import os, pathlib, hydra
import math
import re

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
    # SpringMassPlaneEntity:
    #   3 masses on a flat plane (slope=0), connected by springs in series
    #   + optional end spring connecting first and last mass.
    #   All springs have damping, so energy dissipates over time.
    #
    # Body names:  "springmassplaneentity_0.mass-{i}"
    # Tendon names: "springmassplaneentity_0.spring-{i}"
    #   spring-0: mass-0 ↔ mass-1 (series)
    #   spring-1: mass-1 ↔ mass-2 (series)
    #   spring-2: mass-0 ↔ mass-2 (end spring, if configured)


    ## ASSERTION CODE

    # ---- Setup ----
    entity = scene.entities[0]
    num_masses = len(entity.mass_values)
    g_mag = abs(data["global"]["gravity"][0])

    N = len(data["global"]["time"])
    times = np.array(data["global"]["time"])
    dt = times[1] - times[0]

    print(f"\n{'='*60}")
    print(f"PHYSICS ASSERTIONS — SpringMassPlaneEntity")
    print(f"  {N} timesteps, {num_masses} masses, damping={entity.damping}")
    print(f"{'='*60}\n")

    # Collect body data
    body_names = []
    body_data = {}
    for i in range(num_masses):
        name = f"{entity.name}.mass-{i}"
        body_names.append(name)
        if name in data:
            body_data[name] = {
                "position": np.array(data[name]["position"]),        # (N, 3)
                "velocity": np.array(data[name]["velocity_linear"]),  # (N, 3)
                "KE": np.array(data[name]["kinetic_energy_linear"]),  # (N,)
                "mass": data[name]["mass"][0],
            }
            if "acceleration_linear" in data[name]:
                body_data[name]["acceleration"] = np.array(data[name]["acceleration_linear"])
            if "net_force_linear" in data[name]:
                body_data[name]["net_force"] = np.array(data[name]["net_force_linear"])
            print(f"  {name}: mass={body_data[name]['mass']} kg, "
                  f"pos0=({body_data[name]['position'][0, 0]:.3f}, "
                  f"{body_data[name]['position'][0, 2]:.3f})")
        else:
            print(f"  WARNING: {name} not in data keys")

    # Collect tendon data
    tendon_keys = [k for k in data.keys()
                   if k != "global" and re.search(r'\.spring-\d+$', k)
                   and "length" in data[k]]
    tendon_data = {}
    for tname in tendon_keys:
        td = data[tname]
        tendon_data[tname] = {
            "length": np.array(td["length"]),
            "velocity": np.array(td["velocity"]),
            "force": np.array(td["force"]),
            "stiffness": np.array(td["stiffness"]),
        }
        print(f"  {tname}: k={tendon_data[tname]['stiffness'][0]:.2f} N/m, "
              f"L0 initial length={tendon_data[tname]['length'][0]:.4f} m")

    print()


    # ================================================================
    # 1. INITIAL KE VERIFICATION
    #    Verify KE at t=0 matches ½mv² for each mass.
    # ================================================================
    print("--- [1] Checking initial kinetic energy ---")
    for name in body_names:
        if name not in body_data:
            continue
        bd = body_data[name]
        ke0 = bd["KE"][0]
        v0 = bd["velocity"][0]
        m = bd["mass"]
        expected_ke0 = 0.5 * m * np.dot(v0, v0)
        print(f"  {name}: KE={ke0:.6f}, expected={expected_ke0:.6f}")
        np.testing.assert_allclose(ke0, expected_ke0, rtol=1e-3,
            err_msg=f"Initial KE mismatch for {name}")
    print("  ✓ Initial kinetic energy verified.\n")


    # ================================================================
    # 2. NEWTON'S SECOND LAW (F = m·a) for all masses
    # ================================================================
    print("--- [2] Checking Newton's second law (F = m·a) ---")
    check_start = 10
    check_end = N - 10
    if check_end <= check_start:
        check_start, check_end = 0, N

    for name in body_names:
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
    # 3. SPRING FORCE CONSISTENCY
    #    Tendon force = k * (L - L0) + damping * v_tendon
    #    With damping > 0, force includes both elastic and damping terms.
    # ================================================================
    print("--- [3] Checking spring force consistency ---")
    for tname, td in tendon_data.items():
        L = td["length"]
        v_tendon = td["velocity"]
        force = td["force"]
        k_val = td["stiffness"]

        if len(L) == 0 or len(force) == 0:
            print(f"  {tname}: empty data — skipping")
            continue

        # Expected: k * (L - L0) + damping * v
        # L0 is the springlength (rest length from MuJoCo model)
        # The stiffness and springlength are baked into the tendon;
        # force = stiffness * (length - springlength) + damping * velocity
        # We don't have springlength directly, but we can verify consistency:
        # force - k * L - damping * v should be constant (= -k * L0)
        residual = force - k_val * L - entity.damping * v_tendon
        residual_drift = np.max(residual) - np.min(residual)
        residual_scale = np.max(np.abs(force)) + 1e-12

        print(f"  {tname}: residual drift = {residual_drift:.6e}, "
              f"rel = {residual_drift/residual_scale:.6e}")
        assert residual_drift / residual_scale < 5e-2, (
            f"Spring {tname} force inconsistency: residual drift = {residual_drift:.6e}"
        )
    print("  ✓ Spring forces consistent with Hooke's law + damping.\n")


    # ================================================================
    # 4. VELOCITY-POSITION CONSISTENCY
    #    v(t) ≈ (pos(t+1) - pos(t)) / dt
    # ================================================================
    print("--- [4] Checking velocity-position consistency ---")
    cs = 10
    ce = min(N - 1, N - 10)
    if ce <= cs:
        cs, ce = 0, N - 1

    for name in body_names:
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
    # 5. SPRING LENGTH vs MASS POSITION CONSISTENCY
    #    For series springs: L_i = |pos(mass_i) - pos(mass_{i+1})|
    #    For end spring:     L_end = |pos(mass_0) - pos(mass_last)|
    #    Note: tendons measure site-to-site distance (center sites).
    # ================================================================
    print("--- [5] Checking spring length vs mass positions ---")

    # Series springs: spring-i connects mass-i to mass-(i+1)
    num_series_springs = num_masses - 1
    for i in range(num_series_springs):
        tname = f"{entity.name}.spring-{i}"
        if tname not in tendon_data:
            print(f"  {tname}: not in tendon data — skipping")
            continue
        name_left = f"{entity.name}.mass-{i}"
        name_right = f"{entity.name}.mass-{i + 1}"
        if name_left not in body_data or name_right not in body_data:
            continue

        pos_left = body_data[name_left]["position"]
        pos_right = body_data[name_right]["position"]
        computed_L = np.linalg.norm(pos_right - pos_left, axis=1)
        recorded_L = tendon_data[tname]["length"]

        error = np.max(np.abs(computed_L - recorded_L))
        scale = np.max(recorded_L) + 1e-12
        print(f"  {tname} (mass-{i} ↔ mass-{i+1}): "
              f"max |computed - recorded| = {error:.6e}, rel = {error/scale:.6e}")
        assert error / scale < 0.1, (
            f"Spring {tname} length inconsistency: error = {error:.6e}"
        )

    # End spring: spring-(num_series) connects mass-0 to mass-(last)
    if entity.end_spring_config is not None and num_masses > 1:
        end_idx = num_series_springs  # index of end spring
        tname = f"{entity.name}.spring-{end_idx}"
        if tname in tendon_data:
            name_first = f"{entity.name}.mass-0"
            name_last = f"{entity.name}.mass-{num_masses - 1}"
            if name_first in body_data and name_last in body_data:
                pos_first = body_data[name_first]["position"]
                pos_last = body_data[name_last]["position"]
                computed_L = np.linalg.norm(pos_last - pos_first, axis=1)
                recorded_L = tendon_data[tname]["length"]

                error = np.max(np.abs(computed_L - recorded_L))
                scale = np.max(recorded_L) + 1e-12
                print(f"  {tname} (mass-0 ↔ mass-{num_masses-1}, end spring): "
                      f"max |computed - recorded| = {error:.6e}, rel = {error/scale:.6e}")
                assert error / scale < 0.1, (
                    f"End spring {tname} length inconsistency: error = {error:.6e}"
                )
    print("  ✓ Spring lengths match mass positions.\n")


    # ================================================================
    # 6. ENERGY DISSIPATION (with damping, energy must decrease)
    #    Total mechanical energy = sum(KE) + sum(elastic PE)
    #    With damping > 0, total E should monotonically decrease
    #    (no spontaneous energy increase beyond numerical tolerance).
    # ================================================================
    print("--- [6] Checking energy dissipation with damping ---")

    total_KE = np.zeros(N)
    total_spring_PE = np.zeros(N)

    for name in body_names:
        if name not in body_data:
            continue
        total_KE += body_data[name]["KE"]

    for tname, td in tendon_data.items():
        L = td["length"]
        k_val = td["stiffness"]
        # We need L0: use residual from test 3 to infer it
        # force = k*(L-L0) + damping*v  =>  when v=0 at t=0 (if masses start at rest):
        #   L0 = L[0] - force[0]/k[0]
        force0 = td["force"][0]
        v0 = td["velocity"][0]
        k0 = k_val[0]
        L0_inferred = L[0] - (force0 - entity.damping * v0) / k0

        total_spring_PE += 0.5 * k_val * (L - L0_inferred)**2

    total_E = total_KE + total_spring_PE

    E0 = total_E[0]
    E_final = total_E[-1]
    # Check that energy doesn't increase significantly above initial
    E_max = np.max(total_E)
    E_increase = E_max - E0

    print(f"  E(0) = {E0:.6f}, E(end) = {E_final:.6f}")
    print(f"  E_max = {E_max:.6f}")
    print(f"  Max energy increase above E(0): {E_increase:.6e}")
    if entity.damping > 0:
        print(f"  Energy dissipated: {E0 - E_final:.6f} ({100*(E0-E_final)/max(E0,1e-12):.1f}%)")
    assert E_increase / max(abs(E0), 1e-12) < 5e-2, (
        f"Energy increased significantly: {E_increase:.6e} from E0={E0:.6f}"
    )
    print("  ✓ Energy behaves correctly (no spurious increase).\n")


    # ================================================================
    # 7. MASSES STAY ON PLANE (z-coordinate ~ constant)
    #    Since the plane is flat (slope=0) and there's no vertical force
    #    except gravity balanced by the plane, z shouldn't change much.
    # ================================================================
    print("--- [7] Checking masses stay on plane ---")
    for name in body_names:
        if name not in body_data:
            continue
        bd = body_data[name]
        z = bd["position"][:, 2]
        z_drift = np.max(z) - np.min(z)
        print(f"  {name}: z range = [{np.min(z):.6f}, {np.max(z):.6f}], drift = {z_drift:.6e}")
        assert z_drift < 0.1, (
            f"{name} left the plane: z drift = {z_drift}"
        )
    print("  ✓ All masses remain on the plane.\n")


    print(f"{'='*60}")
    print("ALL PHYSICS ASSERTIONS PASSED ✓")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
