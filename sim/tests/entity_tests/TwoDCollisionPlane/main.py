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
    # ipdb> type(data)
    # <class 'collections.defaultdict'>
    # ipdb> data.keys()
    # dict_keys(['global', 'twodcollisionplane_0.sphere-0', 'twodcollisionplane_0.sphere-1', 'twodcollisionplane_0.sphere-2', 'contact', 'friction'])
    # ipdb> type(data["contact"])
    # <class 'collections.defaultdict'>
    # ipdb> data["contact"].keys()
    # dict_keys(['twodcollisionplane_0.sphere-0_twodcollisionplane_0.sphere-1'])
    # ipdb> len(data["contact"]["twodcollisionplane_0.sphere-0_twodcollisionplane_0.sphere-1"])
    # 10001
    # ipdb> data["contact"]["twodcollisionplane_0.sphere-0_twodcollisionplane_0.sphere-1"][0]
    # array([0., 0., 0.])
    # ipdb> data["twodcollisionplane_0.sphere-0"].keys()
    # dict_keys(['position', 'inertia', 'com_offset', 'velocity', 'velocity_linear', 'velocity_angular', 'com_external_force', 'com_applied_force', 'mass', 'acceleration', 'acceleration_linear', 'acceleration_angular', 'net_force', 'net_force_linear', 'net_torque', 'gravcomp', 'displacement', 'momentum_linear', 'momentum_angular', 'kinetic_energy', 'kinetic_energy_linear', 'kinetic_energy_angular', 'potential_energy', 'inertia_z', 'em_PE'])
    # ipdb> data["twodcollisionplane_0.sphere-0"]["velocity_linear"][0]
    # array([-1.01      , -0.11683116, -0.0674525 ])
    # ipdb> len(data["twodcollisionplane_0.sphere-0"]["velocity_linear"])
    # 10001
    # ipdb>

    ## ASSERTION CODE (to validate if the physics is correct)

    # ---- Setup: extract sphere names and scene parameters ----
    sphere_names = [
        "twodcollisionplane_0.sphere-0",
        "twodcollisionplane_0.sphere-1",
        "twodcollisionplane_0.sphere-2",
    ]
    masses = {name: data[name]["mass"][0] for name in sphere_names}
    # Expected initial conditions from recorder.txt / main.yaml
    expected_initial = {
        sphere_names[0]: {"mass": 2.0, "vel": np.array([-1.01, -0.13]), "radius": 0.13},
        sphere_names[1]: {"mass": 5.0, "vel": np.array([0.56, -1.13]),  "radius": 0.28},
        sphere_names[2]: {"mass": 4.0, "vel": np.array([1.01, 0.99]),   "radius": 0.27},
    }

    # ---- Slope parameters ----
    plane_slope_deg = scene.entities[0].plane_slope  # degrees
    theta = math.radians(plane_slope_deg)
    g = 9.81  # magnitude of gravity
    total_mass = sum(masses[n] for n in sphere_names)

    # In-plane "up-slope" unit vector in 3D world coords: (0, cos(θ), sin(θ))
    # Gravity vector in world coords: (0, 0, -g)
    # Component of gravity along slope = g⃗ · ê_slope = -g*sin(θ)
    # So gravitational acceleration projected into the plane (3D world coords):
    #   along slope: -g*sin(θ) * (0, cos(θ), sin(θ))
    #   along X: 0 (gravity has no X component)
    g_slope_3d = np.array([0.0, -g * math.sin(theta) * math.cos(theta),
                                -g * math.sin(theta) * math.sin(theta)])
    print(f"  Plane slope: {plane_slope_deg}°, θ={theta:.4f} rad")
    print(f"  Gravity along slope (3D): ({g_slope_3d[0]:.4f}, {g_slope_3d[1]:.4f}, {g_slope_3d[2]:.4f})")

    # Read coefficient of restitution from scene YAML parameters
    # resolution_coefficient_list entries are [sphere_idx_A, sphere_idx_B, e]
    cor_list = scene.entities[0].resolution_coefficient_list
    collision_pairs = []  # list of (nameA, nameB, e)
    for entry in cor_list:
        nameA, nameB, e_val = entry[0], entry[1], float(entry[2])
        collision_pairs.append((
            nameA,
            nameB,
            e_val,
        ))
    print(f"  Collision pairs from YAML: {[(a.split('.')[-1], b.split('.')[-1], e) for a, b, e in collision_pairs]}")

    N = len(data["global"]["time"])  # total timesteps
    print(f"\n{'='*60}")
    print(f"PHYSICS ASSERTIONS — {N} timesteps recorded")
    print(f"{'='*60}\n")

    # ================================================================
    # 1. CHECK INITIAL KE MATCHES EXPECTED ANALYTICAL VALUES
    # ================================================================
    print("--- [1] Checking initial KE matches expected values ---")
    for name, props in expected_initial.items():
        expected_KE = 0.5 * props["mass"] * np.linalg.norm(props["vel"]) ** 2
        # velocity_linear has 3 components (x, y, z); only x,y should matter 
        recorded_KE = data[name]["kinetic_energy_linear"][0]
        print(f"  {name}: expected_KE={expected_KE:.6f}, recorded_KE={recorded_KE:.6f}")
        np.testing.assert_allclose(
            recorded_KE, expected_KE, rtol=5e-2,
            err_msg=f"Initial KE mismatch for {name}"
        )
    print("  ✓ All initial KE values match expected.\n")

    # ================================================================
    # 2. TOTAL MOMENTUM vs EXPECTED GRAVITATIONAL IMPULSE (every timestep)
    #    On a sloped plane, gravity exerts a net force along the slope.
    #    p_total(t) = p_total(0) + total_mass * g_slope_3d * t
    #    On a flat plane (slope=0), g_slope_3d = 0 and this reduces to
    #    momentum conservation.
    # ================================================================
    print("--- [2] Checking total momentum vs expected gravitational impulse ---")
    # Compute total momentum at each timestep (all 3 components)
    total_momentum = np.zeros((N, 3))
    for name in sphere_names:
        mom = np.array(data[name]["momentum_linear"])  # shape (N, 3)
        total_momentum += mom

    times = np.array(data["global"]["time"])
    p_initial = total_momentum[0]

    # Expected momentum at each timestep: p(0) + M_total * g_slope * t
    expected_momentum = p_initial[None, :] + total_mass * g_slope_3d[None, :] * times[:, None]
    momentum_error = total_momentum - expected_momentum
    max_p_deviation = np.max(np.abs(momentum_error))
    p_magnitude = np.max(np.linalg.norm(total_momentum, axis=1))
    print(f"  Initial total momentum: ({p_initial[0]:.6f}, {p_initial[1]:.6f}, {p_initial[2]:.6f})")
    print(f"  Max absolute deviation from expected: {max_p_deviation:.6e}")
    print(f"  Relative deviation: {max_p_deviation / max(p_magnitude, 1e-12):.6e}")
    assert max_p_deviation < 5e-2, (
        f"Total momentum doesn't match expected gravitational impulse: max deviation = {max_p_deviation:.6e}"
    )
    print("  ✓ Total momentum matches expected gravitational impulse.\n")

    # ================================================================
    # 3. TOTAL MECHANICAL ENERGY CONSERVATION WITHIN FREE-FLIGHT SEGMENTS
    #    On a sloped frictionless plane, KE alone is not conserved —
    #    gravity does work. But total mechanical energy (KE + gravitational PE)
    #    should be constant within each free-flight segment.
    #    PE_i = m_i * g * z_i  (gravitational PE relative to z=0)
    #    On a flat plane (slope=0), PE is constant and this reduces to KE check.
    # ================================================================
    print("--- [3] Checking total mechanical energy constancy within free-flight segments ---")
    total_KE = np.zeros(N)
    total_PE = np.zeros(N)
    for name in sphere_names:
        ke = np.array(data[name]["kinetic_energy_linear"])  # shape (N,)
        total_KE += ke
        # Gravitational PE = m * g * z (z is the height, index 2 in position)
        pos = np.array(data[name]["position"])  # shape (N, 3)
        m = masses[name]
        total_PE += m * g * pos[:, 2]  # z-coordinate gives height

    total_E = total_KE + total_PE  # total mechanical energy

    # Build a mask of timesteps where ANY contact is active
    any_contact_active = np.zeros(N, dtype=bool)
    for pair_key in data["contact"]:
        cf = np.array(data["contact"][pair_key])
        any_contact_active |= (np.linalg.norm(cf, axis=1) > 1e-6)

    # Add a small margin around contact events (±5 steps) to avoid
    # transition artifacts from the penalty spring
    margin = 5
    contact_indices = np.where(any_contact_active)[0]
    for idx in contact_indices:
        lo = max(0, idx - margin)
        hi = min(N, idx + margin + 1)
        any_contact_active[lo:hi] = True

    free_flight_mask = ~any_contact_active
    n_contact = np.sum(~free_flight_mask)

    # Find contiguous free-flight segments
    ff_diff = np.diff(free_flight_mask.astype(int))
    ff_starts = np.where(ff_diff == 1)[0] + 1
    ff_ends = np.where(ff_diff == -1)[0] + 1
    if free_flight_mask[0]:
        ff_starts = np.insert(ff_starts, 0, 0)
    if free_flight_mask[-1]:
        ff_ends = np.append(ff_ends, N)
    n_ff_segs = min(len(ff_starts), len(ff_ends))

    E_initial = total_E[0]
    print(f"  Initial total KE: {total_KE[0]:.6f}")
    print(f"  Initial total PE: {total_PE[0]:.6f}")
    print(f"  Initial total mechanical energy: {E_initial:.6f}")
    print(f"  Timesteps in/near contact: {n_contact}")
    print(f"  Free-flight segments: {n_ff_segs}")

    max_drift_overall = 0.0
    for s in range(n_ff_segs):
        seg_E = total_E[ff_starts[s]:ff_ends[s]]
        if len(seg_E) < 2:
            continue
        seg_ref = seg_E[0]
        seg_drift = np.max(np.abs(seg_E - seg_ref))
        max_drift_overall = max(max_drift_overall, seg_drift)
        print(f"    Segment {s+1} (steps {ff_starts[s]}-{ff_ends[s]-1}): "
              f"E={seg_ref:.6f}, max drift={seg_drift:.6e}")

    print(f"  Max energy drift within any segment: {max_drift_overall:.6e}")
    assert max_drift_overall < 5e-2, (
        f"Total mechanical energy not constant within a free-flight segment: max drift = {max_drift_overall:.6e}"
    )
    print("  ✓ Total mechanical energy constant within each free-flight segment.\n")

    # ================================================================
    # 4. PER-BALL INDIVIDUAL KE CHECK ACROSS COLLISIONS
    #    For each collision pair, detect collision events, compute
    #    expected post-collision velocity for EACH ball using collision
    #    physics (normal/tangential decomposition in 3D), and verify
    #    each ball's recorded KE matches the expected value.
    #
    #    Collision physics (works in 3D, generalizes to any plane tilt):
    #      - Collision normal n̂ = (posB - posA) / |posB - posA| (3D)
    #      - Decompose: v_n = (v · n̂) n̂,  v_t = v - v_n
    #      - Along normal (1D collision with COR e):
    #          v1n' = ((m1 - e*m2)*v1n + (1+e)*m2*v2n) / (m1+m2)
    #          v2n' = ((m2 - e*m1)*v2n + (1+e)*m1*v1n) / (m1+m2)
    #      - Tangential unchanged: v_t' = v_t
    #      - Expected KE_i = 0.5 * m_i * |v_n' + v_t|²
    #
    #    On a sloped plane, gravity changes velocity during the contact
    #    window. We subtract the expected gravitational delta-v from the
    #    recorded post-collision velocity before comparing.
    # ================================================================
    print("--- [4] Checking per-ball individual KE across collisions ---")
    for (nameA, nameB, e) in collision_pairs:
        contact_key = f"{nameA}_{nameB}"
        if contact_key not in data["contact"]:
            contact_key = f"{nameB}_{nameA}"
        if contact_key not in data["contact"]:
            print(f"  No contact data found for pair ({nameA}, {nameB}); skipping.")
            continue

        contact_forces = np.array(data["contact"][contact_key])  # (N, 3)
        contact_magnitude = np.linalg.norm(contact_forces, axis=1)

        # Detect collision events
        in_contact = contact_magnitude > 1e-6
        diff_c = np.diff(in_contact.astype(int))
        collision_starts = np.where(diff_c == 1)[0] + 1
        collision_ends = np.where(diff_c == -1)[0] + 1 + 10
        if in_contact[0]:
            collision_starts = np.insert(collision_starts, 0, 0)
        if in_contact[-1]:
            collision_ends = np.append(collision_ends, N - 1)

        n_collisions = min(len(collision_starts), len(collision_ends))
        print(f"  Pair {contact_key} (e={e}): detected {n_collisions} collision event(s)")

        mA = masses[nameA]
        mB = masses[nameB]
        vel_A = np.array(data[nameA]["velocity_linear"])  # (N, 3)
        vel_B = np.array(data[nameB]["velocity_linear"])
        pos_A = np.array(data[nameA]["position"])          # (N, 3)
        pos_B = np.array(data[nameB]["position"])
        ke_A = np.array(data[nameA]["kinetic_energy_linear"])
        ke_B = np.array(data[nameB]["kinetic_energy_linear"])

        for i in range(n_collisions):
            pre_idx = max(0, collision_starts[i] - 1)
            post_idx = min(N - 1, collision_ends[i])
            # Collision timestep (for computing normal from positions)
            col_idx = collision_starts[i]

            # --- Full 3D velocities from data ---
            vA_pre_3d = vel_A[pre_idx]
            vB_pre_3d = vel_B[pre_idx]
            vA_post_3d = vel_A[post_idx]
            vB_post_3d = vel_B[post_idx]

            # --- Subtract gravitational delta-v during contact window ---
            # Gravity accelerates the balls during the contact window;
            # remove this effect so we compare collision physics only.
            dt_contact = times[post_idx] - times[pre_idx]
            grav_delta_v = g_slope_3d * dt_contact
            vA_post_corrected = vA_post_3d - grav_delta_v
            vB_post_corrected = vB_post_3d - grav_delta_v

            # --- Collision normal from positions at moment of impact ---
            # Use full 3D positions (normal lies in the tilted plane)
            pA = pos_A[col_idx]
            pB = pos_B[col_idx]
            n_vec = pB - pA
            n_norm = np.linalg.norm(n_vec)
            if n_norm < 1e-12:
                print(f"    Collision {i+1}: degenerate normal, skipping.")
                continue
            n_hat = n_vec / n_norm  # unit collision normal (A→B) in 3D

            # --- Pre-collision velocities (full 3D) ---
            vA_pre = vA_pre_3d
            vB_pre = vB_pre_3d

            # Decompose into normal and tangential
            vAn = np.dot(vA_pre, n_hat)  # scalar: A's velocity along normal
            vBn = np.dot(vB_pre, n_hat)
            vA_t = vA_pre - vAn * n_hat  # tangential (unchanged)
            vB_t = vB_pre - vBn * n_hat

            # 1D collision along normal with COR e
            vAn_post = ((mA - e * mB) * vAn + (1 + e) * mB * vBn) / (mA + mB)
            vBn_post = ((mB - e * mA) * vBn + (1 + e) * mA * vAn) / (mA + mB)

            # Recompose expected post-collision velocity (3D)
            vA_post_expected = vAn_post * n_hat + vA_t
            vB_post_expected = vBn_post * n_hat + vB_t

            # KE from analytical expected velocity (3D)
            expected_KE_A = 0.5 * mA * np.linalg.norm(vA_post_expected) ** 2
            expected_KE_B = 0.5 * mB * np.linalg.norm(vB_post_expected) ** 2

            # KE from gravity-corrected recorded velocity (3D)
            recorded_KE_A_corr = 0.5 * mA * np.linalg.norm(vA_post_corrected) ** 2
            recorded_KE_B_corr = 0.5 * mB * np.linalg.norm(vB_post_corrected) ** 2

            # Recorded KE from data (includes all 3 components, no correction)
            recorded_KE_A_full = ke_A[post_idx]
            recorded_KE_B_full = ke_B[post_idx]

            print(f"    Collision {i+1} (steps {collision_starts[i]}-{collision_ends[i]}):")
            print(f"      Normal n̂ = ({n_hat[0]:.4f}, {n_hat[1]:.4f}, {n_hat[2]:.4f})")
            print(f"      Positions at impact: A=({pA[0]:.4f},{pA[1]:.4f},{pA[2]:.4f}), B=({pB[0]:.4f},{pB[1]:.4f},{pB[2]:.4f}), dist={n_norm:.4f}")
            print(f"      Gravity delta-v during contact: ({grav_delta_v[0]:.6f}, {grav_delta_v[1]:.6f}, {grav_delta_v[2]:.6f}), dt={dt_contact:.6f}")
            print(f"      --- Pre-collision velocities ---")
            print(f"      {nameA} vel_pre (3D): ({vA_pre_3d[0]:.6f}, {vA_pre_3d[1]:.6f}, {vA_pre_3d[2]:.6f})")
            print(f"      {nameB} vel_pre (3D): ({vB_pre_3d[0]:.6f}, {vB_pre_3d[1]:.6f}, {vB_pre_3d[2]:.6f})")
            print(f"      v_approach_normal: vAn={vAn:.6f}, vBn={vBn:.6f}, v_rel_n={vAn-vBn:.6f}")
            print(f"      --- Post-collision velocities ---")
            print(f"      {nameA} vel_post ACTUAL (3D):     ({vA_post_3d[0]:.6f}, {vA_post_3d[1]:.6f}, {vA_post_3d[2]:.6f})")
            print(f"      {nameA} vel_post CORRECTED (3D):  ({vA_post_corrected[0]:.6f}, {vA_post_corrected[1]:.6f}, {vA_post_corrected[2]:.6f})")
            print(f"      {nameA} vel_post EXPECTED (3D):   ({vA_post_expected[0]:.6f}, {vA_post_expected[1]:.6f}, {vA_post_expected[2]:.6f})")
            print(f"      {nameB} vel_post ACTUAL (3D):     ({vB_post_3d[0]:.6f}, {vB_post_3d[1]:.6f}, {vB_post_3d[2]:.6f})")
            print(f"      {nameB} vel_post CORRECTED (3D):  ({vB_post_corrected[0]:.6f}, {vB_post_corrected[1]:.6f}, {vB_post_corrected[2]:.6f})")
            print(f"      {nameB} vel_post EXPECTED (3D):   ({vB_post_expected[0]:.6f}, {vB_post_expected[1]:.6f}, {vB_post_expected[2]:.6f})")
            print(f"      --- KE comparison ---")
            print(f"      {nameA}: KE_expected={expected_KE_A:.6f}, KE_corrected={recorded_KE_A_corr:.6f}, KE_actual(data)={recorded_KE_A_full:.6f}")
            print(f"      {nameB}: KE_expected={expected_KE_B:.6f}, KE_corrected={recorded_KE_B_corr:.6f}, KE_actual(data)={recorded_KE_B_full:.6f}")
            print(f"      Total: KE_expected={expected_KE_A+expected_KE_B:.6f}, KE_actual(data)={recorded_KE_A_full+recorded_KE_B_full:.6f}, KE_pre={ke_A[pre_idx]+ke_B[pre_idx]:.6f}")

            # Assert using the gravity-corrected KE
            np.testing.assert_allclose(
                recorded_KE_A_corr, expected_KE_A, rtol=5e-2,
                err_msg=f"Individual KE mismatch for {nameA} after collision {i+1}"
            )
            np.testing.assert_allclose(
                recorded_KE_B_corr, expected_KE_B, rtol=5e-2,
                err_msg=f"Individual KE mismatch for {nameB} after collision {i+1}"
            )
        print(f"  ✓ Per-ball KE verified for all collisions of {contact_key}.\n")

    # ================================================================
    # 5. FREE-FLIGHT VELOCITY EVOLUTION
    #    Between collisions, each sphere's velocity should evolve
    #    linearly due to gravitational acceleration along the slope:
    #      v(t) = v(t0) + g_slope_3d * (t - t0)
    #    On a flat plane (slope=0), g_slope_3d = 0 and this reduces to
    #    velocity constancy.
    # ================================================================
    print("--- [5] Checking free-flight velocity evolution ---")

    # Build a mask of "any collision active" per timestep for each sphere
    sphere_in_collision = {name: np.zeros(N, dtype=bool) for name in sphere_names}
    for pair_key in data["contact"]:
        contact_forces = np.array(data["contact"][pair_key])
        contact_mag = np.linalg.norm(contact_forces, axis=1)
        in_contact = contact_mag > 1e-6

        # Identify which spheres are involved in this pair
        for name in sphere_names:
            if name in pair_key:
                sphere_in_collision[name] |= in_contact

    for name in sphere_names:
        vel = np.array(data[name]["velocity_linear"])  # (N, 3)
        free_mask = ~sphere_in_collision[name]

        if not np.any(free_mask):
            print(f"  {name}: always in contact — skipping free-flight check.")
            continue

        # Find contiguous free-flight segments
        diff = np.diff(free_mask.astype(int))
        seg_starts = np.where(diff == 1)[0] + 1
        seg_ends = np.where(diff == -1)[0] + 1
        if free_mask[0]:
            seg_starts = np.insert(seg_starts, 0, 0)
        if free_mask[-1]:
            seg_ends = np.append(seg_ends, N)

        n_segs = min(len(seg_starts), len(seg_ends))
        max_vel_drift = 0.0
        for s in range(n_segs):
            s_start = seg_starts[s] + 10
            s_end = seg_ends[s] - 10
            if s_end <= s_start + 1:
                continue
            seg_vel = vel[s_start:s_end]  # full 3D
            seg_times = times[s_start:s_end]
            # Subtract expected gravitational velocity change:
            # v_corrected(t) = v(t) - g_slope_3d * (t - t0)
            # should be constant within the segment
            t0 = seg_times[0]
            dt = seg_times - t0  # shape (len_seg,)
            expected_vel = seg_vel[0][None, :] + g_slope_3d[None, :] * dt[:, None]
            drift = np.max(np.abs(seg_vel - expected_vel))
            max_vel_drift = max(max_vel_drift, drift)

        print(f"  {name}: {n_segs} free-flight segment(s), max velocity drift from expected = {max_vel_drift:.6e}")
        assert max_vel_drift < 5e-2, (
            f"Free-flight velocity drift too large for {name}: {max_vel_drift:.6e}"
        )
    print("  ✓ Free-flight velocity evolution verified.\n")

    print(f"{'='*60}")
    print("ALL PHYSICS ASSERTIONS PASSED ✓")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()

