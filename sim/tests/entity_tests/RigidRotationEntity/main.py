from sim.scene import parse_scene, Scene
import os, pathlib, hydra
import math

from omegaconf import DictConfig, OmegaConf
import ipdb
import numpy as np

from recorder.recorder import Recorder, SCENE_TYPE_TO_CATEGORY_MAP


def quat_to_rotmat(q):
    """Convert quaternion (w, x, y, z) to rotation matrix."""
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def body_inertia_tensor_local(spec):
    """Return local COM inertia tensor for supported rigid body specs."""
    m = spec["mass"]
    body_type = spec["body_type"]
    if body_type == "sphere":
        r = spec["radius"]
        i = (2.0 / 5.0) * m * r * r
        return np.diag([i, i, i])
    if body_type == "bar":
        l = spec["length"]
        w = spec["width"]
        h = spec["height"]
        ixx = (1.0 / 12.0) * m * (w * w + h * h)
        iyy = (1.0 / 12.0) * m * (l * l + h * h)
        izz = (1.0 / 12.0) * m * (l * l + w * w)
        return np.diag([ixx, iyy, izz])
    raise ValueError(f"Unsupported body_type for inertia check: {body_type}")


def estimate_omega_from_positions(com_pos, joint_pos, joint_axis, cs, ce, dt):
    """Estimate angular speed around hinge axis from COM trajectory."""
    r = com_pos - joint_pos
    r_perp = r - np.outer(r @ joint_axis, joint_axis)

    e1 = np.mean(r_perp[cs:ce], axis=0)
    n1 = np.linalg.norm(e1)
    if n1 < 1e-9:
        e1 = r_perp[cs]
        n1 = np.linalg.norm(e1)
    e1 = e1 / (n1 + 1e-12)
    e2 = np.cross(joint_axis, e1)
    e2 = e2 / (np.linalg.norm(e2) + 1e-12)

    x = r_perp @ e1
    y = r_perp @ e2
    theta = np.unwrap(np.arctan2(y, x))
    omega = np.gradient(theta, dt)
    return omega


def expected_inertia_about_axis(entity, body_data, joint_pos, joint_axis):
    """Compute expected total inertia about hinge axis from YAML geometry."""
    spec_by_suffix = {}
    for idx, spec in enumerate(entity.rigid_bodies):
        spec_by_suffix[f"{spec['body_type']}-{idx}"] = spec

    i_expected = 0.0
    for full_name, bd in body_data.items():
        suffix = full_name.split(".")[-1]
        if suffix not in spec_by_suffix:
            matched = [k for k in spec_by_suffix if suffix.startswith(k.split("-")[0])]
            if not matched:
                continue
            suffix = matched[0]
        spec = spec_by_suffix[suffix]
        i_local = body_inertia_tensor_local(spec)
        rmat = quat_to_rotmat(spec.get("quat", [1, 0, 0, 0]))
        i_world = rmat @ i_local @ rmat.T
        com0 = bd["com_position"][0]
        d = com0 - joint_pos
        d_perp = d - np.dot(d, joint_axis) * joint_axis
        i_expected += joint_axis @ i_world @ joint_axis + spec["mass"] * np.dot(d_perp, d_perp)
    return i_expected

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
    # RigidRotationEntity: rigid bodies rotating around a hinge joint
    #   - Bar + Sphere welded together, rotating around hinge at (-0.12, 0.18, 0.11) along x-axis
    #   - Gravity causes rotation

    entity = scene.entities[0]
    g_mag = abs(data["global"]["gravity"][0])

    N = len(data["global"]["time"])
    times = np.array(data["global"]["time"])
    dt = times[1] - times[0]

    print(f"\n{'='*60}")
    print(f"PHYSICS ASSERTIONS — RigidRotationEntity")
    print(f"  {N} timesteps, g={g_mag}, {len(entity.rigid_bodies)} rigid bodies")
    if entity.joint:
        print(f"  Joint: pos={entity.joint['position']}, axis={entity.joint.get('axis')}")
    print(f"{'='*60}\n")

    # Find body data keys belonging to this entity
    entity_keys = [k for k in data.keys() if k != "global" and entity.name in k
                   and "position" in data[k]]
    print(f"  Bodies found: {entity_keys}")

    body_data = {}
    for key in entity_keys:
        bd = data[key]
        com_offset = np.array(bd["com_offset"]) if "com_offset" in bd else np.zeros_like(np.array(bd["position"]))
        position = np.array(bd["position"])
        com_position = position + com_offset
        body_data[key] = {
            "position": position,
            "com_position": com_position,
            "velocity": np.array(bd["velocity_linear"]),
            "KE": np.array(bd["kinetic_energy_linear"]),
            "KE_total": np.array(bd["kinetic_energy"]) if "kinetic_energy" in bd else np.array(bd["kinetic_energy_linear"]),
            "mass": bd["mass"][0],
        }
        if "acceleration_linear" in bd:
            body_data[key]["acceleration"] = np.array(bd["acceleration_linear"])
        if "net_force_linear" in bd:
            body_data[key]["net_force"] = np.array(bd["net_force_linear"])
    print()


    # ================================================================
    # 1. INITIAL KINETIC ENERGY
    # ================================================================
    print("--- [1] Checking initial kinetic energy ---")
    for name, bd in body_data.items():
        ke0 = bd["KE"][0]
        v0 = bd["velocity"][0]
        m = bd["mass"]
        expected_ke0 = 0.5 * m * np.dot(v0, v0)
        print(f"  {name}: KE={ke0:.6f}, expected={expected_ke0:.6f}")
        np.testing.assert_allclose(ke0, expected_ke0, rtol=1e-2,
            err_msg=f"Initial KE mismatch for {name}")
    print("  ✓ Initial kinetic energy verified.\n")


    # ================================================================
    # 2. HINGE GEOMETRY CONSISTENCY
    #    For hinge about axis a through point p:
    #      (a) distance to axis is constant
    #      (b) projection on axis is constant
    # ================================================================
    print("--- [2] Checking hinge geometry consistency ---")
    joint_pos = np.array(entity.joint["position"], dtype=float)
    joint_axis = np.array(entity.joint.get("axis", [1, 0, 0]), dtype=float)
    joint_axis = joint_axis / (np.linalg.norm(joint_axis) + 1e-12)
    cs, ce = 10, N - 10
    if ce <= cs:
        cs, ce = 0, N

    for name, bd in body_data.items():
        com_pos = bd["com_position"]
        r = com_pos - joint_pos
        axial = r @ joint_axis
        r_perp = r - np.outer(axial, joint_axis)
        rho = np.linalg.norm(r_perp, axis=1)

        rho_drift = np.max(np.abs(rho[cs:ce] - np.mean(rho[cs:ce])))
        axial_drift = np.max(np.abs(axial[cs:ce] - np.mean(axial[cs:ce])))
        print(
            f"  {name}: axis-distance drift={rho_drift:.6e}, axis-projection drift={axial_drift:.6e}"
        )
        assert rho_drift < 5e-2, f"Distance-to-axis drift too large for {name}: {rho_drift:.6e}"
        assert axial_drift < 5e-2, f"Axis-projection drift too large for {name}: {axial_drift:.6e}"
    print("  ✓ Hinge geometry consistency verified.\n")

    # Position-derived system angular speed used by later checks.
    omega_series = []
    for bd in body_data.values():
        omega_series.append(
            estimate_omega_from_positions(
                bd["com_position"], joint_pos, joint_axis, cs, ce, dt
            )
        )
    omega_stack = np.vstack(omega_series)
    omega_med = np.median(omega_stack, axis=0)
    i_expected = expected_inertia_about_axis(entity, body_data, joint_pos, joint_axis)

    # ================================================================
    # 3. TOTAL ENERGY CONSERVATION
    #    E = 0.5 * I_expected * omega^2 + sum(m*g*z_com)
    # ================================================================
    print("--- [3] Checking total energy conservation ---")
    total_KE = 0.5 * i_expected * (omega_med ** 2)
    total_PE = np.zeros(N)
    for bd in body_data.values():
        total_PE += bd["mass"] * g_mag * bd["com_position"][:, 2]
    total_E = total_KE + total_PE
    E0 = total_E[0]
    E_drift = np.max(np.abs(total_E - E0))
    E_rel = E_drift / max(abs(E0), 1e-12)
    print(f"  E(0) = {E0:.6f}, max drift = {E_drift:.6e}, rel = {E_rel:.6e}")
    assert E_rel < 0.1, f"Energy not conserved: drift = {E_drift:.6e}"
    print("  ✓ Total energy conserved.\n")


    # ================================================================
    # 4. RIGID BODY CONSTRAINT
    #    Distance between body centers should remain constant
    #    (if they are welded together as one rigid assembly).
    # ================================================================
    print("--- [4] Checking rigid body constraint ---")
    names = list(body_data.keys())
    if len(names) >= 2:
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                pos_i = body_data[names[i]]["com_position"]
                pos_j = body_data[names[j]]["com_position"]
                dist = np.linalg.norm(pos_i - pos_j, axis=1)
                d_mean = np.mean(dist)
                d_drift = np.max(np.abs(dist - d_mean))
                print(f"  |{names[i].split('.')[-1]} - {names[j].split('.')[-1]}|: "
                      f"mean={d_mean:.6f}, max drift={d_drift:.6e}")
                assert d_drift < 0.05, (
                    f"Rigid constraint violated between {names[i]} and {names[j]}: drift={d_drift}"
                )
        print("  ✓ Rigid body constraint maintained.\n")
    else:
        print("  Only one body — skipping rigid constraint check.\n")

    # ================================================================
    # 5. RIGID ROTATION RATE CONSISTENCY (POSITION-ONLY)
    #    Estimate angular speed from COM trajectories around hinge axis
    #    and ensure all welded bodies share the same omega(t).
    # ================================================================
    print("--- [5] Checking rigid rotation rate consistency ---")
    for name, bd in body_data.items():
        omega = estimate_omega_from_positions(
            bd["com_position"], joint_pos, joint_axis, cs, ce, dt
        )
        print(f"  {name}: omega range [{np.min(omega[cs:ce]):.4f}, {np.max(omega[cs:ce]):.4f}] rad/s")

    omega_ref = np.median(omega_stack, axis=0)
    omega_spread = np.max(np.abs(omega_stack - omega_ref[None, :]), axis=0)
    spread_rel = np.max(omega_spread[cs:ce] / (np.abs(omega_ref[cs:ce]) + 1e-6))
    print(f"  max relative omega spread across bodies = {spread_rel:.6e}")
    assert spread_rel < 2.5e-1, f"Welded bodies do not share a consistent angular rate: {spread_rel:.6e}"
    print("  ✓ Rotation-rate consistency verified.\n")

    # ================================================================
    # 6. EFFECTIVE INERTIA ABOUT HINGE AXIS (ANALYTIC vs OBSERVED)
    # ================================================================
    print("--- [6] Checking effective inertia about hinge axis ---")
    # Dynamic estimate via torque balance about hinge axis: I*alpha = tau_gravity.
    g_vec = np.array([0.0, 0.0, -g_mag])
    tau = np.zeros(N)
    for bd in body_data.values():
        r = bd["com_position"] - joint_pos
        fg = bd["mass"] * g_vec[None, :]
        tau += np.einsum("ij,j->i", np.cross(r, fg), joint_axis)
    alpha = np.gradient(omega_med, dt)
    valid = (np.abs(alpha) > 1e-2)
    if np.any(valid):
        i_observed_series = np.abs(tau[valid] / alpha[valid])
        i_observed = np.median(i_observed_series)
        rel_i_err = abs(i_observed - i_expected) / (abs(i_expected) + 1e-12)
        print(f"  I_expected = {i_expected:.6f} kg·m^2")
        print(f"  I_observed (from tau/alpha) = {i_observed:.6f} kg·m^2")
        print(f"  Relative error = {rel_i_err:.6e}")
        assert rel_i_err < 2.5e-1, f"Effective inertia mismatch too large: {rel_i_err:.6e}"
        print("  ✓ Effective inertia matches analytic expectation.\n")
    else:
        print("  Angular acceleration too small for stable inertia estimation — skipped.\n")


    print(f"{'='*60}")
    print("ALL PHYSICS ASSERTIONS PASSED ✓")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
