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
    # PendulumEntity: simple pendulum
    #   - rope_length=0.79, mass=2.09 kg, angle=24.05° from vertical
    #   - init_velocity: sphere = (-0.14, 0.94, 0) (may be linear vel of bob)
    #   - Gravity g=9.81 m/s²
    #   - Bob body: "pendulumentity_0.sphere" or similar

    entity = scene.entities[0]
    g_mag = abs(data["global"]["gravity"][0])
    L = entity.rope_length
    mass = entity.mass_value
    angle_deg = entity.angle

    N = len(data["global"]["time"])
    times = np.array(data["global"]["time"])
    dt = times[1] - times[0]

    print(f"\n{'='*60}")
    print(f"PHYSICS ASSERTIONS — PendulumEntity")
    print(f"  {N} timesteps, L={L} m, mass={mass} kg, angle={angle_deg}°")
    print(f"  g={g_mag} m/s²")
    print(f"{'='*60}\n")

    # Find the bob body name
    bob_name = None
    for key in data.keys():
        if key != "global" and entity.name in key and "sphere" in key:
            bob_name = key
            break
    if bob_name is None:
        # Try other patterns
        for key in data.keys():
            if key != "global" and entity.name in key:
                if "position" in data[key]:
                    bob_name = key
                    break
    
    assert bob_name is not None, f"Could not find bob body in data. Keys: {list(data.keys())}"
    print(f"  Bob body: {bob_name}")

    bd = data[bob_name]
    pos = np.array(bd["position"])
    vel = np.array(bd["velocity_linear"])
    KE = np.array(bd["kinetic_energy_linear"])

    if "acceleration_linear" in bd:
        accel = np.array(bd["acceleration_linear"])
    if "net_force_linear" in bd:
        net_force = np.array(bd["net_force_linear"])


    # ================================================================
    # 1. INITIAL KINETIC ENERGY
    # ================================================================
    print("--- [1] Checking initial kinetic energy ---")
    ke0 = KE[0]
    v0 = vel[0]
    expected_ke0 = 0.5 * mass * np.dot(v0, v0)
    print(f"  KE(0) = {ke0:.6f}, expected = {expected_ke0:.6f}")
    np.testing.assert_allclose(ke0, expected_ke0, rtol=1e-2,
        err_msg="Initial KE mismatch")
    print("  ✓ Initial kinetic energy verified.\n")


    # ================================================================
    # 2. VELOCITY-POSITION CONSISTENCY
    # ================================================================
    print("--- [2] Checking velocity-position consistency ---")
    cs, ce = 10, min(N - 1, N - 10)
    if ce <= cs: cs, ce = 0, N - 1
    vel_fd = np.diff(pos, axis=0) / dt
    vel_mid = 0.5 * (vel[:-1] + vel[1:])
    error = np.max(np.abs(vel_fd[cs:ce] - vel_mid[cs:ce]))
    scale = np.max(np.abs(vel_mid[cs:ce])) + 1e-12
    print(f"  max |v_fd - v_mid| = {error:.6e}, rel = {error/scale:.6e}")
    assert error / scale < 5e-2, f"Velocity-position inconsistency: error = {error:.6e}"
    print("  ✓ Velocity-position consistency verified.\n")


    # ================================================================
    # 3. TOTAL ENERGY CONSERVATION
    #    E = KE + m*g*z should be conserved (no damping).
    # ================================================================
    print("--- [3] Checking total energy conservation ---")
    total_E = KE + mass * g_mag * pos[:, 2]
    E0 = total_E[0]
    E_drift = np.max(np.abs(total_E - E0))
    E_rel = E_drift / max(abs(E0), 1e-12)
    print(f"  E(0) = {E0:.6f}")
    print(f"  Max drift = {E_drift:.6e}, relative = {E_rel:.6e}")
    assert E_rel < 0.1, f"Energy not conserved: rel drift = {E_rel:.6e}"
    print("  ✓ Total energy conserved.\n")


    # ================================================================
    # 4. CONSTANT DISTANCE FROM PIVOT
    #    The bob should maintain ~constant distance from the pivot
    #    (constraint of rigid rope/rod). The pivot is at the top of the rope.
    # ================================================================
    print("--- [4] Checking constant distance from pivot ---")
    # The pivot is likely at the entity position or the top of the pendulum.
    # We infer it from the bob position + rope_length * unit_vector pointing up.
    # At t=0, the pivot should be at pos[0] + L * (towards vertical).
    # Alternative: check that the distance from the mean pivot is constant.
    
    # Assume the pivot is at the suspension point, which for a pendulum
    # at angle theta from vertical is:
    #   bob_pos = pivot + L * (sin(theta), 0, -cos(theta))
    # So pivot = bob_pos - L * direction
    # We compute pivot from first frame and check distance stays constant.
    
    # Compute distances between consecutive positions to find effective radius
    # Or simply: the pendulum bob should trace a circular arc,
    # so |pos - pivot| should be approximately L for all timesteps.
    
    # Try to find pivot by fitting: pivot is the point that minimizes
    # variance of distances from all bob positions.
    # For simplicity, compute pivot from initial geometry:
    pivot_guess = pos[0] + np.array([0, 0, L * math.cos(math.radians(angle_deg))])
    pivot_guess[0] -= L * math.sin(math.radians(angle_deg))
    
    distances = np.linalg.norm(pos - pivot_guess, axis=1)
    d_mean = np.mean(distances)
    d_drift = np.max(np.abs(distances - d_mean))
    
    print(f"  Pivot guess: ({pivot_guess[0]:.4f}, {pivot_guess[1]:.4f}, {pivot_guess[2]:.4f})")
    print(f"  Mean distance = {d_mean:.6f} m (rope_length = {L} m)")
    print(f"  Max distance drift = {d_drift:.6e}")
    # The distance should be nearly constant (rigid constraint)
    assert d_drift < 0.1, f"Bob distance varies too much: drift = {d_drift:.6e}"
    print("  ✓ Constant distance from pivot.\n")


    # ================================================================
    # 5. OSCILLATORY MOTION CHECK
    #    The pendulum should oscillate. Detect turning points using
    #    vertical velocity sign changes (axis-independent for 3D motion).
    # ================================================================
    print("--- [5] Checking oscillatory motion ---")
    x = pos[:, 0] - pos[0, 0]
    x_sign_changes = np.sum(np.diff(np.sign(x)) != 0)
    print(f"  x-position sign changes: {x_sign_changes}")

    vz = vel[:, 2].copy()
    eps = 1e-6
    vz[np.abs(vz) < eps] = 0.0
    vz_sign = np.sign(vz)

    # Fill zero-sign samples with nearest non-zero neighbor to avoid
    # spurious/missed sign flips near turning points.
    for i in range(1, len(vz_sign)):
        if vz_sign[i] == 0:
            vz_sign[i] = vz_sign[i - 1]
    for i in range(len(vz_sign) - 2, -1, -1):
        if vz_sign[i] == 0:
            vz_sign[i] = vz_sign[i + 1]

    turning_points = np.sum(np.diff(vz_sign) != 0)
    print(f"  vertical-velocity sign changes (turning points): {turning_points}")

    if abs(angle_deg) > 5:
        assert turning_points >= 2, (
            f"Expected oscillation but only {turning_points} turning points"
        )
        print("  ✓ Oscillatory motion detected.\n")
    else:
        print("  Small angle — skipping oscillation check.\n")


    print(f"{'='*60}")
    print("ALL PHYSICS ASSERTIONS PASSED ✓")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
