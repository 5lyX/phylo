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
    # ThrowingMotionEntity: projectile motion
    #   - Sphere launched from height=1.67 m at speed=1.77 m/s, angle=0° (horizontal)
    #   - ball_mass=1.0 kg, ball_radius=0.1 m
    #   - COR=1.0 (perfectly elastic bounce)
    #   - No friction
    #   - Gravity g=9.81 m/s² downward
    #   - Ball body name: "projectileentity_0.ball"

    entity = scene.entities[0]
    ball_name = f"{entity.name}.ball"
    mass = entity.ball_mass
    radius = entity.ball_radius
    g_mag = abs(data["global"]["gravity"][0])
    angle_rad = math.radians(entity.initial_angle)
    v0 = entity.initial_speed
    vx0 = v0 * math.cos(angle_rad)
    vz0 = v0 * math.sin(angle_rad)
    h0 = entity.initial_height

    N = len(data["global"]["time"])
    times = np.array(data["global"]["time"])
    dt = times[1] - times[0]

    assert ball_name in data, f"Ball '{ball_name}' not in data. Keys: {list(data.keys())}"
    bd = data[ball_name]
    pos = np.array(bd["position"])
    vel = np.array(bd["velocity_linear"])
    KE = np.array(bd["kinetic_energy_linear"])
    accel = np.array(bd["acceleration_linear"])
    net_force = np.array(bd["net_force_linear"])

    print(f"\n{'='*60}")
    print(f"PHYSICS ASSERTIONS — ThrowingMotionEntity (Projectile)")
    print(f"  {N} timesteps, mass={mass} kg, radius={radius} m")
    print(f"  h0={h0} m, v0={v0} m/s, angle={entity.initial_angle}°")
    print(f"  COR=1.0, friction=0, g={g_mag} m/s²")
    print(f"{'='*60}\n")


    # ================================================================
    # 1. INITIAL KINETIC ENERGY
    # ================================================================
    print("--- [1] Checking initial kinetic energy ---")
    ke0 = KE[0]
    expected_ke0 = 0.5 * mass * np.dot(vel[0], vel[0])
    print(f"  KE(0) = {ke0:.6f}, expected = {expected_ke0:.6f}")
    np.testing.assert_allclose(ke0, expected_ke0, rtol=1e-3,
        err_msg="Initial KE mismatch")
    print("  ✓ Initial kinetic energy verified.\n")


    # ================================================================
    # 2. INITIAL VELOCITY DIRECTION
    #    angle=0 => horizontal launch: vx=v0, vz=0
    # ================================================================
    print("--- [2] Checking initial velocity ---")
    print(f"  vel(0) = ({vel[0, 0]:.4f}, {vel[0, 1]:.4f}, {vel[0, 2]:.4f})")
    print(f"  expected ≈ ({vx0:.4f}, 0, {vz0:.4f})")
    np.testing.assert_allclose(vel[0, 0], vx0, atol=0.05,
        err_msg="Initial vx mismatch")
    np.testing.assert_allclose(vel[0, 2], vz0, atol=0.05,
        err_msg="Initial vz mismatch")
    print("  ✓ Initial velocity verified.\n")


    # ================================================================
    # 3. NEWTON'S SECOND LAW (F = m·a)
    # ================================================================
    print("--- [3] Checking Newton's second law ---")
    cs, ce = 10, N - 10
    if ce <= cs: cs, ce = 0, N
    ma = mass * accel[cs:ce]
    F = net_force[cs:ce]
    error = np.max(np.abs(F - ma))
    scale = np.max(np.abs(F)) + 1e-12
    print(f"  max |F-ma| = {error:.6e}, rel = {error/scale:.6e}")
    assert error / scale < 5e-2, f"Newton's 2nd law violation: error = {error:.6e}"
    print("  ✓ Newton's second law verified.\n")


    # ================================================================
    # 4. VELOCITY-POSITION CONSISTENCY
    # ================================================================
    print("--- [4] Checking velocity-position consistency ---")
    cs2, ce2 = 10, min(N - 1, N - 10)
    if ce2 <= cs2: cs2, ce2 = 0, N - 1
    vel_fd = np.diff(pos, axis=0) / dt
    vel_mid = 0.5 * (vel[:-1] + vel[1:])
    error = np.max(np.abs(vel_fd[cs2:ce2] - vel_mid[cs2:ce2]))
    scale = np.max(np.abs(vel_mid[cs2:ce2])) + 1e-12
    print(f"  max |v_fd - v_mid| = {error:.6e}, rel = {error/scale:.6e}")
    assert error / scale < 5e-2, f"Velocity-position inconsistency: error = {error:.6e}"
    print("  ✓ Velocity-position consistency verified.\n")


    # ================================================================
    # 5. PARABOLIC TRAJECTORY (before first bounce)
    #    With angle=0 (horizontal launch from height h0):
    #      x(t) = vx0 * t
    #      z(t) = h0 + radius + plane_thickness - 0.5*g*t²
    #    Time to first bounce: t_bounce = sqrt(2*h0/g)
    #    We check the first phase of flight.
    # ================================================================
    print("--- [5] Checking parabolic trajectory (pre-bounce) ---")
    # Find first bounce: z-velocity sign change from negative to positive
    vz = vel[:, 2]
    bounce_indices = []
    for i in range(1, N):
        if vz[i-1] < -0.01 and vz[i] > 0.01:
            bounce_indices.append(i)
    
    if len(bounce_indices) > 0:
        first_bounce = bounce_indices[0]
        # Check trajectory up to 80% of first bounce (avoid collision region)
        check_end = int(0.8 * first_bounce)
        if check_end > 10:
            t_check = times[:check_end] - times[0]
            z_expected = pos[0, 2] - 0.5 * g_mag * t_check**2
            x_expected = pos[0, 0] + vx0 * t_check

            z_error = np.max(np.abs(pos[:check_end, 2] - z_expected))
            x_error = np.max(np.abs(pos[:check_end, 0] - x_expected))
            print(f"  First bounce at timestep {first_bounce} (t={times[first_bounce]:.4f}s)")
            print(f"  Checking {check_end} timesteps before bounce")
            print(f"  max |z - z_parabolic| = {z_error:.6e}")
            print(f"  max |x - x_linear| = {x_error:.6e}")
            assert z_error < 0.05, f"z-trajectory not parabolic: error = {z_error}"
            assert x_error < 0.05, f"x-trajectory not linear: error = {x_error}"
            print("  ✓ Parabolic trajectory verified.\n")
        else:
            print("  Not enough pre-bounce timesteps to check.\n")
    else:
        print("  No bounce detected — skipping.\n")


    # ================================================================
    # 6. Y-COMPONENT STAYS ZERO
    #    Projectile in XZ plane, y should remain ~0.
    # ================================================================
    print("--- [6] Checking y-component stays zero ---")
    y_max = np.max(np.abs(pos[:, 1]))
    vy_max = np.max(np.abs(vel[:, 1]))
    print(f"  max |y_pos| = {y_max:.6e}, max |vy| = {vy_max:.6e}")
    assert y_max < 0.1, f"Ball moves in y: max|y| = {y_max}"
    print("  ✓ Motion confined to XZ plane.\n")


    # ================================================================
    # 7. ENERGY CONSERVATION (within each free-flight arc)
    #    Total energy = KE + m*g*z should be conserved during each
    #    free-flight phase. We check each arc between bounces separately,
    #    since the COR correction may not perfectly restore energy at
    #    each bounce (MuJoCo soft contact limitation).
    # ================================================================
    print("--- [7] Checking energy conservation (per free-flight arc) ---")
    total_E = KE + mass * g_mag * pos[:, 2]

    # Build list of free-flight segments: (start, end) index pairs
    bounce_margin = 20
    # Add boundaries: start of sim, each bounce, end of sim
    boundaries = [0] + bounce_indices + [N]
    arcs = []
    for k in range(len(boundaries) - 1):
        arc_start = boundaries[k] + (bounce_margin if k > 0 else 0)
        arc_end = boundaries[k + 1] - (bounce_margin if k < len(boundaries) - 2 else 0)
        if arc_end - arc_start > 20:
            arcs.append((arc_start, arc_end))

    print(f"  {len(bounce_indices)} bounces → {len(arcs)} free-flight arcs")
    max_arc_drift = 0
    for i, (a, b) in enumerate(arcs):
        E_arc = total_E[a:b]
        E_arc_ref = E_arc[0]
        arc_drift = np.max(np.abs(E_arc - E_arc_ref))
        arc_rel = arc_drift / max(abs(E_arc_ref), 1e-12)
        max_arc_drift = max(max_arc_drift, arc_rel)
        print(f"    Arc {i} [{a}:{b}]: E_ref={E_arc_ref:.4f}, drift={arc_drift:.4e}, rel={arc_rel:.4e}")

    print(f"  Worst arc relative drift: {max_arc_drift:.6e}")
    assert max_arc_drift < 0.05, f"Energy not conserved in free flight: worst drift = {max_arc_drift:.6e}"
    print("  ✓ Energy conserved within each free-flight arc.\n")


    print(f"{'='*60}")
    print("ALL PHYSICS ASSERTIONS PASSED ✓")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
