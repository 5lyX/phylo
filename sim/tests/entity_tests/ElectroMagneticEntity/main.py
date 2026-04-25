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
    # ElectroMagneticEntity:
    #   A single charged particle in EM fields.
    #   - mass = 0.67 kg, charge q = 2.77 C
    #   - init_velocity = (-0.63, 1.09, 0, 0, 0, 0)  [vx, vy, vz, wx, wy, wz]
    #   - One uniform electric field:
    #       strength = 0.22, angle = 296.15°
    #       shape = box, range = (1.36, 0.56), position = (0.88, 0.51)
    #       mode = uniform_electric
    #
    # Physics:
    #   - No gravity ("gravity are negligible")
    #   - F = qE when particle is inside the field region
    #   - F = 0 when particle is outside the field region
    #   - Electric field direction: E = strength * (cos(angle), sin(angle), 0)
    #
    # Particle body name: "electromagneticentity_0.particle"


    ## ASSERTION CODE

    # ---- Setup ----
    entity = scene.entities[0]
    particle_name = f"{entity.name}.particle"
    mass = entity.mass
    q = entity.q
    vx0, vy0, vz0 = entity.init_velocity[0], entity.init_velocity[1], entity.init_velocity[2]

    N = len(data["global"]["time"])
    times = np.array(data["global"]["time"])
    dt = times[1] - times[0]

    print(f"\n{'='*60}")
    print(f"PHYSICS ASSERTIONS — ElectroMagneticEntity")
    print(f"  {N} timesteps, mass={mass} kg, q={q} C")
    print(f"  init_vel=({vx0}, {vy0}, {vz0})")
    print(f"  Number of fields: {len(entity.field_configs)}")
    for i, fc in enumerate(entity.field_configs):
        print(f"    Field {i}: type={fc['field_type']}, mode={fc['mode']}, "
              f"strength={fc['field_strength']}")
    print(f"{'='*60}\n")

    # Get particle data
    assert particle_name in data, f"Particle '{particle_name}' not in data. Keys: {list(data.keys())}"
    pd = data[particle_name]
    pos = np.array(pd["position"])          # (N, 3)
    vel = np.array(pd["velocity_linear"])   # (N, 3)
    KE = np.array(pd["kinetic_energy_linear"])  # (N,)
    accel = np.array(pd["acceleration_linear"])  # (N, 3)
    net_force = np.array(pd["net_force_linear"])  # (N, 3)
    mass_rec = pd["mass"][0]

    print(f"  Particle mass (recorded): {mass_rec} kg")
    print(f"  Particle pos(0): ({pos[0, 0]:.4f}, {pos[0, 1]:.4f}, {pos[0, 2]:.4f})")
    print(f"  Particle vel(0): ({vel[0, 0]:.4f}, {vel[0, 1]:.4f}, {vel[0, 2]:.4f})")
    print()


    # ================================================================
    # 1. INITIAL KINETIC ENERGY
    # ================================================================
    print("--- [1] Checking initial kinetic energy ---")
    ke0 = KE[0]
    v0 = vel[0]
    expected_ke0 = 0.5 * mass * np.dot(v0, v0)
    print(f"  KE(0) = {ke0:.6f}, expected = {expected_ke0:.6f}")
    np.testing.assert_allclose(ke0, expected_ke0, rtol=1e-3,
        err_msg="Initial KE mismatch")
    print("  ✓ Initial kinetic energy verified.\n")


    # ================================================================
    # 2. INITIAL VELOCITY
    # ================================================================
    print("--- [2] Checking initial velocity ---")
    print(f"  vel(0) = ({vel[0, 0]:.6f}, {vel[0, 1]:.6f}, {vel[0, 2]:.6f})")
    print(f"  expected = ({vx0}, {vy0}, {vz0})")
    np.testing.assert_allclose(vel[0, 0], vx0, atol=1e-3,
        err_msg="Initial vx mismatch")
    np.testing.assert_allclose(vel[0, 1], vy0, atol=1e-3,
        err_msg="Initial vy mismatch")
    np.testing.assert_allclose(vel[0, 2], vz0, atol=1e-3,
        err_msg="Initial vz mismatch")
    print("  ✓ Initial velocity verified.\n")


    # ================================================================
    # 3. NEWTON'S SECOND LAW (F = m·a)
    # ================================================================
    print("--- [3] Checking Newton's second law (F = m·a) ---")
    check_start = 10
    check_end = N - 10
    if check_end <= check_start:
        check_start, check_end = 0, N

    ma = mass * accel[check_start:check_end]
    F = net_force[check_start:check_end]
    error = np.max(np.abs(F - ma))
    scale = np.max(np.abs(F)) + 1e-12
    print(f"  max |F-ma| = {error:.6e}, rel = {error/scale:.6e}")
    assert error / scale < 5e-2, (
        f"Newton's 2nd law violation: error = {error:.6e}"
    )
    print("  ✓ Newton's second law verified.\n")


    # ================================================================
    # 4. VELOCITY-POSITION CONSISTENCY
    # ================================================================
    print("--- [4] Checking velocity-position consistency ---")
    cs = 10
    ce = min(N - 1, N - 10)
    if ce <= cs:
        cs, ce = 0, N - 1

    vel_fd = np.diff(pos, axis=0) / dt
    vel_mid = 0.5 * (vel[:-1] + vel[1:])
    error = np.max(np.abs(vel_fd[cs:ce] - vel_mid[cs:ce]))
    scale = np.max(np.abs(vel_mid[cs:ce])) + 1e-12
    print(f"  max |v_fd - v_mid| = {error:.6e}, rel = {error/scale:.6e}")
    assert error / scale < 5e-2, (
        f"Velocity-position inconsistency: error = {error:.6e}"
    )
    print("  ✓ Velocity-position consistency verified.\n")


    # ================================================================
    # 5. Z-COMPONENT STAYS ZERO
    #    The EM fields are in the XY plane; no gravity.
    #    z-position and z-velocity should remain ~0 throughout.
    # ================================================================
    print("--- [5] Checking z-component stays zero ---")
    z_pos_max = np.max(np.abs(pos[:, 2]))
    z_vel_max = np.max(np.abs(vel[:, 2]))
    print(f"  max |z_pos| = {z_pos_max:.6e}")
    print(f"  max |z_vel| = {z_vel_max:.6e}")
    assert z_pos_max < 0.1, f"Particle moves in z: max|z| = {z_pos_max}"
    assert z_vel_max < 0.1, f"Particle has z-velocity: max|vz| = {z_vel_max}"
    print("  ✓ Motion confined to XY plane.\n")


    # ================================================================
    # 6. FORCE DIRECTION CONSISTENCY WITH FIELD
    #    For a uniform electric field:
    #      E = strength * (cos(angle), sin(angle), 0)
    #      F = q * E
    #    The force should be constant when inside the field region,
    #    and zero when outside.
    #    We verify the net_force direction matches expected E direction
    #    at timesteps where the particle is inside the field region.
    # ================================================================
    print("--- [6] Checking force direction matches field ---")
    fc = entity.field_configs[0]
    if fc["mode"] in ["uniform_electric", "uniform_magnetic"]:
        strength = fc["field_strength"]
        angle_deg = fc.get("field_angle", 0.0)
        angle_rad = math.radians(angle_deg)

        if fc["field_type"] == "electric":
            Ex = strength * math.cos(angle_rad)
            Ey = strength * math.sin(angle_rad)
            expected_F = q * np.array([Ex, Ey, 0.0])
            print(f"  E field: ({Ex:.6f}, {Ey:.6f}, 0) N/C")
            print(f"  Expected F = q*E: ({expected_F[0]:.6f}, {expected_F[1]:.6f}, 0) N")
        elif fc["field_type"] == "magnetic":
            Bz = strength
            # F = qv × B => not constant, depends on v
            print(f"  B field: (0, 0, {Bz}) T — force direction depends on velocity")
            expected_F = None  # Can't check constant direction for magnetic

        # Check field region
        field_pos = fc.get("field_position", (0, 0))
        field_range = fc.get("field_range", None)
        field_shape = fc.get("field_shape", None)

        # Identify timesteps where particle is inside the field
        inside_mask = np.ones(N, dtype=bool)
        if field_range is not None and field_shape == "box":
            half_w = field_range[0] / 2
            half_h = field_range[1] / 2
            inside_mask = (
                (np.abs(pos[:, 0] - field_pos[0]) <= half_w) &
                (np.abs(pos[:, 1] - field_pos[1]) <= half_h)
            )
        elif field_range is not None and field_shape == "circle":
            r = np.sqrt((pos[:, 0] - field_pos[0])**2 + (pos[:, 1] - field_pos[1])**2)
            inside_mask = (r >= field_range[0]) & (r <= field_range[1])

        n_inside = np.sum(inside_mask)
        n_outside = np.sum(~inside_mask)
        print(f"  Timesteps inside field: {n_inside}, outside: {n_outside}")

        if fc["field_type"] == "electric" and expected_F is not None and n_inside > 20:
            # Check force at inside timesteps
            F_inside = net_force[inside_mask]
            F_expected = np.tile(expected_F, (n_inside, 1))
            force_error = np.max(np.abs(F_inside - F_expected))
            force_scale = np.linalg.norm(expected_F) + 1e-12
            print(f"  max |F - qE| inside field = {force_error:.6e}, "
                  f"rel = {force_error/force_scale:.6e}")
            assert force_error / force_scale < 0.1, (
                f"Force inside field doesn't match q*E: error = {force_error:.6e}"
            )
            print("  ✓ Force inside field matches q*E.")

        if n_outside > 20:
            # Check force is zero outside the field
            F_outside = net_force[~inside_mask]
            F_out_max = np.max(np.abs(F_outside))
            print(f"  max |F| outside field = {F_out_max:.6e}")
            assert F_out_max < 0.1, (
                f"Non-zero force outside field region: max = {F_out_max}"
            )
            print("  ✓ Force outside field is ~zero.")
    else:
        print(f"  Skipping (mode={fc['mode']}, not uniform)")
    print()


    # ================================================================
    # 7. WORK-ENERGY THEOREM
    #    ΔKE should equal the work done by the electric force.
    #    For constant force F inside field:
    #      W = F · Δx (cumulative)
    #    Note: with magnetic fields, B does no work (F ⊥ v always).
    # ================================================================
    print("--- [7] Checking work-energy theorem ---")
    # Only meaningful for electric fields (magnetic does no work)
    has_electric = any(fc["field_type"] == "electric" for fc in entity.field_configs)
    if has_electric:
        delta_KE = KE[-1] - KE[0]
        # Approximate work: sum of F · dx over all timesteps
        dx = np.diff(pos, axis=0)  # (N-1, 3)
        F_avg = 0.5 * (net_force[:-1] + net_force[1:])  # (N-1, 3)
        work = np.sum(F_avg * dx)  # scalar
        print(f"  ΔKE = {delta_KE:.6f}")
        print(f"  Work (∫F·dx) = {work:.6f}")
        error = abs(delta_KE - work)
        scale = max(abs(delta_KE), abs(work), 1e-12)
        print(f"  |ΔKE - W| = {error:.6e}, rel = {error/scale:.6e}")
        assert error / scale < 0.1, (
            f"Work-energy theorem violation: |ΔKE - W| = {error:.6e}"
        )
        print("  ✓ Work-energy theorem verified.\n")
    else:
        print("  No electric field — skipping (B does no work).\n")


    print(f"{'='*60}")
    print("ALL PHYSICS ASSERTIONS PASSED ✓")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
