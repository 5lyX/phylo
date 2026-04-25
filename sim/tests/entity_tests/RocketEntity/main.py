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
    # RocketEntity: rocket launch from planet surface
    #   - rocket_mass=0.109 kg (initial), v_exhaust=26.01 m/s, dm_dt=-0.018 kg/s
    #   - min_mass=0.065 kg (fuel exhausted)
    #   - planet_mass=2165.36 kg, planet_radius=1.006 m
    #   - launch_angle=0 (radial launch upward)
    #   - Gravity: planet gravitational field (no MuJoCo gravity, applied by recorder)

    entity = scene.entities[0]
    rocket_name = None
    planet_name = None

    # Find the rocket and planet body names
    for key in data.keys():
        if key == "global":
            continue
        if "rocket" in key.lower():
            rocket_name = key
        elif "planet" in key.lower():
            planet_name = key

    N = len(data["global"]["time"])
    times = np.array(data["global"]["time"])
    dt = times[1] - times[0]

    print(f"\n{'='*60}")
    print(f"PHYSICS ASSERTIONS — RocketEntity")
    print(f"  {N} timesteps, rocket_mass={entity.rocket_mass} kg")
    print(f"  v_exhaust={entity.v_exhaust}, dm_dt={entity.dm_dt}")
    print(f"  planet_mass={entity.planet_mass}, planet_radius={entity.planet_radius}")
    print(f"  Rocket body: {rocket_name}")
    print(f"  Planet body: {planet_name}")
    print(f"{'='*60}\n")

    assert rocket_name is not None, f"Rocket not found. Keys: {list(data.keys())}"
    rd = data[rocket_name]
    pos = np.array(rd["position"])
    vel = np.array(rd["velocity_linear"])
    KE = np.array(rd["kinetic_energy_linear"])

    if "acceleration_linear" in rd:
        accel = np.array(rd["acceleration_linear"])
    if "net_force_linear" in rd:
        net_force = np.array(rd["net_force_linear"])


    # ================================================================
    # 1. INITIAL KINETIC ENERGY (rocket starts at rest)
    # ================================================================
    print("--- [1] Checking initial kinetic energy ---")
    ke0 = KE[0]
    print(f"  KE(0) = {ke0:.6f} (should be ~0 at launch)")
    assert ke0 < 1.0, f"Rocket should start at rest, KE(0) = {ke0}"
    print("  ✓ Initial KE verified.\n")


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
    # 3. ROCKET MOVES AWAY FROM PLANET
    #    The rocket should gain altitude over time.
    # ================================================================
    print("--- [3] Checking rocket gains altitude ---")
    # Planet is at origin; rocket distance should increase
    dist_from_origin = np.linalg.norm(pos, axis=1)
    d0 = dist_from_origin[0]
    d_final = dist_from_origin[-1]
    d_max = np.max(dist_from_origin)
    print(f"  Distance(0) = {d0:.4f}, Distance(end) = {d_final:.4f}, max = {d_max:.4f}")
    assert d_max > d0, "Rocket never moved away from planet"
    print("  ✓ Rocket gains altitude.\n")


    # ================================================================
    # 4. ROCKET SPEED INCREASES DURING THRUST
    #    While fuel is being burned (mass > min_mass), thrust produces
    #    acceleration. Speed should generally increase early on.
    # ================================================================
    print("--- [4] Checking speed increase during thrust ---")
    speed = np.linalg.norm(vel, axis=1)
    # Check speed mid-way is > initial speed
    mid_idx = N // 4  # Early phase
    print(f"  speed(0) = {speed[0]:.4f}, speed({mid_idx}) = {speed[mid_idx]:.4f}")
    assert speed[mid_idx] > speed[0] + 0.01, "Rocket should accelerate during thrust phase"
    print("  ✓ Rocket accelerates during thrust.\n")


    # ================================================================
    # 5. MOTION IN LAUNCH DIRECTION
    #    launch_angle=0 => radial upward from planet surface.
    #    For a planet at origin, the rocket should move radially outward.
    # ================================================================
    print("--- [5] Checking motion direction ---")
    # The rocket starts on the planet surface. With launch_angle=0,
    # it should move mostly in a radial direction (upward).
    # The initial position defines the radial direction.
    r0 = pos[0]
    r0_mag = np.linalg.norm(r0)
    if r0_mag > 1e-6:
        radial_dir = r0 / r0_mag
        # Project velocity onto radial direction
        vr = np.dot(vel, radial_dir)
        # Most of the velocity should be radial at early times
        v_total = np.linalg.norm(vel, axis=1)
        early_check = min(N // 4, 100)
        if early_check > 10:
            radial_frac = np.mean(np.abs(vr[10:early_check]) / (v_total[10:early_check] + 1e-12))
            print(f"  Radial velocity fraction (early): {radial_frac:.4f}")
            # For a radial launch, most velocity should be radial
            assert radial_frac > 0.5, f"Rocket motion not aligned with launch direction"
    print("  ✓ Motion direction verified.\n")


    print(f"{'='*60}")
    print("ALL PHYSICS ASSERTIONS PASSED ✓")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
