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
    # RollingPlaneEntity: mesh body rolling on inclined plane
    #   - slope=45°, bowl (radius=0.27, height=0.11, thickness=0.23)
    #   - body_angle=60°, friction=0.5
    #   - Bowl on a tilted plane under gravity

    entity = scene.entities[0]
    g_mag = abs(data["global"]["gravity"][0])

    N = len(data["global"]["time"])
    times = np.array(data["global"]["time"])
    dt = times[1] - times[0]

    print(f"\n{'='*60}")
    print(f"PHYSICS ASSERTIONS — RollingPlaneEntity")
    print(f"  {N} timesteps, slope={entity.slope}°, g={g_mag}")
    print(f"{'='*60}\n")

    # Find the mesh body in data
    mesh_name = None
    for key in data.keys():
        if key == "global":
            continue
        if entity.name in key and "plane" not in key:
            if "position" in data[key]:
                mesh_name = key
                break

    if mesh_name is None:
        # Fall back to first non-global key with position data
        for key in data.keys():
            if key != "global" and "position" in data[key]:
                mesh_name = key
                break

    assert mesh_name is not None, f"Mesh body not found. Keys: {list(data.keys())}"
    print(f"  Mesh body: {mesh_name}")

    bd = data[mesh_name]
    pos = np.array(bd["position"])
    vel = np.array(bd["velocity_linear"])
    KE = np.array(bd["kinetic_energy_linear"])
    mass = bd["mass"][0]

    if "acceleration_linear" in bd:
        accel = np.array(bd["acceleration_linear"])
    if "net_force_linear" in bd:
        net_force = np.array(bd["net_force_linear"])

    print(f"  Mass: {mass} kg")
    print(f"  pos(0): ({pos[0, 0]:.4f}, {pos[0, 1]:.4f}, {pos[0, 2]:.4f})")
    print()


    # ================================================================
    # 1. INITIAL KINETIC ENERGY
    # ================================================================
    print("--- [1] Checking initial kinetic energy ---")
    ke0 = KE[0]
    expected_ke = 0.5 * mass * np.dot(vel[0], vel[0])
    print(f"  KE(0) = {ke0:.6f}, expected = {expected_ke:.6f}")
    np.testing.assert_allclose(ke0, expected_ke, rtol=1e-2,
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
    assert error / scale < 5e-2, f"Velocity-position inconsistency: {error:.6e}"
    print("  ✓ Velocity-position consistency verified.\n")


    # ================================================================
    # 3. ENERGY BEHAVIOR
    #    With friction, total energy (KE + gravitational PE) should
    #    NOT increase. It can decrease due to friction.
    # ================================================================
    print("--- [3] Checking energy non-increase ---")
    total_E = KE + mass * g_mag * pos[:, 2]
    E0 = total_E[0]
    E_max = np.max(total_E)
    E_increase = E_max - E0
    print(f"  E(0) = {E0:.6f}, E_max = {E_max:.6f}")
    print(f"  Max energy increase above E(0): {E_increase:.6e}")
    assert E_increase / max(abs(E0), 1e-12) < 0.1, (
        f"Energy increased significantly: {E_increase:.6e}"
    )
    print("  ✓ Energy non-increase verified.\n")


    # ================================================================
    # 4. BODY DESCENDS ON SLOPE
    #    With slope=45°, the body should roll down the incline.
    #    Its z-coordinate should decrease over time (descending).
    # ================================================================
    print("--- [4] Checking body descends on slope ---")
    z0 = pos[0, 2]
    z_final = pos[-1, 2]
    z_min = np.min(pos[:, 2])
    print(f"  z(0) = {z0:.4f}, z(end) = {z_final:.4f}, z_min = {z_min:.4f}")
    assert z_min < z0, "Body didn't descend on the incline"
    print("  ✓ Body descends as expected.\n")


    print(f"{'='*60}")
    print("ALL PHYSICS ASSERTIONS PASSED ✓")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
