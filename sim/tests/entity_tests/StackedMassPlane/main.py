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
    # StackedMassPlane scene:
    #   - StackedMassPlane entity: 2 stacked masses (6.29, 1.08 kg) on a 30° inclined plane
    #   - Connected to a FixedPulleyEntity and a MassWithFixedPulley via two tendons
    #   - Top mass pulled by string over pulley, bottom mass on the incline

    g_mag = abs(data["global"]["gravity"][0])
    N = len(data["global"]["time"])
    times = np.array(data["global"]["time"])
    dt = times[1] - times[0]

    print(f"\n{'='*60}")
    print(f"PHYSICS ASSERTIONS — StackedMassPlane")
    print(f"  {N} timesteps, g={g_mag}")
    print(f"{'='*60}\n")

    body_data = {}
    for key in data.keys():
        if key == "global" or "position" not in data[key]:
            continue
        bd = data[key]
        body_data[key] = {
            "position": np.array(bd["position"]),
            "velocity": np.array(bd["velocity_linear"]),
            "KE": np.array(bd["kinetic_energy_linear"]),
            "mass": bd["mass"][0],
        }
        if "acceleration_linear" in bd:
            body_data[key]["acceleration"] = np.array(bd["acceleration_linear"])
        if "net_force_linear" in bd:
            body_data[key]["net_force"] = np.array(bd["net_force_linear"])
        print(f"  {key}: mass={body_data[key]['mass']:.3f} kg")
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
    # 2. NEWTON'S SECOND LAW
    # ================================================================
    print("--- [2] Checking Newton's second law ---")
    cs, ce = 10, N - 10
    if ce <= cs: cs, ce = 0, N
    for name, bd in body_data.items():
        if "acceleration" not in bd or "net_force" not in bd:
            continue
        m = bd["mass"]
        ma = m * bd["acceleration"][cs:ce]
        F = bd["net_force"][cs:ce]
        error = np.max(np.abs(F - ma))
        scale = np.max(np.abs(F)) + 1e-12
        print(f"  {name}: max |F-ma| = {error:.6e}, rel = {error/scale:.6e}")
        assert error / scale < 5e-2, f"F=ma violation for {name}"
    print("  ✓ Newton's second law verified.\n")

    # ================================================================
    # 3. VELOCITY-POSITION CONSISTENCY
    # ================================================================
    print("--- [3] Checking velocity-position consistency ---")
    cs2, ce2 = 10, min(N - 1, N - 10)
    if ce2 <= cs2: cs2, ce2 = 0, N - 1
    for name, bd in body_data.items():
        pos = bd["position"]
        vel = bd["velocity"]
        vel_fd = np.diff(pos, axis=0) / dt
        vel_mid = 0.5 * (vel[:-1] + vel[1:])
        error = np.max(np.abs(vel_fd[cs2:ce2] - vel_mid[cs2:ce2]))
        scale = np.max(np.abs(vel_mid[cs2:ce2])) + 1e-12
        print(f"  {name}: max |v_fd - v_mid| = {error:.6e}, rel = {error/scale:.6e}")
        assert error / scale < 5e-2, f"Velocity-position inconsistency for {name}"
    print("  ✓ Velocity-position consistency verified.\n")

    # ================================================================
    # 4. ENERGY NON-INCREASE
    # ================================================================
    print("--- [4] Checking energy behavior ---")
    total_KE = np.zeros(N)
    total_PE = np.zeros(N)
    for name, bd in body_data.items():
        total_KE += bd["KE"]
        total_PE += bd["mass"] * g_mag * bd["position"][:, 2]
    total_E = total_KE + total_PE
    E0 = total_E[0]
    E_max = np.max(total_E)
    E_increase = E_max - E0
    E_rel = E_increase / max(abs(E0), 1e-12)
    print(f"  E(0) = {E0:.6f}, E_max = {E_max:.6f}")
    print(f"  Max increase = {E_increase:.6e}, rel = {E_rel:.6e}")
    assert E_rel < 0.1, f"Energy increased: {E_increase:.6e}"
    print("  ✓ Energy behavior verified.\n")

    print(f"{'='*60}")
    print("ALL PHYSICS ASSERTIONS PASSED ✓")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
