from sim.scene import parse_scene
import os, pathlib, hydra
import math

from omegaconf import DictConfig
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
    # SolarSystemEntity: orbital mechanics
    #   - Star at origin, fixed (no joint), mass=3.45, radius=1.14
    #   - 2 planets with free joints in XY plane
    #   - MuJoCo gravity is 0; gravitational force applied by recorder
    #   - Planet bodies: "solarsystementity_0.planet-{i}"
    #   - Star body: "solarsystementity_0.star"

    entity = scene.entities[0]
    num_planets = len(entity.planets)

    N = len(data["global"]["time"])
    times = np.array(data["global"]["time"])
    dt = times[1] - times[0]

    print(f"\n{'='*60}")
    print(f"PHYSICS ASSERTIONS — SolarSystemEntity (Orbital)")
    print(f"  {N} timesteps, star_mass={entity.star_mass}, {num_planets} planets")
    print(f"{'='*60}\n")

    # Collect planet data
    planet_data = {}
    for i in range(num_planets):
        pname = f"{entity.name}.planet-{i}"
        if pname in data:
            planet_data[pname] = {
                "position": np.array(data[pname]["position"]),
                "velocity": np.array(data[pname]["velocity_linear"]),
                "KE": np.array(data[pname]["kinetic_energy_linear"]),
                "mass": data[pname]["mass"][0],
            }
            if "acceleration_linear" in data[pname]:
                planet_data[pname]["acceleration"] = np.array(data[pname]["acceleration_linear"])
            if "net_force_linear" in data[pname]:
                planet_data[pname]["net_force"] = np.array(data[pname]["net_force_linear"])
            p = entity.planets[i]
            print(f"  planet-{i}: mass={planet_data[pname]['mass']}, "
                  f"pos0=({p['position'][0]}, {p['position'][1]}), "
                  f"vel0=({p['velocity'][0]}, {p['velocity'][1]})")
        else:
            print(f"  WARNING: {pname} not in data")
    print()


    # ================================================================
    # 1. INITIAL KINETIC ENERGY
    # ================================================================
    print("--- [1] Checking initial kinetic energy ---")
    for pname, pd in planet_data.items():
        ke0 = pd["KE"][0]
        v0 = pd["velocity"][0]
        m = pd["mass"]
        expected_ke0 = 0.5 * m * np.dot(v0, v0)
        print(f"  {pname}: KE={ke0:.6f}, expected={expected_ke0:.6f}")
        np.testing.assert_allclose(ke0, expected_ke0, rtol=1e-3,
            err_msg=f"Initial KE mismatch for {pname}")
    print("  ✓ Initial kinetic energy verified.\n")


    # ================================================================
    # 2. NEWTON'S SECOND LAW (F = m·a)
    # ================================================================
    print("--- [2] Checking Newton's second law ---")
    cs, ce = 10, N - 10
    if ce <= cs: cs, ce = 0, N
    for pname, pd in planet_data.items():
        if "acceleration" not in pd or "net_force" not in pd:
            continue
        m = pd["mass"]
        ma = m * pd["acceleration"][cs:ce]
        F = pd["net_force"][cs:ce]
        error = np.max(np.abs(F - ma))
        scale = np.max(np.abs(F)) + 1e-12
        print(f"  {pname}: max |F-ma| = {error:.6e}, rel = {error/scale:.6e}")
        assert error / scale < 5e-2, f"F=ma violation for {pname}: error = {error:.6e}"
    print("  ✓ Newton's second law verified.\n")


    # ================================================================
    # 3. VELOCITY-POSITION CONSISTENCY
    # ================================================================
    print("--- [3] Checking velocity-position consistency ---")
    cs2, ce2 = 10, min(N - 1, N - 10)
    if ce2 <= cs2: cs2, ce2 = 0, N - 1
    for pname, pd in planet_data.items():
        pos = pd["position"]
        vel = pd["velocity"]
        vel_fd = np.diff(pos, axis=0) / dt
        vel_mid = 0.5 * (vel[:-1] + vel[1:])
        error = np.max(np.abs(vel_fd[cs2:ce2] - vel_mid[cs2:ce2]))
        scale = np.max(np.abs(vel_mid[cs2:ce2])) + 1e-12
        print(f"  {pname}: max |v_fd - v_mid| = {error:.6e}, rel = {error/scale:.6e}")
        assert error / scale < 5e-2, f"Velocity-position inconsistency for {pname}"
    print("  ✓ Velocity-position consistency verified.\n")


    # ================================================================
    # 4. Z-COMPONENT STAYS ZERO (all orbits in XY plane)
    # ================================================================
    print("--- [4] Checking z-component stays zero ---")
    for pname, pd in planet_data.items():
        z_max = np.max(np.abs(pd["position"][:, 2]))
        vz_max = np.max(np.abs(pd["velocity"][:, 2]))
        print(f"  {pname}: max|z|={z_max:.6e}, max|vz|={vz_max:.6e}")
        assert z_max < 0.1, f"Planet {pname} moves in z: max|z| = {z_max}"
    print("  ✓ Orbits confined to XY plane.\n")


    # ================================================================
    # 5. ANGULAR MOMENTUM CONSERVATION (about star at origin)
    #    L = m * (x*vy - y*vx) for each planet
    #    With purely central force from the star, L should be conserved.
    # ================================================================
    print("--- [5] Checking angular momentum conservation ---")
    for pname, pd in planet_data.items():
        m = pd["mass"]
        pos = pd["position"]
        vel = pd["velocity"]
        # L_z = m * (x*vy - y*vx)
        Lz = m * (pos[:, 0] * vel[:, 1] - pos[:, 1] * vel[:, 0])
        Lz0 = Lz[0]
        Lz_drift = np.max(np.abs(Lz - Lz0))
        Lz_rel = Lz_drift / max(abs(Lz0), 1e-12)
        print(f"  {pname}: L_z(0)={Lz0:.6f}, max drift={Lz_drift:.6e}, "
              f"rel={Lz_rel:.6e}")
        assert Lz_rel < 0.05, (
            f"Angular momentum not conserved for {pname}: drift={Lz_drift:.6e}"
        )
    print("  ✓ Angular momentum conserved.\n")


    # ================================================================
    # 6. TOTAL MECHANICAL ENERGY CONSERVATION
    #    E_total = sum(KE_i + PE_i), with PE_i = -mu*m_i/r_i and mu = G*M_star.
    #    Also check each planet's specific energy consistency to avoid cancellation.
    # ================================================================
    print("--- [6] Checking total mechanical energy conservation ---")
    mu = entity.G * entity.star_mass
    print(f"  Using mu = G*M_star = {entity.G:.6f} * {entity.star_mass:.6f} = {mu:.6f}")

    total_KE = np.zeros(N)
    total_PE = np.zeros(N)

    for pname, pd in planet_data.items():
        total_KE += pd["KE"]
        r = np.linalg.norm(pd["position"][:, :3], axis=1)
        r = np.maximum(r, 1e-9)
        total_PE += -mu * pd["mass"] / r

        v2 = np.sum(pd["velocity"][:, :3] ** 2, axis=1)
        eps = 0.5 * v2 - mu / r  # specific energy
        eps0 = eps[0]
        eps_drift = np.max(np.abs(eps - eps0))
        eps_rel = eps_drift / max(abs(eps0), 1e-12)
        print(f"  {pname}: eps(0)={eps0:.6f}, max drift={eps_drift:.6e}, rel={eps_rel:.6e}")
        assert eps_rel < 0.1, f"Specific energy drift too high for {pname}: {eps_drift:.6e}"

    total_E = total_KE + total_PE
    E0 = total_E[0]
    E_drift = np.max(np.abs(total_E - E0))
    E_rel = E_drift / max(abs(E0), 1e-12)
    print(f"  System E(0)={E0:.6f}, max drift={E_drift:.6e}, rel={E_rel:.6e}")
    assert E_rel < 0.1, f"Total mechanical energy not conserved: drift = {E_drift:.6e}"
    print("  ✓ Total mechanical energy conserved.\n")


    # ================================================================
    # 7. ORBITAL PERIOD CONSISTENCY
    #    Expected T from initial state:
    #      a = -mu / (2*eps0), T_expected = 2*pi*sqrt(a^3/mu) for bound orbit (eps0<0)
    #    Measured T from angular sweep:
    #      T_measured = 2*pi * total_time / |delta_theta_unwrapped|
    # ================================================================
    print("--- [7] Checking orbital period consistency ---")
    t_span = times[-1] - times[0]
    for pname, pd in planet_data.items():
        pos = pd["position"][:, :3]
        vel = pd["velocity"][:, :3]
        r0 = np.linalg.norm(pos[0])
        v0_sq = float(np.dot(vel[0], vel[0]))
        eps0 = 0.5 * v0_sq - mu / max(r0, 1e-9)

        if eps0 >= 0:
            print(f"  {pname}: unbound orbit (eps0={eps0:.6f}) -> skipping period check")
            continue

        a = -mu / (2.0 * eps0)
        T_expected = 2.0 * math.pi * math.sqrt((a ** 3) / mu)

        theta = np.unwrap(np.arctan2(pos[:, 1], pos[:, 0]))
        angle_swept = abs(theta[-1] - theta[0])

        if angle_swept < 1.5 * math.pi:
            print(
                f"  {pname}: insufficient angular sweep ({angle_swept:.3f} rad) "
                f"for stable period estimate -> skipping"
            )
            continue

        T_measured = 2.0 * math.pi * t_span / angle_swept
        rel_err = abs(T_measured - T_expected) / max(T_expected, 1e-12)
        print(
            f"  {pname}: T_expected={T_expected:.6f}, T_measured={T_measured:.6f}, "
            f"rel_err={rel_err:.6e}"
        )
        assert rel_err < 0.2, f"Orbital period mismatch for {pname}: rel_err={rel_err:.6e}"
    print("  ✓ Orbital period consistency verified.\n")


    print(f"{'='*60}")
    print("ALL PHYSICS ASSERTIONS PASSED ✓")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
