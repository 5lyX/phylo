import mujoco, copy, math, re
import numpy as np

from ipdb import set_trace as st

TANGENTIAL_STIFFNESS = 100
STATIC_VEL_THRESHOLD = 1e-4

def get_data_val(model, query):
    n_text = model.name_textadr.shape[0]
    dict_val = {}
    for i in range(n_text):
        start_adr = model.name_textadr[i]
        end_adr = model.names[start_adr:].find(b'\x00')
        name_i = str(model.names[start_adr:start_adr+end_adr], 'utf-8')

        if query in name_i:
            start_adr_val = model.text_adr[i]
            l = model.text_size[i]
            dict_val[name_i] = str(model.text_data[start_adr_val:start_adr_val+l-1], 'utf-8')
    return  dict_val

def parse_custom_data(data):
    # Define a regular expression to match tuples within parentheses
    tuple_pattern = re.compile(r"\((.*?)\)")

    # Find all tuples in the input string
    matches = tuple_pattern.findall(data)

    parsed_result = []

    for match in matches:
        # Split the tuple elements by comma and strip whitespace
        elements = [elem.strip() for elem in match.split(',')]
        
        # Parse each element into int, float, or keep as string
        parsed_tuple = []
        for elem in elements:
            if elem.isdigit():
                parsed_tuple.append(int(elem))
            else:
                try:
                    parsed_tuple.append(float(elem))
                except ValueError:
                    parsed_tuple.append(elem)
        
        # Append the parsed tuple to the result list
        parsed_result.append(tuple(parsed_tuple))

    return parsed_result

def get_text(model, name):
    n_text = model.name_textadr.shape[0]

    for i in range(n_text):
        start_adr = model.name_textadr[i]
        end_adr = model.names[start_adr:].find(b'\x00')
        name_i = str(model.names[start_adr:start_adr+end_adr], 'utf-8')

        if name == name_i:
            break
    else:
        # print(KeyError(f'Name {name} not found in names!'))
        return ""
    
    start_adr = model.text_adr[i]
    l = model.text_size[i]

    return str(model.text_data[start_adr:start_adr+l], 'utf-8')

def process_tension_sensor(text):
    if text == "": return []
    text = text.split(',')

    out = []
    for tendon in text:
        tendon = tendon[:-1]
        out.append(tendon)
    
    return out

def process_coefficients_friction(text):
    if text == "": return []
    text = text.split('-')

    out = []
    for surface_pair in text:
        g1, g2, cs, cd, r = surface_pair[1:-2].split(',')
        out.append((g1, g2, float(cs), float(cd), eval(r)))
    
    return out

def process_coefficients_restitution(text):
    if text == "": return []
    text = text.split('-')

    text[-1] = text[-1][:-1]

    out = []
    for surface_pair in text:
        g1, g2, ce = surface_pair[1:-1].split(',')
        out.append((g1, g2, float(ce)))
    
    return out

def cross(a):
    '''
    Convert cross product a x b into Matrix multiplication Ab
    '''
    a0, a1, a2 = a[0], a[1], a[2]
    return np.array([[0, -a2, a1],
                     [a2, 0, -a0],
                     [-a1, a0, 0]])

def impedance_scaling(r, solimp):
    d0, dw, w, mp, p = tuple(solimp)

    r = abs(r)
    if r >= w:
        return dw
    else:
        # Normalize r in [0, 1]
        x = r / w
                
        if x < mp:
            y = 0.5 * (x / mp) ** p
        else:
            y = 1 - 0.5 * ((1 - x) / (1 - mp)) ** p

        return d0 + (dw - d0) * y


def calculate_contact_force_v2(coefficient_friction, coefficient_restitution, model, data, damping = None):
    '''
    Calculate friction force to be applied on a body
    '''
    # if damping is None: damping = 0

    nv = data.cdof.shape[0]
    inertia = np.zeros((nv, nv))
    mujoco.mj_fullM(model, inertia, data.qM)
    
    Fq = data.qfrc_applied + data.qfrc_actuator + data.qfrc_passive - data.qfrc_bias

    applied_normal_frc = {}
    applied_friction_frc = {}

    # Add all normal forces
    uniques = {}
    arr = np.zeros(nv)
    damping_forces = np.zeros(nv)
    custom_normal_forces = np.zeros(len(data.contact))
    for i in range(len(data.contact)):
        c = data.contact[i]
        geom_1, geom_2 = tuple(c.geom)
        
        ordered = min(geom_1, geom_2), max(geom_1, geom_2)
        geom_1, geom_2 = ordered
        
        if c.efc_address == -1 : 
            # uniques[ordered] = np.zeros(3)
            continue

        body_1, body_2 = model.geom_bodyid[geom_1], model.geom_bodyid[geom_2]

        contact_point = c.pos
        r_vec1, r_vec2 = contact_point - data.xpos[body_1], contact_point - data.xpos[body_2]        
        
        normal_mag = data.efc_force[c.efc_address]
        normal_dir = c.frame[:3]
        _ = np.sqrt(np.dot(normal_dir, normal_dir))
        normal_dir = np.zeros_like(normal_dir) if _ < 1e-5 else normal_dir / _  
        normal_dir = normal_dir * np.sign(np.dot(normal_dir, -r_vec1))

        Jl1, Jr1 = np.zeros((3, nv)), np.zeros((3, nv))
        mujoco.mj_jacBodyCom(model, data, Jl1, Jr1, body_1)

        Jcom1 = np.concatenate((Jl1, Jr1), axis = 0)

        Jl2, Jr2 = np.zeros((3, nv)), np.zeros((3, nv))
        mujoco.mj_jacBodyCom(model, data, Jl2, Jr2, body_2)

        Jcom2 = np.concatenate((Jl2, Jr2), axis = 0)
        
        inert_jac1, inert_jac_t1 = np.block([[Jr1], [Jl1]]).reshape((6, nv)), np.block([Jr1.T, Jl1.T]).reshape((nv, 6))
        inert_jac2, inert_jac_t2 = np.block([[Jr2], [Jl2]]).reshape((6, nv)), np.block([Jr2.T, Jl2.T]).reshape((nv, 6))

        inertia_1 = inert_jac1 @ inertia @ inert_jac_t1
        inertia_2 = inert_jac2 @ inertia @ inert_jac_t2

        Jl1, Jr1 = np.zeros((3, nv)), np.zeros((3, nv))
        mujoco.mj_jac(model, data, Jl1, Jr1, contact_point, body_1)

        Jcon1 = np.concatenate((Jl1, Jr1), axis = 0)

        Jl2, Jr2 = np.zeros((3, nv)), np.zeros((3, nv))
        mujoco.mj_jac(model, data, Jl2, Jr2, contact_point, body_2)

        Jcon2 = np.concatenate((Jl2, Jr2), axis = 0)
        
        for tup in coefficient_restitution: 
            g1, g2, ce = tup
            if (model.geom(g1).id == geom_1 and model.geom(g2).id == geom_2) or (model.geom(g2).id == geom_1 and model.geom(g1).id == geom_2):
                break
        else:
            ce = 1

        if ce != 1:
            RVEC1, RVEC2 = cross(r_vec1), cross(r_vec2)
            convert1 = np.block([np.eye(3), -RVEC1])
            convert2 = np.block([np.eye(3), -RVEC2])

            v = convert1 @ Jcon1 @ data.qvel - convert2 @ Jcon2 @ data.qvel
            v = np.dot(v, normal_dir)

            m_eff = np.linalg.pinv(normal_dir[:, None].T@(convert1@Jcon1@np.linalg.pinv(inertia)@Jcon1.T + convert2@Jcon2@np.linalg.pinv(inertia)@Jcon2.T)@ np.block([normal_dir, np.zeros_like(normal_dir)])[:, None])
            tau = c.solref[0]
            mu = c.solref[1]

            if tau > 0 or mu > 0:
                k = 1/ tau**2 / mu**2
                _b = 2 / tau
            else:
                k = -tau
                _b = -mu

            k *= impedance_scaling(c.dist, c.solimp)

            b = -2*math.sqrt(k*m_eff)*math.log(max(ce, 1e-16))/np.pi - _b

            mujoco.mj_applyFT(model, data, -b * v * normal_dir, np.zeros(3), contact_point, body_1, damping_forces)
            mujoco.mj_applyFT(model, data, b * v * normal_dir, np.zeros(3), contact_point, body_2, damping_forces)

            normal_mag += b * v

        if ordered in uniques:
            uniques[ordered] += normal_mag * normal_dir
        else:
            uniques[ordered] = normal_mag * normal_dir 
        
        custom_normal_forces[i] = normal_mag
        
        mujoco.mj_applyFT(model, data, normal_mag * normal_dir, np.zeros(3), contact_point, body_1, arr)
        mujoco.mj_applyFT(model, data, -normal_mag * normal_dir, np.zeros(3), contact_point, body_2, arr)

    Fq += arr

    for k in uniques.keys():
        uniques[k] = np.sqrt(np.dot(uniques[k], uniques[k]))
        applied_normal_frc[k] = uniques[k]

    qfrc = np.zeros(nv)
    
    for i in range(len(data.contact)):
        c = data.contact[i]
        geom_1, geom_2 = tuple(c.geom)
        ordered = min(geom_1, geom_2), max(geom_1, geom_2)
        
        body_1, body_2 = model.geom_bodyid[geom_1], model.geom_bodyid[geom_2]

        for tup in coefficient_friction: 
            g1, g2, cs, cd, r = tup
            if (model.geom(g1).id == geom_1 and model.geom(g2).id == geom_2) or (model.geom(g2).id == geom_1 and model.geom(g1).id == geom_2):
                break
        else:
            continue

        Jl1, Jr1 = np.zeros((3, nv)), np.zeros((3, nv))
        mujoco.mj_jacBodyCom(model, data, Jl1, Jr1, body_1)

        Jcom1 = np.concatenate((Jl1, Jr1), axis = 0)

        Jl2, Jr2 = np.zeros((3, nv)), np.zeros((3, nv))
        mujoco.mj_jacBodyCom(model, data, Jl2, Jr2, body_2)

        Jcom2 = np.concatenate((Jl2, Jr2), axis = 0)
        
        contact_point = c.pos
        if c.efc_address == -1 : continue
        normal_mag = custom_normal_forces[i] # data.efc_force[c.efc_address]
        r_vec1, r_vec2 = contact_point - data.xpos[body_1], contact_point - data.xpos[body_2]        
        RVEC1, RVEC2 = cross(r_vec1), cross(r_vec2)

        Fnet1, Fnet2 = Jcom1 @ Fq, Jcom2 @ Fq

        fric_mat1 = np.block([[np.eye(3)],
                              [RVEC1]])
        fric_mat2 = np.block([[np.eye(3)],
                              [RVEC2]])
        
        inert_jac1, inert_jac_t1 = np.block([[Jr1], [Jl1]]).reshape((6, nv)), np.block([Jr1.T, Jl1.T]).reshape((nv, 6))
        inert_jac2, inert_jac_t2 = np.block([[Jr2], [Jl2]]).reshape((6, nv)), np.block([Jr2.T, Jl2.T]).reshape((nv, 6))

        inertia_1 = inert_jac1 @ inertia @ inert_jac_t1
        inertia_2 = inert_jac2 @ inertia @ inert_jac_t2

        convert1 = np.block([np.eye(3), -RVEC1])
        convert2 = np.block([np.eye(3), -RVEC2])

        normal_dir = c.frame[:3]
        _ = np.sqrt(np.dot(normal_dir, normal_dir))
        normal_dir = np.zeros_like(normal_dir) if _ < 1e-5 else normal_dir / _  
        normal_dir = normal_dir * np.sign(np.dot(normal_dir, -r_vec1))

        P_lin = np.eye(3) - np.outer(normal_dir, normal_dir)  # 3x3 tangent projector
        P = np.block([
            [P_lin,       np.zeros((3, 3))],
            [np.zeros((3, 3)), np.eye(3)]
        ])  # 6x6 projector

        A = (
            np.linalg.pinv(inertia_2) @ fric_mat2 + 
            np.linalg.pinv(inertia_1 @ np.linalg.pinv(convert1) @ convert2) @ fric_mat1
        )
        b = (
            np.linalg.pinv(inertia_2) @ Fnet2 - 
            np.linalg.pinv(inertia_1 @ np.linalg.pinv(convert1) @ convert2) @ Fnet1
        )

        # f = np.linalg.pinv(P @ A) @ (P @ b) # Project with P so that you are solving for the tangent space
        f = np.linalg.pinv(A) @ b
        # f = np.linalg.pinv(np.linalg.pinv(inertia_2) @ fric_mat2 + np.linalg.pinv(inertia_1 @ np.linalg.pinv(convert1) @ convert2) @ fric_mat1) @ (np.linalg.pinv(inertia_2) @ Fnet2 - np.linalg.pinv(inertia_1 @ np.linalg.pinv(convert1) @ convert2) @ Fnet1)
        
        friction_mag = np.sqrt(np.dot(f, f))
        friction_dir = np.zeros(3) if friction_mag < 1e-5 else f / friction_mag

        vel1 = np.zeros((6,))
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_1, vel1, 0)
        vel2 = np.zeros((6,))
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_2, vel2, 0)

        acc1 = np.zeros((6,))
        mujoco.mj_objectAcceleration(model, data, mujoco.mjtObj.mjOBJ_BODY, body_1, acc1, 0)
        acc2 = np.zeros((6,))
        mujoco.mj_objectAcceleration(model, data, mujoco.mjtObj.mjOBJ_BODY, body_2, acc2, 0)
        
        # vel1 -= acc1 * model.opt.timestep
        # vel2 -= acc2 * model.opt.timestep
        
        rel_vel = convert1 @ vel1 - convert2 @ vel2
        rel_vel_mag = np.sqrt(np.dot(rel_vel, rel_vel))

        if rel_vel_mag > 1e-3 : 
            coefficient = cd
            # print('slip')
        else: 
            coefficient = cs
            # print('no slip')

        distribution_factor = 0 if uniques[ordered] < 1e-5 else (normal_mag / uniques[ordered])

        _ = min(friction_mag * distribution_factor, normal_mag * coefficient)
        f = friction_dir * _
        # if np.abs(_) < 5e-2: st()

        # print(np.dot(friction_dir, np.array([-np.cos(np.pi/3), 0, np.sin(np.pi/3)])))
 
        mujoco.mj_applyFT(model, data, f, np.zeros(3), contact_point if r else data.xipos[body_1], body_1, qfrc)
        mujoco.mj_applyFT(model, data, -f, np.zeros(3), contact_point if r else data.xipos[body_2], body_2, qfrc)

        applied_friction_frc[ordered] = applied_friction_frc.get(ordered, np.zeros(3)) + f

    return qfrc + damping_forces, applied_normal_frc, applied_friction_frc

def calculate_contact_force_v3(coefficient_friction, coefficient_restitution, model, data, damping = None):
    '''
    Calculate friction force to be applied on a body
    '''
    # if damping is None: damping = 0

    nv = data.cdof.shape[0]
    inertia = np.zeros((nv, nv))
    mujoco.mj_fullM(model, inertia, data.qM)
    
    Fq = data.qfrc_applied + data.qfrc_actuator + data.qfrc_passive - data.qfrc_bias

    applied_normal_frc = {}
    applied_friction_frc = {}

    # Add all normal forces
    uniques = {}
    arr = np.zeros(nv)
    damping_forces = np.zeros(nv)
    custom_normal_forces = np.zeros(len(data.contact))
    for i in range(len(data.contact)):
        c = data.contact[i]
        geom_1, geom_2 = tuple(c.geom)
        
        ordered = min(geom_1, geom_2), max(geom_1, geom_2)
        
        if c.efc_address == -1 : 
            continue

        body_1, body_2 = model.geom_bodyid[geom_1], model.geom_bodyid[geom_2]

        contact_point = c.pos
        r_vec1, r_vec2 = contact_point - data.xpos[body_1], contact_point - data.xpos[body_2]        
        
        normal_mag = data.efc_force[c.efc_address]
        normal_dir = c.frame[:3]
        _ = np.sqrt(np.dot(normal_dir, normal_dir))
        normal_dir = np.zeros_like(normal_dir) if _ < 1e-5 else normal_dir / _  
        normal_dir = normal_dir * np.sign(np.dot(normal_dir, -r_vec1))

        Jl1, Jr1 = np.zeros((3, nv)), np.zeros((3, nv))
        mujoco.mj_jacBodyCom(model, data, Jl1, Jr1, body_1)

        Jcom1 = np.concatenate((Jl1, Jr1), axis = 0)

        Jl2, Jr2 = np.zeros((3, nv)), np.zeros((3, nv))
        mujoco.mj_jacBodyCom(model, data, Jl2, Jr2, body_2)

        Jcom2 = np.concatenate((Jl2, Jr2), axis = 0)

        inert_jac1, inert_jac_t1 = np.block([[Jr1], [Jl1]]).reshape((6, nv)), np.block([Jr1.T, Jl1.T]).reshape((nv, 6))
        inert_jac2, inert_jac_t2 = np.block([[Jr2], [Jl2]]).reshape((6, nv)), np.block([Jr2.T, Jl2.T]).reshape((nv, 6))

        inertia_1 = inert_jac1 @ inertia @ inert_jac_t1
        inertia_2 = inert_jac2 @ inertia @ inert_jac_t2

        Jl1, Jr1 = np.zeros((3, nv)), np.zeros((3, nv))
        mujoco.mj_jac(model, data, Jl1, Jr1, contact_point, body_1)

        Jcon1 = np.concatenate((Jl1, Jr1), axis = 0)

        Jl2, Jr2 = np.zeros((3, nv)), np.zeros((3, nv))
        mujoco.mj_jac(model, data, Jl2, Jr2, contact_point, body_2)

        Jcon2 = np.concatenate((Jl2, Jr2), axis = 0)

        for tup in coefficient_restitution: 
            g1, g2, ce = tup
            if (model.geom(g1).id == geom_1 and model.geom(g2).id == geom_2) or (model.geom(g2).id == geom_1 and model.geom(g1).id == geom_2):
                break
        else:
            ce = 1

        if ce != 1:
            RVEC1, RVEC2 = cross(r_vec1), cross(r_vec2)
            convert1 = np.block([np.eye(3), -RVEC1])
            convert2 = np.block([np.eye(3), -RVEC2])

            v = convert1 @ Jcon1 @ data.qvel - convert2 @ Jcon2 @ data.qvel
            v = np.dot(v, normal_dir)

            m_eff = np.linalg.pinv(normal_dir[:, None].T@(convert1@Jcon1@np.linalg.pinv(inertia)@Jcon1.T + convert2@Jcon2@np.linalg.pinv(inertia)@Jcon2.T)@ np.block([normal_dir, np.zeros_like(normal_dir)])[:, None])
            tau = c.solref[0]
            mu = c.solref[1]

            if tau > 0 or mu > 0:
                k = 1/ tau**2 / mu**2
                _b = 2 / tau
            else:
                k = -tau
                _b = -mu

            k *= impedance_scaling(c.dist, c.solimp)

            b = -2*math.sqrt(k*m_eff)*math.log(max(ce, 1e-16))/np.pi - _b

            mujoco.mj_applyFT(model, data, -b * v * normal_dir, np.zeros(3), contact_point, body_1, damping_forces)
            mujoco.mj_applyFT(model, data, b * v * normal_dir, np.zeros(3), contact_point, body_2, damping_forces)

            normal_mag += b * v

        if ordered in uniques:
            uniques[ordered] += normal_mag * normal_dir
        else:
            uniques[ordered] = normal_mag * normal_dir 
        
        custom_normal_forces[i] = normal_mag
        
        mujoco.mj_applyFT(model, data, normal_mag * normal_dir, np.zeros(3), contact_point, body_1, arr)
        mujoco.mj_applyFT(model, data, -normal_mag * normal_dir, np.zeros(3), contact_point, body_2, arr)

    Fq += damping_forces

    for k in uniques.keys():
        uniques[k] = np.sqrt(np.dot(uniques[k], uniques[k]))
        applied_normal_frc[k] = uniques[k]

    qfrc = np.zeros(nv)
    
    for i in range(len(data.contact)):
        c = data.contact[i]
        geom_1, geom_2 = tuple(c.geom)
        ordered = min(geom_1, geom_2), max(geom_1, geom_2)
        
        body_1, body_2 = model.geom_bodyid[geom_1], model.geom_bodyid[geom_2]

        for tup in coefficient_friction: 
            g1, g2, cs, cd, r = tup
            if (model.geom(g1).id == geom_1 and model.geom(g2).id == geom_2) or (model.geom(g2).id == geom_1 and model.geom(g1).id == geom_2):
                break
        else:
            continue

        Jl1, Jr1 = np.zeros((3, nv)), np.zeros((3, nv))
        mujoco.mj_jacBodyCom(model, data, Jl1, Jr1, body_1)

        Jcom1 = np.concatenate((Jl1, Jr1), axis = 0)

        Jl2, Jr2 = np.zeros((3, nv)), np.zeros((3, nv))
        mujoco.mj_jacBodyCom(model, data, Jl2, Jr2, body_2)

        Jcom2 = np.concatenate((Jl2, Jr2), axis = 0)
        
        contact_point = c.pos
        if c.efc_address == -1 : continue
        normal_mag = custom_normal_forces[i] # data.efc_force[c.efc_address]
        r_vec1, r_vec2 = contact_point - data.xpos[body_1], contact_point - data.xpos[body_2]        
        RVEC1, RVEC2 = cross(r_vec1), cross(r_vec2)

        Fnet1, Fnet2 = Jcom1 @ Fq, Jcom2 @ Fq

        fric_mat1 = np.block([[np.eye(3)],
                              [RVEC1]])
        fric_mat2 = np.block([[np.eye(3)],
                              [RVEC2]])
        
        inert_jac1, inert_jac_t1 = np.block([[Jr1], [Jl1]]).reshape((6, nv)), np.block([Jr1.T, Jl1.T]).reshape((nv, 6))
        inert_jac2, inert_jac_t2 = np.block([[Jr2], [Jl2]]).reshape((6, nv)), np.block([Jr2.T, Jl2.T]).reshape((nv, 6))

        inertia_1 = inert_jac1 @ inertia @ inert_jac_t1
        inertia_2 = inert_jac2 @ inertia @ inert_jac_t2

        convert1 = np.block([np.eye(3), -RVEC1])
        convert2 = np.block([np.eye(3), -RVEC2])

        normal_dir = c.frame[:3]
        _ = np.sqrt(np.dot(normal_dir, normal_dir))
        normal_dir = np.zeros_like(normal_dir) if _ < 1e-5 else normal_dir / _  
        normal_dir = normal_dir * np.sign(np.dot(normal_dir, -r_vec1))

        # 1) build tangent basis T ∈ ℝ³×2
        n = normal_dir
        # pick an arbitrary axis not parallel to n
        axis = np.array([1.,0,0.])
        if abs(n.dot(axis)) > 0.9:
            axis = np.array([0.,1,0.])
        t1 = np.cross(n, axis)
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(n, t1)     # already unit length
        T = np.stack([t1, t2], axis=1)  # shape=(3,2)

        A = (
            np.linalg.pinv(inertia_2) @ fric_mat2 + 
            np.linalg.pinv(inertia_1 @ np.linalg.pinv(convert1) @ convert2) @ fric_mat1
        )
        b = (
            np.linalg.pinv(inertia_2) @ Fnet2 - 
            np.linalg.pinv(inertia_1 @ np.linalg.pinv(convert1) @ convert2) @ Fnet1
        )

        # 2) project A into tangent coordinates: A_tan ∈ ℝ⁶×2
        A_tan = A @ T              # (6×3)·(3×2) → (6×2)
        # b stays the same shape (6,)

        f_tan = np.linalg.pinv(A_tan) @ b # Project with T so that you are solving for the tangent space
        # f = np.linalg.pinv(A) @ b
        # f = np.linalg.pinv(np.linalg.pinv(inertia_2) @ fric_mat2 + np.linalg.pinv(inertia_1 @ np.linalg.pinv(convert1) @ convert2) @ fric_mat1) @ (np.linalg.pinv(inertia_2) @ Fnet2 - np.linalg.pinv(inertia_1 @ np.linalg.pinv(convert1) @ convert2) @ Fnet1)

        # 3) convert f_tan back to world coordinates
        f = T @ f_tan # (3×2)·(2,) → (3,)
        
        friction_mag = np.sqrt(np.dot(f, f))
        friction_dir = np.zeros(3) if friction_mag < 1e-5 else f / friction_mag

        vel1 = np.zeros((6,))
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_1, vel1, 0)
        vel2 = np.zeros((6,))
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_2, vel2, 0)

        acc1 = np.zeros((6,))
        mujoco.mj_objectAcceleration(model, data, mujoco.mjtObj.mjOBJ_BODY, body_1, acc1, 0)
        acc2 = np.zeros((6,))
        mujoco.mj_objectAcceleration(model, data, mujoco.mjtObj.mjOBJ_BODY, body_2, acc2, 0)
        
        # vel1 -= acc1 * model.opt.timestep
        # vel2 -= acc2 * model.opt.timestep
        
        rel_vel = convert1 @ vel1 - convert2 @ vel2
        rel_vel_mag = np.sqrt(np.dot(rel_vel, rel_vel))

        if rel_vel_mag > 1e-3 : 
            coefficient = cd
            # print('slip')
        else: 
            coefficient = cs
            # print('no slip')

        distribution_factor = 0 if uniques[ordered] < 1e-5 else (normal_mag / uniques[ordered])

        _ = min(friction_mag * distribution_factor, normal_mag * coefficient)
        f = friction_dir * _
        
        if data.time > 0.5: st()

        # print(np.dot(friction_dir, np.array([-np.cos(np.pi/3), 0, np.sin(np.pi/3)])))

        mujoco.mj_applyFT(model, data, f, np.zeros(3), contact_point if r else data.xipos[body_1], body_1, qfrc)
        mujoco.mj_applyFT(model, data, -f, np.zeros(3), contact_point if r else data.xipos[body_2], body_2, qfrc)

        applied_friction_frc[ordered] = applied_friction_frc.get(ordered, np.zeros(3)) + f

    return qfrc + damping_forces, applied_normal_frc, applied_friction_frc

def calculate_contact_force_v4(coefficient_friction, coefficient_restitution, model, data, damping = None):
    '''
    Calculate friction force to be applied on a body
    '''
    # if damping is None: damping = 0

    nv = data.cdof.shape[0]
    inertia = np.zeros((nv, nv))
    mujoco.mj_fullM(model, inertia, data.qM)
    inv_inertia = np.linalg.pinv(inertia)
    
    Fq = data.qfrc_applied + data.qfrc_actuator + data.qfrc_passive - data.qfrc_bias

    applied_normal_frc = {}
    applied_friction_frc = {}

    # Add all normal forces
    uniques = {}
    arr = np.zeros(nv)
    damping_forces = np.zeros(nv)
    custom_normal_forces = np.zeros(len(data.contact))
    for i in range(len(data.contact)):
        c = data.contact[i]
        geom_1, geom_2 = tuple(c.geom)
        
        ordered = min(geom_1, geom_2), max(geom_1, geom_2)
        
        if c.efc_address == -1 : 
            continue

        body_1, body_2 = model.geom_bodyid[geom_1], model.geom_bodyid[geom_2]

        contact_point = c.pos
        r_vec1, r_vec2 = contact_point - data.xpos[body_1], contact_point - data.xpos[body_2]        
        
        normal_mag = data.efc_force[c.efc_address]
        normal_dir = c.frame[:3]
        _ = np.sqrt(np.dot(normal_dir, normal_dir))
        normal_dir = np.zeros_like(normal_dir) if _ < 1e-5 else normal_dir / _  
        normal_dir = normal_dir * np.sign(np.dot(normal_dir, -r_vec1))

        Jl1, Jr1 = np.zeros((3, nv)), np.zeros((3, nv))
        mujoco.mj_jacBodyCom(model, data, Jl1, Jr1, body_1)

        Jcom1 = np.concatenate((Jl1, Jr1), axis = 0)

        Jl2, Jr2 = np.zeros((3, nv)), np.zeros((3, nv))
        mujoco.mj_jacBodyCom(model, data, Jl2, Jr2, body_2)

        Jcom2 = np.concatenate((Jl2, Jr2), axis = 0)

        inert_jac1, inert_jac_t1 = np.block([[Jr1], [Jl1]]).reshape((6, nv)), np.block([Jr1.T, Jl1.T]).reshape((nv, 6))
        inert_jac2, inert_jac_t2 = np.block([[Jr2], [Jl2]]).reshape((6, nv)), np.block([Jr2.T, Jl2.T]).reshape((nv, 6))

        inertia_1 = inert_jac1 @ inertia @ inert_jac_t1
        inertia_2 = inert_jac2 @ inertia @ inert_jac_t2

        Jl1, Jr1 = np.zeros((3, nv)), np.zeros((3, nv))
        mujoco.mj_jac(model, data, Jl1, Jr1, contact_point, body_1)

        Jcon1 = np.concatenate((Jl1, Jr1), axis = 0)

        Jl2, Jr2 = np.zeros((3, nv)), np.zeros((3, nv))
        mujoco.mj_jac(model, data, Jl2, Jr2, contact_point, body_2)

        Jcon2 = np.concatenate((Jl2, Jr2), axis = 0)

        for tup in coefficient_restitution: 
            g1, g2, ce = tup
            if (model.geom(g1).id == geom_1 and model.geom(g2).id == geom_2) or (model.geom(g2).id == geom_1 and model.geom(g1).id == geom_2):
                break
        else:
            ce = 1

        if ce != 1:
            RVEC1, RVEC2 = cross(r_vec1), cross(r_vec2)
            convert1 = np.block([np.eye(3), -RVEC1])
            convert2 = np.block([np.eye(3), -RVEC2])

            v = convert1 @ Jcon1 @ data.qvel - convert2 @ Jcon2 @ data.qvel
            v = np.dot(v, normal_dir)

            m_eff = np.linalg.pinv(normal_dir[:, None].T@(convert1@Jcon1@np.linalg.pinv(inertia)@Jcon1.T + convert2@Jcon2@np.linalg.pinv(inertia)@Jcon2.T)@ np.block([normal_dir, np.zeros_like(normal_dir)])[:, None])
            tau = c.solref[0]
            mu = c.solref[1]

            if tau > 0 or mu > 0:
                k = 1/ tau**2 / mu**2
                _b = 2 / tau
            else:
                k = -tau
                _b = -mu

            k *= impedance_scaling(c.dist, c.solimp)

            b = -2*math.sqrt(k*m_eff)*math.log(max(ce, 1e-16))/np.pi - _b

            mujoco.mj_applyFT(model, data, -b * v * normal_dir, np.zeros(3), contact_point, body_1, damping_forces)
            mujoco.mj_applyFT(model, data, b * v * normal_dir, np.zeros(3), contact_point, body_2, damping_forces)

            normal_mag += b * v

        if ordered in uniques:
            uniques[ordered] += normal_mag * normal_dir
        else:
            uniques[ordered] = normal_mag * normal_dir 
        
        custom_normal_forces[i] = normal_mag
        
        mujoco.mj_applyFT(model, data, normal_mag * normal_dir, np.zeros(3), contact_point, body_1, arr)
        mujoco.mj_applyFT(model, data, -normal_mag * normal_dir, np.zeros(3), contact_point, body_2, arr)

    Fq += damping_forces

    for k in uniques.keys():
        uniques[k] = np.sqrt(np.dot(uniques[k], uniques[k]))
        applied_normal_frc[k] = uniques[k]

    qfrc = np.zeros(nv)

    # — preamble: after you’ve computed
    #   • inertia (nv×nv full M)
    #   • inv_inertia = np.linalg.pinv(inertia)           # new
    #   • data.qvel
    #   • data.contact list
    #   • for each contact i you have:
    #       body_1[i], body_2[i]
    #       contact_point[i], normal_dir[i]
    #       Jcon1[i], Jcon2[i]        # each 6×nv from mj_jac
    #       fric_mat1[i], fric_mat2[i] # each 6×3 = [I; cross(r)]
    #       # net‐force terms
    #       Fnet1[i] = Jcom1[i] @ Fq   # 6‑vector
    #       Fnet2[i] = Jcom2[i] @ Fq   # 6‑vector

    # 1) build 2×3 tangent basis T_i for each contact
    T_list = []
    Jrel_list = []
    fric_list = []
    b_list = []
    
    for i in range(len(data.contact)):
        c = data.contact[i]
        geom_1, geom_2 = tuple(c.geom)
        ordered = min(geom_1, geom_2), max(geom_1, geom_2)
        
        body_1, body_2 = model.geom_bodyid[geom_1], model.geom_bodyid[geom_2]

        for tup in coefficient_friction: 
            g1, g2, cs, cd, r = tup
            if (model.geom(g1).id == geom_1 and model.geom(g2).id == geom_2) or (model.geom(g2).id == geom_1 and model.geom(g1).id == geom_2):
                break
        else:
            continue

        Jl1, Jr1 = np.zeros((3, nv)), np.zeros((3, nv))
        mujoco.mj_jacBodyCom(model, data, Jl1, Jr1, body_1)

        Jcom1 = np.concatenate((Jl1, Jr1), axis = 0)

        Jl2, Jr2 = np.zeros((3, nv)), np.zeros((3, nv))
        mujoco.mj_jacBodyCom(model, data, Jl2, Jr2, body_2)

        Jcom2 = np.concatenate((Jl2, Jr2), axis = 0)
        
        contact_point = c.pos
        if c.efc_address == -1 : continue
        normal_mag = custom_normal_forces[i] # data.efc_force[c.efc_address]
        r_vec1, r_vec2 = contact_point - data.xpos[body_1], contact_point - data.xpos[body_2]        
        RVEC1, RVEC2 = cross(r_vec1), cross(r_vec2)

        Fnet1, Fnet2 = Jcom1 @ Fq, Jcom2 @ Fq

        fric_mat1 = np.block([[np.eye(3)],
                              [RVEC1]])
        fric_mat2 = np.block([[np.eye(3)],
                              [RVEC2]])
        
        inert_jac1, inert_jac_t1 = np.block([[Jr1], [Jl1]]).reshape((6, nv)), np.block([Jr1.T, Jl1.T]).reshape((nv, 6))
        inert_jac2, inert_jac_t2 = np.block([[Jr2], [Jl2]]).reshape((6, nv)), np.block([Jr2.T, Jl2.T]).reshape((nv, 6))

        inertia_1 = inert_jac1 @ inertia @ inert_jac_t1
        inertia_2 = inert_jac2 @ inertia @ inert_jac_t2

        convert1 = np.block([np.eye(3), -RVEC1])
        convert2 = np.block([np.eye(3), -RVEC2])

        normal_dir = c.frame[:3]
        _ = np.sqrt(np.dot(normal_dir, normal_dir))
        normal_dir = np.zeros_like(normal_dir) if _ < 1e-5 else normal_dir / _  
        normal_dir = normal_dir * np.sign(np.dot(normal_dir, -r_vec1))

        # your existing code already has:
        #   normal_dir  (3,)
        #   r1 = contact_point - data.xpos[body_1], same for r2
        #   Jcon1, Jcon2  (6×nv)
        #   fric_mat1, fric_mat2  (6×3)
        #   Fnet1, Fnet2  (6,)

        # --- new: tangent basis on plane perpendicular to normal_dir
        n = normal_dir
        # pick any vector not collinear with n
        t0 = np.array([1.,0,0.])
        if abs(n.dot(t0)) > 0.9:
            t0 = np.array([0.,1,0.])
        t1 = np.cross(n, t0)
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(n, t1)  
        T = np.stack([t1, t2], axis=1)        # shape (3,2)
        T_list.append(T)

        # --- build the *relative* Jacobian 3×nv for this contact:
        #    v_rel = (v + ω×r1) - (v + ω×r2)
        #    use your convert1, convert2 (3×6) and Jcon1,Jcon2 (6×nv)
        Jrel = convert1 @ Jcon1 - convert2 @ Jcon2  # shape (3, nv)
        Jrel_list.append(Jrel)

        # --- store fric_mats so we can ask “how f_j leaks into the 6D wrench”
        fric_list.append((fric_mat1, fric_mat2))

        # --- external slip‐side: b_i = -Tᵀ · (Jrel @ qvel)
        b_i = - T.T @ (Jrel @ data.qvel)           # shape (2,)
        b_list.append(b_i)

    # now n = len(T_list)
    n = len(T_list)

    if n == 0:
        return qfrc + damping_forces, applied_normal_frc, applied_friction_frc

    G = np.zeros((2*n, 2*n))
    d = np.concatenate(b_list)  # shape (2n,)
    
    # 2) assemble full 2n×2n coupling matrix G
    for i in range(n):
        T_i = T_list[i]       # 3×2
        Jrel_i = Jrel_list[i] # 3×nv
        for j in range(n):
            fr1_j, fr2_j = fric_list[j]  # each 6×3
            T_j = T_list[j]              # 3×2

            # A_ij maps contact‐j’s 3D friction into contact‐i’s 3D relative accel:
            #   A_ij = Jrel_i @ invM @ fric_mat_j_total
            # where invM is nv×nv, fric_mat_j_total is 6×3:
            #    we must sum the two bodies’ contributions:
            try:
                Aij = (Jrel_i @ inv_inertia @ fr1_j) \
                    - (Jrel_i @ inv_inertia @ fr2_j)  # shape (3×3)
            except:
                st()

            # project into tangent → 2×2 block
            G_block = T_i.T @ Aij @ T_j          # (2×3)·(3×3)·(3×2) = (2×2)
            G[2*i:2*i+2, 2*j:2*j+2] = G_block

    # 3) solve the 2n×2n linear system
    f2D_all = np.linalg.solve(G, d)  # shape (2n,)

    f_all_cache = []
    # 4) lift back to 3D friction for each contact
    for i in range(n):
        T_i = T_list[i]
        f2 = f2D_all[2*i:2*i+2]        # tangent coefficients
        f3 = T_i @ f2                  # world‐space 3D friction
        # now apply f3 (and any rolling torque) at contact i:
        mujoco.mj_applyFT(
        model, data,
        f3,         # linear
        np.zeros(3),# or your rolling torque T
        contact_point[i],
        body_1[i],
        qfrc
        )
        f_all_cache.append(f3)

    if data.time > 0.5: st()

    return qfrc + damping_forces, applied_normal_frc, applied_friction_frc

def calculate_contact_force_old(coefficient_friction, coefficient_restitution, model, data, damping = None, tangential_stiffness=TANGENTIAL_STIFFNESS, static_vel_threshold=STATIC_VEL_THRESHOLD):
    '''
    Calculate custom damping (for restitution) and friction forces to be applied on bodies in contact.
    These forces are ADDED to the simulation using data.qfrc_applied.
    MuJoCo's solver still runs based on its solimp/solref and other forces.
    This function calculates the ADDITIONAL forces only.

    Args:
        coefficient_friction: Kinetic friction coefficient (mu_k). Could be a list/tuple for per-contact.
                              For simplicity, using a single value mu_k here.
                              For proper static friction, you'd also need mu_s. Let's assume mu_s = mu_k for this simple model,
                              or pass as (mu_s, mu_k) and use appropriately. Using a single mu_k for now.
        coefficient_restitution: List of tuples (geom_name1, geom_name2, restitution_coeff) for custom restitution.
        model: MuJoCo model
        data: MuJoCo data
        tangential_stiffness: Stiffness for the friction damping term (tuning parameter).
        static_vel_threshold: Relative tangential speed below which static friction behavior is attempted.

    Returns:
        custom_qfrc: The calculated generalized forces to be added to data.qfrc_applied.
        applied_normal_damping_frc_mag: Dictionary for logging custom normal damping force magnitudes.
        applied_friction_frc_mag: Dictionary for logging custom friction force magnitudes.
        estimated_normal_mags: Dictionary for logging estimated normal force magnitudes used for friction limit.
    '''
    nv = model.nv # Use model.nv for degrees of freedom

    # Array to accumulate the generalized forces from custom contacts
    # This will be added to data.qfrc_applied BEFORE mj_step
    custom_qfrc = np.zeros(nv)

    applied_normal_damping_frc_mag = {} # For logging/debugging
    applied_friction_frc_mag = {} # For logging/debugging
    estimated_normal_mags = {} # For logging/debugging

    # Pre-calculate geom ID to friction mapping for quick lookup
    # Using the first element of geom_friction for tangential friction
    # This assumes you've set model.geom_friction appropriately in your XML/model loading
    # If not, you might need to pass friction coefficients per-pair like restitution
    geom_id_to_friction = { i: model.geom_friction[i, 0] for i in range(model.ngeom) }

    for i in range(data.ncon): # Iterate over active contacts
        c = data.contact[i]

        # Skip disabled contacts or contacts not handled by EFC (if any)
        # Basing friction calculation on data.efc_force requires efc_address to be valid.
        if c.efc_address == -1:
             continue

        geom_1, geom_2 = c.geom

        # Store forces by ordered geom pair for potential unique tracking
        ordered_geoms = min(geom_1, geom_2), max(geom_1, geom_2)

        geom_1, geom_2 = ordered_geoms
        body_1, body_2 = model.geom_bodyid[geom_1], model.geom_bodyid[geom_2]

        contact_point = c.pos
        r_vec1, r_vec2 = contact_point - data.xpos[body_1], contact_point - data.xpos[body_2]

        # Get contact normal (pointing from geom2 towards geom1 by convention in c.frame)
        # Ensure it's normalized
        normal_dir = c.frame[:3]
        norm_normal = np.linalg.norm(normal_dir)
        if norm_normal < 1e-6:
             # print(f"Warning: Degenerate normal at contact {i}") # Avoid spamming console
             continue # Skip this contact if normal is zero

        normal_dir /= norm_normal
        normal_dir = normal_dir * np.sign(np.dot(normal_dir, -r_vec1))
        normal_mag = data.efc_force[c.efc_address] # Normal force magnitude from MuJoCo's EFC

        # --- Normal Force (Damping for Restitution) ---
        # Get the coefficient of restitution for this pair
        ce = 1.0 # Default to elastic if not found
        for g1_name, g2_name, restit_coeff in coefficient_restitution:
            try:
                g1_id = model.geom(g1_name).id
                g2_id = model.geom(g2_name).id
                if (g1_id == geom_1 and g2_id == geom_2) or (g2_id == geom_1 and g1_id == geom_2):
                    ce = restit_coeff
                    break
            except:
                 # Handle cases where geom names might be invalid in the input list
                 print(f"Warning: Geom names '{g1_name}' or '{g2_name}' not found in model.")
                 continue

        F_normal_damping_vec = np.zeros(3) # Initialize damping force

        if ce < 1.0: # Only calculate/add damping if restitution is less than perfect
            # Calculate relative velocity at the contact point
            # OLD (BUG): mj_jac returns (jacp, jacr) for ONE body, not two bodies.
            #   J2_point was the rotational Jacobian of body_1, not body_2's positional Jacobian.
            # J1_point, J2_point = np.zeros((3, nv)), np.zeros((3, nv))
            # mujoco.mj_jac(model, data, J1_point, J2_point, contact_point, body_1)
            # v_rel_world = J1_point @ data.qvel - J2_point @ data.qvel
            # FIX: compute positional Jacobian for BOTH bodies at the contact point
            J1_pos, J1_rot = np.zeros((3, nv)), np.zeros((3, nv))
            mujoco.mj_jac(model, data, J1_pos, J1_rot, contact_point, body_1)
            J2_pos, J2_rot = np.zeros((3, nv)), np.zeros((3, nv))
            mujoco.mj_jac(model, data, J2_pos, J2_rot, contact_point, body_2)
            v_rel_world = J1_pos @ data.qvel - J2_pos @ data.qvel

            # Relative normal velocity (velocity of body 1 point along normal away from body 2)
            v_rel_normal = np.dot(v_rel_world, normal_dir) # positive means separating

            # Use MuJoCo's calculated inverse effective mass for the normal constraint
            # Effective mass m_eff = 1 / data.efc_R[c.efc_address]
            m_eff_normal = 0.0
            if abs(data.efc_R[c.efc_address]) > 1e-9:
                 m_eff_normal = 1.0 / data.efc_R[c.efc_address] # Use MuJoCo's calculated effective mass

            # MuJoCo's impedance parameters (k, d) derived from solref (tau, mu)
            tau = c.solref[0]
            mu_solref = c.solref[1] # This mu in solref is damping ratio related, not friction coeff

            k_solref = 0.0
            d_solref = 0.0
            if tau > 0 or mu_solref > 0:
                k_solref = 1/ tau**2 / mu_solref**2 if mu_solref > 1e-9 else 1/ tau**2 / 1e-9**2 # Avoid div by zero
                d_solref = 2 / tau if tau > 1e-9 else 2 / 1e-9 # Avoid div by zero
            elif tau < 0 or mu_solref < 0: # Negative solref means direct (stiffness, damping)
                k_solref = -tau
                d_solref = -mu_solref

            # Apply impedance scaling based on distance
            scale = impedance_scaling(c.dist, c.solimp)
            k_scaled = k_solref * scale
            d_scaled = d_solref * scale
            k_solref = k_scaled
            d_solref = d_scaled

            # Calculate ADDITIONAL damping 'b' based on desired 'ce'
            b_additional_normal = 0.0
            zeta_desired = 0.0
            d_total_desired = 0.0
            if m_eff_normal > 1e-9 and k_solref >= 0: # Need positive mass and non-negative stiffness
                log_ce = math.log(max(ce, 1e-16)) # Ensure ce is > 0
                # Desired total damping coefficient d_total = 2 * zeta_desired * sqrt(k_eff * m_eff)
                # zeta_desired = -log(ce) / sqrt(pi^2 + log(ce)^2)
                sqrt_term = math.sqrt(math.pi**2 + log_ce**2)
                zeta_desired = -log_ce / sqrt_term if sqrt_term > 1e-9 else 0

                d_total_desired = 2 * zeta_desired * math.sqrt(k_solref * m_eff_normal)

                # Additional damping needed = Total desired - MuJoCo's base damping (approximated from solref)
                # This formula is from the user's code logic - assumes d_solref is the base damping
                b_additional_normal = d_total_desired - d_solref

            # --- DEBUG: COR parameters (first 3 steps + every 10th) ---
            if not hasattr(calculate_contact_force_old, '_cor_debug_count'):
                calculate_contact_force_old._cor_debug_count = 0
            calculate_contact_force_old._cor_debug_count += 1
            _n = calculate_contact_force_old._cor_debug_count
            if _n <= 3 or _n % 10 == 0:
                print(f"  [COR DEBUG step={_n} t={data.time:.4f}] ce={ce}, solref=({c.solref[0]}, {c.solref[1]})")
                print(f"    k_raw={-c.solref[0] if c.solref[0]<0 else 'N/A'}, scale={scale:.6f}, k_scaled={k_scaled:.2f}, d_base={d_scaled:.6f}")
                print(f"    m_eff={m_eff_normal:.6f}, efc_R={data.efc_R[c.efc_address]:.6e}, dist={c.dist:.6e}")
                print(f"    zeta_desired={zeta_desired:.6f}, d_total_desired={d_total_desired:.6f}, d_solref(scaled)={d_scaled:.6f}")
                print(f"    b_additional={b_additional_normal:.6f}, v_rel_normal={v_rel_normal:.6f}")
                print(f"    |F_damping|={abs(b_additional_normal * v_rel_normal):.6f}, normal_mag(mujoco)={normal_mag:.6f}")
                print(f"    solimp={list(c.solimp)}")

            # The force should oppose the closing velocity.
            # If v_rel_normal < 0 (closing), F_normal_damping_vec should be positive * normal_dir
            # Force vector: -b_additional_normal * v_rel_normal * normal_dir
            F_normal_damping_vec = -b_additional_normal * v_rel_normal * normal_dir

            # Add this additional normal damping force to the custom qfrc
            mujoco.mj_applyFT(model, data, F_normal_damping_vec, np.zeros(3), contact_point, body_1, custom_qfrc)
            mujoco.mj_applyFT(model, data, -F_normal_damping_vec, np.zeros(3), contact_point, body_2, custom_qfrc)

            # Store magnitude for logging
            # applied_normal_damping_frc_mag[ordered_geoms] = np.linalg.norm(F_normal_damping_vec)
        
        estimated_normal_mags[i] = F_normal_damping_vec + normal_mag * normal_dir # Store estimated normal force magnitude
        applied_normal_damping_frc_mag[ordered_geoms] = applied_normal_damping_frc_mag.get(ordered_geoms, 0) + F_normal_damping_vec + normal_mag * normal_dir

    # --- Friction Force ---
    # Estimate the total normal force magnitude acting at the contact.
    # This is the tricky part in a hybrid model. We need an estimate *before* MuJoCo's solver finishes.
    # Option 1 (used here): Sum magnitudes of MuJoCo's elastic force estimate (from efc_force)
    # and our custom normal damping force. This is an approximation.
    # Option 2: Base it only on the custom damping force magnitude. (Ignores MuJoCo's base elastic force)
    # Option 3: Base it on gravity/external forces projected onto the normal. (Complex, local effect)
    # Let's use Option 1 as it aligns with the user's initial normal force calculation idea.
    # Note: data.efc_force[c.efc_address] is the force *computed by MuJoCo's solver* (potentially from a previous iteration or step).
    # Using it directly as an input for forces applied *before* the solver is conceptually inconsistent
    # but often works as a practical heuristic in these hybrid force-based methods.
    for i in range(data.ncon):
        c = data.contact[i]

        # Skip disabled contacts or contacts not handled by EFC (if any)
        # Basing friction calculation on data.efc_force requires efc_address to be valid.
        if c.efc_address == -1:
            continue

        geom_1, geom_2 = c.geom

        # Store forces by ordered geom pair for potential unique tracking
        ordered_geoms = min(geom_1, geom_2), max(geom_1, geom_2)

        geom_1, geom_2 = ordered_geoms
        body_1, body_2 = model.geom_bodyid[geom_1], model.geom_bodyid[geom_2]

        contact_point = c.pos
        r_vec1, r_vec2 = contact_point - data.xpos[body_1], contact_point - data.xpos[body_2]

        # Get contact normal (pointing from geom2 towards geom1 by convention in c.frame)
        # Ensure it's normalized
        normal_dir = c.frame[:3]
        norm_normal = np.linalg.norm(normal_dir)
        if norm_normal < 1e-6:
            # print(f"Warning: Degenerate normal at contact {i}") # Avoid spamming console
            continue # Skip this contact if normal is zero

        normal_dir /= norm_normal
        normal_dir = normal_dir * np.sign(np.dot(normal_dir, -r_vec1))
        normal_mag = data.efc_force[c.efc_address] # Normal force magnitude from MuJoCo's EFC

        estimated_normal_mag = estimated_normal_mags.get(i, 0)
        estimated_normal_mag = np.linalg.norm(estimated_normal_mag)

        # Get friction coefficient (using the provided mu_k)
        # A more robust model would use mu_s for static and mu_k for kinetic based on speed threshold
        # For simplicity, let's use the passed coefficient_friction as mu_k and a simple damping model for friction
        # If you need mu_s, pass (mu_s, mu_k) and use them here.
        for g1_name, g2_name, cs, cd, r in coefficient_friction:
            g1_id, g2_id = model.geom(g1_name).id, model.geom(g2_name).id

            if (g1_id == geom_1 and g2_id == geom_2) or (g2_id == geom_1 and g1_id == geom_2):
                mu_k = cd
                mu_s = cs # Simple assumption if only mu_k is provided

                rolling = r

                break
        else: continue # Skip if no matching friction coefficient found

        friction_limit_kinetic = mu_k * estimated_normal_mag
        friction_limit_static = mu_s * estimated_normal_mag

        # Calculate relative tangential velocity
        J1_point_lin, J1_point_rot, J2_point_lin, J2_point_rot = (
            np.zeros((3, nv)), np.zeros((3, nv)),
            np.zeros((3, nv)), np.zeros((3, nv)), # Re-calculate Jacs if needed, or reuse from normal section
        )
        mujoco.mj_jac(model, data, J1_point_lin, J1_point_rot, contact_point, body_1)
        mujoco.mj_jac(model, data, J2_point_lin, J2_point_rot, contact_point, body_2)
        v_rel_world = J1_point_lin @ data.qvel - J2_point_lin @ data.qvel # Relative velocity of body 1 pt wrt body 2 pt

        v_rel_normal_vec = np.dot(v_rel_world, normal_dir) * normal_dir
        v_rel_tangent_vec = v_rel_world - v_rel_normal_vec # Tangential component
        v_rel_tangent_speed = np.linalg.norm(v_rel_tangent_vec)

        F_friction_vec = np.zeros(3) # Initialize friction force vector

        # Determine friction force based on tangential velocity and normal force limit
        if v_rel_tangent_speed > static_vel_threshold:
            # Kinetic Friction: Apply force opposite velocity with magnitude mu_k * N
            if v_rel_tangent_speed > 1e-6: # Avoid division by zero
                friction_dir = -v_rel_tangent_vec / v_rel_tangent_speed
                F_friction_vec = friction_limit_kinetic * friction_dir
        else:
            # Static Friction Regime (attempt to stop motion):
            # Calculate force needed to stop motion using a tangential stiffness (damping)
            # This models the "stickiness" - higher stiffness means a smaller velocity
            # will generate the required force to stop motion.
            # The required force vector is -tangential_stiffness * v_rel_tangent_vec
            F_needed_static = -tangential_stiffness * v_rel_tangent_vec
            F_needed_static_mag = np.linalg.norm(F_needed_static)

            if F_needed_static_mag <= friction_limit_static:
                # Static friction can hold: Apply the required force
                F_friction_vec = F_needed_static
            else:
                # Static friction limit exceeded: Transition to kinetic friction
                # Apply kinetic friction force in the direction opposite current velocity
                if v_rel_tangent_speed > 1e-6: # Should be true if we exceed static limit and vel > threshold, but safe check
                    friction_dir = -v_rel_tangent_vec / v_rel_tangent_speed
                    F_friction_vec = friction_limit_kinetic * friction_dir


        # Apply the friction force
        # This force is added to custom_qfrc, which will be added to data.qfrc_applied
        point_of_application1 = contact_point if rolling else data.xipos[body_1] # Apply at contact point or body center
        point_of_application2 = contact_point if rolling else data.xipos[body_2] # Apply at contact point or body center
        mujoco.mj_applyFT(model, data, F_friction_vec, np.zeros(3), point_of_application1, body_1, custom_qfrc)
        mujoco.mj_applyFT(model, data, -F_friction_vec, np.zeros(3), point_of_application2, body_2, custom_qfrc)

        # Store magnitudes for logging
        applied_friction_frc_mag[ordered_geoms] = applied_friction_frc_mag.get(ordered_geoms, 0) + F_friction_vec

    # Return the total accumulated custom qfrc
    # This should be added to data.qfrc_applied before calling mj_step
    # Also return logged forces for inspection
    
    return custom_qfrc, applied_normal_damping_frc_mag, applied_friction_frc_mag # , estimated_normal_mags

def calculate_contact_force(coefficient_friction, coefficient_restitution, model, data, damping=None, tangential_stiffness=TANGENTIAL_STIFFNESS, static_vel_threshold=STATIC_VEL_THRESHOLD):
    '''
    Calculate custom friction forces to be applied on bodies in contact.
    COR (restitution) is handled separately by apply_restitution_correction(),
    which should be called AFTER mj_step.

    This function delegates to calculate_contact_force_old with an empty
    restitution list so that the COR damping block is skipped entirely.
    '''
    return calculate_contact_force_old(
        coefficient_friction,
        [],  # Empty restitution list → skips COR damping (ce defaults to 1.0)
        model, data, damping=damping,
        tangential_stiffness=tangential_stiffness,
        static_vel_threshold=static_vel_threshold,
    )


def apply_restitution_correction(coefficient_restitution, model, data, cor_state):
    '''
    Apply post-collision velocity correction to enforce coefficient of restitution.

    Instead of applying continuous damping during contact (which interferes with
    MuJoCo's constraint solver), this function:
      1. Detects when a collision STARTS → records pre-collision relative normal velocity
      2. Lets MuJoCo handle contact dynamics normally (elastic bounce)
      3. Detects when a collision ENDS → applies a one-time impulse to adjust
         the relative normal velocity to match the desired COR

    This directly enforces the COR definition: e = -v_rel_post / v_rel_pre
    while preserving momentum and respecting all joint constraints.

    Args:
        coefficient_restitution: List of (geom_name1, geom_name2, e) tuples
        model: MuJoCo model
        data: MuJoCo data
        cor_state: Dict for tracking contact state across calls.
                   Initialize as empty dict {} before the simulation loop.
                   Keys managed internally:
                     'prev_contacts': set of active contact pairs from last step
                     'pre_collision_data': dict of stored pre-collision info per pair

    Returns:
        None (modifies data.qvel in-place)
    '''
    if not coefficient_restitution:
        return

    # Initialize state on first call
    if 'prev_contacts' not in cor_state:
        cor_state['prev_contacts'] = set()
        cor_state['pre_collision_data'] = {}

    nv = model.nv

    # Build lookup: ordered geom pair → (ce, geom_name1, geom_name2)
    cor_lookup = {}
    for g1_name, g2_name, ce in coefficient_restitution:
        try:
            g1_id = model.geom(g1_name).id
            g2_id = model.geom(g2_name).id
            ordered = (min(g1_id, g2_id), max(g1_id, g2_id))
            cor_lookup[ordered] = ce
        except Exception as e:
            print(f"  [COR IMPULSE] WARNING: geom lookup failed for '{g1_name}'/'{g2_name}': {e}")
            continue

    # # Debug: print lookup table on first call
    # if not hasattr(apply_restitution_correction, '_debug_init'):
    #     apply_restitution_correction._debug_init = True
    #     print(f"  [COR IMPULSE] cor_lookup = {cor_lookup}")
    #     print(f"  [COR IMPULSE] coefficient_restitution = {coefficient_restitution}")

    # Identify current active contacts with COR pairs
    current_contacts = set()
    contact_normals = {}  # ordered pair → normal direction

    for i in range(data.ncon):
        c = data.contact[i]
        if c.efc_address == -1:
            continue

        geom_1, geom_2 = c.geom
        ordered = (min(geom_1, geom_2), max(geom_1, geom_2))

        if ordered not in cor_lookup:
            continue

        if cor_lookup[ordered] >= 1.0:
            continue  # No correction needed for elastic collisions

        current_contacts.add(ordered)

        # Store contact normal (for use at start of collision)
        body_1 = model.geom_bodyid[ordered[0]]
        body_2 = model.geom_bodyid[ordered[1]]
        normal_dir = c.frame[:3].copy()
        norm = np.linalg.norm(normal_dir)
        if norm < 1e-6:
            continue
        normal_dir /= norm
        # Orient normal to point away from body_1
        r_vec1 = c.pos - data.xpos[body_1]
        normal_dir *= np.sign(np.dot(normal_dir, -r_vec1))

        contact_normals[ordered] = {
            'normal': normal_dir,
            'contact_point': c.pos.copy(),
            'body_1': body_1,
            'body_2': body_2,
        }

    # --- Detect NEW contacts (collision start) ---
    new_contacts = current_contacts - cor_state['prev_contacts']
    # if new_contacts:
    #     print(f"  [COR IMPULSE t={data.time:.4f}] NEW contacts: {new_contacts}, current={current_contacts}, prev={cor_state['prev_contacts']}")
    for pair in new_contacts:
        if pair not in contact_normals:
            continue
        info = contact_normals[pair]
        body_1, body_2 = info['body_1'], info['body_2']
        normal = info['normal']
        contact_point = info['contact_point']

        # Compute relative normal velocity at contact point
        J1_pos, J1_rot = np.zeros((3, nv)), np.zeros((3, nv))
        mujoco.mj_jac(model, data, J1_pos, J1_rot, contact_point, body_1)
        J2_pos, J2_rot = np.zeros((3, nv)), np.zeros((3, nv))
        mujoco.mj_jac(model, data, J2_pos, J2_rot, contact_point, body_2)

        v_rel = J1_pos @ data.qvel - J2_pos @ data.qvel
        v_rel_n = np.dot(v_rel, normal)

        cor_state['pre_collision_data'][pair] = {
            'v_rel_pre_n': v_rel_n,
            'normal': normal.copy(),
            'ce': cor_lookup[pair],
            'body_1': body_1,
            'body_2': body_2,
        }

    # --- Detect ENDED contacts (collision end) → apply velocity correction ---
    ended_contacts = cor_state['prev_contacts'] - current_contacts
    for pair in ended_contacts:
        if pair not in cor_state['pre_collision_data']:
            continue

        pre_data = cor_state['pre_collision_data'].pop(pair)
        v_rel_pre_n = pre_data['v_rel_pre_n']
        normal = pre_data['normal']
        ce = pre_data['ce']
        body_1 = pre_data['body_1']
        body_2 = pre_data['body_2']

        # Current relative normal velocity (post elastic collision from MuJoCo)
        J1_pos, J1_rot = np.zeros((3, nv)), np.zeros((3, nv))
        mujoco.mj_jacBodyCom(model, data, J1_pos, J1_rot, body_1)
        J2_pos, J2_rot = np.zeros((3, nv)), np.zeros((3, nv))
        mujoco.mj_jacBodyCom(model, data, J2_pos, J2_rot, body_2)

        v_rel_post = J1_pos @ data.qvel - J2_pos @ data.qvel
        v_rel_post_n = np.dot(v_rel_post, normal)

        # Desired relative normal velocity from COR definition
        v_rel_desired_n = -ce * v_rel_pre_n

        # Velocity correction needed
        delta_v_n = v_rel_desired_n - v_rel_post_n

        if abs(delta_v_n) < 1e-10:
            continue  # No correction needed

        # Compute effective mass: 1/m_eff = n^T (J1 M^-1 J1^T + J2 M^-1 J2^T) n
        M_full = np.zeros((nv, nv))
        mujoco.mj_fullM(model, M_full, data.qM)
        M_inv = np.linalg.inv(M_full)

        W = J1_pos @ M_inv @ J1_pos.T + J2_pos @ M_inv @ J2_pos.T  # 3x3
        inv_m_eff = normal @ W @ normal

        if abs(inv_m_eff) < 1e-12:
            continue

        m_eff = 1.0 / inv_m_eff

        # Impulse magnitude
        J_impulse_scalar = m_eff * delta_v_n

        # Apply impulse in generalized coordinates
        # Body 1 gets +J along normal, Body 2 gets -J along normal
        impulse_cartesian = J_impulse_scalar * normal
        data.qvel += M_inv @ (J1_pos.T - J2_pos.T) @ impulse_cartesian

    # Update state for next call
    cor_state['prev_contacts'] = current_contacts