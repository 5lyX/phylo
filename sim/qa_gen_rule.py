import ipdb
st = ipdb.set_trace

import os
from datetime import datetime

os.environ["MUJOCO_GL"] = "glfw"
import json, pathlib
from tqdm import tqdm
import numpy as np

from omegaconf import OmegaConf, DictConfig
import hydra
import wandb
import pandas as pd
import asyncio
import random
from concurrent.futures import ProcessPoolExecutor

from collections import defaultdict
import re, ast

from sim.utils import replace_all, find_tags, restructure_data

from sim.scene import parse_scene, create_mappings
from recorder.recorder import Recorder
from sim.entities.pulley_entities import ConstantForceFixedPulley, FixedPulleyEntity

def _get_valid_string_names(restructured_data):
    """
    Keep only rope-like string channels and skip spring channels.
    Spring traces may not correspond to named scene tendons used in NLQ.
    """
    return [
        k
        for k in restructured_data.keys()
        if 'length' in restructured_data[k] and k.split('.')[-1][:6] != "spring"
    ]

POTENTIAL_FIND_QUANTITIES = {
    "pulley": {
                    "categories": ["BasicPulley", "IntermediatePulley", "BasicInclinedPlaneFriction", 
                                   "IntermediateInclinedPlaneFriction", "AdvancedInclinedPlaneFriction",
                                   "IntermediateHybrid", "AdvancedHybrid", "DifficultPulley"],
                    "masses": {
                        "net_force_linear":"magnitude of the net force", 
                        "acceleration_linear":"magnitude of the acceleration", 
                        "displacement":"magnitude of the displacement", 
                        "velocity_linear":"magnitude of the velocity",
                        "kinetic_energy": "net kinetic energy (rotation if any + linear)",
                        "kinetic_energy_linear": "linear kinetic energy",
                        "potential_energy": "change in potential energy (gravitational)",
                        },
                    "strings": {
                        "force": "tension",
                        }  # force is the tension in the string which is scalar, length is also a scalar
                },
    "collision": {
                    "categories": ["BasicCollision", "IntermediateCollision", "AdvancedCollision", 
                                   "DifficultProjectile"],
                    "masses": {
                        "velocity_linear": "magnitude of the velocity",
                        "momentum_linear": "magnitude of the linear momentum of COM",
                        "momentum_angular": "magnitude of the angular momentum about the COM",
                        "kinetic_energy": "net kinetic energy (rotation if any + linear)",
                        "kinetic_energy_linear": "linear kinetic energy",
                        "kinetic_energy_angular": "rotational kinetic energy",
                        },
                },
    "spring": { # Temporary. Add spring force
                    "categories": ["SpringBlockSystems", 
                                   "DifficultSpringMass"],
                    "masses": {
                        "velocity_linear": "magnitude of the velocity",
                        "momentum_linear": "magnitude of the linear momentum of COM",
                        "momentum_angular": "magnitude of the angular momentum about the COM",
                        "kinetic_energy": "net kinetic energy (rotation if any + linear)",
                        "kinetic_energy_linear": "linear kinetic energy",
                        "kinetic_energy_angular": "rotational kinetic energy",
                        },
                },
    "rotation": {
                    "categories": ["Rotation", "RigidBodyRotation", ], 
                    "masses": {
                        "momentum_linear": "magnitude of linear momentum of the COM",
                        "momentum_angular": "magnitude of angular momentum about the COM",
                        "acceleration_linear": "magnitude of linear acceleration of the COM",
                        "acceleration_angular": "magnitude of angular acceleration about the COM",
                        "velocity_angular": "magnitude of angular velocity about the COM",
                        "velocity_linear": "magnitude of linear velocity of the COM",
                        "kinetic_energy": "net kinetic energy (rotation if any + linear)",
                        "kinetic_energy_linear": "linear kinetic energy",
                        "kinetic_energy_angular": "rotational kinetic energy",
                        "potential_energy": "change in potential energy (gravitational)",     
                        "com_offset": "distance between COM and the body's frame",   
                        "inertia_z": "moment of inertia about axis passing through COM, parallel to z axis",  
                        "net_force_linear": "magnitude of the net force",
                        "net_torque": "magnitude of the net torque about the COM",
                        },
                },
    "orbital": {
                    "categories": ["OrbitalMotion", "DifficultOrbitalMotion", "GeneralCelestial", "DifficultRocket"],
                    "masses": {
                        "net_force_linear":"magnitude of the net force", 
                        "acceleration_linear":"magnitude of the acceleration", 
                        "displacement":"magnitude of the displacement", 
                        "velocity_linear":"magnitude of the velocity",
                        "kinetic_energy_linear": "linear kinetic energy",
                        "potential_energy": "change in potential energy (gravitational)",
                        },
                },
    "em": {
                    "categories": ["DifficultElectroMagnetic"],
                    "masses": {
                        "net_force_linear":"magnitude of the net force", 
                        "acceleration_linear":"magnitude of the acceleration", 
                        "displacement":"magnitude of the displacement", 
                        "velocity_linear":"magnitude of the velocity",
                        "kinetic_energy_linear": "linear kinetic energy",
                        "potential_energy": "change in potential energy (electrostatic)",
                        },
                },
}

VECTOR_QUANTITIES = [
    "net_force_linear",
    "net_torque",
    "acceleration_linear",
    "acceleration_angular",
    "velocity_linear",
    "velocity_angular",
    "momentum_linear",
    "momentum_angular",
    "com_offset",
]



def remove_empty_keys(data):
    """
    Recursively removes keys with empty values (empty lists or dictionaries) from a nested dictionary.

    :param data: The hierarchical dictionary to process.
    :return: A cleaned dictionary with empty keys removed.
    """
    if isinstance(data, dict):
        return {
            key: remove_empty_keys(value)
            for key, value in data.items()
            if (isinstance(value, dict) and value) or (isinstance(value, list) and value)
        }
    return data  # If not a dictionary, return the value as is

def run_async_task(task):
    """
    Runs an asynchronous task safely, ensuring compatibility with both existing event loops 
    and standalone execution.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        return asyncio.create_task(task)
    else:
        return asyncio.run(task)

def data_gen(
    scene_yaml,
    cfg,
    recorder_cfg,
    seed_offset=0,
    seed=None,
    scene=None,
    data=None,
    restructured_data=None,
):
    generation_cfg = cfg.question_generation

    if scene is None:
        try:
            scene = parse_scene("", scene_data_dict=scene_yaml)
        except Exception as e:
            raise RuntimeError(f"Failed to parse scene. Error: {e}")

    description = scene.get_nlq()

    ''' +++ 1. Create a question to ask +++ '''
    ''' 1.1 Get Mass and Spatial names in this scene '''
    potential_find_quantities = {
        'masses': {
            'net_force': 'magnitude of the net force',
            'acceleration': 'magnitude of the acceleration',
            'displacement': 'magnitude of the displacement',
            'velocity': 'magnitude of the velocity',
            'kinetic_energy': 'net kinetic energy (rotation if any + linear)',
            'kinetic_energy_linear': 'linear kinetic energy',
            'potential_energy': 'change in potential energy (gravitational)',
        },
        'strings': {
            'force': 'tension',
        }   # force is the tension in the string which is scalar, length is also a scalar
    }

    ''' 1.2 Get Simulation Data '''
    if data is None or restructured_data is None:
        recorder = Recorder(scene=scene, cfg=recorder_cfg, scene_folder="") # scene_folder is not used in this function
        data, metadata, instability = recorder.simulate()
        data = remove_empty_keys(data)
        restructured_data = restructure_data(data)
    keys = {k: list(v.keys()) for k, v in restructured_data.items()}

    entity_names = [e.name for e in scene.entities if type(e) not in [ConstantForceFixedPulley, FixedPulleyEntity]]
    string_names = _get_valid_string_names(restructured_data)

    simulation_mappings = []

    if seed is None:
        np.random.seed(datetime.now().microsecond + seed_offset)
    else:
        np.random.seed(seed + seed_offset)

    proposed_question = ''

    available_modes = []
    if entity_names: available_modes.append("masses")
    if string_names: available_modes.append("strings")

    if not available_modes:
        raise RuntimeError("No valid entities or strings found in scene data.")

    mode = random.choice(available_modes)


    if mode == "masses":
        entity_to_ask = random.choice(entity_names)
        subentity_to_ask = random.choice(list(restructured_data[entity_to_ask].keys()))

        _quantity_to_ask = random.choice(list(potential_find_quantities['masses'].keys()))
        quantity_to_ask = potential_find_quantities['masses'][_quantity_to_ask]

        attribute_to_ask = (entity_to_ask, subentity_to_ask, quantity_to_ask)
    elif mode == "strings":
        string_to_ask = random.choice(string_names)

        _quantity_to_ask = random.choice(list(potential_find_quantities['strings'].keys()))
        quantity_to_ask = potential_find_quantities['strings'][_quantity_to_ask]

        attribute_to_ask = (string_to_ask, quantity_to_ask)

    try:
        idx = random.choice(range(len(data['global']['time']) // 2, len(data['global']['time'])))
    except Exception as e:
        raise RuntimeError(f"Failed to select a valid time index for scene. Error: {e}")

    time_to_ask = data['global']['time'][idx]

    if mode == "masses":
        selected_item = f"{entity_to_ask}.{subentity_to_ask}"
    elif mode == "strings":
        selected_item = f"{string_to_ask}"

    simulation_mapping = "{{\n    'attribute': '{}.{}',\n    'time': {},\n    'component': 'norm'\n}}".format(selected_item, _quantity_to_ask, time_to_ask)
    simulation_mappings.append(simulation_mapping)

    proposed_question += scene.get_question(time_to_ask, attribute_to_ask, mode, keys)
    
    if mode == "strings" and not string_names:
        mode = "masses"

    if mode == "masses":
        answer = restructured_data[entity_to_ask][subentity_to_ask][_quantity_to_ask][idx]
    elif mode == "strings":
        answer = restructured_data[string_to_ask][_quantity_to_ask][idx]

    if _quantity_to_ask in ['net_force', 'acceleration', 'velocity', 'position']:
        if answer.ndim > 0:
            answer = np.linalg.norm(answer[-3:])    # get the magnitude of a vector for now, later we can ask for the vector along a certain axis or direction.

    general_info = "Assume acceleration due to gravity as 9.81 m/s^2, all strings inextensible, and all surfaces frictionless unless otherwise stated." 
    q = description + '\n' + proposed_question + ' ' + general_info
    a = answer

    if generation_cfg.numerical:
        Final_Problem_with_Answer = {
            "text": q,   # problem description
            "image": None,
            "answer": str(a),
            "is_symbolic": False,
            "simulation_mapping": simulation_mapping,
            "model_name": 'heuristic',
            "given_variable_mapping": None,    # given variables in the problem
        }
    elif generation_cfg.symbolic:
        # not implemented error
        raise NotImplementedError("Symbolic question generation is not implemented yet.")
    else:
        raise ValueError("Invalid mode")

    return Final_Problem_with_Answer
            
def get_numerical_qs(generation_cfg, potential_find_quantities, string_names, entity_names, restructured_data, data, scene_dir, scene, keys, seed_offset = 0):
    proposed_questions = []
    answers = []
    attribute_to_asks = []
    time_to_asks = []
    simulation_mappings = []
    entity_names = [n for n in entity_names if n in restructured_data]

    if len(entity_names) == 0: return proposed_questions, answers, attribute_to_asks, time_to_asks, simulation_mappings

    for attempt in range(generation_cfg.num_generations_per_problem):
        # set random seed for reproducibility
        np.random.seed(datetime.now().microsecond)
        proposed_question = ''
        
        mode = random.choice(list(potential_find_quantities.keys())) # Choose between mass or string
        if len(string_names) == 0: mode = "masses"

        if mode == "masses":
            entity_to_ask = random.choice(entity_names)
            subentity_to_ask = random.choice(
                list(restructured_data[entity_to_ask].keys())
            )
            
            _quantity_to_ask = random.choice(
                list(potential_find_quantities['masses'].keys())
            )
            quantity_to_ask = potential_find_quantities['masses'][_quantity_to_ask]

            attribute_to_ask = (entity_to_ask, subentity_to_ask, quantity_to_ask)
        elif mode == "strings":
            string_to_ask = random.choice(string_names)
            
            _quantity_to_ask = random.choice(
                list(potential_find_quantities['strings'].keys())
            )
            quantity_to_ask = potential_find_quantities['strings'][_quantity_to_ask]
            
            attribute_to_ask = (string_to_ask, quantity_to_ask)

        try:
            idx = random.choice(range(len(data['global']['time']) //2, len(data['global']['time'])))
        except:
            print("scene_dir: ", scene_dir)
            
        time_to_ask = data['global']['time'][idx]
        
        if mode == "masses":
            selected_item = f"{entity_to_ask}.{subentity_to_ask}"
        elif mode == "strings":
            selected_item = f"{string_to_ask}"
        
        simulation_mapping="{{\n    'attribute': '{}.{}',\n    'time': {},\n    'component': 'norm'\n}}".format(selected_item, _quantity_to_ask, time_to_ask)
        simulation_mappings.append(simulation_mapping)
        
        proposed_question += scene.get_question(time_to_ask, attribute_to_ask, mode, keys)
        
        if mode == "masses":
            try: answer = restructured_data[entity_to_ask][subentity_to_ask][_quantity_to_ask][idx]
            except: st()
        elif mode == "strings":
            answer = restructured_data[string_to_ask][_quantity_to_ask][idx]
        
        # if _quantity_to_ask == 'net_force' or _quantity_to_ask == 'acceleration' or _quantity_to_ask == 'velocity' or _quantity_to_ask == 'position':
        if _quantity_to_ask in VECTOR_QUANTITIES:
            assert answer.ndim > 0
            answer = np.linalg.norm(answer)    # get the magnitude of a vector for now, later we can ask for the vector along a certain axis or direction.

        proposed_questions.append(proposed_question)
        answers.append(answer)
        attribute_to_asks.append('.'.join(attribute_to_ask))
        time_to_asks.append(time_to_ask)
        
    return proposed_questions, answers, attribute_to_asks, time_to_asks, simulation_mappings

def get_ans(restructured_data, mode, entity_to_ask, string_to_ask, subentity_to_ask, _quantity_to_ask, time):
    idx = np.argmin([np.abs(x - time) for x in restructured_data['global']['time']])
    if mode == "masses":
        answer = restructured_data[entity_to_ask][subentity_to_ask][_quantity_to_ask][idx]
    elif mode == "strings":
        answer = restructured_data[string_to_ask][_quantity_to_ask][idx]
    
    if _quantity_to_ask in VECTOR_QUANTITIES:
        assert answer.ndim > 0
        answer = np.linalg.norm(answer)    # get the magnitude of a vector for now, later we can ask for the vector along a certain axis or direction.

    return answer

def run(cfg, recorder_cfg):
    generation_cfg = cfg.question_generation
    root_dir = cfg.root_dir
    try:
        with open('cost.json', 'r') as f:   # cost.json is a file that contains the cost of tokens for each model
            cost = json.load(f)
    except:
        from llm.utils.basic_utils import cost_dict
        cost = cost_dict
    
    if cfg.solve_locally:
        pass
    else:
        cost_per_model = cost[cfg.model_name]
    
    
    scene_type_metrics = defaultdict(int)


    total_cost = 0
    num_descriptions = 0
    wandb.init(project='physics-qa-gen', config=dict(cfg), mode = 'disabled')
    
    all_logs = []
    
    scene_type_dirs = [os.path.join(root_dir, _dir) for _dir in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, _dir))]
    scene_dirs = sum([[os.path.join(_dir, _d) for _d in os.listdir(_dir) if os.path.isdir(os.path.join(_dir, _d))] for _dir in scene_type_dirs], [])
    scene_dirs = sorted(scene_dirs)
    
    if cfg.factor_id != -1 and cfg.num_factors != -1:
        scene_dirs = scene_dirs[cfg.factor_id::cfg.num_factors]
        
    # Load general info
    with open('llm/prompts/general_info.txt', 'r') as f:
        gi = f.read() 

    processed_count = 0
    random.shuffle(scene_dirs)
    total_len = len(scene_dirs)
    
    for scene_dir in tqdm(scene_dirs, desc="Processing scene dirs"):
        yaml_file = os.path.join(scene_dir, 'scene_output.yaml')
        if not os.path.exists(yaml_file):
            continue
        try:
            scene = parse_scene(yaml_file)
        except Exception as e:
            print("Failed to parse scene: ", yaml_file)
            print("Error: ", e)
            continue
        description = scene.get_nlq()
        scene_type = yaml_file.split('/')[-3]
        
        with open(os.path.join(scene_dir,  'scene_output_desc.txt'), 'w') as f:
            f.write(description)
            num_descriptions += 1

        num_input_tokens = 0 
        num_output_tokens = 0
        num_cached_tokens = 0        
        
        table_logs= [{} for _ in range(generation_cfg.num_generations_per_problem)]
        print('*****************Processing: ', scene_dir, '*****************')

        ''' +++ 1. Generate natural language description for the DSL +++ '''
        ''' Not Required Anymore '''
        
        ''' +++ 2. Save the natural language description +++ '''
        ''' Not Required Anymore '''
        
        ''' +++ 3. Create a question to ask +++ '''
        ''' 3.1 Get Mass and Spatial names in this scene '''
        category = [k for k in POTENTIAL_FIND_QUANTITIES if scene_type in POTENTIAL_FIND_QUANTITIES[k]["categories"]]
        assert len(category) <= 1, f"Multiple categories found for scene type: {scene_type}"
        if len(category) == 0: raise ValueError(f"No category found for scene type: {scene_type}")
        category = category[0] 
        potential_find_quantities = {k: v for k, v in POTENTIAL_FIND_QUANTITIES[category].items() if k!= "categories"}

        ''' 3.2 Get simulation data '''
        current_dir = os.path.dirname(os.path.realpath(__file__))   # PHO/sim
        
        recorder = Recorder(scene=scene, cfg=recorder_cfg, scene_folder=current_dir, category=category)
        
        data, metadata, instability = recorder.simulate()
        data = remove_empty_keys(data)
        
        restructured_data = restructure_data(data)
        keys = {k:list(v.keys()) for k, v in restructured_data.items()}
        
        entity_names = [e.name for e in scene.entities if type(e) not in [ConstantForceFixedPulley, FixedPulleyEntity]]
        
        string_names = _get_valid_string_names(restructured_data)
        
        ''' 4. Continuous question: get the attribute of a random item to ask about'''
        ''' 4.1 Determine the question to ask '''
        (
            proposed_questions,
            answers,
            attribute_to_asks,
            time_to_asks, 
            simulation_mappings 
        ) = get_numerical_qs(
            generation_cfg, potential_find_quantities, string_names, entity_names, restructured_data, data, scene_dir, scene, keys
        )
        
        ''' 4.2 Finalize the problem '''
        Final_Problems_with_Answers = [(description + '\n' +  q + ' ' + gi, a) for q,a in zip(proposed_questions, answers)]
        
        
        ''' +++ 5. Generate symbolic representation of the problem +++ '''
        description, sym_mapping = scene.get_nlq(symbolic=True)
        if generation_cfg.symbolic:
            with open(os.path.join(scene_dir,  'scene_output_symbolic_desc.txt'), 'w') as f:
                f.write(description)
            with open(os.path.join(scene_dir, 'symbolic_mapping.json'), 'w') as f:
                json.dump(sym_mapping, f, indent=4)

            # Resample new numerical qs instead of converting existing numerical qs for symbolic q gen 
            (
                proposed_questions,
                answers,
                attribute_to_asks,
                time_to_asks, 
                symbolic_simulation_mappings 
            ) = get_numerical_qs(
                generation_cfg, potential_find_quantities, string_names, entity_names, restructured_data, data, scene_dir, scene, keys, seed_offset=generation_cfg.num_generations_per_problem
            )
            
            symbolic_proposed_questions = [q.replace(f"{time_to_asks[-(generation_cfg.num_generations_per_problem - idx)]:.2f}", 't') for idx, q in enumerate(proposed_questions)]
            Final_Symbolic_Problems_with_Answers = [(description + '\n' + q + ' ' + gi, a) for q,a in zip(symbolic_proposed_questions, answers)]
        
        ''' +++ 6. Generate reverse in time problems +++ '''
        if generation_cfg.reverse:
            mask_key = random.choice(list(sym_mapping.keys()))
            mask_value = sym_mapping[mask_key]

            description = replace_all(description, {k:str(v) for k, v in sym_mapping.items() if k != mask_key})
            description = replace_all(description, {mask_key: 'x'})

            def get_reverse_from_normal(q, a, idx):
                symbolic_q = q.replace(f"{time_to_asks[-(generation_cfg.num_generations_per_problem - idx)]:.2f}", 't')

                mode = random.choice(sum([list(d.keys()) for d in potential_find_quantities.values()], start = []) + ['time']) # Choose between all possibble find variables

                if mode == "time":
                    proposed_q = symbolic_q[8:-1].capitalize() + f" is {a:.2f}. What is the value of t?"
                    ans = round(time_to_asks[-(generation_cfg.num_generations_per_problem - idx)], 2)
                else:
                    try:
                        proposed_q = f"What is the value of x given that {q[8:-1]} is {a:.2f}?"
                    except:
                        st()
                    ans = mask_value
                
                return proposed_q, ans
            
            # Resample new numerical qs instead of converting existing numerical qs for reverse q gen
            (
                proposed_questions,
                answers,
                attribute_to_asks,
                time_to_asks, 
                reverse_simulation_mappings 
            ) = get_numerical_qs(
                generation_cfg, potential_find_quantities, string_names, entity_names, restructured_data, data, scene_dir, scene, keys, seed_offset=generation_cfg.num_generations_per_problem*2
            )

            reverse_proposed_questions = [get_reverse_from_normal(q, a, idx) for idx, (q, a) in enumerate(zip(proposed_questions, answers))] # q[8:-1] removes the "What is " and "?" from the question
            Final_Reverse_Problems_with_Answers = [(description + '\n' + q + ' ' + gi, a) for q, a in reverse_proposed_questions]

        if generation_cfg.numerical:
            question_answer_pair_dir = os.path.join(scene_dir, f'question_numerical_answer_pair')
            os.makedirs(question_answer_pair_dir, exist_ok=True)
        if generation_cfg.symbolic:
            symbolic_question_answer_pair_dir = os.path.join(scene_dir, f'symbolic_question_answer_pair')
            os.makedirs(symbolic_question_answer_pair_dir, exist_ok=True)
        if generation_cfg.reverse:
            reverse_question_answer_pair_dir = os.path.join(scene_dir, f'reverse_question_answer_pair')
            os.makedirs(reverse_question_answer_pair_dir, exist_ok=True)    
        
        if generation_cfg.numerical:
            for i, (q, a) in enumerate(Final_Problems_with_Answers):
                Final_Problem_with_Answer = "<problem>\n{problem}\n</problem>\n\n<answer>\n{answer}\n</answer>\n\n<simulation_mapping>\n{simulation_mapping}\n</simulation_mapping>".format(
                    problem=q,
                    answer=a,
                    simulation_mapping=simulation_mappings[i]
                ).replace(r"\n", "\n")
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                
                with open(os.path.join(question_answer_pair_dir, f'qa_{i}_{timestamp}.txt'), 'w') as f:
                    f.write(Final_Problem_with_Answer)
                
        if generation_cfg.symbolic:
            for i, (q, a) in enumerate(Final_Symbolic_Problems_with_Answers):
                t = time_to_asks[-(generation_cfg.num_generations_per_problem - i)]
                modified_mapping = {k:str(v) for k, v in sym_mapping.items()}
                modified_mapping["t"] = str(t)

                modifed_mapping = json.dumps(modified_mapping, indent=4)

                Final_Problem_with_Answer = "<problem>\n{problem}\n</problem>\n\n<answer>\n{answer}\n</answer>\n\n<simulation_mapping>\n{simulation_mapping}\n</simulation_mapping>\n\n<mapping>\n{mapping}\n</mapping>".format(
                    problem=q,
                    answer=a,
                    simulation_mapping=symbolic_simulation_mappings[i],
                    mapping=modifed_mapping
                ).replace(r"\n", "\n")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

                with open(os.path.join(symbolic_question_answer_pair_dir, f'qa_{i}_{timestamp}.txt'), 'w') as f:
                    f.write(Final_Problem_with_Answer)

                # with open(os.path.join(scene_dir, f'q_symbolic_{i}_{timestamp}.txt'), 'w') as f:
                #     f.write(q)

        if generation_cfg.reverse:
            for i, (q, a) in enumerate(Final_Reverse_Problems_with_Answers):
                Final_Problem_with_Answer = "<problem>\n{problem}\n</problem>\n\n<answer>\n{answer}\n</answer>\n\n<simulation_mapping>\n{simulation_mapping}\n</simulation_mapping>".format(
                    problem=q,
                    answer=a,
                    simulation_mapping=reverse_simulation_mappings[i]
                ).replace(r"\n", "\n")
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                
                with open(os.path.join(reverse_question_answer_pair_dir, f'qa_{i}_{timestamp}.txt'), 'w') as f:
                    f.write(Final_Problem_with_Answer)

        try:
            del continual_prompts
        except:
            pass

        all_logs.extend(table_logs)
        processed_count += 1
        
        log_data = {'processed_count': processed_count, 'percent_complete': processed_count / total_len}
        
        df = pd.DataFrame(
            {
                "proposed_questions": proposed_questions,
                "answers": answers,
                "attribute_to_asks": attribute_to_asks,
                "time_to_asks": time_to_asks
            }
        )
        if generation_cfg.symbolic:
            df["proposed_symbolic_questions"] = symbolic_proposed_questions
            
        df.to_csv(os.path.join(scene_dir, 'qa_logs.csv'))
        
        if not cfg.solve_locally:
            current_cost_tmp = cost_per_model['input'] * num_input_tokens + cost_per_model['output'] * num_output_tokens + cost_per_model['cached'] * num_cached_tokens
            current_cost = current_cost_tmp / 1e6
            total_cost += current_cost
            average_cost = total_cost / processed_count
            log_data['costing/cost'] = current_cost
            log_data['costing/total_cost'] = total_cost
            log_data['costing/average_cost'] = average_cost
        
        wandb.log(log_data)
        scene_type_metrics[scene_type] += 1
        print(scene_type_metrics)
        print('*****************Processed: ', scene_dir, '*****************')
        if processed_count > cfg.max_samples and cfg.max_samples != -1:
            break

def extract_tag(text, tag):
    pattern = fr"<{tag}>(.*?)</{tag}>"
    m = re.search(pattern, text, re.DOTALL)
    return m.group(1).strip() if m else None

def _simulate_child(args):
    """Module-level worker: simulate one child scene. Must be at module level to be picklable."""
    child_file, recorder_cfg, current_dir = args
    child = os.path.basename(child_file)
    try:
        scene = parse_scene(child_file)
    except Exception as e:
        print(f"Failed to parse scene: {child_file}, Error: {e}")
        return child, None
    scene_type = child_file.split('/')[-4]
    category = [k for k in POTENTIAL_FIND_QUANTITIES if scene_type in POTENTIAL_FIND_QUANTITIES[k]["categories"]]
    if len(category) != 1:
        return child, None
    category = category[0]
    try:
        recorder = Recorder(scene=scene, cfg=recorder_cfg, scene_folder=current_dir, category=category)
    except Exception:
        return child, None
    data, _metadata, _instability = recorder.simulate()
    data = remove_empty_keys(data)
    restructured_data = restructure_data(data)
    entity_names = [e.name for e in scene.entities if type(e) not in [ConstantForceFixedPulley, FixedPulleyEntity]]
    string_names = _get_valid_string_names(restructured_data)
    return child, {
        'restructured_data': restructured_data,
        'entity_names': entity_names,
        'string_names': string_names
    }


def get_child(cfg, recorder_cfg):
    root_dir = cfg.root_dir


    wandb.init(project='physics-qa-gen', config=dict(cfg), mode = 'disabled')
    
    scene_type_dirs = [os.path.join(root_dir, _dir) for _dir in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, _dir))]
    scene_dirs = sum([[os.path.join(_dir, _d) for _d in os.listdir(_dir) if os.path.isdir(os.path.join(_dir, _d))] for _dir in scene_type_dirs], [])
    scene_dirs = sorted(scene_dirs)
    
    if cfg.factor_id != -1 and cfg.num_factors != -1:
        scene_dirs = scene_dirs[cfg.factor_id::cfg.num_factors]
        
    random.shuffle(scene_dirs)
    
    num_workers = getattr(cfg.recorder, 'num_workers', os.cpu_count())
    current_dir = os.path.dirname(os.path.realpath(__file__))   # PHO/sim

    for scene_dir in tqdm(scene_dirs, desc="Processing scene dirs"):
        print(f"***Doing scene {scene_dir}***")
        child_files = [
            os.path.join(scene_dir, 'child_scenes', c)
            for c in os.listdir(os.path.join(scene_dir, 'child_scenes'))
            if os.path.exists(os.path.join(scene_dir, 'child_scenes', c))
        ]
        worker_args = [(cf, recorder_cfg, current_dir) for cf in child_files]
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            results = list(tqdm(pool.map(_simulate_child, worker_args), total=len(worker_args), desc="  children", leave=False))
        data_dict = {child: entry for child, entry in results if entry is not None}
        
        # Add shortcut scene
        try:
            shortcut_file = os.path.join(scene_dir,'scene_output.yaml')
            
            shortcut_scene = parse_scene(shortcut_file)
            made_changes = shortcut_scene.get_shortcut()

            if made_changes:
                scene_type = shortcut_file.split('/')[-3]
                category = [k for k in POTENTIAL_FIND_QUANTITIES if scene_type in POTENTIAL_FIND_QUANTITIES[k]["categories"]]
                assert len(category) <= 1, f"Multiple categories found for scene type: {scene_type}"
                if len(category) == 0: raise ValueError(f"No category found for scene type: {scene_type}")
                category = category[0] 
                
                try:
                    recorder = Recorder(scene=shortcut_scene, cfg=recorder_cfg, scene_folder=current_dir, category=category)
                except:
                    continue

                data, metadata, instability = recorder.simulate()
                
                data = remove_empty_keys(data)
                
                restructured_data = restructure_data(data)
                
                entity_names = [e.name for e in shortcut_scene.entities if type(e) not in [ConstantForceFixedPulley, FixedPulleyEntity]]
                
                string_names = _get_valid_string_names(restructured_data)

                data_dict['shortcut'] = {
                    'restructured_data': restructured_data,
                    'entity_names': entity_names,
                    'string_names': string_names
                }
        except:
            raise ValueError("Failed to add shortcut scene")
        
        print("Len data dict:", len(data_dict))
        numeric_qs_path = os.path.join(scene_dir, 'question_numerical_answer_pair')
        reverse_qs_path = os.path.join(scene_dir, 'question_reverse_answer_pair')
        symbolic_qs_path = os.path.join(scene_dir, 'symbolic_question_answer_pair')

        qs = {}
        if os.path.exists(numeric_qs_path):
            for txt_file in os.listdir(numeric_qs_path):
                with open(os.path.join(numeric_qs_path, txt_file), 'r') as f:
                    qs[os.path.join(numeric_qs_path, txt_file)] = f.read()
        if os.path.exists(reverse_qs_path):
            for txt_file in os.listdir(reverse_qs_path):
                with open(os.path.join(reverse_qs_path, txt_file), 'r') as f:
                    qs[os.path.join(reverse_qs_path, txt_file)] = f.read()
        if os.path.exists(symbolic_qs_path):
            for txt_file in os.listdir(symbolic_qs_path):
                with open(os.path.join(symbolic_qs_path, txt_file), 'r') as f:
                    qs[os.path.join(symbolic_qs_path, txt_file)] = f.read()
            
        valid_qs = []
        for fp in qs:
            q = qs[fp]
            sim_mapping = extract_tag(q, 'simulation_mapping')
            sim_mapping = ast.literal_eval(sim_mapping)

            attr = sim_mapping['attribute']
            time = sim_mapping['time']
            main_ans = float(extract_tag(q, 'answer').strip())

            sub_attrs = attr.split('.')

            entity_to_ask = ''
            string_to_ask = ''
            subentity_to_ask = ''
            _quantity_to_ask = ''            

            if len(sub_attrs) == 3:
                entity_to_ask, subentity_to_ask, _quantity_to_ask = sub_attrs
                mode = 'masses'
            else:
                string_to_ask, _quantity_to_ask = '.'.join(sub_attrs[:-1]), sub_attrs[-1]
                mode = 'strings'
            
            can_keep = True
            for child in data_dict:
                try:
                    child_ans = get_ans(data_dict[child]['restructured_data'], mode, entity_to_ask, string_to_ask, subentity_to_ask, _quantity_to_ask, time)
                except KeyError as e:
                    continue
                if abs(abs(main_ans) - abs(child_ans)) / abs(main_ans) <= 5e-2:
                    # Can shortcut, dont keep this question
                    can_keep = False
                    break

            if can_keep:
                valid_qs.append(fp)

        with open(os.path.join(scene_dir, 'valid_qs.txt'), 'w') as f:
            f.write('\n'.join(valid_qs))
    
        print(f"***Done scene {scene_dir}***")

@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    recorder_cfg = cfg.recorder
    random.seed(cfg.seed)

    if cfg.question_generation.build_child_scenes:
        get_child(cfg, recorder_cfg)

    else:
        run(cfg, recorder_cfg)
    
if __name__ == "__main__":
    main()

