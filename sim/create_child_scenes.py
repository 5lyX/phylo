import yaml, copy, os
from sim.scene import Scene
from itertools import combinations

from omegaconf import DictConfig
import hydra

def powerset(lst):
    out = []
    n = len(lst)
    for r in range(n + 1):
        out.extend(combinations(lst, r))
    return out

@hydra.main(config_path="../config", config_name="config")
def generate_children(cfg: DictConfig):
    data_folder = cfg.root_dir

    scene_types = os.listdir(data_folder)
    for scene_type in scene_types:

        scene_type_path = os.path.join(data_folder, scene_type)
        if not os.path.isdir(scene_type_path):
            continue

        print("Starting scene type:", scene_type)

        scenes = os.listdir(scene_type_path)
        num_child = 0
        for scene_folder in scenes:

            scene_folder_path = os.path.join(scene_type_path, scene_folder)
            if not os.path.isdir(scene_folder_path):
                continue

            folder_name = os.path.join(scene_folder_path, "child_scenes")
            os.makedirs(folder_name, exist_ok=True)

            yaml_path_in = os.path.join(scene_folder_path, "scene_output.yaml")
            if not os.path.exists(yaml_path_in):
                continue

            with open(yaml_path_in, "r") as f:
                data = yaml.safe_load(f)

            scene = data["scene"]
            entities = scene["entities"]

            ps = powerset(entities)

            for i, subset in enumerate(ps):
                if len(subset) in [0, len(entities)]:
                    continue

                sub_scene = copy.deepcopy(data)
                sub_scene["scene"]["entities"] = list(subset)

                # compute absent entities
                absent = [e["name"] for e in entities if e not in subset]
                # print(absent)

                # filter connections
                connections = [
                    conn for conn in sub_scene["scene"]["connections"]
                    if all(e["entity"] not in absent for e in conn["tendon"])
                ]

                sub_scene["scene"]["connections"] = connections

                yaml_path_out = os.path.join(folder_name, f"child_{i}.yaml")
                with open(yaml_path_out, "w", encoding="utf-8") as f:
                    yaml.dump(sub_scene, f, sort_keys=False)
                    num_child += 1

        print("Num child:", num_child)

if __name__ == '__main__':
    generate_children()