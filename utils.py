import os
import config

def get_labels(target_name):
    """対象データセットのクラスラベルリストを返す"""
    if target_name.lower().startswith("voc"):
        return config.voc_labels
    return config.fluid_labels


def get_num_classes(target_name):
    """クラス数を返す"""
    return len(get_labels(target_name))


def get_data_folder(target_name):
    """データセットフォルダのパスを自動検索して返す"""
    base_dir = os.path.dirname(__file__)
    candidate_paths = [
        os.path.join(base_dir, f"Dataset_{target_name}"),
        os.path.join(base_dir, f"Dataset_{target_name.lower()}"),
        os.path.join(base_dir, f"Dataset_{target_name.upper()}"),
    ]
    if target_name.lower() == "fluid":
        candidate_paths.extend([
            os.path.join(base_dir, "Dataset_fluid"),
            os.path.join(base_dir, "Dataset_Fluid"),
        ])

    for path in candidate_paths:
        if os.path.isdir(path):
            return path
    return candidate_paths[0]
