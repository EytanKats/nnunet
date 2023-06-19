from pathlib import Path
import json
import argparse

def dump_training_json(task_path):
    task_path = Path(task_path)
    for postfix in ["Tr", "Ts"]:
        image_folder = "images"+postfix
        label_folder = "labels"+postfix
        lbl_path = task_path/label_folder
        img_path = task_path/image_folder
        if postfix == 'Tr':
            images = [f"./{image_folder}/"+p.name for p in lbl_path.iterdir()]
        else:
            images = [f"./{image_folder}/"+p.name for p in img_path.iterdir()]
        labels = [f"./{label_folder}/"+p.name for p in lbl_path.iterdir()]
        pairs = [dict(image=img, label=lbl) for img, lbl in zip(images, labels)]
        print(f"Built {len(pairs)} pairs for {postfix}")
        with open(Path(task_path).joinpath(f"dataset_paths_{postfix}.json"), "w") as f:
            json.dump(pairs, f, indent=4, sort_keys=True)
        if postfix == 'Ts':
            with open(Path(task_path).joinpath(f"dataset_paths_imagelist_{postfix}.json"), "w") as f:
                json.dump(images, f, indent=4, sort_keys=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build nnUNet dataset json.')
    parser.add_argument('task_path', type=str,
                        help='Path containing nnUNet task: .../Task999_Name')

    args = parser.parse_args()
    assert "Task" in args.task_path or "Dataset" in args.task_path
    dump_training_json(args.task_path)