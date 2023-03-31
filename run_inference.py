import argparse
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from tqdm import tqdm

from datasets.semantic_kitti import (
    SemanticKitti,
    SemanticITTIK,
    class_names,
    map_inv,
    splits,
)

from utils.evaluation import Eval
from models import deeplab

parser = argparse.ArgumentParser("Run lidar bug inference")
parser.add_argument("--checkpoint-path", required=True, type=Path)
parser.add_argument("--output-path", required=True, type=Path)
parser.add_argument("--semantic-kitti-dir", required=True, type=Path)
parser.add_argument("--semantic-ittik-dir", required=False, type=Path)
parser.add_argument("--split", default="val", type=str)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    if args.semantic_ittik_dir:
        dataset = SemanticITTIK(args.semantic_ittik_dir, args.semantic_ittik_dir, args.split)
    else: 
        dataset = SemanticKitti(args.semantic_kitti_dir, args.split)

    loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False,
    )
    for seq in splits[args.split]:
        seq_dir = args.output_path / "sequences" / f"{seq:0>2}" / "predictions"
        seq_dir.mkdir(parents=True, exist_ok=True)
    model = deeplab.resnext101_aspp_kp(19)
    model.to(device)
    model.load_state_dict(torch.load(args.checkpoint_path))
    print("Runnign validation")
    model.eval()
    eval_metric = Eval(19, 255)
    with torch.no_grad():
        for step, items in tqdm(enumerate(loader), total=len(loader)):
            images = items["image"].to(device)
            labels = items["labels"]
            py = items["py"].float().to(device)
            px = items["px"].float().to(device)
            pxyz = items["points_xyz"].float().to(device)
            knns = items["knns"].long().to(device)
            predictions = model(images, px, py, pxyz, knns)
            _, predictions_argmax = torch.max(predictions, 1)
            predictions_points = predictions_argmax.cpu().numpy()
            eval_metric.update(predictions_points, labels)
            predictions_points = np.vectorize(map_inv.get)(predictions_points).astype(
                np.uint32
            )
            seq, sweep = items["seq"][0], items["sweep"][0]
            out_file = (
                args.output_path
                / "sequences"
                / f"{seq}"
                / "predictions"
                / f"{sweep}.label"
            )
            predictions_points.tofile(out_file.as_posix())

        miou, ious = eval_metric.getIoU()
        print(f"mIou {miou}")
        print("Per class Ious: ")
        for class_name, iou in zip(class_names, ious):
            print(f"{class_name}: {iou}")


if __name__ == "__main__":
    main()
