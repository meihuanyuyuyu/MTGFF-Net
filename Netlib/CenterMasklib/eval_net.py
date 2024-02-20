import torch
import numpy as np
from tqdm import tqdm
import argparse
import os

import time
import psutil
from tools.utils import draw_instance_map
from tools.metric import dice_score
from centermask.config import centermask_config
from CentermaskInfer import CenteMaskInfer

parser = argparse.ArgumentParser(description="infering")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--ds", type=str, default="conic")
parser.add_argument("--ds_num", type=int, default=1)
parser.add_argument("--config", type=str, default="centermaskres50")
process_arg = parser.parse_args()

arg = getattr(centermask_config, process_arg.config)(process_arg.ds,process_arg.device,process_arg.ds_num)

class InferEngine:
    def __init__(self) -> None:
        print("initlizing model...")
        print("Check setted Config:\n")
        for _ in dir(arg):
            if not _.startswith("_"):
                print(_, "=", getattr(arg, _))
        self.infer = CenteMaskInfer(arg=arg)

    def prepare_data(self):
        if arg.ds == "conic":
            data = (
                torch.from_numpy(
                    np.load("data/CoNIC_Challenge/images.npy").astype(np.float64) / 255
                )
                .float()
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            gts = np.load("data/CoNIC_Challenge/labels.npy")
            index = arg.TEST_SET_INDEX
            data = data[index]
            gts = gts[index.numpy()]
            return data, gts

        else:
            from PIL import Image
            from torchvision.transforms.functional import pil_to_tensor

            data_list = os.listdir("data/PANNUKE/images")
            data_list.sort(key=lambda x: int(x.split(".")[0]))
            imgs = torch.tensor([])
            index = arg.TEST_SET_INDEX
            for _, idx in enumerate(index):
                file_path = os.path.join("data/PANNUKE/images", data_list[idx.long().item()])
                img = pil_to_tensor(Image.open(file_path).convert("RGB")) / 255
                imgs = torch.cat([imgs, img[None]], dim=0)
                
            gts = np.load("data/PANNUKE/masks.npy")[index.numpy()]
            return imgs, gts
    
    def generate_results(self, data: torch.Tensor, gts):
        preds = np.zeros((0, 256, 256, 2), dtype=np.uint16)
        _gts = np.zeros((0, 256, 256, 2), dtype=np.uint8)
        bar = tqdm(range(len(data)))

        for idx in bar:
            img = data[idx : idx + 1].to(device=arg.DEVICE)
            gt = gts[idx : idx + 1]

            pred = self.infer(img)
            preds = np.concatenate([preds, pred], axis=0)
            _gts = np.concatenate([_gts, gt], axis=0)
        print(
            "进程的内存使用：%.4f GB"
            % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024)
        )
        return preds, _gts
    
    def save_predictions(self, preds, _gts):

        pred_dir = os.path.join(
            arg.numpy_result_dir, arg.TEST_RESULT_DIR
        )
        figure_dir = os.path.join(
            "figures/" + arg.ds + "/results", arg.__class__.__name__
        )
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
        if not os.path.exists(figure_dir):
            os.makedirs(figure_dir)
        pred_path = pred_dir + "/preds.npy"
        true_path = pred_dir + "/gts.npy"
        np.save(pred_path, preds)
        np.save(true_path, _gts)
        return pred_path, true_path, figure_dir
            
    def print_metric(self, pred_path, true_path):
        pred =np.load(pred_path)
        target = np.load(true_path)
        pred = torch.from_numpy(pred.astype(np.uint8)).long() # [N, 256, 256, 2]
        target = torch.from_numpy(target.astype(np.uint8)).long() # [N, 256, 256, 2]
        dice = dice_score(pred[..., 1], target[..., 1])
        print(f"dice score:{dice:.6f}")

    def run(self):
        # 准备数据
        data, gts = self.prepare_data()
        start = time.time()
        preds, _gts = self.generate_results(data, gts)
        end = time.time()
        print(f"Cost {(end-start):.1f} seconds.")
        # 保存
        pred_path, true_path, figure_dir = self.save_predictions(preds, _gts)
        print("saving results...")
        print(f"figure dir{figure_dir},pred shape:{pred_path},true_path:{true_path}")
        draw_instance_map(data, preds, fp=figure_dir)
        # 指标计算
        self.print_metric(pred_path, true_path)
        os.system(
            f"python compute_metric/compute_stats.py --mode=seg_class --ds={arg.ds} --pred={pred_path} --true={true_path} >> log/out_log/{arg.__class__.__name__}_{arg.ds_num}.txt 2>&1"
        )

if __name__ == "__main__":
    engine = InferEngine()
    engine.run()

