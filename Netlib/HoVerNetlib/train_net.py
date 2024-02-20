import argparse
import torch
import os
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import MultiStepLR, _LRScheduler
from torch.nn.functional import mse_loss
from model.HoverNet import val_np_tp_hv
from tools.metric import jaccard, mask_category_acc
from tools.augmentation import Random_flip, MyGausssianBlur, MyColorjitter
from torch.cuda.amp import GradScaler, autocast
from tools.utils import dice_loss, focal_loss
from tools.dataset import (
    HoverNetCoNIC,
    PannukeHover,
    DataLoader,
    Subset,
)
from tqdm import tqdm

from config import hovernet_config


def setup_seed(seed, benchmark=True):
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed_all(seed=seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = benchmark
    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.enabled = benchmark


parser = argparse.ArgumentParser(description="train hovernet")
parser.add_argument("--split_num", type=int, default=1)
parser.add_argument("--ds", type=str, default="conic")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--config", type=str, default="HoverConfig1")
process_arg = parser.parse_args()


arg = getattr(hovernet_config, process_arg.config)(
    process_arg.ds, process_arg.device, process_arg.split_num
)
writer = SummaryWriter(arg.exp_data_dir)
scaler = GradScaler()


def _train(data: DataLoader, optimizer: Optimizer, model: torch.nn.Module):
    model.train()
    bar = tqdm(data, colour="CYAN", disable=False)
    losses = []
    for i, (data) in enumerate(bar):
        img, hv, np, tp = data
        img = img.to(device=process_arg.device)
        hv = hv.to(device=process_arg.device)
        np = np.to(device=process_arg.device)
        tp = tp.to(device=process_arg.device)
        with autocast():
            optimizer.zero_grad()
            outputs = model(img)
            loss_tp = focal_loss(outputs["tp"], tp) + dice_loss(outputs["tp"], tp)
            loss_np = 2 * (focal_loss(outputs["np"], np) + dice_loss(outputs["np"], np))
            loss_hv = 3 * mse_loss(outputs["hv"], hv)
            loss: torch.Tensor = loss_tp + loss_np + loss_hv
            if loss.isnan():
                bar.set_description(
                    f"skip nan loss:{loss.item():.4f},loss_tp:{loss_tp.item():.4f},loss_np:{loss_np.item():.4f},loss_hv:{loss_hv.item():.4f}"
                )
                continue
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        if not loss.isnan():
            losses.append(loss.item())
        bar.set_description(
            f"train_loss:{loss.item():.4f},loss_tp:{loss_tp.item():.4f},loss_np:{loss_np.item():.4f},loss_hv:{loss_hv.item():.4f}"
        )
    losses = torch.mean(torch.tensor(losses)).item()
    print(f"train_loss:{losses:.4f},lr:{optimizer.param_groups[0]['lr']:.6f},")
    return losses


@torch.no_grad()
def _val(data: DataLoader, model: torch.nn.Module):
    model.eval()
    bar = tqdm(data, colour="red", disable=True)
    ious = []
    acces = []
    for i, data in enumerate(bar):
        img, hv, np, tp = data
        img = img.to(device=process_arg.device)
        hv = hv.to(device=process_arg.device)
        np = np.to(device=process_arg.device)
        tp = tp.to(device=process_arg.device)
        with autocast():
            out_dict: dict = model(img)
            pred_tp, pred_np, pred_hv = out_dict.values()
            iou = jaccard(pred_np.argmax(1), np)
            acc = mask_category_acc(pred_tp.argmax(1), tp)
            ious.append(iou)
            acces.append(acc)
            bar.set_description(f"masks iou:{iou:.4f},category acc:{acc:.4f}")
    return torch.tensor(ious).mean().item(), torch.tensor(acces).mean().item()


class Module_process:
    def __init__(self, module: torch.nn.Module, dataset) -> None:
        self.ds = dataset
        self.module = module
        if dataset == "conic":
            print("loading conic dataset...")
            data_set = HoverNetCoNIC(
                arg.img_size,
                [
                    Random_flip(),
                    MyGausssianBlur(3, sigma=(0.2, 0.6)),
                    MyColorjitter(0.07, 0.07, 0.07, 0.07),
                ],
            )
            train_indexes = arg.TRAIN_SET_INDEX
            val_indexes = arg.VAL_SET_INDEX
            self.train_set = Subset(data_set, train_indexes)
            self.val_set = Subset(data_set, val_indexes)

        if dataset == "pannuke":
            print("loading pannuke dataset...")
            data_set = PannukeHover(
                arg.img_size,
                [
                    Random_flip(),
                    MyGausssianBlur(3, sigma=(0.2, 0.6)),
                    MyColorjitter(0.07, 0.07, 0.07, 0.07),
                ],
            )
            train_indexes = arg.TRAIN_SET_INDEX
            val_indexes = arg.VAL_SET_INDEX
            self.train_set = Subset(data_set, train_indexes)
            self.val_set = Subset(data_set, val_indexes)


class Training_process:
    def __init__(self) -> None:
        print("Check setted Config:\n")
        for _ in dir(arg):
            if not _.startswith("_"):
                print(_, "=", getattr(arg, _))
        self.net = arg.model(arg.num_classes)
        self.net.to(device=arg.device)
        self.optimizer = AdamW(
            self.net.parameters(), lr=arg.lr, weight_decay=arg.weight_decay
        )
        self.lrs = MultiStepLR(self.optimizer, arg.lr_s, gamma=0.1)
        self.module_process = Module_process(self.net, arg.ds)
        self.loading_data()

    def loading_data(self):
        train_set = self.module_process.train_set
        test_set = self.module_process.val_set
        train_data = DataLoader(
            train_set, batch_size=arg.batch, shuffle=True, num_workers=8
        )
        test_data = DataLoader(
            test_set, batch_size=arg.batch, shuffle=False, num_workers=2
        )
        setattr(self, "train_data", train_data)
        setattr(self, "test_data", test_data)

    def run(self):
        for _ in range(arg.epoch):
            losses = _train(self.train_data, optimizer=self.optimizer, model=self.net)
            self.lrs.step()
            torch.cuda.empty_cache()
            writer.add_scalar("Training loss", losses, _)

            if _ % 10 == 6:
                ious, acces = _val(self.test_data, self.net)
                writer.add_scalar("Hover:ious", ious, _)
                writer.add_scalar("Hover:category acc", acces, _)
                # if best_iou < ious or best_acc < acces:
                print(
                    f"best_iou:{ious:.4f},best_acc:{acces:.4f},\n epoch:{_} saving model "
                )
                torch.save(
                    self.net.state_dict(), os.path.join(arg.model_para, "best.pth")
                )


if __name__ == "__main__":
    setup_seed(24)
    engine = Training_process()
    engine.run()
