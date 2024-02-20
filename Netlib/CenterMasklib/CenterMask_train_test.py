import argparse
import os

import torch
from torch.cuda.amp import GradScaler, autocasta
from detectron2.utils.events import EventStorage
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import MultiStepLR, _LRScheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from centermask.config import centermask_config, color, setup_seed
from centermask.train_net.tools.augmentation import (
    MyColorjitter,
    MyGausssianBlur,
    Random_flip,
)
from centermask.train_net.tools.dataset import (
    CentermaskCONICData,
    CentermaskPNCData,
    DataLoader,
    Subset,
    collect_fn_detectron,
)
from centermask.train_net.tools.metric import instance2results
from centermask.train_net.tools.visualize_utils import Visualizer

parser = argparse.ArgumentParser(description="train or test centermask")
parser.add_argument("--ds", type=str, default="conic")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--split_num", type=int, default=1)
parser.add_argument("--config", type=str, default="centermaskres50")
process_arg = parser.parse_args()
arg = getattr(centermask_config, process_arg.config)(
    process_arg.ds, process_arg.device, process_arg.split_num
)
writer = SummaryWriter(arg.exp_data_dir)


class Moudule_process:
    def __init__(self, moudule: torch.nn.Module, dataset) -> None:
        super().__init__()
        self.moudule = moudule
        if dataset == "conic":
            data_set = CentermaskCONICData(
                dataset_dir=arg.DATASET_DIR,
                masks_size=28,
                transfs=[
                    Random_flip(),
                    MyGausssianBlur(3, (0.3, 0.5)),
                    MyColorjitter(0.08, 0.08, 0.08, 0.08),
                ],
            )
            train_indexes, val_indexes = arg.TRAIN_SET_INDEX, arg.VAL_SET_INDEX
            self.train_set = Subset(data_set, train_indexes)
            self.val_set = Subset(data_set, val_indexes)
        if dataset == "pannuke":
            data_set = CentermaskPNCData(
                dataset_dir=arg.DATASET_DIR,
                masks_size=28,
                transfs=[
                    Random_flip(),
                ],
            )
            train_indexes, val_indexes = arg.TRAIN_SET_INDEX, arg.VAL_SET_INDEX
            self.train_set = Subset(data_set, train_indexes)
            self.val_set = Subset(data_set, val_indexes)

        self.best_val = 0
        self.validation = _val


def _train(
    data: DataLoader,
    opt: Optimizer,
    model: torch.nn.Module,
    lr_s: _LRScheduler,
    mode_process: Moudule_process,
):
    mode_process.moudule.train()
    bar = tqdm(data, colour="CYAN", disable=True)
    losses = []
    scaler = GradScaler()
    for data in bar:
        # data: List[Dict[str, Tensor]]
        # if data[i]['instances'] is None:
        # skip training

        data_dict, label = data

        idx = 0
        # label:tensor: [batch_size,2,h,w]
        # 遇到没有标注的图片跳过，instances为None，跳过训练
        for i in range(len(data_dict)):
            if data_dict[i]["instances"] is None:
                idx = 1
        if idx == 1:
            continue
        with autocast():
            with EventStorage():
                loss = model(data_dict)
            opt.zero_grad()
            loss = sum(_ for _ in loss.values())
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if not loss.isnan():
                losses.append(loss.item())
            # show loss and lr in tqdm
            # show mask iou, box iou, category acc in tqdm
            bar.set_description(
                f"losses:{loss.item():.4f},lr:{lr_s.get_last_lr()[0]:.4f}"
            )
    losses = torch.tensor(losses).mean().item()
    print(f"losses:{losses:.4f},lr:{lr_s.get_last_lr()[0]:.4f}")
    lr_s.step()
    return losses


@torch.no_grad()
def _val(data: DataLoader, model: torch.nn.Module):
    # 显示模型在验证集的表现
    model.eval()
    bar = tqdm(data, colour="red", disable=True)
    s2_boxes_iou = []
    s2_f1 = []
    s2_cls_acc = []
    masks_iou = []
    for i, data in enumerate(bar):
        data_dict, label = data
        with autocast():
            preds = model(data_dict)  # preds: Dict[str, Tensor]
        # stage1_iou,stage1_acc,stage1_recall= proposal_targetboxes_miou(stage1_rois,boxes)
        box_iou, cls_acc, mask_iou, prc, recall, f1 = instance2results(preds, data_dict)
        v = Visualizer(data_dict[0]["image"], gt=label[0])
        v.draw_predict(preds[0])
        v.save(arg.figure_dir, i)
        s2_cls_acc.append(cls_acc)
        s2_boxes_iou.append(box_iou)
        masks_iou.append(mask_iou)
        s2_f1.append(f1)
        bar.set_description(
            f"masks iou:{mask_iou},boxes iou:{box_iou},category acc:{cls_acc}"
        )
    s2_f1 = torch.stack(s2_f1, dim=0).mean(dim=0)
    return {
        "Val stage2 boxes iou": torch.tensor(s2_boxes_iou).mean().item(),
        "Val stage2 category accuracy": torch.tensor(s2_cls_acc).mean().item(),
        "Val stage2 masks iou": torch.tensor(masks_iou).mean().item(),
        "Val stage2 boxes f1_0.5": s2_f1[0].item(),
        "Val stage2 boxes f1_0.6": s2_f1[1].item(),
        "Val stage2 boxes f1_0.7": s2_f1[-1].item(),
    }


class Traning_process:
    def __init__(self) -> None:
        print("Check setted Config:\n")
        for _ in dir(arg):
            if not _.startswith("_"):
                print(_, "=", getattr(arg, _))

        self.net: torch.nn.Module = arg.model
        # 加载预训练模型
        self.optimizer = AdamW(
            self.net.parameters(), lr=arg.lr, weight_decay=arg.weight_decay
        )
        self.lrs = MultiStepLR(self.optimizer, arg.lr_s, gamma=0.1)
        self.module_process = Moudule_process(self.net, dataset=process_arg.ds)
        self.generating_dataloader()

    def generating_dataloader(self):
        train_set = self.module_process.train_set
        val_set = self.module_process.val_set
        train_data = DataLoader(
            train_set,
            batch_size=arg.batch,
            shuffle=True,
            num_workers=arg.NUM_WORKERS,
            collate_fn=collect_fn_detectron,
        )
        val_data = DataLoader(
            val_set, 1, shuffle=False, num_workers=1, collate_fn=collect_fn_detectron
        )
        setattr(self, "train_data", train_data)
        setattr(self, "val_data", val_data)

    def run(self):
        self.net.load_state_dict(torch.load(arg.load_model_para))
        for _ in range(arg.epoch):
            losses = _train(
                self.train_data,
                opt=self.optimizer,
                model=self.net,
                lr_s=self.lrs,
                mode_process=self.module_process,
            )
            torch.cuda.empty_cache()
            writer.add_scalar("Training loss", losses, _)
            if _ % 5 == 1:
                metrics: dict = self.module_process.validation(self.val_data, self.net)
                for k, v in metrics.items():
                    writer.add_scalar(k, v, _)
                # 将最好的模型保存下来

                print(
                    f"Epoch:{_}, category:{metrics['Val stage2 category accuracy']},\n F1_0.5:{metrics['Val stage2 boxes f1_0.5']},F1_0.6:{metrics['Val stage2 boxes f1_0.6']},F1_0.7:{metrics['Val stage2 boxes f1_0.7']},\n saving model to {arg.model_para+'/best.pth'}"
                )
                torch.save(
                    self.net.state_dict(), os.path.join(arg.model_para, "best.pth")
                )


if __name__ == "__main__":
    setup_seed(24)
    color = color.to(device=arg.DEVICE)
    train = Traning_process()
    train.run()
