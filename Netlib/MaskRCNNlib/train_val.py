import argparse
import os
import json
import Config
import torch
from Config.mrcnn_configs import color, setup_seed
from tools.augmentation import (
    MyColorjitter,
    MyGausssianBlur,
    Random_flip,
    MyRandomCrop,
    MyCenterCrop,
)
from tools.dataset import (
    DataLoader,
    Subset,
    collect_fn_semantic,
    MRCNNLizardDataset,
    ConicDataset,
    collect_fn,
)
from tools.visualize_utils import Visualizer
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import MultiStepLR, _LRScheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from warnings import filterwarnings
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler

filterwarnings("ignore")
print("start time:", datetime.now())
# change the mode
parser = argparse.ArgumentParser(description="train MRCNN")
parser.add_argument("--ds", type=str, default="conic")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--split_num", type=int, default=1)
parser.add_argument("--config", type=str, default="MRCNNConfig1")
process_arg = parser.parse_args()
arg = getattr(Config, process_arg.config)(
    ds=process_arg.ds,
    device=process_arg.device,
    ds_num=process_arg.split_num,
)
writer = SummaryWriter(arg.exp_data_dir)


class Moudule_mode:
    def __init__(self, dataset) -> None:
        
        self.validation = val_CellMaskNet_v2
        
            # self.moudule.load_state_dict(torch.load(arg.model_para), strict=True)
        if dataset == "lizard":
            self.train_set = [
                MRCNNLizardDataset(
                    process_arg.ds_path,
                    [
                        MyRandomCrop(256),
                        Random_flip(),
                        MyGausssianBlur(3, sigma=(0.2, 0.3)),
                        MyColorjitter(0.08, 0.08, 0.08, 0.08),
                    ],
                    mask_size=arg.ROI_RESOLUTION,
                    set_indices=i,
                )
                for i in range(1,4)
            ]
            self.val_set = [MRCNNLizardDataset(
                    process_arg.ds_path,
                    [
                        MyCenterCrop(256),
                        Random_flip(),
                        MyGausssianBlur(3, sigma=(0.2, 0.3)),
                        MyColorjitter(0.08, 0.08, 0.08, 0.08),
                    ],
                    mask_size=arg.ROI_RESOLUTION,
                    set_indices=i,
                )
                for i in range(1,4)]

        if dataset == "conic":
            sets = ConicDataset( masks_size=arg.ROI_RESOLUTION, transfs=[
                    Random_flip(),
                    MyColorjitter(0.08, 0.08, 0.08, 0.08),
                ],)
            self.train_set = Subset(sets, arg.TRAIN_SET_INDEX)
            self.val_set = Subset(sets, arg.VAL_SET_INDEX)

        if dataset == "pannuke":
            self.train_set = MRCNNLizardDataset(
                masks_size=arg.ROI_RESOLUTION,
                transfs=[
                    Random_flip(),
                    MyGausssianBlur(3, sigma=(0.2, 0.3)),
                    MyColorjitter(0.05, 0.05, 0.05, 0.05),
                ],
            )



def train_CellMaskNet(
    data: DataLoader,
    opt: Optimizer,
    model: torch.nn.Module,
    lr_s: _LRScheduler,
):
    model.train()
    scaler = GradScaler()
    bar = tqdm(data, colour="CYAN", disable=False)
    losses = []
    for data in bar:
        imgs, labels, boxes, masks, cls = data
        imgs = imgs.to(device=process_arg.device)
        masks = [mask.to(device=process_arg.device) for mask in masks]
        boxes = [box.to(device=process_arg.device) for box in boxes]
        cls = [_.to(device=process_arg.device) for _ in cls]
        with autocast():
            loss: dict = model(imgs, boxes, masks, cls)
            opt.zero_grad()
            loss_ = sum([_ if _ is not None else 0 for _ in loss.values()])
            scaler.scale(loss_).backward()
            scaler.step(opt)
            if not loss_.isnan():
                losses.append(loss_.item())
        # show every loss in tqdm
        bar.set_description(f"losses:{loss_.item():.4f},{loss},lr:{lr_s.get_last_lr()}")
    losses = torch.tensor(losses).mean().item()
    # print(f"losses:{losses:.4f}")
    lr_s.step()
    return losses


@torch.no_grad()
def val_CellMaskNet_v2(data: DataLoader, model: torch.nn.Module):
    model.eval()
    bar = tqdm(data, colour="red", disable=False)
    s1_box_f1 = []
    s2_boxes_f1 = []
    s2_aps = []
    boxes_acc = []
    masks_iou = []
    semantic_ious = []
    for i, data in enumerate(bar):
        imgs, labels, boxes, masks, cls = data
        if len(boxes[0]) == 0:
            continue
        v = Visualizer(imgs[0], labels[0], label=Visualizer.CONIC_CLASSES)
        imgs = imgs.to(device=arg.device)
        masks = [mask.to(device=arg.device) for mask in masks]
        boxes = [box.to(device=arg.device) for box in boxes]
        cls = [_.to(device=arg.device) for _ in cls]
        #semantic = semantic_map.to(device=process_arg.device, dtype=torch.int64)

        with autocast():
            preds, metircs = model(imgs, boxes, masks, cls)
            v.draw_predict(preds, stage1_rois)
            v.save(arg.figure_dir, i)
            if metircs is not None:
                s1_f1, s2_f1,s2_ap, masks_jaccard, categ_accuracy, semantic_iou = metircs
                s1_box_f1.append(s1_f1)
                s2_boxes_f1.append(s2_f1)
                s2_aps.append(s2_ap)
                boxes_acc.append(categ_accuracy)
                masks_iou.append(masks_jaccard)
                #semantic_ious.append(semantic_iou)
            else:continue
            # maskrcnn_visualization(stage1_rois,rois,out_cls,out_masks,labels,arg.val_img_fp+f'/{i}.png',color)
        # pickle the result

    s2_f1 = torch.stack(s2_boxes_f1, dim=0).mean(dim=0)
    s2_ap = torch.stack(s2_aps,dim=0).mean(dim=0) 

    results = {
        "Val stage1 boxes f1_0.5": torch.tensor(s1_box_f1).mean().item(),
        "Val stage2 boxes f1_0.5": s2_f1[0].item(),
        "Val stage2 boxes f1_0.6": s2_f1[1].item(),
        "Val stage2 boxes f1_0.7": s2_f1[-1].item(),
        "Val stage2 boxes AP_0.5":s2_ap[0].item(),
        "Val stage2 boxes AP_0.6":s2_ap[1].item(),
        "Val stage2 boxes AP_0.7":s2_ap[-1].item(),
        "Val stage2 category accuracy": torch.tensor(boxes_acc).mean().item(),
        "Val stage2 masks iou": torch.tensor(masks_iou).mean().item(),
    }
    if arg.USE_SEMANTIC:
        results["Val semantic iou"] = torch.tensor(semantic_ious).mean().item()
    json.dump(results, open(os.path.join(arg.exp_data_dir, "val_result.json"), "a+"))
    return results


class Traning_process:
    def __init__(self) -> None:
        print("Check setted Config:\n")
        for _ in dir(arg):
            if not _.startswith("_"):
                print(_, "=", getattr(arg, _))

        self.net = arg.model(
            anchors=arg.anchor_wh.to(arg.device),
            backbone=arg.BACKBONE,
            bottom_up=arg.BOTTOM_UP,
            proposal_generator=arg.PROPAOSAL_GENERATOR,
            stride=arg.STRIDE,
            rpn_pos_threshold=arg.RPN_POS_THRESHOLD,
            rpn_fraction_ratio=arg.RPN_FRACTION_RATIO,
            nms_threshold=arg.NMS_THRESHOLD,
            pre_nms_k=arg.PRE_NMS_K,
            post_nms_k=arg.POST_NMS_K,
            roi_head=arg.ROI_HEAD,
            box_detection=arg.BOX_DETECTION,
            expand=arg.EXPAND,
            expand_ratio=arg.EXPAND_RATIO,
            use_gt_box=arg.USE_GT_BOX,
            roi_resolution=arg.ROI_RESOLUTION,
            stage2_max_proposal=arg.STAGE2_MAX_PROPOSAL,
            stage2_sample_ratio=arg.STAGE2_SAMPLE_RATIO,
            box_weight=arg.BOX_WEIGHT,
            roi_pos_threshold=arg.ROI_POS_THRESHOLD,
            post_decttion_score_threshold=arg.POST_DETECTION_SCORE_THRESHOLD,
            detection_per_img=arg.DETECTION_PER_IMG,
            num_classes=arg.NUM_CLASSES,
            use_semantic=arg.USE_SEMANTIC,
            seg_stride=arg.SEG_STRIDE,
            fuse_feature=arg.FUSE_FEATURE,
        ).to(device=arg.device)

        self.optimizer = AdamW(
            self.net.parameters(), lr=arg.lr, weight_decay=arg.weight_decay
        )
        self.lrs = MultiStepLR(self.optimizer, arg.lr_s, gamma=0.1)
        self.module_process = Moudule_mode(
            dataset=process_arg.ds
        )


        def stage1_boxes_hook_for_val(moudle, input, output):
            global stage1_rois
            stage1_rois = input[0]

        self.net.roi_head.register_forward_hook(stage1_boxes_hook_for_val)
        self.generating_data()

    def generating_data(self):
        train_set = self.module_process.train_set
        val_set = self.module_process.val_set
        train_data = DataLoader(
            train_set,
            batch_size=arg.batch,
            shuffle=True,
            num_workers=8,
            collate_fn=collect_fn,
        )
        val_data = DataLoader(
            val_set, 1, shuffle=False, num_workers=4, collate_fn=collect_fn
        )
        setattr(self, "train_data", train_data)
        setattr(self, "val_data", val_data)

    def run(self):
        max_categ = 0
        max_iou = 0
        path = arg.model_para
        if not os.path.exists(path):
            os.makedirs(path)
        for _ in range(arg.epoch):
            losses = train_CellMaskNet(
                self.train_data,
                opt=self.optimizer,
                model=self.net,
                lr_s=self.lrs, 
            )
            torch.cuda.empty_cache()
            writer.add_scalar("Training loss", losses, _)
            if _ % 20 == 10:
                metrics: dict = self.module_process.validation(self.val_data, self.net)
                if (
                    metrics["Val stage2 category accuracy"] > max_categ
                    or metrics["Val stage2 boxes f1_0.5"] > max_iou
                ):
                    max_categ = metrics["Val stage2 category accuracy"]
                    max_iou = metrics["Val stage2 boxes f1_0.5"]
                    torch.save(self.net.state_dict(), arg.model_para + '/best.pth')
                for k, v in metrics.items():
                    writer.add_scalar(k, v, _)


if __name__ == "__main__":
    setup_seed(43, benchmark=True)
    color = color.to(device=arg.device)
    process = Traning_process()
    process.run()
