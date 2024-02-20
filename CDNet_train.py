import argparse
import torch
from Netlib.CDNetlib.Config import CDNet_config, setup_seed, color
from Netlib.CDNetlib.tools.logger import Logger
from Netlib.CDNetlib.tools.augmentation import Random_flip, MyGausssianBlur, MyColorjitter,MyRandomCrop,MyCenterCrop
from Netlib.CDNetlib import CDNetDataset, model_unet_MandDandP
from torch.utils.data import DataLoader,Subset
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import MultiStepLR, _LRScheduler
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

parser = argparse.ArgumentParser(description="train CDNet")
parser.add_argument("--ds", type=str, default="monuseg")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument('--split_num',type=int,default=1)
parser.add_argument("--config", type=str, default="CDNet_MoNuSeg")
process_arg = parser.parse_args()
arg = getattr(CDNet_config, process_arg.config)(process_arg.ds, process_arg.device,process_arg.split_num)

Logger.init(logfile_level='info',log_file=arg.LOG_DIR + "/train.log")

writer = SummaryWriter(arg.exp_data_dir)
scaler = GradScaler()

def train_CDNet(
    model: torch.nn.Module,
    samples: DataLoader,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    epoch: int,
    writer: SummaryWriter,
    logger: Logger,
):
    model.train()
    bar = tqdm(samples, colour="CYAN", disable=True)
    losses = []
    for i, (imgs, classes, boundry, point, direction) in enumerate(bar):
        target = dict()
        target["classes"] = classes.to(arg.DEVICE)
        target["boundry"] = boundry.to(arg.DEVICE)
        target["point"] = point.to(arg.DEVICE)
        target["direction"] = direction.to(arg.DEVICE)

        imgs = imgs.to(arg.DEVICE)
        with autocast():
            optimizer.zero_grad()
            loss = model(imgs, target)
            _losses = sum(loss.values())
            scaler.scale(_losses).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(_losses.item())
            bar.set_description(f"train_loss:{_losses.item():.4f}")
    losses = torch.mean(torch.tensor(losses)).item()
    print(f"train_loss:{losses:.4f},epoch:{epoch},lr:{optimizer.param_groups[0]['lr']:.6f},")
    writer.add_scalar("train_loss", losses, epoch)
    logger.info(
        f"train_loss:{losses:.4f},epoch:{epoch},lr:{optimizer.param_groups[0]['lr']:.6f},"
    )
    scheduler.step()
    return losses

@torch.no_grad()
def _val(
    model: torch.nn.Module,
    samples: DataLoader,
    epoch: int,
    writer: SummaryWriter,
    logger: Logger,
):
    model.eval()
    bar = tqdm(samples, colour="red")
    metric = []
    val_losses = []
    for imgs, classes, boundry, point, direction in bar:
        target = dict()
        target["classes"] = classes.to(arg.DEVICE)
        target["boundry"] = boundry.to(arg.DEVICE)
        target["point"] = point.to(arg.DEVICE)
        target["direction"] = direction.to(arg.DEVICE)
        imgs = imgs.to(arg.DEVICE)
        with autocast():
            preds,loss = model(imgs, target)
            val_losses.append(sum(loss.values()).item())
            m = model.metric(preds, target)
            k = list(m.keys())
            metric.append(list(m.values()))
            bar.set_description(f"val_metric:{m}")

    val_losses = torch.mean(torch.tensor(val_losses)).item()
    mean_values = torch.mean(torch.tensor(metric), dim=0)
    metrics = dict(zip(k, mean_values))
    logger.info(f"val_metric:{metrics},epoch:{epoch},val_loss:{val_losses}")
    for k in metrics.keys():
        writer.add_scalar(k, metrics[k], epoch)
    writer.add_scalar("val_loss", val_losses, epoch)
    return metrics


if __name__ == "__main__":
    setup_seed(24)
    color = color.to(arg.DEVICE)
    Logger.info(
        f"Check setted Config:\n\
                =====================================\n"
    )
    for _ in dir(arg):
        Logger.info(f"{_}= {getattr(arg, _)}") if not _.startswith("_") else None

    net = model_unet_MandDandP.Unet(
        arg.BACKBONE_NAME, classes=arg.CLASSES,encoder_freeze=False, pretrained=True
    )
    net = net.to(arg.DEVICE)
    
    optimizer = AdamW(net.parameters(), lr=arg.LR, weight_decay=arg.WEIGHT_DECAY)
    scheduler = MultiStepLR(optimizer, milestones=arg.MILESTONES, gamma=arg.GAMMA)

    if arg.ds == "conic":
        train_set = CDNetDataset.CDNetDataset_CONIC(
            arg.DATASET_FP,
            [
                Random_flip(),
                MyGausssianBlur(kernel_size=3, sigma=(0.2, 0.3)),
                MyColorjitter(0.06, 0.06, 0.06, 0.06),
            ],
        )
        val_set = CDNetDataset.CDNetDataset_CONIC(arg.DATASET_FP)
    elif arg.ds == "pannuke":
        train_set = CDNetDataset.CDNetDataset_PANNUKE(
            arg.DATASET_FP,
            [
                Random_flip(),
                MyGausssianBlur(kernel_size=3, sigma=(0.2, 0.3)),
                MyColorjitter(0.06, 0.06, 0.06, 0.06),
            ],
        )
        val_set = CDNetDataset.CDNetDataset_PANNUKE(arg.DATASET_FP)
    elif arg.ds == "monuseg":
        train_set = CDNetDataset.CDNetDataset_MoNuSeg(arg.DATASET_FP,[
            MyRandomCrop((256,256)),
            Random_flip(),
            MyGausssianBlur(kernel_size=3, sigma=(0.2, 0.3)),
            MyColorjitter(0.06, 0.06, 0.06, 0.06)
            ]
        )
        val_set = CDNetDataset.CDNetDataset_MoNuSeg(arg.DATASET_FP,[MyCenterCrop((256,256))])
    elif arg.ds == "dsb":
        train_set = CDNetDataset.CDNetDataset_DSB(arg.DATASET_FP,[
            MyRandomCrop((256,256)),
            Random_flip(),
            MyGausssianBlur(kernel_size=3, sigma=(0.2, 0.3)),
            MyColorjitter(0.06, 0.06, 0.06, 0.06)
            ],
            train=True
        )
        val_set = CDNetDataset.CDNetDataset_DSB(arg.DATASET_FP,[MyCenterCrop((256,256))])
    elif arg.ds == "cpm":
        train_set = CDNetDataset.CDNetDataset_CPM(arg.DATASET_FP,[
            MyRandomCrop((256,256)),
            Random_flip(),
            MyGausssianBlur(kernel_size=3, sigma=(0.2, 0.3)),
            MyColorjitter(0.06, 0.06, 0.06, 0.06)
            ]
        )
        val_set = CDNetDataset.CDNetDataset_CPM(arg.DATASET_FP,[MyCenterCrop((256,256))])

    train_set = Subset(
        train_set,
        arg.TRAIN_SET_INDEX,
    )
    val_set = Subset(val_set, arg.VAL_SET_INDEX)

    train_samples = DataLoader(
        train_set,
        batch_size=arg.BATCH_SIZE,
        shuffle=True,
        num_workers=arg.NUM_WORKERS,
        collate_fn=None,
    )
    val_samples = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1)

    Logger.info(f"train_set:{len(train_set)},val_set:{len(val_set)}")
    best_metric_c = 0
    for epoch in range(arg.EPOCHS):
        train_loss = train_CDNet(
            net, train_samples, optimizer, scheduler, epoch, writer, Logger
        )
        if epoch % arg.VAL_EPOCH == 1:
            metric = _val(net, val_samples, epoch, writer, Logger)
            if metric["boundry"] > best_metric_c:
                best_metric_c = metric["boundry"]
                torch.save(net.state_dict(), arg.model_para + "/best.pth")
        # checkpoint
