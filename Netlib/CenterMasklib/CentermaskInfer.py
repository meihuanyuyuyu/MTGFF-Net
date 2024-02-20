import torch
from detectron2.structures import Instances


class CenteMaskInfer:
    def __init__(self, arg) -> None:
        self.net = arg.model
        self.net.to(arg.DEVICE)
        self.net.load_state_dict(torch.load(arg.load_model_para, map_location=arg.DEVICE))
        self.net.eval()
        self.dataset = arg.ds

    @torch.no_grad()
    def __call__(self, img: torch.Tensor):
        results = torch.zeros(img.shape[0], 2, *img.shape[2:], device=img.device) # 1, 2, 256, 256
        inputs = [{"image": i} for i in img]
        preds = self.net(inputs)
        for pred, result in zip(preds, results):
            pred_instance: Instances = pred["instances"]
            classes = pred_instance.pred_classes + 1
            masks: torch.Tensor = pred_instance.pred_masks #  n ,256, 256
            masks = masks.long()
            if len(masks) != 0:
                for m, c in zip(masks, classes):
                    result[1] = torch.where(
                        torch.logical_and(m, result[1] == 0), result[1] + c, result[1]
                    )
                result[0] = torch.cat(
                    [torch.zeros_like(masks[0:1]), masks], dim=0
                ).argmax(dim=0)
        results = results.permute(0, 2, 3, 1).cpu().numpy()
        return results

