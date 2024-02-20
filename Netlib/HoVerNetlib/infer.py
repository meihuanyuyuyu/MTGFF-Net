import torch
from .model.HoverNet import HoverNet,proc_np_hv
from torch.cuda.amp import autocast

class HovernetInfer:
    def __init__(self,arg) -> None:
        self.net = HoverNet(arg.num_classes)
        self.net.load_state_dict(torch.load(arg.load_model_para,map_location=arg.device))
        self.net.eval()
        self.net.to(device=arg.device)
        self.dataset = arg.ds
        self.img_size = arg.img_size

    @torch.no_grad()
    def __call__(self,img:torch.Tensor):
        with autocast():
            predict:dict =self.net(img)
            tp,_np,hv = predict.values()
            _np = _np.argmax(dim=1)
            tp = tp.argmax(dim=1).squeeze(0)
        pred = torch.stack([_np,*hv.unbind(dim=1)],dim=-1).cpu().squeeze(0)
        proc_pred =torch.from_numpy(proc_np_hv(pred))
        pred = torch.zeros(2, *self.img_size).long()
        pred[0] = proc_pred
        for idx in range(1, proc_pred.max() + 1):
            mask = proc_pred == idx
            if mask.sum()==0:
                continue
            cls = tp[mask].mode().values
            if cls == 0:
                pred[0][mask] = 0
            else:
                pred[1][mask] = cls
        return pred.unsqueeze(0).permute(0,2,3,1).cpu().numpy()


  