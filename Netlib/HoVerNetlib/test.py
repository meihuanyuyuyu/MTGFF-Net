
from config.hovernet_config import HoverConfig1
from tools.dataset import HoverNetCoNIC
import torch

data = HoverNetCoNIC([256,256])
arg = HoverConfig1("conic","cuda",1)
model = arg.model().to(arg.device)
model.load_state_dict(torch.load(arg.load_model_para,map_location=arg.device))
model.eval()

d = data[0]
img, hv, np, nc = d
img = img.unsqueeze(0).to(arg.device)
hv = hv.unsqueeze(0).to(arg.device)
np = np.unsqueeze(0).to(arg.device)
nc = nc.unsqueeze(0).to(arg.device)

from torch.cuda.amp import autocast
out = []

def hook(module,input,output):
    out.append(output)

for name, module in model.named_children():
    print(name)
    getattr(model,name).register_forward_hook(hook)

with autocast():
    preds = model(img)
    tp, _np, hv = preds.values()



