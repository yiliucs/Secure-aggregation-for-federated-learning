from models import Nets
from torchstat import stat
from utils.options import args_parser
import torch
import torchvision.models as models
args = args_parser()
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
model = Nets.CNNCeleba(args=args).to(args.device)

stat(model, (3, 128, 128))

print("resnet18 have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))