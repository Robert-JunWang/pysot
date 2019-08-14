from pysot.models.backbone import PeleeNet7a
import torch

# converted_model_file = '../../pretrained_models/peleenet21a_tracking.pth'
# state_dict = torch.load('../../pretrained_models/peleenet21a.pth')
state_dict = torch.load('/home/rwang/workspace/PeleeNetV2/weights/pnet17v1_7387_cpu.pth.tar')['state_dict']
converted_model_file = '../pretrained_models/peleenet17a_tracking.pth'

model = PeleeNet7a()
t = {k:v for k,v in state_dict.items() if k in model.state_dict()}
model.load_state_dict(t)
torch.save(model.state_dict(), converted_model_file)