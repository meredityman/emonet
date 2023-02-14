from pathlib import Path
import torch
import torch.onnx
import subprocess


import tensorrt



from emonet.models import EmoNet

BATCH_SIZE=1

# dummy_input=torch.randn(BATCH_SIZE, 3, 256, 256)

# state_dict_path = Path(__file__).parent.joinpath('pretrained', f'emonet_8.pth')
# state_dict = torch.load(str(state_dict_path), map_location='cpu')
# state_dict = {k.replace('module.',''):v for k,v in state_dict.items()}
# net = EmoNet(n_expression=8)
# net.load_state_dict(state_dict, strict=False)
# net.eval()

# torch.onnx.export(net, dummy_input, "models/emonet.onnx", verbose=False)

cmd = """trtexec --onnx=models/emonet.onnx --saveEngine=models/emonet16.trt  \
    --explicitBatch --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16"""

subprocess.call(cmd)
