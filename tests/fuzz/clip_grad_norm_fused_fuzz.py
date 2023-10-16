import sys
import atheris
import torch
import torch_npu
import torch.nn as nn
import random
from apex import amp
with atheris.instrument_imports():
    from apex.optimizers import NpuFusedSGD

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(120, 84)
        self.linear2 = nn.Linear(84, 48)
    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x

@atheris.instrument_func
def Test_NpuFusedSGD(input_bytes):

    check_list_input = input_bytes.decode('utf-8', 'ignore').strip().split(',')
    if not check_list_input or len(check_list_input) != 6:
        return False
    try:
        for i in range(2):
            check_list_input[i] = float(check_list_input[i])
    except Exception as e:
        return False

    max_norm = random.random() * check_list_input[0]
    norm_type = random.random() * check_list_input[1]
    
    try:
        ip = torch.randn([1, 120]).npu().abs()
        ip.requires_grad = True
        input_list = [True, False]
        params = []
        for i in range(5):
            input_tensor_size = [random.randint(0, 5) for _ in range(random.randint(0, 5))]
            if input_list[random.randint(0, 1)]:
                input_tensor = torch.randn(input_tensor_size).float().npu()
            else:
                input_tensor = torch.randn(input_tensor_size).half().npu()
            params.append(input_tensor)
        for i, p in enumerate(params):
            if i < len(params) - 1:
                p.requires_grad = True
                p.grad = p.clone().detach() / 100
        optimizer = NpuFusedSGD(params, lr=0.1)
        model = Net().npu()
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', loss_scale=128, combine_grad=True)
        loss = model(ip).sum()
        loss.backward()
        print(loss)
        optimizer.step()
        optimizer.clip_optimizer_grad_norm_fused(max_norm=max_norm, norm_type=norm_type)
    except Exception as e:
        print(e)
        return True

if __name__ == "__main__":
    atheris.Setup(sys.argv, Test_NpuFusedSGD)
    atheris.Fuzz()
