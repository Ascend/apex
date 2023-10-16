import atheris
import random
import sys

import torch
import torch_npu

with atheris.instrument_imports():
    from apex.optimizers import NpuFusedSGD


@atheris.instrument_func
def Test_zero_grad(input_bytes):
    input_list = [True, False]
    check_list_input = input_bytes.decode('utf-8', 'ignore').strip().split(',')
    if not check_list_input or len(check_list_input) != 6:
        return False
    try:
        for i in range(6):
            check_list_input[i] = float(check_list_input[i])
    except Exception as e:
        return False

    lr = random.random() * check_list_input[1]
    momentum = random.random() * check_list_input[2]
    dampening = random.random() * check_list_input[3]
    weight_decay = random.random() * check_list_input[4]
    number_of_params = int(check_list_input[5])

    try:
        params = []
        for i in range(number_of_params):
            input_tensor_size = [random.randint(0, check_list_input[0]) for _ in range(random.randint(0, 5))]
            if input_list[random.randint(0, 1)]:
                input_tensor = torch.randn(input_tensor_size).float().npu()
            else:
                input_tensor = torch.randn(input_tensor_size).half().npu()
            params.append(input_tensor)
        for i, p in enumerate(params):
            if i < len(params) - 1:
                p.requires_grad = True
                p.grad = p.clone().detach() / 100
        opt = NpuFusedSGD(params, lr, momentum, dampening, weight_decay, input_list[random.randint(0, 1)])
        opt.zero_grad()
    except Exception as e:
        print(e)
        return True


if __name__ == "__main__":
    atheris.Setup(sys.argv, Test_zero_grad)
    atheris.Fuzz()
