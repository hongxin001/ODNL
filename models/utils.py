from models.wrn import WideResNet
import torch

def build_model(model_type, num_classes, device, args):
    if model_type == "wrn":
        net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate)
    elif model_type == "resnet":
        from models.resnet import ResNet34
        net = ResNet34(num_classes=num_classes)
    net.to(device)
    if args.gpu is not None and len(args.gpu) > 1:
        gpu_list = [int(s) for s in args.gpu.split(',')]
        net = torch.nn.DataParallel(net, device_ids=gpu_list)
    return net