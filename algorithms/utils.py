

def create_alg(args, device, num_classes, train_loader):
    if args.alg == "standard":
        from algorithms.standard import Standard
        alg_obj = Standard(args, device, num_classes, train_loader)
    elif args.alg == "odnl":
        from algorithms.odnl import ODNL
        alg_obj = ODNL(args, device, num_classes, train_loader)
    return alg_obj