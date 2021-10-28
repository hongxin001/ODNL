import torch
import numpy as np
import torch.nn.functional as F


def get_ood_scores(args, net, loader, ood_num_examples, device, in_dist=False):
    _score = []
    _right_score = []
    _wrong_score = []

    concat = lambda x: np.concatenate(x, axis=0)
    to_np = lambda x: x.data.cpu().numpy()

    with torch.no_grad():
        for batch_idx, examples in enumerate(loader):
            data, target = examples[0], examples[1]
            if batch_idx >= ood_num_examples // args.test_bs and in_dist is False:
                break

            data = data.to(device)

            output = net(data)
            smax = to_np(F.softmax(output, dim=1))

            if args.score == 'ce':
                _score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1))))
            elif args.score == 'energy':
                _score.append(-to_np((args.T * torch.logsumexp(output / args.T, dim=1))))
            else: # original MSP and Mahalanobis (but Mahalanobis won't need this returned)
                _score.append(-np.max(smax, axis=1))

            if in_dist:
                preds = np.argmax(smax, axis=1)
                targets = target.numpy().squeeze()
                right_indices = preds == targets
                wrong_indices = np.invert(right_indices)

                if args.score == 'ce':
                    _right_score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1)))[right_indices])
                    _wrong_score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1)))[wrong_indices])
                else:
                    _right_score.append(-np.max(smax[right_indices], axis=1))
                    _wrong_score.append(-np.max(smax[wrong_indices], axis=1))

    if in_dist:
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    else:
        return concat(_score)[:ood_num_examples].copy()