import copy
import math

import torch
from torch import nn


class NeuroFuzzy(nn.Module):
    def __init__(self, classes=3, rules=12):
        super().__init__()
        blocks, previous = [], 1
        for channels, kernel in zip([64, 128, 128, 256, 256], [9, 5, 5, 3, 3]):
            blocks.extend([nn.Conv1d(previous, channels, kernel, padding=kernel // 2),
                           nn.GroupNorm(8, channels), nn.ReLU(), nn.MaxPool1d(2)])
            previous = channels
        self.encoder = nn.Sequential(*blocks, nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(256, 32))
        self.projection = nn.Linear(32, 8)
        self.centers = nn.Parameter(torch.randn(rules, 8) * .2)
        self.widths = nn.Parameter(torch.zeros(rules, 8))
        self.consequent = nn.Parameter(torch.randn(rules, classes, 9) * .02)
        self.log_temperature = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        z = self.projection(self.encoder(x))
        width = nn.functional.softplus(self.widths).square() + 1e-7
        log_membership = -.5 * (z[:, None] - self.centers).square() / width
        firing = torch.log(torch.exp(log_membership) + 1e-7).sum(-1).softmax(-1)
        augmented = torch.cat([z, torch.ones_like(z[:, :1])], -1)
        consequent = torch.einsum('bd,rcd->brc', augmented, self.consequent)
        logits = (firing[:, :, None] * consequent).sum(1) / (nn.functional.softplus(self.log_temperature) + 1e-6)
        return logits.softmax(-1), firing


def quantize(delta, residual, enabled=True):
    transmitted, remainder, payload = {}, {}, 0
    for key, value in delta.items():
        if not enabled:
            transmitted[key] = value.clone()
            remainder[key] = torch.zeros_like(value)
            payload += 4 * value.numel()
            continue
        corrected = value + residual.get(key, torch.zeros_like(value))
        maximum = float(corrected.abs().max())
        scale = maximum / 127 if maximum > 0 else 1.
        normalized = corrected / scale
        lower = normalized.floor()
        integer = (lower + (torch.rand_like(lower) < normalized - lower)).clamp(-127, 127).to(torch.int8)
        transmitted[key] = integer.float() * scale
        remainder[key] = corrected - transmitted[key]
        payload += value.numel() + 4
    return transmitted, remainder, payload


def reliability(entropy, brier):
    def normalize(values):
        values = torch.as_tensor(values, dtype=torch.float32)
        spread = values.max() - values.min()
        return (values - values.min()) / spread if spread > 0 else torch.zeros_like(values)
    return .2 + .8 * torch.exp(-normalize(entropy) - 1.5 * normalize(brier))


def summarize(model, x, y, batch):
    model.eval()
    probabilities, firing = [], []
    with torch.no_grad():
        for start in range(0, len(x), batch):
            p, pi = model(x[start:start + batch])
            probabilities.append(p)
            firing.append(pi)
    p, pi = torch.cat(probabilities), torch.cat(firing)
    entropy = -(pi * pi.clamp_min(1e-7).log()).sum(-1) / math.log(pi.shape[-1])
    brier = (p - nn.functional.one_hot(y, p.shape[-1])).square().sum(-1)
    return entropy, brier


def local_update(global_model, x, y, epochs, batch, lr, variant, seed):
    model = copy.deepcopy(global_model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    counts = torch.bincount(y, minlength=model.consequent.shape[1])
    class_weight = (counts + 1).float().rsqrt()
    class_weight /= class_weight.sum()
    generator = torch.Generator().manual_seed(seed)
    anchor = {k: v.detach() for k, v in global_model.named_parameters()}
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(x), generator=generator)
        for ids in order.split(batch):
            batch_x = x[ids].clone()
            shifts = torch.randint(-64, 65, (len(ids),), generator=generator)
            scales = torch.tensor([.9, .95, 1., 1.05, 1.1])[torch.randint(5, (len(ids),), generator=generator)]
            for j in range(len(ids)):
                batch_x[j] = batch_x[j].roll(int(shifts[j]), -1) * scales[j]
            p, firing = model(batch_x)
            target = y[ids]
            correct = p[torch.arange(len(ids)), target].clamp_min(1e-7)
            loss = -(class_weight[target] * (1 - correct).square() * correct.log()).mean()
            if variant != 'no_brier':
                loss += .2 * (p - nn.functional.one_hot(target, p.shape[1])).square().sum(-1).mean()
            if variant != 'no_rule_regularization':
                loss += .01 * (firing.mean(0) - 1 / firing.shape[1]).square().sum()
            if variant == 'fedprox':
                loss += .005 * sum((v - anchor[k]).square().sum() for k, v in model.named_parameters())
            optimizer.zero_grad()
            loss.backward()
            if variant not in ('fedavg', 'fedprox', 'no_class_support'):
                model.consequent.grad[:, counts == 0] = 0
            optimizer.step()
    return model, counts
