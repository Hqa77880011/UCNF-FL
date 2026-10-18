import argparse
import csv
import json
import math
from fractions import Fraction
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import loadmat
from scipy.signal import butter, resample_poly, sosfiltfilt
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from .method import NeuroFuzzy, local_update, quantize, reliability, summarize


def prepare(args):
    records = list(csv.DictReader(open(args.manifest, encoding='utf-8')))
    x, y, client, rig, group, split = [], [], [], [], [], []
    assignments, intervals = {}, {}
    for row in records:
        if row['group'] in assignments and assignments[row['group']] != row['split']:
            raise ValueError('A bearing/acquisition group crosses local partitions')
        assignments[row['group']] = row['split']
        source = Path(args.manifest).parent / row['path']
        raw = np.asarray(loadmat(source)[row['key']] if source.suffix == '.mat' else np.load(source)).squeeze()
        if raw.ndim != 1:
            raise ValueError('Each selected source must contain one channel')
        fs = float(row['fs'])
        start, stop = int(row.get('start') or 0), int(row.get('stop') or len(raw))
        source_key = str(source.resolve()) + ':' + row.get('key', '')
        for old_start, old_stop, old_split in intervals.get(source_key, []):
            if old_split != row['split'] and max(start, old_start) - min(stop, old_stop) < math.ceil(2048 * fs / 12800):
                raise ValueError('Source intervals across partitions require a 2048-target-sample guard')
        intervals.setdefault(source_key, []).append((start, stop, row['split']))
        raw = raw[start:stop]
        raw = sosfiltfilt(butter(4, 10, 'highpass', fs=fs, output='sos'), raw - raw.mean())
        ratio = Fraction(12800 / fs).limit_denominator(1000)
        raw = resample_poly(raw, ratio.numerator, ratio.denominator)
        for t in range(0, len(raw) - 2047, 1024):
            x.append(raw[t:t + 2048])
            y.append(int(row['label']))
            client.append(row['client'])
            rig.append(row['rig'])
            group.append(row['group'])
            split.append(row['split'])
    x, client, split = np.asarray(x, dtype=np.float32), np.asarray(client), np.asarray(split)
    for identifier in np.unique(client):
        train = (client == identifier) & (split == 'train')
        if not train.any():
            raise ValueError('Every client needs training data')
        mean, std = x[train].mean(), max(x[train].std(), 1e-6)
        x[client == identifier] = (x[client == identifier] - mean) / std
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, x=x[:, None], y=y, client=client, rig=rig, group=group, split=split)


def train(args):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    d = np.load(args.data)
    model = NeuroFuzzy(int(d['y'].max()) + 1, args.rules)
    clients = list(np.unique(d['client']))
    train_data, val_data = {}, {}
    for client in clients:
        for partition, destination in [('train', train_data), ('val', val_data)]:
            selected = (d['client'] == client) & (d['split'] == partition)
            if not selected.any():
                raise ValueError('Each client needs separate nonempty train and validation groups')
            destination[client] = (torch.tensor(d['x'][selected]), torch.tensor(d['y'][selected], dtype=torch.long))
    uplink_residual = {c: {} for c in clients}
    downlink_residual, history = {}, []
    total_up, total_down, total_summary = 0, 0, 0
    enabled = args.variant not in ('fedavg', 'fedprox', 'no_quantization')
    for round_index in range(args.rounds):
        participants = list(rng.choice(clients, args.participation or len(clients), replace=False))
        updates, counts, entropies, briers, class_entropy, class_brier = [], [], [], [], [], []
        lr = 1e-5 + .5 * (.001 - 1e-5) * (1 + np.cos(np.pi * round_index / max(1, args.rounds - 1)))
        current = model.state_dict()
        for client in participants:
            x, y = train_data[client]
            local, count = local_update(model, x, y, args.local_epochs, args.batch, lr, args.variant,
                                        args.seed + round_index * len(clients) + clients.index(client))
            vx, vy = val_data[client]
            entropy, brier = summarize(local, vx, vy, args.batch)
            entropies.append(float(entropy.mean()))
            briers.append(float(brier.mean()))
            class_entropy.append([float(entropy[vy == c].mean()) if (vy == c).any() else float(entropy.mean()) for c in range(len(count))])
            class_brier.append([float(brier[vy == c].mean()) if (vy == c).any() else float(brier.mean()) for c in range(len(count))])
            delta = {k: local.state_dict()[k] - current[k] for k in current}
            if args.variant not in ('fedavg', 'fedprox', 'no_class_support'):
                delta['consequent'][:, count == 0] = 0
                if 'consequent' in uplink_residual[client]:
                    uplink_residual[client]['consequent'][:, count == 0] = 0
            transmitted, uplink_residual[client], payload = quantize(delta, uplink_residual[client], enabled)
            total_up += payload
            total_summary += 4 * (2 + 3 * len(count))
            updates.append(transmitted)
            counts.append(count)
        support = torch.stack(counts).float()
        r = reliability(entropies, briers)
        if args.variant in ('fedavg', 'fedprox', 'no_reliability'):
            r = torch.ones_like(r)
        alpha = support.sum(1) * r
        alpha /= alpha.sum()
        class_alpha = torch.zeros_like(support)
        for c in range(support.shape[1]):
            present = support[:, c] > 0
            if present.any():
                rc = reliability(np.array(class_entropy)[present, c], np.array(class_brier)[present, c])
                if args.variant in ('fedavg', 'fedprox', 'no_reliability'):
                    rc = torch.ones_like(rc)
                class_alpha[present, c] = support[present, c] * rc
                class_alpha[:, c] /= class_alpha[:, c].sum()
        aggregate = {}
        for key in current:
            if key == 'consequent' and args.variant not in ('fedavg', 'fedprox', 'no_class_support'):
                aggregate[key] = sum(update[key] * class_alpha[j][None, :, None] for j, update in enumerate(updates))
            else:
                aggregate[key] = sum(update[key] * alpha[j] for j, update in enumerate(updates))
        retained_residual = downlink_residual.get('consequent', torch.zeros_like(aggregate['consequent'])).clone()
        transmitted, downlink_residual, payload = quantize(aggregate, downlink_residual, enabled)
        if args.variant not in ('fedavg', 'fedprox', 'no_class_support'):
            absent = support.sum(0) == 0
            transmitted['consequent'][:, absent] = 0
            downlink_residual['consequent'][:, absent] = retained_residual[:, absent]
        total_down += payload * len(clients)
        model.load_state_dict({k: current[k] + transmitted[k] for k in current})
        history.append({'round': round_index + 1, 'participants': [str(c) for c in participants],
                        'reliability': r.tolist(), 'shared_weights': alpha.tolist(), 'class_weights': class_alpha.tolist(),
                        'uplink_bytes': total_up, 'downlink_bytes': total_down, 'summary_bytes': total_summary})
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({'state': model.state_dict(), 'rules': args.rules, 'classes': support.shape[1],
                'payload': {'uplink_bytes': total_up, 'downlink_bytes': total_down, 'summary_bytes': total_summary}}, out / 'model.pt')
    (out / 'rounds.json').write_text(json.dumps(history, indent=2))


def metrics(y, probability):
    prediction = probability.argmax(1)
    confidence = probability.max(1)
    ece = 0.
    for low in np.arange(15) / 15:
        selected = (confidence > low) & (confidence <= low + 1 / 15)
        if selected.any():
            ece += selected.mean() * abs((prediction[selected] == y[selected]).mean() - confidence[selected].mean())
    return {'accuracy': accuracy_score(y, prediction), 'balanced_accuracy': balanced_accuracy_score(y, prediction),
            'macro_f1': f1_score(y, prediction, average='macro', zero_division=0), 'ece': float(ece),
            'brier': float(np.mean(np.square(probability - np.eye(probability.shape[1])[y]).sum(1)))}


def evaluate(args):
    d = np.load(args.data)
    saved = torch.load(args.model, map_location='cpu', weights_only=False)
    model = NeuroFuzzy(saved['classes'], saved['rules'])
    model.load_state_dict(saved['state'])
    model.eval()
    ids = np.flatnonzero(d['split'] == 'test')
    x = d['x'][ids].copy()
    if args.snr is not None:
        rng = np.random.default_rng(args.seed)
        scale = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) / 10 ** (args.snr / 10))
        x += rng.normal(size=x.shape) * scale
    with torch.no_grad():
        probabilities = np.concatenate([model(torch.tensor(x[i:i + args.batch]))[0].numpy() for i in range(0, len(x), args.batch)])
    truth = d['y'][ids]
    by_client = {str(c): metrics(truth[d['client'][ids] == c], probabilities[d['client'][ids] == c]) for c in np.unique(d['client'][ids])}
    by_rig = {str(r): metrics(truth[d['rig'][ids] == r], probabilities[d['rig'][ids] == r]) for r in np.unique(d['rig'][ids])}
    result = {'overall': metrics(truth, probabilities), 'clients': by_client, 'rigs': by_rig,
              'equal_rig_macro_f1': float(np.mean([r['macro_f1'] for r in by_rig.values()])),
              'worst_client_macro_f1': min(r['macro_f1'] for r in by_client.values()),
              'client_macro_f1_std': float(np.std([r['macro_f1'] for r in by_client.values()])),
              'parameters': sum(p.numel() for p in model.parameters()), **saved['payload']}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'metrics.json').write_text(json.dumps(result, indent=2))
    np.savez_compressed(out / 'predictions.npz', probability=probabilities, truth=truth, client=d['client'][ids], rig=d['rig'][ids], group=d['group'][ids])
    plt.bar(list(by_client), [v['macro_f1'] for v in by_client.values()])
    plt.xticks(rotation=45)
    plt.ylabel('Client Macro-F1')
    plt.tight_layout()
    plt.savefig(out / 'clients.png', dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    p = sub.add_parser('train')
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--variant', choices=['full', 'fedavg', 'fedprox', 'no_reliability', 'no_class_support', 'no_brier', 'no_rule_regularization', 'no_quantization'], default='full')
    for key, default in [('rounds', 150), ('local-epochs', 5), ('batch', 64), ('rules', 12), ('participation', 0), ('seed', 17)]:
        p.add_argument('--' + key, type=int, default=default)
    p = sub.add_parser('evaluate')
    for key in ['data', 'model', 'output']:
        p.add_argument('--' + key, required=True)
    p.add_argument('--snr', type=float)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--batch', type=int, default=64)
    args = parser.parse_args()
    {'prepare': prepare, 'train': train, 'evaluate': evaluate}[args.command](args)
