"""Calibrate every model confidence that the dashboard shows, using a fit split and a DIFFERENT evaluation split.

Sections (run one with --only): corrosion, yolo, patchcore, rca, forecast.
Outputs models/calibration/calibration.json (parameters + before/after reliability metrics). Nothing here changes a model
or a decision threshold; calibrated numbers are shown beside the raw ones.

Metrics: ECE (10 equal-width bins, top-label for multiclass), Brier score, NLL. Lower is better.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
OUT = ROOT / 'models/calibration/calibration.json'


def ece(p, y, bins=10):
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if mask.any():
            total += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(total)


def brier(p, y):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def reliability(p, y, bins=5):
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if mask.sum():
            rows.append({'range': [round(lo, 2), round(hi, 2)], 'n': int(mask.sum()), 'mean_predicted': round(float(p[mask].mean()), 3),
                         'observed_rate': round(float(y[mask].mean()), 3)})
    return rows


def metrics(p, y):
    return {'ece': round(ece(p, y), 4), 'brier': round(brier(p, y), 4), 'n': int(len(y))}


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_platt(score, y):
    """p = sigmoid(a * score + b)."""
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(C=1e4, max_iter=1000).fit(np.asarray(score, float).reshape(-1, 1), np.asarray(y, int))
    return float(model.coef_[0][0]), float(model.intercept_[0])


def platt(score, a, b):
    return 1 / (1 + np.exp(-(a * np.asarray(score, float) + b)))


def section_corrosion():
    import torch
    import train_rust as tr
    manifest = json.loads(tr.MANIFEST.read_text(encoding='utf-8'))
    split = {n: [r for r in manifest['records'] if r['split'] == n] for n in ('validation', 'test')}
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    artifact = torch.load(tr.OUT / 'rust_classifier.pt', map_location='cpu', weights_only=True)
    model = tr.build_model()
    model.load_state_dict(artifact['state_dict'])
    model = model.to(device)
    probs = {n: tr.predict(model, split[n], device) for n in split}
    result = {'fit_split': 'validation (rust photos)', 'eval_split': 'test (rust photos)', 'artifact_sha256': hashlib.sha256((tr.OUT / 'rust_classifier.pt').read_bytes()).hexdigest()}
    y = {n: np.array([r['corrosion'] for r in split[n]], int) for n in split}
    a, b = fit_platt(logit(probs['validation'][:, 0]), y['validation'])
    result['corrosion'] = {'method': 'Platt scaling on the logit of the raw probability', 'params': {'a': a, 'b': b},
        'before': metrics(probs['test'][:, 0], y['test']), 'after': metrics(platt(logit(probs['test'][:, 0]), a, b), y['test']),
        'reliability_after': reliability(platt(logit(probs['test'][:, 0]), a, b), y['test'])}
    pos = {n: [i for i, r in enumerate(split[n]) if r['corrosion']] for n in split}
    ys = {n: np.array([split[n][i]['severe'] for i in pos[n]], int) for n in split}
    sa, sb = fit_platt(logit(probs['validation'][pos['validation'], 1]), ys['validation'])
    sp = probs['test'][pos['test'], 1]
    result['severe'] = {'method': 'Platt scaling, corrosion-positive images only', 'params': {'a': sa, 'b': sb},
        'before': metrics(sp, ys['test']), 'after': metrics(platt(logit(sp), sa, sb), ys['test'])}
    return result


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter, 1e-9)


def section_yolo():
    import cv2
    from core import config  # noqa: F401
    from scripts.prepare_neu import CLASSES
    from vision.yolo_detector import ProxyDetector
    detector = ProxyDetector()
    rows = {}
    for name in ('val', 'test'):
        records = []
        for image in sorted((ROOT / f'data/processed/neu-det/images/{name}').glob('*.jpg')):
            frame = cv2.imread(str(image))
            height, width = frame.shape[:2]
            truth = []
            for line in (ROOT / f'data/processed/neu-det/labels/{name}/{image.stem}.txt').read_text().split('\n'):
                if line.strip():
                    c, cx, cy, w, h = line.split()
                    cx, cy, w, h = float(cx) * width, float(cy) * height, float(w) * width, float(h) * height
                    truth.append((int(c), [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]))
            used = set()
            for det in sorted(detector.detect(frame)['detections'], key=lambda d: -d['confidence']):
                best = max(((iou(det['bbox_xyxy_px'], box), i) for i, (c, box) in enumerate(truth) if i not in used and CLASSES[c] == det['label']), default=(0, -1))
                hit = best[0] >= .5
                if hit:
                    used.add(best[1])
                records.append((det['label'], det['confidence'], int(hit)))
        rows[name] = records
    from sklearn.isotonic import IsotonicRegression
    fit = np.array([(c, h) for _, c, h in rows['val']]); ev = np.array([(c, h) for _, c, h in rows['test']])
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip').fit(fit[:, 0], fit[:, 1])
    calibrated = iso.predict(ev[:, 0])
    per_class = {}
    for label in sorted({r[0] for r in rows['test']}):
        mask = np.array([r[0] == label for r in rows['test']])
        per_class[label] = {'detections': int(mask.sum()), 'raw_mean_conf': round(float(ev[mask, 0].mean()), 3),
                            'observed_precision': round(float(ev[mask, 1].mean()), 3), 'calibrated_mean': round(float(calibrated[mask].mean()), 3)}
    return {'method': 'isotonic regression, all classes pooled (per-class samples are too small)', 'artifact_sha256': detector.digest,
            'fit_split': f'NEU val ({len(fit)} detections at conf >= .25)',
            'eval_split': f'NEU test ({len(ev)} detections)', 'iou_match': .5,
            'params': {'x': [round(float(v), 4) for v in iso.X_thresholds_], 'y': [round(float(v), 4) for v in iso.y_thresholds_]},
            'before': metrics(ev[:, 0], ev[:, 1]), 'after': metrics(calibrated, ev[:, 1]), 'reliability_after': reliability(calibrated, ev[:, 1]),
            'per_class_test': per_class, 'note': 'Precision of a reported box (IoU >= .5 with a labelled box of the same class), not an image-level defect probability.'}


def section_patchcore():
    evaluation = json.loads((ROOT / 'models/patchcore/evaluation.json').read_text(encoding='utf-8'))
    rows = evaluation['scores']
    get = lambda s: (np.array([r['anomaly_score'] for r in rows if r['split'] == s]), np.array([r['label'] == 'defect' for r in rows if r['split'] == s], int))
    (xv, yv), (xt, yt) = get('validation'), get('test')
    a, b = fit_platt(xv, yv)
    prevalence = float(yv.mean())
    return {'method': 'Platt scaling on the raw anomaly score', 'fit_split': f'casting validation ({len(yv)} images)', 'eval_split': f'casting test ({len(yt)} images)',
            'params': {'a': a, 'b': b}, 'dataset_defect_prevalence': round(prevalence, 3),
            'after': metrics(platt(xt, a, b), yt), 'reliability_after': reliability(platt(xt, a, b), yt),
            'before': {'note': 'raw distances are not probabilities; no "before" ECE exists'},
            'note': 'P(defect | score) at the validation prevalence (83% defective, unlike production). Re-weight with the prior-shift formula for a real defect rate. Valid for the casting-impeller PatchCore bank only.'}


def section_rca():
    from scipy.optimize import minimize_scalar
    from xgboost import DMatrix
    from analytics import xgboost_rca as rca
    booster, metadata = rca._load(str(rca.MODEL_DIR))
    data = np.load(ROOT / 'models/xgboost/synthetic_history.npz')
    x, y, lots = data['features'], data['labels'], data['lot_ids']
    lo, hi = metadata['split']['test_lots']
    in_test = (lots >= lo) & (lots <= hi)
    fit_mask, eval_mask = in_test & (lots % 2 == 0), in_test & (lots % 2 == 1)   # parity split: lots 400-449 are unusually easy, so a half split misleads
    prob = lambda m: booster.predict(DMatrix(x[m], feature_names=list(rca.FEATURES)))
    pf, pe = prob(fit_mask), prob(eval_mask)
    yf, ye = y[fit_mask], y[eval_mask]

    def nll(t, p, yy):
        q = np.exp(np.log(np.clip(p, 1e-9, 1)) / t); q /= q.sum(axis=1, keepdims=True)
        return -np.mean(np.log(np.clip(q[np.arange(len(yy)), yy], 1e-9, 1)))
    temperature = float(minimize_scalar(lambda t: nll(t, pf, yf), bounds=(.3, 10), method='bounded').x)

    def scaled(p, t):
        q = np.exp(np.log(np.clip(p, 1e-9, 1)) / t); return q / q.sum(axis=1, keepdims=True)
    top = lambda p: (p.max(axis=1), (p.argmax(axis=1) == ye).astype(int))
    raw_c, raw_ok = top(pe); cal_c, cal_ok = top(scaled(pe, temperature))
    improves = (ece(cal_c, cal_ok) < ece(raw_c, raw_ok) - .005) and (nll(temperature, pe, ye) < nll(1.0, pe, ye))
    return {'method': 'temperature scaling of the XGBoost class probabilities (applied only if it helps on the held-out lots)', 'params': {'temperature': round(temperature, 4)},
            'apply_correction': bool(improves),
            'decision': 'temperature scaling applied' if improves else 'NO correction applied: the raw probability already matches observed accuracy on held-out lots',
            'fit_split': 'synthetic held-out even-numbered lots', 'eval_split': 'synthetic held-out odd-numbered lots',
            'accuracy_eval': round(float(raw_ok.mean()), 4), 'mean_confidence_before': round(float(raw_c.mean()), 4), 'mean_confidence_after': round(float(cal_c.mean()), 4),
            'before': {**metrics(raw_c, raw_ok), 'nll': round(float(nll(1.0, pe, ye)), 4)}, 'after': {**metrics(cal_c, cal_ok), 'nll': round(float(nll(temperature, pe, ye)), 4)},
            'reliability_after': reliability(cal_c, cal_ok),
            'note': 'Calibrated against the SYNTHETIC process simulator only. It does not say how often the cause is right on a real machine.'}


def section_forecast():
    import joblib
    from train_forecast import synthetic_lots
    artifact_path = ROOT / 'models/plsr/forecast.joblib'
    artifact = joblib.load(artifact_path)
    x, y = synthetic_lots(2026, 1800)
    train_end, cal_end = int(len(y) * .6), int(len(y) * .8)
    pred = np.clip(artifact['model'].predict(x[cal_end:]).ravel(), 0, 1)
    truth = y[cal_end:]
    residuals = np.asarray(artifact['residuals'], float)
    q_lo, q_hi = np.quantile(residuals, [.025, .975])

    def coverage(p, t, k):
        return float(np.mean((t >= np.clip(p + k * q_lo, 0, 1)) & (t <= np.clip(p + k * q_hi, 0, 1))))
    half = len(truth) // 2
    result = {'method': 'empirical coverage of the 95% simulated interval on chronologically later rows',
              'artifact_sha256': hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
              'residuals_from': 'calibration segment (rows 60-80%)', 'fit_split': f'test first half ({half} lots)', 'eval_split': f'test second half ({len(truth) - half} lots)',
              'coverage_before': {'in_sample_residual_segment': round(coverage(np.clip(artifact['model'].predict(x[train_end:cal_end]).ravel(), 0, 1), y[train_end:cal_end], 1.0), 3),
                                  'test_first_half': round(coverage(pred[:half], truth[:half], 1.0), 3), 'test_second_half': round(coverage(pred[half:], truth[half:], 1.0), 3)}}
    scale = 1.0
    if result['coverage_before']['test_first_half'] < .93:
        for k in np.arange(1.0, 3.01, .05):
            scale = float(round(k, 2))
            if coverage(pred[:half], truth[:half], k) >= .95:
                break
    result['interval_scale'] = scale
    result['coverage_after'] = {'test_first_half': round(coverage(pred[:half], truth[:half], scale), 3), 'test_second_half': round(coverage(pred[half:], truth[half:], scale), 3)}
    result['note'] = ('The interval already covers about 95% of later lots; no widening is applied.' if scale == 1.0 else
                      f'Residual quantiles are widened x{scale} so later lots are covered about 95% of the time. Synthetic data only.')
    return result


SECTIONS = {'corrosion': section_corrosion, 'yolo': section_yolo, 'patchcore': section_patchcore, 'rca': section_rca, 'forecast': section_forecast}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', choices=SECTIONS)
    args = parser.parse_args()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table = json.loads(OUT.read_text(encoding='utf-8')) if OUT.is_file() else {}
    for name, function in SECTIONS.items():
        if args.only and name != args.only:
            continue
        started = time.time()
        table[name] = function() | {'calibrated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
        OUT.write_text(json.dumps(table, indent=2), encoding='utf-8')
        print(f'[{name}] {time.time() - started:.0f}s', json.dumps({k: table[name].get(k) for k in ('before', 'after', 'coverage_before', 'coverage_after', 'interval_scale', 'params')}, indent=None)[:900], flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
