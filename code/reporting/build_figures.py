"""Read saved results and produce report figures/data/Markdown; never train."""
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'paper/data'
FIG = ROOT / 'paper/figures'
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'axes.unicode_minus': False, 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.titleweight': 'bold', 'savefig.facecolor': 'white'})
COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']
BASE = 'sampling_bias/lambda_1'
BASES = [(BASE, 'Synthetic 0+2000'), ('fresh10', 'Fresh 200+1800'), ('fixed_real', 'Fixed 2000+2000')]
config = json.loads((ROOT / 'settings/gao_unbalanced_true_gmm.json').read_text(encoding='utf-8'))
centers = np.array(config['means'])
with np.load(ROOT / 'outputs/datas/initial_samples_n2000_seed42.npz') as z:
    real = z['X']

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

@lru_cache(None)
def reports(condition):
    with ThreadPoolExecutor(max_workers=8) as pool:
        result = [list(pool.map(read, sorted((ROOT / 'outputs' / condition / f'seed_{seed}/metrics').glob('generation_*.json')))) for seed in range(5)]
    assert all((r[-1]['generation'] == 2000 for r in result)), condition
    print('Loaded', condition, flush=True)
    return result

def plot_line(ax, condition, key, label, color, style='-', group=None):
    runs = reports(condition)
    generations = [r['generation'] for r in runs[0]]
    if group:
        values = [[np.mean([c[key] for c in r['per_class'] if c['is_rare'] == (group == 'rare') and c[key] is not None]) for r in run] for run in runs]
    else:
        values = [[r['overall'][key] for r in run] for run in runs]
    values = np.asarray(values, float)
    ax.plot(generations, values.mean(axis=0), color=color, linestyle=style, label=label, linewidth=1.7)
    ax.fill_between(generations, values.min(axis=0), values.max(axis=0), color=color, alpha=0.09)
    ax.set_xlabel('Generation')
    ax.grid(alpha=0.18)
figures = []

def save(fig, number, slug, title, caption):
    filename = f'F{number:02d}_{slug}'
    fig.savefig(FIG / f'{filename}.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIG / f'{filename}.pdf', bbox_inches='tight')
    plt.close(fig)
    figures.append({'number': number, 'file': filename, 'title': title, 'caption': caption})
    print('Saved', filename, flush=True)
fig, axes = plt.subplots(1, 4, figsize=(9, 3.15), layout='constrained')
arrays = [real]
for generation in (1, 200, 2000):
    with np.load(ROOT / 'outputs' / BASE / f'seed_0/datas/generation_{generation:04d}.npz') as z:
        arrays.append(z['X'])
lo = np.vstack(arrays).min(axis=0) - 0.2
hi = np.vstack(arrays).max(axis=0) + 0.2
for ax, x, title in zip(axes, arrays, ('Reference', 'Generation 1', 'Generation 200', 'Generation 2000')):
    ax.scatter(x[:, 0], x[:, 1], s=4, alpha=0.55, color=COLORS[0], edgecolors='none', rasterized=True)
    rare = centers[config['rare_component_indices_zero_based']]
    ax.scatter(rare[:, 0], rare[:, 1], s=35, facecolors='none', edgecolors=COLORS[1], linewidths=1)
    ax.set(title=title, xlabel='x', ylabel='y', xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]))
    ax.set_aspect('equal')
save(fig, 1, 'collapse_snapshots', '', '')
fig, axes = plt.subplots(1, 3, figsize=(9, 3.35), layout='constrained')
for ax, key, title in zip(axes, ('total_variance', 'mean_component_variance', 'recall'), ('Total variance', 'Component variance (log)', 'Recall')):
    plot_line(ax, BASE, key, 'Fully synthetic', COLORS[0])
    ax.set_title(title)
axes[1].set_yscale('log')
axes[2].set_ylim(0, 1.03)
save(fig, 2, 'diversity_metrics', '', '')
fig, axes = plt.subplots(1, 3, figsize=(9, 3.65), layout='constrained')
for j, (condition, label) in enumerate(BASES):
    for ax, key in zip(axes, ('recall', 'rare_mean_recall', 'nll_initial')):
        plot_line(ax, condition, key, label, COLORS[j], ['-', '--', '-.'][j])
for ax, title in zip(axes, ('Overall recall', 'Rare-class recall', 'Reference NLL (log)')):
    ax.set_title(title)
axes[0].set_ylim(0, 1.03)
axes[1].set_ylim(0, 1.03)
axes[2].set_yscale('log')
axes[0].legend(fontsize=10, loc='center right')
save(fig, 3, 'training_loops', '', '')
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), layout='constrained')
for ax, condition, title in zip(axes, (BASE, 'fresh10'), ('Fully synthetic', '10% fresh real data')):
    for group, label, color, style in (('rare', 'Rare classes (macro)', COLORS[1], '--'), ('common', 'Common classes (macro)', COLORS[0], '-')):
        plot_line(ax, condition, 'recall', label, color, style, group=group)
    ax.set(title=title, ylim=(0, 1.03))
    ax.set_xscale('symlog', linthresh=10)
    ax.set_xlim(1, 2000)
    ax.legend(fontsize=10)
save(fig, 4, 'rare_common', '', '')
fig, axes = plt.subplots(1, 3, figsize=(9, 3.55), layout='constrained')
for j, value in enumerate((1, 0.8, 0.5)):
    for ax, key in zip(axes, ('recall', 'precision', 'mean_component_variance')):
        plot_line(ax, f'sampling_bias/lambda_{value:g}', key, f'λ={value:g}', COLORS[j], ['-', '--', '-.'][j])
axes[0].set(xlim=(1, 200), ylim=(0, 1.03), title='Early recall')
axes[0].set_xscale('symlog', linthresh=10)
axes[1].set(ylim=(0, 1.03), title='Precision')
axes[2].set_yscale('log')
axes[2].set_title('Component variance (log)')
axes[0].legend(fontsize=10)
save(fig, 5, 'sampling_bias', '', '')
fig, axes = plt.subplots(1, 3, figsize=(9, 3.55), layout='constrained')
for j, size in enumerate((500, 2000, 8000)):
    for ax, key in zip(axes, ('fid_2d', 'recall', 'rare_mean_recall')):
        plot_line(ax, f'initialization_memory/n_{size}', key, f'Initial n={size}', COLORS[j], ['-', '--', '-.'][j])
for ax, title in zip(axes, ('2D Frechet distance', 'Overall recall', 'Rare-class recall')):
    ax.set_title(title)
    ax.set_xscale('symlog', linthresh=10)
    ax.set_xlim(1, 2000)
axes[1].set_ylim(0.9, 1.0)
axes[2].set_ylim(0.45, 1.0)
axes[0].legend(fontsize=10)
save(fig, 6, 'initial_memory', '', '')
effective = read(ROOT / 'outputs/effective_sample_size/effective_size.json')
calibration = read(ROOT / 'outputs/effective_sample_size/calibration/summary.json')['rows']
nr = next((r for r in calibration if r['n_real'] == 200))
fig, axes = plt.subplots(1, 2, figsize=(9, 3.65), layout='constrained')
for j, value in enumerate((1.0, 0.8)):
    selected = sorted([r for r in effective['conditions'] if r['lambda'] == value], key=lambda r: r['n_synthetic'])
    x = np.array([r['n_synthetic'] for r in selected])
    y = np.array([r['mean_distance'] for r in selected])
    axes[0].plot(x, y, marker='o', color=COLORS[j], linestyle=['-', '--'][j], label=f'λ={value:g}')
    axes[0].fill_between(x, [r['min_seed_distance'] for r in selected], [r['max_seed_distance'] for r in selected], color=COLORS[j], alpha=0.12)
    means, low, high = ([], [], [])
    for ns in x:
        runs = reports(f'effective_sample_size/lambda_{value:g}/ns_{ns}')
        vals = [np.mean([r['overall']['precision'] for r in run if r['generation'] > 1500]) for run in runs]
        means.append(np.mean(vals))
        low.append(min(vals))
        high.append(max(vals))
    axes[1].plot(x, means, marker='o', color=COLORS[j], linestyle=['-', '--'][j], label=f'λ={value:g}')
    axes[1].fill_between(x, low, high, color=COLORS[j], alpha=0.12)
axes[0].axhline(nr['mean'], color='#555555', linestyle=':', label='Real-only baseline (n=200)')
axes[0].axhspan(nr['mean'] - nr['standard_error'], nr['mean'] + nr['standard_error'], color='gray', alpha=0.12)
axes[0].set(title='Tail sliced W2 distance', ylabel='Squared sliced W2', xlabel='Synthetic samples per generation')
axes[1].set(title='Tail precision', ylabel='Precision', xlabel='Synthetic samples per generation', ylim=(0, 1.03))
for ax in axes:
    ax.grid(alpha=0.18)
    ax.legend(fontsize=10)
save(fig, 7, 'synthetic_benefit', '', '')
fig, ax = plt.subplots(figsize=(7.5, 3.45), layout='constrained')
ax.errorbar([r['n_real'] for r in calibration], [r['mean'] for r in calibration], yerr=[r['standard_error'] for r in calibration], color=COLORS[0], marker='o', capsize=3, label='Real-only calibration (mean +/- SE)')
target = next((r for r in effective['conditions'] if r['lambda'] == 0.8 and r['n_synthetic'] == 1800))
ne = target['n_effective']
ax.axhline(target['mean_distance'], color=COLORS[1], linestyle='--', label='Recursive tail distance')
ax.vlines(ne, 0, target['mean_distance'], color=COLORS[1], linestyle=':')
ax.scatter([ne], [target['mean_distance']], color=COLORS[1], s=55, zorder=4)
ax.annotate(f'Effective size = {ne:.0f}\nFresh budget = 200', xy=(ne, target['mean_distance']), xytext=(750, 0.034), arrowprops={'arrowstyle': '->', 'color': COLORS[1]}, fontsize=10)
ax.set(xscale='log', xlabel='Real-only training sample size', ylabel='Squared sliced W2 distance', title='Distance-matched effective sample size', ylim=(0, 0.06))
ax.grid(alpha=0.18)
ax.legend(fontsize=9, loc='upper right')
save(fig, 8, 'effective_sample_size', 'Effective sample size', '')
audit = []
for path in sorted((ROOT / 'outputs').glob('**/summary/final.json')):
    if 'report' in path.parts:
        continue
    data = read(path)
    for r in data['trajectories']:
        audit.append({'condition': path.parent.parent.relative_to(ROOT / 'outputs').as_posix(), 'seed': r['seed'], 'generation': r['last_evaluated_generation'], 'nonconverged': len(r.get('nonconverged_generations', [])), 'failure': r.get('failure')})
assert len(audit) == 85 and all((r['generation'] == 2000 and (not r['failure']) for r in audit))
assert len(list((ROOT / 'outputs/effective_sample_size/calibration').glob('n_*_repeat_*.json'))) == 400
final_rows = []
for condition, label in BASES:
    runs = reports(condition)
    row = {'Condition': label}
    for key in ('fid_2d', 'precision', 'recall', 'rare_mean_recall', 'common_mean_recall', 'nll_initial'):
        row[key] = float(np.mean([run[-1]['overall'][key] for run in runs]))
    final_rows.append(row)
memory = []
for size in (500, 2000, 8000):
    runs = reports(f'initialization_memory/n_{size}')
    memory.append({'initial_n': size, 'initial_recall': float(np.mean([r[0]['overall']['recall'] for r in runs])), 'tail_recall': float(np.mean([np.mean([r['overall']['recall'] for r in run if r['generation'] > 1500]) for run in runs])), 'tail_rare_recall': float(np.mean([np.mean([r['overall']['rare_mean_recall'] for r in run if r['generation'] > 1500]) for run in runs]))})
bias = []
for value in (1.0, 0.8, 0.5):
    runs = reports(f'sampling_bias/lambda_{value:g}')
    bias.append({'lambda': value, 'first_below_half': [next((r['generation'] for r in run if r['overall']['recall'] < 0.5)) for run in runs], 'final_recall': float(np.mean([run[-1]['overall']['recall'] for run in runs])), 'final_precision': float(np.mean([run[-1]['overall']['precision'] for run in runs]))})
tail_metrics = []
for condition in effective["conditions"]:
    path = f"effective_sample_size/lambda_{condition['lambda']:g}/ns_{condition['n_synthetic']}"
    row = {"lambda": condition["lambda"], "n_synthetic": condition["n_synthetic"]}
    for key in ("precision", "recall", "rare_mean_recall", "nll_initial"):
        row[key] = float(np.mean([np.mean([r["overall"][key] for r in run if r["generation"] > 1500]) for run in reports(path)]))
    tail_metrics.append(row)
data = {"audit": audit, "final_comparison": final_rows, "memory": memory, "bias": bias,
        "effective_conditions": effective["conditions"], "tail_metrics": tail_metrics,
        "calibration": calibration, "primary_control": BASE}
(OUT / "results.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
print("English figures and numerical data ready", flush=True)
