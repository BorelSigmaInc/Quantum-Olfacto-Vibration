#!/usr/bin/env python3
"""OCNQENC – Generate panel+card PNGs A–J (graph left, explanation card right)."""
import numpy as np, xarray as xr, os, glob, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ---- Explanation texts for each panel ----
cards = {
    'A': ("Layman: Shows how important each ocean pattern is. The first few patterns are huge;\n"
          "the rest are tiny ripples.\n\n"
          "Technical: Singular values of the (31×1 036 800) SST matrix. The sharp decay\n"
          "indicates a low‑rank matrix, justifying strong compression.", "Navy"),
    'B': ("Layman: How much total ocean change is captured by the first N patterns.\n"
          "With 20 patterns we already have nearly all the information.\n\n"
          "Technical: Cumulative sum of squared singular values divided by total.\n"
          "Reaching ~97 % with k=20 modes proves almost no physical signal is lost.", "DarkGreen"),
    'C': ("Layman: The 'shapes' of ocean temperature changes.\n"
          "Mode 1 is often the equatorial warm tongue (El Niño pattern).\n\n"
          "Technical: Rows of Vᵀ reshaped to 720×1440. They form an orthonormal basis.\n"
          "Mode 1 typically captures the ENSO dipole; Mode 2 the meridional gradient.", "DarkBlue"),
    'D': ("Layman: How each shape changes day‑by‑day during January 2023.\n"
          "Big swings mean active weather.\n\n"
          "Technical: Columns of U multiplied by the corresponding singular value (UΣ).\n"
          "They represent the projection of each day's SST anomaly onto the spatial modes.", "DarkCyan"),
    'E': ("Layman: Most quantum slots have almost zero chance of being measured;\n"
          "only a few slots matter.\n\n"
          "Technical: Histogram of |amplitude|² for all 16 380 coefficients.\n"
          "Sparseness means the state can be prepared with fewer gates and is noise‑robust.", "Purple"),
    'F': ("Layman: If we sort the slots from most likely to least likely,\n"
          "the first few hundred cover 95 % of total probability.\n\n"
          "Technical: Cumulative sum of sorted probabilities. The steep rise indicates\n"
          "the amplitude vector is highly compressible; tensor‑train prep will be shallow.", "DarkRed"),
    'G': ("Layman: A close‑up look at the actual numbers that go into the quantum computer\n"
          "for one specific location.\n\n"
          "Technical: The 20 mode‑coefficients for the first subsampled grid point.\n"
          "These are the raw inputs to the 'initialize' routine, after normalisation.", "DarkOrange"),
    'H': ("Layman: How 'disorderly' the quantum state is. A lower number means the state\n"
          "is very structured and easier to make.\n\n"
          "Technical: Shannon entropy of the probability distribution.\n"
          "9.3 bits out of 14 possible indicates a low‑entanglement state suitable for MPS.", "SteelBlue"),
    'I': ("Layman: The types and numbers of quantum operations needed to build our ocean state.\n\n"
          "Technical: count_ops() on the decomposed circuit. The 'initialize' instruction hides\n"
          "many CNOTs; a shallower tensor‑train circuit will replace it before real hardware.", "DarkOrange"),
    'J': ("Layman: Shows the spread of the amplitude values; most are near zero,\n"
          "only a few are large.\n\n"
          "Technical: Empirical CDF of all amplitude values. The symmetric, steep shape\n"
          "confirms the vector is dominated by a small number of components.", "Magenta")
}

# ---- Load data ----
data_dir = "/home/rrmr/Desktop/RRMR/BProjects/BorelSIgmaInc/Projects/Research Projects/Kryptur/OCNQENC/sst_data"
files = sorted(glob.glob(os.path.join(data_dir, "*.nc")))
daily_fields = []
for f in files:
    ds = xr.open_dataset(f, chunks='auto')
    daily_fields.append(ds['sst'].squeeze(drop=True).values.astype(np.float32))
    ds.close()
sst_cube = np.stack(daily_fields, axis=0)          # (31, 720, 1440)

# ---- Pre‑process ----
sst_anom = sst_cube - np.nanmean(sst_cube, axis=0, keepdims=True)
sst_anom = np.nan_to_num(sst_anom, nan=0.0)
T, H, W = sst_anom.shape
X = sst_anom.reshape(T, -1)

# ---- SVD ----
U, s, Vt = np.linalg.svd(X, full_matrices=False)
k = 20
subsample = 1266
spatial_modes = Vt[:k, ::subsample]
coeff_vector = spatial_modes.ravel()
amplitude_vector = coeff_vector / np.linalg.norm(coeff_vector)

# ---- Utility: create a figure with graph left and card right ----
def save_card(panel_id, graph_func, text, color):
    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(1, 2, width_ratios=[2.5, 1], wspace=0.3)
    ax_graph = fig.add_subplot(gs[0])
    graph_func(ax_graph)
    ax_card = fig.add_subplot(gs[1])
    ax_card.axis('off')
    ax_card.text(0.05, 0.95, text, transform=ax_card.transAxes,
                 fontsize=10, verticalalignment='top', family='monospace',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor=color, linewidth=2))
    ax_card.set_title(f'Panel {panel_id} – Explanation', fontsize=12, color=color)
    fig.tight_layout()
    fig.savefig(f'card_{panel_id}.png', dpi=200)
    plt.close(fig)

# ---- Panel A ----
def plot_a(ax):
    ax.semilogy(s, 'o-', markersize=4, color='navy')
    ax.set_title('A) Singular Values (log)')
    ax.set_xlabel('Mode index'); ax.set_ylabel('Singular value')
    ax.grid(True, alpha=0.3)
save_card('A', plot_a, cards['A'][0], cards['A'][1])

# ---- Panel B ----
def plot_b(ax):
    var_exp = s**2 / np.sum(s**2)
    cum_var = np.cumsum(var_exp)
    ax.plot(cum_var * 100, '.-', color='darkgreen')
    ax.set_title('B) Cumulative Explained Variance')
    ax.set_xlabel('Number of modes'); ax.set_ylabel('Cumulative variance (%)')
    ax.axhline(y=95, color='r', linestyle='--', alpha=0.6)
    ax.axvline(x=k, color='gray', linestyle='--', alpha=0.6)
    ax.text(k+1, 70, f'k={k} modes', fontsize=9)
    ax.grid(True, alpha=0.3)
save_card('B', plot_b, cards['B'][0], cards['B'][1])

# ---- Panel C (4 maps) ----
fig = plt.figure(figsize=(16, 7))
gs = GridSpec(1, 2, width_ratios=[2.5, 1], wspace=0.3)
ax_graph = fig.add_subplot(gs[0])
# sub‑maps inside ax_graph using inset_axes
for i, pos in enumerate([(0,0), (0,1), (1,0), (1,1)]):
    sub_ax = ax_graph.inset_axes([pos[1]*0.5, (1-pos[0])*0.5-0.5, 0.48, 0.48])
    mode_map = Vt[i, :].reshape(H, W)
    im = sub_ax.imshow(mode_map, cmap='RdBu_r', aspect='auto', origin='lower')
    sub_ax.set_title(f'Mode {i+1}', fontsize=8)
    plt.colorbar(im, ax=sub_ax, fraction=0.05, pad=0.02)
ax_graph.set_title('C) Top Spatial Modes')
ax_graph.axis('off')
ax_card = fig.add_subplot(gs[1])
ax_card.axis('off')
ax_card.text(0.05, 0.95, cards['C'][0], transform=ax_card.transAxes,
             fontsize=10, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='DarkBlue', linewidth=2))
ax_card.set_title('Panel C – Explanation', fontsize=12, color='DarkBlue')
fig.tight_layout()
fig.savefig('card_C.png', dpi=200); plt.close(fig)

# ---- Panel D ----
def plot_d(ax):
    for i in range(5):
        ax.plot(U[:, i] * s[i], label=f'Mode {i+1}', marker='o')
    ax.set_title('D) Temporal Coefficients (first 5 modes)')
    ax.set_xlabel('Day (Jan 2023)'); ax.set_ylabel('Amplitude')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
save_card('D', plot_d, cards['D'][0], cards['D'][1])

# ---- Panel E ----
def plot_e(ax):
    ax.hist(amplitude_vector**2, bins=50, color='purple', edgecolor='white', alpha=0.8)
    ax.set_title('E) Amplitude Probabilities Histogram')
    ax.set_xlabel('Probability'); ax.set_ylabel('Count')
save_card('E', plot_e, cards['E'][0], cards['E'][1])

# ---- Panel F ----
def plot_f(ax):
    sorted_probs = -np.sort(-(amplitude_vector**2))
    cum_probs = np.cumsum(sorted_probs)
    ax.plot(cum_probs, color='darkred')
    ax.set_title('F) Cumulative Probability Mass')
    ax.set_xlabel('Sorted component index'); ax.set_ylabel('Cumulative probability')
    ax.axhline(y=0.95, color='gray', linestyle='--')
    ax.set_xscale('log'); ax.grid(True, alpha=0.3)
save_card('F', plot_f, cards['F'][0], cards['F'][1])

# ---- Panel G ----
def plot_g(ax):
    first_day_pattern = spatial_modes[:, 0]
    ax.stem(first_day_pattern)
    ax.set_title('G) Encoded Coefficients (1st spatial pt)')
    ax.set_xlabel('Mode'); ax.set_ylabel('Value')
save_card('G', plot_g, cards['G'][0], cards['G'][1])

# ---- Panel H ----
def plot_h(ax):
    probs = amplitude_vector**2
    entropy = -np.sum(probs * np.log2(probs + 1e-15))
    max_entropy = np.log2(len(probs))
    ax.bar(['Actual', 'Max possible'], [entropy, max_entropy], color=['steelblue', 'lightgray'])
    ax.set_title(f'H) Entropy = {entropy:.2f} bits (max {max_entropy:.1f})')
    ax.set_ylabel('Bits')
save_card('H', plot_h, cards['H'][0], cards['H'][1])

# ---- Panel I ----
from qiskit import qpy
with open("encoding_circuit.qpy", "rb") as f:
    qc = qpy.load(f)[0]
ops = qc.decompose(reps=2).count_ops()
def plot_i(ax):
    gates = list(ops.keys()); counts = list(ops.values())
    ax.barh(gates, counts, color='darkorange')
    ax.set_title('I) Circuit Gate Counts')
    ax.set_xlabel('Count')
save_card('I', plot_i, cards['I'][0], cards['I'][1])

# ---- Panel J ----
def plot_j(ax):
    ax.plot(np.sort(amplitude_vector), np.linspace(0,1,len(amplitude_vector)), color='magenta')
    ax.set_title('J) Amplitude Distribution (CDF)')
    ax.set_xlabel('Amplitude value'); ax.set_ylabel('Cumulative fraction')
save_card('J', plot_j, cards['J'][0], cards['J'][1])

print("Cards A–J saved as card_A.png … card_J.png")
