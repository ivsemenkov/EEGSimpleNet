from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import mne
import numpy as np


def plot_topographies(
        spatial_patterns: np.ndarray, eeg_channels: list[str], montage_name: str = 'standard_1020', 
        save_path: Optional[str] = None
    ) -> plt.Figure:
    montage = mne.channels.make_standard_montage(montage_name)
    info = mne.create_info(eeg_channels, 1, ch_types='eeg')
    info.set_montage(montage)
    spatial_patterns = spatial_patterns.T

    evoked = mne.EvokedArray(
        data=spatial_patterns,
        info=info
    )

    n_patterns = spatial_patterns.shape[1]
    n_cols = min(10, n_patterns)
    n_rows = n_patterns // n_cols
    if (n_patterns % n_cols) != 0:
        n_rows += 1

    fig = evoked.plot_topomap(
        times=np.arange(n_patterns) / evoked.info['sfreq'], ch_type='eeg', units='', scalings={'eeg': 1},
        time_format='', ncols=n_cols, nrows=n_rows, show=False
    )

    axes = fig.axes
    for idx in range(n_patterns):
        axes[idx].set_xlabel(f'Branch: {idx + 1}')

    fig.suptitle('Spatial patterns')

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
    
    return fig


def plot_temporal_patterns(
        fft_freqs: np.ndarray, temporal_patterns: np.ndarray, save_path: Optional[str] = None
    ) -> plt.Figure:
    n_patterns = temporal_patterns.shape[0]
    n_cols = min(3, n_patterns)
    n_rows = n_patterns // n_cols
    if (n_patterns % n_cols) != 0:
        n_rows += 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10), sharex=True, sharey=True)

    for pattern_idx in range(n_patterns):
        row = pattern_idx // n_cols
        col = pattern_idx % n_cols

        if n_rows > 1:
            ax = axes[row, col]
        else:
            ax = axes[col]

        ax.plot(fft_freqs, temporal_patterns[pattern_idx])
        ax.set_title(f'Branch: {pattern_idx + 1}')
        if row == (n_rows - 1):
            ax.set_xlabel('Frequency (Hz)')
    
    fig.suptitle('Temporal patterns')
    
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
    
    return fig
