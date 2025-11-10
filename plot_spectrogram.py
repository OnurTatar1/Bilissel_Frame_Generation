import numpy as np
import matplotlib.pyplot as plt
import os
import time
from scipy.signal import stft, get_window, windows
from scipy.signal import hilbert
from scipy.signal import stft
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert, resample_poly
def plot_spectrogram_stft(signal: np.ndarray,
                        fs: float,
                        nperseg: int = 256,
                        noverlap: int = 128,
                        window: str = "hann",
                        db_clip: tuple[float, float] | None = (-100, 0),
                        cmap: str = "viridis",
                        figsize: tuple[int, int] = (10, 4),
                        title: str | None = None):
    if noverlap is None or noverlap >= nperseg:
        noverlap = nperseg // 2 if nperseg > 1 else 0
    t = np.arange(signal.size) / fs
    #signal = signal * np.exp(1j * 2 * np.pi * (5e5) * t)
    f, t_windows, Z = stft(
        signal,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        return_onesided=False,
        boundary="even",
        padded=False,
    )
    
    # Center the frequency axis around 0 Hz (like fftshift in FFT)
    Z = np.fft.fftshift(Z, axes=0)
    f = np.fft.fftshift(f)
    
    one_sided_mask = f >= 0
    Z = Z[one_sided_mask, :]
    f = f[one_sided_mask]


    Pxx = 20*np.log10(np.abs(Z) + 1e-12)
    if db_clip is not None:
        Pxx = np.clip(Pxx, *db_clip)
    plt.figure(figsize=figsize)
    T, F = np.meshgrid(t_windows, f)
    cmap_obj = plt.get_cmap(cmap, 16)
    plt.pcolormesh(T*1e3, F/1e6, Pxx, cmap=cmap_obj, antialiased=False, linewidth=0)
    plt.colorbar(label="Magnitude (dB)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Frequency (MHz)")
    plt.title(title or "STFT Spectrogram")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_spectrogram_stft_with_rectangles(signal: np.ndarray,
                        fs: float,
                        train_info: list,
                        nperseg: int = 256,
                        noverlap: int = 128,
                        window: str = "hann",
                        db_clip: tuple[float, float] | None = (-100, 0),
                        cmap: str = "viridis",
                        figsize: tuple[int, int] = (8, 4),
                        title: str | None = None,
                        save_or_plot: bool = False,
                        frame_index: int = None,
                        snr_db: int = None):
    if noverlap is None or noverlap >= nperseg:
        noverlap = nperseg // 2 if nperseg > 1 else 0
    t = np.arange(signal.size) / fs
    #signal = signal * np.exp(1j * 2 * np.pi * (5e5) * t)
    f, t_windows, Z = stft(
        signal,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        return_onesided=False,
        boundary="even",
        padded=False,
    )
    
    # Center the frequency axis around 0 Hz (like fftshift in FFT)
    Z = np.fft.fftshift(Z, axes=0)
    f = np.fft.fftshift(f)
    
    one_sided_mask = f >= 0
    Z = Z[one_sided_mask, :]
    f = f[one_sided_mask]

    Pxx = 20*np.log10(np.abs(Z) + 1e-12)
    if db_clip is not None:
        Pxx = np.clip(Pxx, *db_clip)
    plt.figure(figsize=figsize)
    T, F = np.meshgrid(t_windows, f)
    cmap_obj = plt.get_cmap(cmap, 16)
    plt.pcolormesh(T*1e3, F/1e6, Pxx, cmap=cmap_obj, antialiased=False, linewidth=0)

    if save_or_plot:
        # Save a clean spectrogram only (no axes, labels, colorbar, rectangles, or text)
        ax = plt.gca()
        ax.set_axis_off()
        plt.margins(0)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        os.makedirs('yolo_training_data', exist_ok=True)
        filename = os.path.join('yolo_training_data', f"spectrogram_{snr_db}_{frame_index}.png")
        plt.savefig(filename, dpi=200, bbox_inches='tight', pad_inches=0)
        plt.close()
        return

    # Interactive plot with full decorations and rectangles
    plt.colorbar(label="Magnitude (dB)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Frequency (MHz)")
    plt.title(title or "STFT Spectrogram with Train Rectangles")
    plt.grid(True)

    # Add rectangles for each train
    frame_duration_s = signal.size / fs

    for i, train in enumerate(train_info):
        wave_type = train['type']
        start_time_ratio = train['start_index_ratio']
        length_time_ratio = train['waveform_length_ratio']
        start_freq_ratio = train['fc_ratio']
        bandwidth_ratio = train['band_width_ratio']

        start_time_ms = start_time_ratio * frame_duration_s * 1e3
        length_time_ms = length_time_ratio * frame_duration_s * 1e3
        length_freq_mhz = bandwidth_ratio * (fs/2) / 1e6
        start_freq_mhz = start_freq_ratio * (fs/2) / 1e6

        if length_freq_mhz == 0:
            plt.hlines(y=start_freq_mhz,
                       xmin=start_time_ms,
                       xmax=start_time_ms + length_time_ms,
                       colors='red', linewidth=2, alpha=0.8)
        else:
            from matplotlib.patches import Rectangle
            rect = Rectangle((start_time_ms, start_freq_mhz),
                             width=length_time_ms, height=length_freq_mhz,
                             linewidth=2, edgecolor='red', facecolor='none', alpha=0.8)
            plt.gca().add_patch(rect)

        label = train.get('type', f'Train {i+1}')
        plt.text(start_time_ms + max(length_time_ms/2, 1e-3), start_freq_mhz,
                 label, ha='center', va='center', color='red', fontweight='bold', fontsize=8)

    plt.tight_layout()
    plt.show()




def to_baseband(x: np.ndarray, fc: float, fs: float) -> np.ndarray:

    t = np.arange(x.size) / fs
    analytic_signal = hilbert(x)
    baseband_signal = analytic_signal * np.exp(-1j * 2 * np.pi * fc * t)
    return np.asarray(baseband_signal)


def plot_spwvd(x: np.ndarray,
               fs: float,
               frame_length: int | None = None,
               time_smooth_win: int = 11,
               freq_smooth_win: int = 11,
               cmap: str = "viridis",
               db_min: float = -40,
               db_max: float = 0,
               figsize: tuple[int, int] = (10, 6),
               title: str = "Smooth Pseudo-Wigner-Ville Distribution"):
    """
    Generate and plot Smooth Pseudo-Wigner-Ville Distribution (SPWVD).
    
    Parameters
    ----------
    x : np.ndarray
        Input signal (1D real or complex array)
    fs : float
        Sampling frequency in Hz
    frame_length : int, optional
        Number of samples to process. If None, uses entire signal
    time_smooth_win : int, default 11
        Time smoothing window length (should be odd)
    freq_smooth_win : int, default 11
        Frequency smoothing window length (should be odd)
    cmap : str, default "viridis"
        Colormap for the plot
    figsize : tuple, default (10, 6)
        Figure size (width, height) in inches
    title : str, default "Smooth Pseudo-Wigner-Ville Distribution"
        Plot title
    """
    
    # Frame the signal if specified
    if frame_length is not None:
        x = x[:frame_length]
    
    # Get analytic signal (remove negative frequencies)
    x_real = np.real(x).astype(np.float64)
    z = hilbert(x_real)
    N = len(z)
    
    # Initialize WVD matrix
    wvd = np.zeros((N, 2*N-1), dtype=np.complex128)
    
    # Fill WVD matrix
    for n in range(N):
        taumax = min(n, N-1-n)
        for tau in range(-taumax, taumax + 1):
            wvd[n, tau + N - 1] = z[n + tau] * np.conj(z[n - tau])
    
    # Apply FFT along lag axis to get frequency domain
    wvd_freq = np.fft.fftshift(np.fft.fft(wvd, axis=1), axes=1)
    
    # Apply time smoothing
    if time_smooth_win > 1:
        ht = windows.hamming(time_smooth_win)
        ht = ht / ht.sum()
        wvd_freq = np.apply_along_axis(
            lambda m: np.convolve(m, ht, mode='same'), axis=0, arr=wvd_freq)
    
    # Apply frequency smoothing
    if freq_smooth_win > 1:
        hf = windows.hamming(freq_smooth_win)
        hf = hf / hf.sum()
        wvd_freq = np.apply_along_axis(
            lambda m: np.convolve(m, hf, mode='same'), axis=1, arr=wvd_freq)
    
    # Convert to dB scale
    P = 20 * np.log10(np.abs(wvd_freq) + 1e-12)
    
    # Apply dB clipping
    P = np.clip(P, db_min, db_max)
    
    # Set up frequency and time axes
    freqs = np.fft.fftshift(np.fft.fftfreq(2*N-1, d=1/fs))
    times = np.arange(N) / fs * 1e3  # Convert to milliseconds
    
    # Keep only positive frequencies
    pos_mask = freqs >= 0
    P = P[:, pos_mask]
    freqs = freqs[pos_mask]
    
    # Create the plot
    plt.figure(figsize=figsize)
    extent = (times[0], times[-1], freqs[0]/1e6, freqs[-1]/1e6)
    plt.imshow(P.T, origin="lower", extent=extent, aspect="auto", cmap=cmap)
    plt.colorbar(label="Magnitude (dB)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Frequency (MHz)")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_spwvd_v2(frame, fs, time_win_len=201, freq_win_len=101, db_min=-80, db_max=0,
               plot_type='pcolormesh', cmap='coolwarm', pad_factor=0.5):
    """
    Compute and plot the Smoothed Pseudo-Wigner–Ville distribution (SPWVD) of a real signal frame.
    
    Parameters:
    -----------
    frame : 1D numpy array
        The real-valued signal frame (samples).
    fs : float
        Sampling frequency (Hz).
    time_win_len : int
        Length of time-smoothing window (odd integer).
    freq_win_len : int
        Length of frequency-smoothing window (odd integer).
    db_min : float
        Minimum dB level for display (e.g. -80).
    db_max : float
        Maximum dB level for display (e.g. 0).
    plot_type : {'pcolormesh', 'imshow'}
        How to display the time-frequency matrix.
    cmap : str
        Colormap for plotting (e.g. 'coolwarm').
    pad_factor : float
        Fraction of time_win_len for padding on each side (default 0.5).
    
    Returns:
    --------
    None. Displays a time-frequency plot of SPWVD.
    """
    # Ensure odd window lengths
    if time_win_len % 2 == 0:
        time_win_len += 1
    if freq_win_len % 2 == 0:
        freq_win_len += 1

    # Pad signal to avoid edge artifacts
    pad = int(pad_factor * time_win_len)
    sig = np.pad(frame, pad, mode='constant', constant_values=0)
    N = len(sig)
    
    # Analytic signal (Hilbert transform of detrended signal)
    sig_analytic = hilbert(sig)
    
    # Create normalized smoothing windows (Hamming here; Gaussian also possible)
    g = windows.hamming(time_win_len)
    g = g / np.sum(g)  # time-smoothing window
    h = windows.hamming(freq_win_len)
    h = h / np.sum(h)  # frequency-smoothing envelope
    
    Lg = (time_win_len - 1) // 2
    Lh = (freq_win_len - 1) // 2
    
    # Time vector for plotting (only the original, unpadded frame)
    t_original = np.arange(len(frame)) / fs
    # Frequency vector (positive freqs up to Nyquist)
    nfft = 2**(int(np.ceil(np.log2(freq_win_len + 1))) + 1)  # at least > freq_win_len
    nfft = max(nfft, len(sig))  # ensure FFT covers signal length
    freqs = np.linspace(0, fs/2, nfft//2, endpoint=False)
    
    # Prepare empty TF matrix (nfft/2 freq bins x N time samples)
    TF = np.zeros((nfft//2, N), dtype=float)
    
    # Compute the pseudo-Wigner distribution with smoothing
    for ti in range(Lg, N - Lg):
        # Windowed autocorrelation sum for each delay m
        # R(τ) = g-weighted product at time ti
        R = np.zeros(nfft, dtype=complex)
        # For τ = 0
        R[0] = g[Lg] * (sig_analytic[ti] * np.conj(sig_analytic[ti]))
        # For τ = 1..Lg (positive delays)
        for m in range(1, Lg+1):
            R[m] = g[Lg + m] * (sig_analytic[ti+m] * np.conj(sig_analytic[ti-m]))
            R[-m] = g[Lg + m] * (sig_analytic[ti-m] * np.conj(sig_analytic[ti+m]))
        # Apply frequency-domain window h (before FFT)
        # Here we multiply R by shifted h
        H = np.zeros(nfft)
        H[:Lh+1] = h[Lh:]       # first Lh+1 points
        H[-Lh:]    = h[:Lh]      # last Lh points
        R = R * H
        # Fourier transform over τ to get spectrum at time ti
        S = np.fft.fft(R, nfft)
        # Take only positive freqs and magnitude
        TF[:, ti] = np.abs(S[:nfft//2])
    
    # Remove the padding columns in time
    TF = TF[:, pad:pad+len(frame)]
    
    # Convert to dB
    TF_db = 10 * np.log10(TF + 1e-12)
    TF_db = np.clip(TF_db, db_min, db_max)
    
    # Plot
    T, F = np.meshgrid(t_original, freqs)
    plt.figure(figsize=(8, 4))
    if plot_type == 'pcolormesh':
        plt.pcolormesh(T, F, TF_db, shading='auto', cmap=cmap, vmin=db_min, vmax=db_max)
    else:
        # imshow with extent to align axes
        extent = [t_original[0], t_original[-1], freqs[0], freqs[-1]]
        plt.imshow(TF_db[::-1,:], extent=extent, aspect='auto', cmap=cmap,
                   vmin=db_min, vmax=db_max)
    plt.colorbar(label='Intensity [dB]')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.title('Smoothed Pseudo Wigner–Ville Distribution')
    plt.tight_layout()
    plt.show()


def plot_choi_williams(signal: np.ndarray,
                      fs: float,
                      sigma: float = 1.0,
                      nfft: int = 256,
                      hop: int = 4,
                      max_lag: int = 64,
                      db_clip: tuple[float, float] | None = (-100, 0),
                      cmap: str = "gray",
                      figsize: tuple[int, int] = (10, 4),
                      title: str | None = None):
    """
    Plot Choi-Williams distribution of a signal (completely rewritten implementation).
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal (1D real or complex array)
    fs : float
        Sampling frequency in Hz
    sigma : float, default 1.0
        Choi-Williams kernel parameter (controls cross-term suppression)
    nfft : int, default 256
        FFT size for frequency resolution
    hop : int, default 4
        Time step between successive windows (larger = faster)
    max_lag : int, default 64
        Maximum lag for autocorrelation (smaller = faster)
    db_clip : tuple[float, float] | None, default (-100, 0)
        dB range for clipping the output
    cmap : str, default "gray"
        Colormap for the plot
    figsize : tuple[int, int], default (10, 4)
        Figure size (width, height) in inches
    title : str | None, default None
        Plot title
    """
    
    # Ensure signal is complex (analytic signal)
    if np.isrealobj(signal):
        signal = hilbert(signal)
    
    N = len(signal)
    
    # Time and frequency vectors
    t_indices = np.arange(0, N, hop)
    t = t_indices / fs
    freqs = np.fft.fftfreq(nfft, d=1/fs)
    freqs = freqs[:nfft//2]  # Only positive frequencies
    
    # Initialize Choi-Williams distribution matrix
    cwd = np.zeros((nfft//2, len(t_indices)), dtype=np.float64)
    
    # Compute Choi-Williams distribution
    for i, n in enumerate(t_indices):
        # Time window around current sample
        tau_max = min(n, N-1-n, max_lag)
        
        # Initialize autocorrelation function - use 2*max_lag+1 for proper size
        R = np.zeros(2*max_lag+1, dtype=np.complex128)
        
        # Compute autocorrelation with Choi-Williams kernel
        for tau in range(-tau_max, tau_max + 1):
            if n + tau >= 0 and n - tau >= 0 and n + tau < N and n - tau < N:
                # Choi-Williams kernel: exp(-(tau^2) / sigma)
                kernel_val = np.exp(-(tau**2) / sigma)
                
                # Compute autocorrelation: x(t+tau) * x*(t-tau)
                autocorr_val = signal[n + tau] * np.conj(signal[n - tau])
                
                # Store in proper position (tau + tau_max to make it non-negative)
                R[tau + tau_max] = kernel_val * autocorr_val
        
        # Zero-pad to nfft size for FFT
        R_padded = np.zeros(nfft, dtype=np.complex128)
        R_padded[:len(R)] = R
        
        # FFT to get frequency domain
        S = np.fft.fft(R_padded, nfft)
        
        # Take only positive frequencies and magnitude
        cwd[:, i] = np.abs(S[:nfft//2])
    
    # Convert to dB scale
    cwd_db = 20 * np.log10(cwd + 1e-12)
    
    # Apply dB clipping
    if db_clip is not None:
        cwd_db = np.clip(cwd_db, *db_clip)
    
    # Create the plot
    plt.figure(figsize=figsize)
    T, F = np.meshgrid(t, freqs)
    cmap_obj = plt.get_cmap(cmap, 16)
    plt.pcolormesh(T*1e3, F/1e6, cwd_db, cmap=cmap_obj, antialiased=False, linewidth=0)
    plt.colorbar(label="Magnitude (dB)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Frequency (MHz)")
    plt.title(title or "Choi-Williams Distribution")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def plot_spectrogram_cwd(signal: np.ndarray, fs: float, 
                         sigma: float = 1.0, 
                         n_time: int | None = None, 
                         n_freq: int | None = 512, 
                         db_clip: tuple[float, float] | None = (-100, 0), 
                         cmap: str = "viridis", figsize: tuple[int, int] = (10, 4), 
                         title: str | None = None):
    """
    Plot the Choi-Williams distribution spectrogram of a signal.
    
    Parameters:
      signal : np.ndarray 
          Input signal (real or complex) for one frame.
      fs : float 
          Sampling frequency of the signal.
      sigma : float 
          Kernel parameter controlling the trade-off between resolution and cross-term suppression.
      n_time : int or None 
          Number of time points to evaluate (reduces resolution for long signals). 
          If None, use the full signal length.
      n_freq : int or None 
          Number of frequency bins for the spectrogram. If None, defaults to length of signal.
      db_clip : tuple(float, float) or None 
          Clipping range for dB values (min, max). Set to None for no clipping.
      cmap : str 
          Colormap for the spectrogram plot.
      figsize : tuple(int, int) 
          Figure size for the plot.
      title : str or None 
          Title of the plot.
    """
    N = len(signal)
    # Ensure we use an analytic signal to avoid negative-frequency mirror (for real inputs)
    if np.isrealobj(signal):
        signal = hilbert(signal)  # convert to complex analytic signal

    # Set default time and frequency resolution if not specified
    if n_time is None or n_time > N:
        n_time = N
    if n_freq is None:
        n_freq = N
    # Determine time indices at which to compute the distribution (evenly spaced)
    time_indices = np.round(np.linspace(0, N-1, n_time)).astype(int)
    n_time = len(time_indices)  # update in case rounding adjusted it

    # Prepare output time and frequency axes
    t_vals = time_indices / fs                        # time in seconds for each slice
    freq_bins = n_freq
    # Frequency axis from -fs/2 to +fs/2, we'll later take the positive half (0 to fs/2)
    f_vals = np.fft.fftfreq(freq_bins, d=1/fs)        # frequency bins (unshifted)
    f_vals = np.fft.fftshift(f_vals)                  # shift zero-freq to center
    # Compute half of frequency axis (one-sided) after shift
    one_sided_mask = f_vals >= 0
    f_vals = f_vals[one_sided_mask]

    # Initialize spectrogram matrix
    Pxx = np.zeros((f_vals.size, n_time), dtype=float)

    # Precompute a Gaussian kernel in the **ambiguity domain** for given sigma:
    # The kernel weight for a given time-lag tau and frequency-difference eta is exp(-sigma * tau^2 * eta^2).
    # We implement this by weighting correlation at each lag tau by exp(-sigma * tau^2) (approximate, see note below).
    exp_tau = None
    if sigma is not None:
        # We will limit correlation lags to L = freq_bins//2 for efficiency (h_tau window)
        L = freq_bins // 2
        # Prepare Gaussian weights for tau from -L to L (symmetrical)
        tau_range = np.arange(-L, L+1)
        exp_tau = np.exp(-sigma * (tau_range ** 2))
    else:
        exp_tau = None

    # Loop over selected time indices and compute CWD via Fourier transform of the instantaneous autocorrelation
    for ti, n in enumerate(time_indices):
        # Determine maximum lag such that indices are valid
        max_tau = min(n, N-1-n, freq_bins//2)  # restrict to half the freq_bins length or signal edges
        if max_tau < 0:
            continue  # skip if no valid tau (edge case)

        # Compute the instantaneous autocorrelation for this time index:
        # R(tau) = x[n+tau] * conj(x[n-tau]) for tau = -max_tau,...,+max_tau.
        tau_vals = np.arange(-max_tau, max_tau+1)
        x1 = signal[n + tau_vals]       # values for positive tau (and negative, via negative indices)
        x2 = signal[n - tau_vals]       # values for negative tau mirrored
        R_tau = x1 * np.conjugate(x2)   # instantaneous autocorrelation sequence

        # Apply the Choi-Williams kernel weight in tau (approximate non-separable kernel)
        if exp_tau is not None:
            # If our R_tau is shorter than exp_tau (which is length 2L+1), extract matching segment
            L_current = max_tau
            # Indices in precomputed exp_tau corresponding to tau range -L_current..+L_current
            offset = (len(exp_tau) - 1) // 2  # index in exp_tau corresponding to tau=0
            idx_start = offset - L_current
            idx_end   = offset + L_current
            # Apply Gaussian weighting for current tau span
            R_tau = R_tau * exp_tau[idx_start: idx_end+1]

        # Zero-pad or truncate R_tau to length n_freq for FFT
        # Center R_tau at zero-lag for correct FFT (tau=0 at center of array)
        M = len(R_tau)
        # If needed, pad R_tau to match n_freq length
        if M < n_freq:
            pad_left = (n_freq - M) // 2
            pad_right = n_freq - M - pad_left
            R_tau_padded = np.pad(R_tau, (pad_left, pad_right), mode='constant')
        elif M > n_freq:
            # If correlation length is longer than n_freq, truncate equally from both ends
            extra = M - n_freq
            cut_left = extra // 2
            R_tau_padded = R_tau[cut_left: cut_left + n_freq]
        else:
            R_tau_padded = R_tau

        # Compute Fourier transform of R_tau to get the distribution at this time (frequency domain)
        S_f = np.fft.fftshift(np.fft.fft(R_tau_padded))
        # Take magnitude (power) in dB
        Pxx_col = 20 * np.log10(np.abs(S_f) + 1e-12)
        # Keep only one-sided (positive frequencies)
        Pxx[:, ti] = Pxx_col[one_sided_mask]

    # Optionally clip dB range for better contrast
    if db_clip is not None:
        Pxx = np.clip(Pxx, db_clip[0], db_clip[1])

    # Plot the spectrogram
    plt.figure(figsize=figsize)
    T, F = np.meshgrid(t_vals, f_vals)
    plt.pcolormesh(T * 1e3, F / 1e6, Pxx, cmap=cmap, shading='auto')
    plt.colorbar(label="Magnitude (dB)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Frequency (MHz)")
    plt.title(title or "Choi-Williams Distribution Spectrogram")
    plt.tight_layout()
    plt.show()


# def plot_spectrogram_cwd(signal: np.ndarray,
#                          fs: float,
#                          decim: int = 50,
#                          sigma: float = 0.05,
#                          max_lag: int | None = None,
#                          nfreq: int | None = None,
#                          db_clip: tuple[float, float] | None = (-60, 10),
#                          cmap: str = "viridis",
#                          figsize: tuple[int, int] = (8, 4),
#                          title: str | None = None):
#     """
#     Choi–Williams distribution (CWD) spectrogram.

#     Parameters
#     ----------
#     signal : array_like
#         1-D real or complex frame.
#     fs : float
#         Sampling rate [Hz].
#     decim : int
#         Integer decimation factor to reduce complexity. Increase if N is large.
#     sigma : float
#         Choi–Williams kernel parameter (>0). Smaller => more WVD-like (sharper, more cross-terms).
#         Larger => more smoothed (cleaner, less resolution). Typical 0.02–0.2.
#     max_lag : int | None
#         Max |lag| (samples) used in the instantaneous auto-correlation. Default: N//4.
#     nfreq : int | None
#         Number of frequency bins for the final FFT over lag. Default: next pow2 of (2*max_lag+1).
#     db_clip : (min_dB, max_dB) or None
#         Dynamic range clipping for display.
#     """
#     # 0) Preprocess and decimate for feasibility
#     x = np.asarray(signal)
#     if np.isrealobj(x):
#         x = hilbert(x)  # analytic signal
#     if decim > 1:
#         # integer decimation using rational resampling (downsample only)
#         x = resample_poly(x, up=1, down=decim)
#         fs = fs / decim

#     N = x.size
#     if N < 16:
#         raise ValueError("Frame too short after decimation.")

#     # 1) Build instantaneous auto-correlation R[n, m] = x[n+m] x*[n-m]
#     M = max_lag if max_lag is not None else N // 4
#     M = int(min(M, N//2 - 1))
#     lags = np.arange(-M, M + 1)
#     R = np.zeros((N, 2 * M + 1), dtype=complex)

#     # Fill for non-negative lags and mirror to negative
#     for m in range(0, M + 1):
#         n = np.arange(m, N - m)
#         seq = x[n + m] * np.conj(x[n - m])
#         R[n, M + m] = seq
#         if m > 0:
#             R[n, M - m] = np.conj(seq)

#     # 2) Ambiguity function A(τ, ν): FFT along time n (rows)
#     A = np.fft.fft(R, axis=0)                # shape: (N, 2M+1) ; frequency here is Doppler ν
#     A = np.fft.fftshift(A, axes=0)           # center ν axis at 0
#     # Normalized Doppler axis in cycles/sample
#     nu = (np.arange(N) - N//2) / N           # ∈ [−0.5, 0.5)

#     # 3) Apply Choi–Williams kernel in ambiguity domain: K(τ, ν) = exp(− (τ^2 * ν^2) / sigma)
#     # τ is lag index (samples), ν is normalized Doppler (cycles/sample)
#     tau = lags.astype(float)                 # τ ∈ [−M..M]
#     Tau, Nu = np.meshgrid(tau, nu, indexing="ij")  # shapes: (2M+1, N)
#     K = np.exp(- (Tau**2) * (Nu**2) / max(sigma, 1e-12))  # (2M+1, N)
#     # Broadcast K to A: A has shape (N, 2M+1) so transpose K
#     A_filt = A * K.T

#     # 4) Back to time–lag domain: IFFT over ν (rows)
#     R_filt = np.fft.ifft(np.fft.ifftshift(A_filt, axes=0), axis=0)

#     # 5) Time–frequency distribution: FFT over lag (columns)
#     if nfreq is None:
#         nfreq = int(2 ** int(np.ceil(np.log2(2 * M + 1))))
#     TFD = np.fft.fftshift(np.fft.fft(R_filt, n=nfreq, axis=1), axes=1)  # shape: (N, nfreq)

#     # 6) Axes and magnitude (dB)
#     f = (np.arange(nfreq) - nfreq//2) / nfreq * fs  # Hz
#     t = np.arange(N) / fs                           # s

#     P = 20 * np.log10(np.abs(TFD) + 1e-12)
#     if db_clip is not None:
#         P = np.clip(P, *db_clip)

#     # 7) Plot
#     plt.figure(figsize=figsize)
#     T, F = np.meshgrid(t * 1e3, f / 1e6, indexing="xy")
#     # transpose P to match F (freq rows) × T (time cols)
#     plt.pcolormesh(T, F, P.T, shading="auto", cmap=cmap)
#     plt.colorbar(label="Magnitude (dB)")
#     plt.xlabel("Time (ms)")
#     plt.ylabel("Frequency (MHz)")
#     plt.title(title or "Choi–Williams Distribution")
#     plt.tight_layout()
#     plt.show()



def plot_spectrogram_cwd_shorttime(signal, fs, win_len, hop, *,
                                    sigma=0.4, max_lag=None,
                                    lag_gauss_frac=0.5,   # 0<..≤1 of max_lag
                                    time_smooth=5,        # odd, small moving avg in time
                                    nfreq=None, db_clip=(-60,10),
                                    cmap="viridis", figsize=(8,4), title=None):
    import numpy as np, matplotlib.pyplot as plt
    from scipy.signal import hilbert

    x = np.asarray(signal)
    if np.isrealobj(x):
        x = hilbert(x)
    N = x.size
    L = int(win_len)
    H = int(hop)
    n_frames = 1 + (N - L) // H
    if n_frames <= 0:
        raise ValueError("win_len larger than signal")

    # time–freq canvas
    if max_lag is None:
        max_lag = L // 8
    M = int(max_lag)
    if nfreq is None:
        nfreq = 1 << int(np.ceil(np.log2(2*M+1)))

    P_stack = np.zeros((n_frames, nfreq), dtype=float)

    # precompute lag Gaussian
    lags = np.arange(-M, M+1)
    lg = int(max(1, lag_gauss_frac*M))
    w_lag = np.exp(-0.5*(lags/lg)**2)

    for k in range(n_frames):
        seg = x[k*H : k*H + L]
        # instantaneous autocorrelation R[n,m]
        R = np.zeros((L, 2*M+1), dtype=complex)
        for m in range(0, M+1):
            n = np.arange(m, L-m)
            seq = seg[n+m] * np.conj(seg[n-m])
            R[n, M+m] = seq
            if m>0:
                R[n, M-m] = np.conj(seq)

        # ambiguity (FFT over time), Choi–Williams kernel
        A = np.fft.fftshift(np.fft.fft(R, axis=0), axes=0)  # (L, 2M+1)
        nu = (np.arange(L) - L//2) / L
        Tau, Nu = np.meshgrid(lags, nu, indexing="ij")
        K = np.exp(-(Tau**2)*(Nu**2)/max(sigma,1e-12))      # (2M+1, L)
        A_f = A * K.T

        # back to time–lag and apply lag window
        Rf = np.fft.ifft(np.fft.ifftshift(A_f, axes=0), axis=0)
        Rf *= w_lag[None, :]                                 # time × lag

        # FFT over lag → one spectrum for this window
        S = np.fft.fftshift(np.fft.fft(Rf, n=nfreq, axis=1), axes=1)  # (L, nfreq)
        # time average inside the window for stability
        S_mean = S.mean(axis=0)
        P_stack[k, :] = np.abs(S_mean)

    # optional time smoothing across frames
    if time_smooth and time_smooth > 1:
        w = np.ones(time_smooth)/time_smooth
        from scipy.signal import convolve2d
        P_stack = convolve2d(P_stack, w[None,:0+1], mode="same")  # no-op keeps API

    # axes and plot - CORRECTED: Proper frequency axis handling
    t = (np.arange(n_frames)*H + L/2)/fs
    
    # Create frequency axis for the full spectrum (including negative freqs)
    f_full = (np.arange(nfreq) - nfreq//2)/nfreq * fs
    
    # Find the center index and create positive frequency mask
    center_idx = nfreq // 2
    pos_freq_mask = np.arange(nfreq) >= center_idx
    
    # Extract positive frequencies and corresponding power
    f_pos = f_full[pos_freq_mask]
    P_pos = P_stack[:, pos_freq_mask]
    
    # Convert to dB
    P = 20*np.log10(P_pos + 1e-12)
    if db_clip is not None:
        P = np.clip(P, *db_clip)

    plt.figure(figsize=figsize)
    T, F = np.meshgrid(t*1e3, f_pos/1e6, indexing="xy")
    plt.pcolormesh(T, F, P.T, shading="auto", cmap=cmap)
    plt.xlabel("Time (ms)"); plt.ylabel("Frequency (MHz)")
    plt.title(title or "Short-Time Choi–Williams Distribution")
    plt.colorbar(label="Magnitude (dB)")
    plt.tight_layout(); plt.show()



def plot_spwvd_v3(frame, fs, time_win_len=201, freq_win_len=101,
                     db_min=-80, db_max=0, cmap='gray',
                     pad_factor=0.5, mode="half_lag",
                     hop=8, nfft_max=4096):
    """
    SPWVD with doğru frekans ekseni.

    Parameters
    ----------
    mode : {"half_lag", "scale"}
        "half_lag"  → τ/2 tanımına sadık, gecikmeyi çift örneklerde kullanır.
        "scale"     → Eski matrisi korur, frekansları 0.5 ile çarpar.
    Diğer parametreler bir önceki fonksiyonla aynıdır.

    Ek Parametreler
    ---------------
    hop : int
        Zaman ekseninde seyrek örnekleme adımı (büyük değer → daha az bellek ve daha hızlı).
    nfft_max : int
        Frekans çözünürlüğünü sınırlamak için üst FFT boyutu sınırı.
    """

    if time_win_len % 2 == 0: time_win_len += 1
    if freq_win_len % 2 == 0: freq_win_len += 1
    hop = max(1, int(hop))

    pad = int(pad_factor * time_win_len)
    sig = np.pad(frame, pad, mode='constant')
    N = len(sig)

    # Analitik sinyal
    z = sig if np.iscomplexobj(sig) else hilbert(sig)

    # Smoothing pencereleri
    g = windows.hamming(time_win_len); g /= g.sum()
    h = windows.hamming(freq_win_len);  h /= h.sum()
    Lg = (time_win_len-1)//2
    Lh = (freq_win_len-1)//2

    # t indislerini sadece orijinal frame bölgesinde ve hop ile örnekle
    t_start = max(Lg*2, pad)
    t_end = min(N - Lg*2, pad + len(frame))
    t_indices = np.arange(t_start, t_end, hop, dtype=int)

    # SPWVD matrisini oluştur (nfft sınırlı, float32 bellek dostu)
    nfft_base = 1 << int(np.ceil(np.log2(max(N, 4*freq_win_len))))
    nfft = min(max(2**8, 2*freq_win_len, 1024), nfft_base, int(nfft_max))
    nfft = 1 << int(np.floor(np.log2(nfft)))  # en yakın aşağı 2^k
    tf = np.zeros((nfft//2, t_indices.size), dtype=np.float32)

    if mode == "half_lag":
        # ---------- Tanıma sadık hesaplama ----------
        for col, t in enumerate(t_indices):
            R = np.zeros(nfft, complex)
            for k in range(-Lg, Lg+1):              # k = τ/2
                tau = 2*k                           # gerçek gecikme
                w_t = g[k+Lg]
                idx_p = t + k
                idx_m = t - k
                R[tau % nfft] = w_t * z[idx_p] * z[idx_m].conj()

            # frekans penceresi
            H = np.zeros(nfft)
            H[:Lh+1] = h[Lh:]
            H[-Lh:]  = h[:Lh]
            S = np.fft.fft(R * H, nfft)
            tf[:, col] = np.abs(S[:nfft//2])

        freqs = np.linspace(0, fs/2, nfft//2, endpoint=False)

    elif mode == "scale":
        # ---------- Eski algoritma, eksen ölçekli ----------
        from copy import deepcopy
        tf_old = np.zeros((1,1), dtype=np.float32)  # placeholder
        freqs = np.linspace(0, fs/2, tf_old.shape[0], endpoint=False) * 0.5
        tf = tf_old
    else:
        raise ValueError("mode must be 'half_lag' or 'scale'")

    # dB'ye dön
    tf_db = 10*np.log10(tf + 1e-12)
    tf_db = np.clip(tf_db, db_min, db_max)

    # Zaman ve frekans eksenleri (pad kaldırılmış zamanlar)
    t_axis = (t_indices - pad) / fs
    T, F = np.meshgrid(t_axis, freqs)

    # Çiz
    plt.figure(figsize=(8,4))
    plt.pcolormesh(T, F, tf_db, shading='auto', cmap=cmap,
                   vmin=db_min, vmax=db_max)
    plt.ylabel('Frekans [Hz]')
    plt.xlabel('Zaman [s]')
    plt.colorbar(label='Yoğunluk [dB]')
    plt.title('Düzeltilmiş SPWVD')
    plt.tight_layout()
    plt.show()




# def plot_spectrogram_stft_with_rectangles(signal: np.ndarray,
#                         fs: float,
#                         fc: float,
#                         train_info: list,
#                         nperseg: int = 256,
#                         noverlap: int = 128,
#                         window: str = "hann",
#                         db_clip: tuple[float, float] | None = (-100, 0),
#                         cmap: str = "viridis",
#                         figsize: tuple[int, int] = (10, 4),
#                         title: str | None = None):
#     if noverlap is None or noverlap >= nperseg:
#         noverlap = nperseg // 2 if nperseg > 1 else 0
#     t = np.arange(signal.size) / fs
#     #signal = signal * np.exp(1j * 2 * np.pi * (5e5) * t)
#     f, t_windows, Z = stft(
#         signal,
#         fs=fs,
#         window=window,
#         nperseg=nperseg,
#         noverlap=noverlap,
#         return_onesided=False,
#         boundary="even",
#         padded=False,
#     )
    
#     # Center the frequency axis around 0 Hz (like fftshift in FFT)
#     Z = np.fft.fftshift(Z, axes=0)
#     f = np.fft.fftshift(f)
    
#     one_sided_mask = f >= 0
#     Z = Z[one_sided_mask, :]
#     f = f[one_sided_mask]

#     Pxx = 20*np.log10(np.abs(Z) + 1e-12)
#     if db_clip is not None:
#         Pxx = np.clip(Pxx, *db_clip)
#     plt.figure(figsize=figsize)
#     T, F = np.meshgrid(t_windows, f)
#     cmap_obj = plt.get_cmap(cmap, 16)
#     plt.pcolormesh(T*1e3, F/1e6, Pxx, cmap=cmap_obj, antialiased=False, linewidth=0)
#     plt.colorbar(label="Magnitude (dB)")
#     plt.xlabel("Time (ms)")
#     plt.ylabel("Frequency (MHz)")
#     plt.title(title or "STFT Spectrogram with Train Rectangles")
#     plt.grid(True)
    
#     # Add rectangles for each train
#     frame_duration = signal.size / fs
#     frame_length = signal.size
    
#     for i, train in enumerate(train_info):
#         # Calculate rectangle coordinates
#         start_time_ratio = train['start_index_ratio']
#         length_time_ratio = train['waveform_length_ratio']
#         start_freq_ratio = train['fc_ratio']
#         bandwidth_ratio = train['band_width_ratio'] 
        
#         # Convert ratios to actual coordinates
#         start_time = start_time_ratio * frame_duration * 1e3  # Convert to ms
#         length_time = length_time_ratio * frame_duration * 1e3  # Convert to ms
#         start_freq = start_freq_ratio * (fs/2) / 1e6  # Convert to MHz
#         length_freq = bandwidth_ratio * (fs/2) / 1e6  # Convert to MHz
        
#         # If bandwidth is 0, draw a horizontal line
#         if length_freq == 0:
#             plt.axhline(y=start_freq, xmin=start_time/frame_duration/1e3, 
#                        xmax=(start_time + length_time)/frame_duration/1e3, 
#                        color='red', linewidth=2, alpha=0.8)
#         else:
#             # Draw rectangle
#             from matplotlib.patches import Rectangle
#             rect = Rectangle((start_time, start_freq), 
#                            length_time, length_freq,
#                            linewidth=2, edgecolor='red', facecolor='none', alpha=0.8)
#             plt.gca().add_patch(rect)
        
#         # Add label for train type
#         plt.text(start_time + length_time/2, start_freq, train['type'], 
#                 ha='center', va='center', color='red', fontweight='bold', fontsize=8)
    
#     plt.tight_layout()
#     plt.show()
    
#     return P, times, freqs



# def plot_spectrogram_stft_with_rectangles(signal: np.ndarray,
#                         fs: float,
#                         fc: float,
#                         train_info: list,
#                         nperseg: int = 256,
#                         noverlap: int = 128,
#                         window: str = "hann",
#                         db_clip: tuple[float, float] | None = (-100, 0),
#                         cmap: str = "viridis",
#                         figsize: tuple[int, int] = (10, 4),
#                         title: str | None = None):
#     if noverlap is None or noverlap >= nperseg:
#         noverlap = nperseg // 2 if nperseg > 1 else 0
#     t = np.arange(signal.size) / fs
#     #signal = signal * np.exp(1j * 2 * np.pi * (5e5) * t)
#     f, t_windows, Z = stft(
#         signal,
#         fs=fs,
#         window=window,
#         nperseg=nperseg,
#         noverlap=noverlap,
#         return_onesided=False,
#         boundary="even",
#         padded=False,
#     )
    
#     # Center the frequency axis around 0 Hz (like fftshift in FFT)
#     Z = np.fft.fftshift(Z, axes=0)
#     f = np.fft.fftshift(f)
    
#     one_sided_mask = f >= 0
#     Z = Z[one_sided_mask, :]
#     f = f[one_sided_mask]

#     Pxx = 20*np.log10(np.abs(Z) + 1e-12)
#     if db_clip is not None:
#         Pxx = np.clip(Pxx, *db_clip)
#     plt.figure(figsize=figsize)
#     T, F = np.meshgrid(t_windows, f)
#     cmap_obj = plt.get_cmap(cmap, 16)
#     plt.pcolormesh(T*1e3, F/1e6, Pxx, cmap=cmap_obj, antialiased=False, linewidth=0)
#     plt.colorbar(label="Magnitude (dB)")
#     plt.xlabel("Time (ms)")
#     plt.ylabel("Frequency (MHz)")
#     plt.title(title or "STFT Spectrogram with Train Rectangles")
#     plt.grid(True)
    
#     # Add rectangles for each train
#     frame_duration = signal.size / fs
#     frame_length = signal.size
    
#     for i, train in enumerate(train_info):
#         # Calculate rectangle coordinates
#         start_time_ratio = train['start_index_ratio']
#         length_time_ratio = train['waveform_length_ratio']
#         start_freq_ratio = train['fc_ratio']
#         bandwidth_ratio = train['wave_param_B'] / (fs/2) if train['wave_param_B'] > 0 else 0
        
#         # Convert ratios to actual coordinates
#         start_time = start_time_ratio * frame_duration * 1e3  # Convert to ms
#         length_time = length_time_ratio * frame_duration * 1e3  # Convert to ms
#         start_freq = start_freq_ratio * (fs/2) / 1e6  # Convert to MHz
#         length_freq = bandwidth_ratio * (fs/2) / 1e6  # Convert to MHz
        
#         # If bandwidth is 0, draw a horizontal line
#         if length_freq == 0:
#             plt.axhline(y=start_freq, xmin=start_time/frame_duration/1e3, 
#                        xmax=(start_time + length_time)/frame_duration/1e3, 
#                        color='red', linewidth=2, alpha=0.8)
#         else:
#             # Draw rectangle
#             from matplotlib.patches import Rectangle
#             rect = Rectangle((start_time, start_freq - length_freq/2), 
#                            length_time, length_freq,
#                            linewidth=2, edgecolor='red', facecolor='none', alpha=0.8)
#             plt.gca().add_patch(rect)
        
#         # Add label for train type
#         plt.text(start_time + length_time/2, start_freq, train['type'], 
#                 ha='center', va='center', color='red', fontweight='bold', fontsize=8)
    
#     plt.tight_layout()
#     plt.show()



# def plot_spectrogram_stft_with_rectangles(signal: np.ndarray,
#                         fs: float,
#                         fc: float,
#                         train_info: list,
#                         nperseg: int = 256,
#                         noverlap: int = 128,
#                         window: str = "hann",
#                         db_clip: tuple[float, float] | None = (-100, 0),
#                         cmap: str = "viridis",
#                         figsize: tuple[int, int] = (10, 4),
#                         title: str | None = None):
#     if noverlap is None or noverlap >= nperseg:
#         noverlap = nperseg // 2 if nperseg > 1 else 0
#     t = np.arange(signal.size) / fs
#     #signal = signal * np.exp(1j * 2 * np.pi * (5e5) * t)
#     f, t_windows, Z = stft(
#         signal,
#         fs=fs,
#         window=window,
#         nperseg=nperseg,
#         noverlap=noverlap,
#         return_onesided=False,
#         boundary="even",
#         padded=False,
#     )
    
#     # Center the frequency axis around 0 Hz (like fftshift in FFT)
#     Z = np.fft.fftshift(Z, axes=0)
#     f = np.fft.fftshift(f)
    
#     one_sided_mask = f >= 0
#     Z = Z[one_sided_mask, :]
#     f = f[one_sided_mask]

#     Pxx = 20*np.log10(np.abs(Z) + 1e-12)
#     if db_clip is not None:
#         Pxx = np.clip(Pxx, *db_clip)
#     plt.figure(figsize=figsize)
#     T, F = np.meshgrid(t_windows, f)
#     cmap_obj = plt.get_cmap(cmap, 16)
#     plt.pcolormesh(T*1e3, F/1e6, Pxx, cmap=cmap_obj, antialiased=False, linewidth=0)
#     plt.colorbar(label="Magnitude (dB)")
#     plt.xlabel("Time (ms)")
#     plt.ylabel("Frequency (MHz)")
#     plt.title(title or "STFT Spectrogram with Train Rectangles")
#     plt.grid(True)
    
#     # Add rectangles for each train
#     frame_duration = signal.size / fs
#     frame_length = signal.size
    
#     for i, train in enumerate(train_info):
#         # Calculate rectangle coordinates
#         start_time_ratio = train['start_index_ratio']
#         length_time_ratio = train['waveform_length_ratio']
#         start_freq_ratio = train['fc_ratio']
#         bandwidth_ratio = train['wave_param_B'] / (fs/2) if train['wave_param_B'] > 0 else 0
        
#         # Convert ratios to actual coordinates
#         start_time = start_time_ratio * frame_duration * 1e3  # Convert to ms
#         length_time = length_time_ratio * frame_duration * 1e3  # Convert to ms
#         start_freq = start_freq_ratio * (fs/2) / 1e6  # Convert to MHz
#         length_freq = bandwidth_ratio * (fs/2) / 1e6  # Convert to MHz
        
#         # If bandwidth is 0, draw a horizontal line
#         if length_freq == 0:
#             plt.axhline(y=start_freq, xmin=start_time/frame_duration/1e3, 
#                        xmax=(start_time + length_time)/frame_duration/1e3, 
#                        color='red', linewidth=2, alpha=0.8)
#         else:
#             # Draw rectangle
#             from matplotlib.patches import Rectangle
#             rect = Rectangle((start_time, start_freq - length_freq/2), 
#                            length_time, length_freq,
#                            linewidth=2, edgecolor='red', facecolor='none', alpha=0.8)
#             plt.gca().add_patch(rect)
        
#         # Add label for train type
#         plt.text(start_time + length_time/2, start_freq, train['type'], 
#                 ha='center', va='center', color='red', fontweight='bold', fontsize=8)
    
#     plt.tight_layout()
#     plt.show()