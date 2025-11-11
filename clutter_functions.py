import numpy as np
import matplotlib.pyplot as plt
from fadin_clutter.clutter_models import get_clutter_model

# Helper function to add SNR control
def add_clutter_with_snr(signal, environment, snr_db, **clutter_params):
    clutter = get_clutter_model(environment, len(signal), **clutter_params)
    signal_power = np.mean(np.abs(signal)**2)
    if signal_power == 0:
        signal_power = 1
    snr_linear = 10**(snr_db / 10)
    desired_clutter_power = signal_power / snr_linear
    current_clutter_power = np.mean(np.abs(clutter)**2)
    if current_clutter_power > 0:
        scale_factor = np.sqrt(desired_clutter_power / current_clutter_power)
        scaled_clutter = clutter * scale_factor
    else:
        scaled_clutter = clutter
    noisy_signal = signal + scaled_clutter
    return scaled_clutter

def add_clutter_to_pulses(waveform, environment, snr_db, **clutter_params):
    """
    Add clutter to individual pulses in a waveform.
    
    Parameters:
    waveform : array-like
        Input radar waveform
    environment : str
        Environment type (e.g., 'urban', 'forest', 'open_sea')
    snr_db : float
        Desired SNR in dB for each pulse
    **clutter_params : dict
        Additional parameters for clutter model
    
    Returns:
    array-like
        Waveform with clutter added to each pulse
    """
    from wave_generator_functions import pulse_indices
    
    # Create a copy of the waveform
    cluttered_waveform = waveform.copy()
    
    # Find pulse indices
    radar_pulse_indices = pulse_indices(waveform)
    if len(radar_pulse_indices) == 0:
        radar_pulse_indices = [f"0:25"]
    # Add clutter to each pulse
    for pulse_index in radar_pulse_indices:
        start_idx, end_idx = pulse_index.split(":")
        start_idx = int(start_idx)
        end_idx = int(end_idx)
        
        # Add clutter to this pulse
        clutter = add_clutter_with_snr(
            environment=environment,
            signal=cluttered_waveform[start_idx:end_idx],
            snr_db=snr_db,
            **clutter_params
        )
        
        # Add clutter to the pulse
        cluttered_waveform[start_idx:end_idx] += clutter
    
    return cluttered_waveform

def snr_clutter_example():
    """
    Example showing how to control clutter SNR
    """
    print("=== SNR-Controlled Clutter Example ===")
    
    # Signal parameters
    fs = 1e9
    duration = 1e-3
    t = np.arange(0, duration, 1/fs)
    fc = 2e6
    
    # Create radar signal
    pulse_width = 1e-5
    pulse_samples = int(pulse_width * fs)
    radar_signal =np.cos(2 * np.pi * fc * t[0:pulse_samples])
    
    
    # Calculate signal power
    signal_power = np.mean(np.abs(radar_signal)**2)
    
    # Different SNR values to test
    snr_values = [0, 5, 10, 15, 20]  # dB
    environment = 'open_sea'
    
    plt.figure(figsize=(15, 10))
    
    for i, snr_db in enumerate(snr_values):
        # Generate clutter with specific SNR using our helper function
        clutter = add_clutter_with_snr(
            signal=radar_signal,
            environment=environment,
            snr_db=snr_db
        )
        noisy_signal=clutter+radar_signal
        # Calculate actual SNR
        actual_snr_db = 10 * np.log10(signal_power / np.mean(np.abs(clutter)**2))
        
        # Plot results
        plt.subplot(2, 3, i+1)
        plt.plot(t[0:pulse_samples]*1000, np.abs(noisy_signal))
        plt.title(f'SNR: {snr_db} dB (Actual: {actual_snr_db:.1f} dB)')
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude')
        plt.grid(True)
    
    
    frame_signal=np.zeros(len(t), dtype=complex)
    frame_signal[int(len(t)*0.3):int(len(t)*0.3)+pulse_samples] = noisy_signal
    # Plot clean signal
    plt.subplot(2, 3, 6)
    plt.plot(t*1000, np.abs(frame_signal))
    plt.title('Clean Signal (No Clutter)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    return frame_signal, clutter, noisy_signal

def add_awgn_noise(frame, snr_db):
    snr_linear = 10**(snr_db / 10)
    noise_power = 1 / snr_linear
    noise_std = np.sqrt(noise_power / 2)  # Her component için std
    noise_real = np.random.normal(0, noise_std, frame.shape)
    noise_imag = np.random.normal(0, noise_std, frame.shape)
    noise = noise_real + 1j * noise_imag
    noisy_frame = frame + noise
    return noisy_frame
#frame_signal, clutter, noisy_signal = snr_clutter_example()
