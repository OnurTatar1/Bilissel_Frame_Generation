import numpy as np
import random
import os
import matplotlib.pyplot as plt
from plot_spectrogram import (
    plot_spectrogram_stft,
    plot_spectrogram_stft_with_rectangles,
    plot_spwvd,
    plot_spwvd_v2,
    plot_spwvd_v3,
    plot_choi_williams,
    plot_spectrogram_cwd,
    plot_spectrogram_cwd_shorttime,
)
from wave_generator_functions import generate_waveform, pri_values, pulse_indices, reflector
# from antenna_Array.array_geometries import (
#     uniform_linear_array_positions,
#     uniform_planar_array_positions,
#     uniform_circular_array_positions,
#     plot_array_positions,
# )
from antenna_Array.angle_estimation_methods import (
    estimate_az_el_from_array_signals_interferometer_2D_planar_array)
from antenna_Array.Music import (run_music_sliding)
from antenna_Array.noise_functions import add_awgn_to_array_signals
from clutter_example import add_clutter_with_snr, add_clutter_to_pulses
###parameters###

frame_duration = 10e-3
fs = 2e7
freq_band=1e8
frame_length = int(frame_duration * fs)
environment = "open_sea"
clutter_params = {
    "shape_param": 1.5,
    "scale_param": 0.5
}
clutter_snr_db = 20
number_of_frames=1

for snr_db in (5, -10, -7, -5,-2):
    for frame_index in range(number_of_frames):
        frame = np.zeros(frame_length, dtype=np.complex128)
        number_of_illuminators=3#random.randint(1, 10)
        number_of_reflectors=0#random.randint(1, 10)
        
        train_info = []
        for i in range(number_of_illuminators):
            
            
            if 1:  #waveform_parametreleri olusturur
                wave_type = random.choice(["rect", "barker", "costas", "frank", "lfm", "p1", "p2", "p3", "p4"])
                fc = 0
                number_of_pulse = random.randint(2, 7)
                PRI_type = random.choice(["stable", "staggered", "jittered", "wobulated", "sliding+", "sliding-"])
                PRI_agile_ampl = random.uniform(0.2, 0.7)
                PRI_mean = random.uniform(3/4*frame_duration/(number_of_pulse),1/7*frame_duration/(number_of_pulse))
                transmitter_ = {
                "number_of_pulse": number_of_pulse,
                "PRI_mode": PRI_type,
                "PRI_mean": PRI_mean,
                "PRI_agile_ampl": PRI_agile_ampl,
                }
                PRI_values=pri_values(transmitter_)
                min_PRI=min(PRI_values)
                pulse_width = random.uniform(min_PRI/1.5, min_PRI/8)
                amplitude = random.uniform(0.5, 10.5)

                
                if wave_type == "barker":
                    wave_param_L = random.choice([2, 7, 11, 13])
                elif wave_type == "costas":
                    wave_param_L = random.choice([7, 11, 13, 17])
                elif wave_type in ["frank", "p1", "p2"]:
                    wave_param_L = random.choice([3, 4, 5, 6, 7])
                elif wave_type in ["p3", "p4"]:
                    wave_param_L = random.randint(4, 50)
                else:
                    wave_param_L = 1

                if wave_type in ["rect", "barker", "p3", "p4"]:
                    wave_param_B = wave_param_L/pulse_width
                elif wave_type in ["frank", "p1", "p2"]:
                    wave_param_B = wave_param_L*wave_param_L/pulse_width
                elif wave_type == "costas":
                    wave_param_B = random.uniform(fs/1e2, 2*fs/1e1)
                else:
                    wave_param_B = random.uniform(fs/1e2, 2*fs/1e1)
                max_fc = fs/2 - wave_param_B - fs/32  # Maximum f_c to avoid aliasing
                min_fc = fs/32         # Minimum f_c to keep signal in positive frequencies
                wave_length=(PRI_mean*number_of_pulse+pulse_width)
                fc+=random.uniform(min_fc, max_fc)
                params = {
                    "wave_param_B": wave_param_B, #wave_param_B,
                    "wave_param_L": wave_param_L,
                    "f_c": fc, 
                    "pulse_width": pulse_width,
                    "PRI_mean": PRI_mean,
                    "number_of_pulse": number_of_pulse,
                    "PRI_mode": "stable",
                    "f_s": fs,
                    "PRI_mode": PRI_type,
                    "PRI_agile_ampl": PRI_agile_ampl,
                    "transmitter_type": wave_type  # Fixed parameter name
                }

            waveform = generate_waveform(params,PRI_values)
            max_start = int(31*frame_length/32 - len(waveform)) 
            min_start = int(frame_length/32)
            start_idx = random.randint(min_start, int(max_start))

            waveform = waveform*amplitude
            cluttered_waveform = add_clutter_to_pulses(
                waveform=waveform,
                environment=environment,
                snr_db=clutter_snr_db,
                **clutter_params
            )
            for i_reflector in range(number_of_reflectors):
                reflection_time_shift=1e-9 #random.uniform(1e-8, 1e-9)
                reflection_snr_db = 20 * np.log10(4 * np.pi * freq_band*reflection_time_shift) + random.uniform(0, 0)
                reflected_waveform = reflector(waveform,reflection_snr_db,reflection_time_shift,fs)
                cluttered_reflected_waveform = add_clutter_to_pulses(
                    waveform=reflected_waveform,
                    environment=environment,
                    snr_db=clutter_snr_db,
                    **clutter_params
                )
                frame[start_idx:start_idx+len(cluttered_reflected_waveform)] += np.complex128(cluttered_reflected_waveform)
            
            frame[start_idx:start_idx+len(waveform)] += np.complex128(cluttered_waveform)
            
            
            # yolo icin labellama
            if 1:
                start_time_ratio=(start_idx / frame_length)-0.005
                length_time_ratio=(len(waveform) / frame_length)+0.01
                bandwidth_ratio=wave_param_B / (fs/2)
                start_freq_ratio=fc / (fs/2)
                if wave_type in ["rect", "barker","frank", "p1", "p2", "p3", "p4"]:
                    start_freq_ratio = start_freq_ratio - bandwidth_ratio / 2 -0.01
                    bandwidth_ratio = bandwidth_ratio + 0.02
                if wave_type == "costas":
                    costas_pulse_bandwidth = wave_param_L/pulse_width
                    costas_pulse_bandwidth_ratio = costas_pulse_bandwidth / (fs/2)
                    bandwidth_ratio=wave_param_B / (fs/2) 
                    start_freq_ratio=fc / (fs/2)-0.01
                train_data = {
                    'type': wave_type,
                    'start_index_ratio': start_time_ratio,
                    'waveform_length_ratio': length_time_ratio,
                    'fc_ratio': start_freq_ratio,
                    'band_width_ratio': bandwidth_ratio,
                }
                train_info.append(train_data)


        #Noise
        if 1:
            snr_linear = 10**(snr_db / 10)
            noise_power = 1 / snr_linear
            noise_std = np.sqrt(noise_power / 2)  # Her component için std
            noise_real = np.random.normal(0, noise_std, frame.shape)
            noise_imag = np.random.normal(0, noise_std, frame.shape)
            noise = noise_real + 1j * noise_imag
            noisy_frame = frame + noise
        
        # plot_spwvd_v3(noisy_frame, fs, time_win_len=501, freq_win_len=211,
        #              db_min=-70, db_max=10, cmap='coolwarm',
        #              pad_factor=0.5, mode="half_lag")
        # plot_choi_williams(noisy_frame, fs,
        #                    sigma=1.0,
        #                    nfft=512,
        #                    hop=2,
        #                    max_lag=128,
        #                    db_clip=(-40, 100),
        #                    cmap="viridis",
        #                    figsize=(8, 4),
        #                    title="Choi-Williams Distribution")
        # plot_spectrogram_cwd(noisy_frame, fs, decim=50, sigma=0.07,
        #              db_clip=(-40, 100), cmap="viridis", figsize=(8,4))

        win_len = int(0.03e-3 * fs)   # 0.5 ms window
        hop     = int(1)   # 0.1 ms hop
        plot_spectrogram_cwd(noisy_frame, fs,
            sigma=0.0001,
            n_time=win_len,
            n_freq=512,
            db_clip=(-40, 100),
            cmap="viridis",
            figsize=(8, 4))


        plot_spectrogram_stft(noisy_frame, fs,
                        nperseg=512,
                        noverlap=256,
                        window="tukey",
                        db_clip=(-40, 100),
                        cmap="viridis",
                        figsize=(8, 4),
                        )
        a=1

        # # Save train information to text file
        # os.makedirs('frame_info_yolo', exist_ok=True)
        # with open(os.path.join('frame_info_yolo', f'train_info_{snr_db}db_{frame_index}.txt'), 'w') as f:
        #     for i, train in enumerate(train_info):
        #         f.write(f"{train['type']}, {train['start_index_ratio']:.6f}, {train['waveform_length_ratio']:.6f}, {train['fc_ratio']:.6f}, {train['band_width_ratio']}\n")






