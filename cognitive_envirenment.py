from emitter_class import Emitter
from emitter_assignments import assign_emitters
import numpy as np
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from plot_spectrogram import (
    plot_spectrogram_stft,
    plot_spectrogram_stft_with_rectangles
)
from wave_generator_functions import generate_waveform, pri_values, pulse_indices, reflector
from labeling import Pulse_train_labeling, Pulse_labeling
# from antenna_Array.array_geometries import (
#     uniform_linear_array_positions,
#     uniform_planar_array_positions,
#     uniform_circular_array_positions,
#     plot_array_positions,
# )
from clutter_functions import add_clutter_to_pulses, add_awgn_noise
global fs
fs=2e7

number_of_frames=1000
frame_duration_seconds=10e-3
simulation_time=number_of_frames* frame_duration_seconds #20 second

print(f"Simulation time: {simulation_time}")

frame_sample_length = int(frame_duration_seconds * fs)
emitters=assign_emitters(fs,frame_duration_seconds)
environment = "open_sea"
clutter_params = {
    "shape_param": 1.5,
    "scale_param": 0.5
}
clutter_snr_db = 20
awgn_power_db = -10
time=0
frame_index=1
train_index=0
pulse_index=0
while time<simulation_time:
    if frame_index%10==0:
        print(f"Frame index: {frame_index}")
        print(f"Time: {time}")
    frame_index+=1
    
    frame=np.zeros(frame_sample_length, dtype=np.complex128)
    YOLO_tren_labels = []
    YOLO_pulse_labels = []
    interleaved_dtk_labels = []
    train_descriptive_words=[]
    for emitter in emitters:
        if emitter.ID!=0:       # or emitter.ID!=1:
            continue
        emitter.update_position(time)
        scan_az_deg, scan_el_deg,target_az,target_el = emitter.scanning()
        target_el=0
        target_az=30
        if emitter.is_scanning==False:
            emitter.first_transmit=True
            continue
        if emitter.first_transmit:
            emitter.generate_waveform()
            emitter.first_transmit=False
        train_index+=1
        amplitude=emitter.antenna_gain_calculator(scan_az_deg, scan_el_deg, target_az, target_el)
        emitter.waveform=emitter.waveform*amplitude
        emitter.calculate_start_index(frame_sample_length)
        
        cluttered_waveform = add_clutter_to_pulses(
            waveform=emitter.waveform,
            environment=environment,
            snr_db=clutter_snr_db,
            **clutter_params
        )
        frame[emitter.start_index:emitter.start_index+len(emitter.waveform)] += np.complex128(cluttered_waveform)
    noisy_frame = add_awgn_noise(frame, AWGN_power_db=awgn_power_db)




    #Labeling
    YOLO_tren_label, train_descriptive_word = Pulse_train_labeling(waveform=emitter.waveform,frame_sample_length=frame_sample_length,
                                            start_idx=emitter.start_index, wave_param_B=emitter.wave_param_B,fs=fs,fc=emitter.fc,
                                            wave_type=emitter.wave_type,wave_param_L=emitter.wave_param_L,pulse_width=emitter.pulse_width,
                                            train_index=train_index,pulse_index=pulse_index,PRI_value=emitter.PRI_values,
                                            PRI_type=emitter.PRI_type,PRI_mean=emitter.PRI_mean, PRI_agile_ampl=emitter.PRI_agile_ampl,
                                            emitter_id=emitter.ID,Scanning_type=emitter.Antenna_Scanning_Type,
                                            Scanning_step=emitter.Antenna_Scanning_step)
    print("train_descriptive_word:")
    print(train_descriptive_word)
    print("YOLO tren label:")
    print(YOLO_tren_label)

    YOLO_pulse_label, dtk_info   = Pulse_labeling(pulse_index=pulse_index, PRI_values=emitter.PRI_values,
                                frame_sample_length=frame_sample_length, start_idx=emitter.start_index, wave_param_B=emitter.wave_param_B,
                                fs=fs, fc=emitter.fc,wave_type=emitter.wave_type, wave_param_L=emitter.wave_param_L,
                                 pulse_width=emitter.pulse_width, amplitude=amplitude, noise_power=awgn_power_db, train_index=train_index)
    pulse_index+=len(emitter.PRI_values)

    print("dtk info:")
    print(dtk_info)
    print("YOLO pulse label:")
    print(YOLO_pulse_label)

    interleaved_dtk_labels.append(dtk_info)
    YOLO_pulse_labels.append(YOLO_pulse_label)

    train_descriptive_words.append(train_descriptive_word)
    YOLO_tren_labels.append(YOLO_tren_label)
    

    plot_spectrogram_stft_with_rectangles(signal=noisy_frame, fs=fs, train_info=YOLO_pulse_label,
                        nperseg=256, noverlap=128, window="hann", db_clip=(-80, 60), cmap="viridis",
                         figsize=(8, 8), title=None, save_or_plot=False, frame_index=frame_index, snr_db=awgn_power_db)
    a=1

    time+=frame_duration_seconds
