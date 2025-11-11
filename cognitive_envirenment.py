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
frame_duration=10e-3
simulation_time=number_of_frames* frame_duration #20 second

print(f"Simulation time: {simulation_time}")

frame_length = int(frame_duration * fs)
emitters=assign_emitters(fs,frame_duration)
environment = "open_sea"
clutter_params = {
    "shape_param": 1.5,
    "scale_param": 0.5
}
clutter_snr_db = 20
awgn_snr_db = 10
time=0
frame_index=1
while time<simulation_time:
    if frame_index%10==0:
        print(f"Frame index: {frame_index}")
        print(f"Time: {time}")
    frame_index+=1
    frame=np.zeros(frame_length, dtype=np.complex128)
    for emitter in emitters:
        emitter.update_position(time)
        emitter.generate_waveform()
        
        emitter.calculate_start_index(frame_length)
        cluttered_waveform = add_clutter_to_pulses(
            waveform=emitter.waveform,
            environment=environment,
            snr_db=clutter_snr_db,
            **clutter_params
        )
        frame[emitter.start_index:emitter.start_index+len(emitter.waveform)] += np.complex128(cluttered_waveform)

    noisy_frame = add_awgn_noise(frame, awgn_snr_db)
    # plot_spectrogram_stft(noisy_frame, fs,
    #                     nperseg=512,
    #                     noverlap=256,
    #                     window="tukey",
    #                     db_clip=(-40, 100),
    #                     cmap="viridis",
    #                     figsize=(8, 4),
    #                     )
    a=1

    time+=frame_duration
