wave_type_to_id = {
    "rect": 0,
    "barker": 1,
    "costas": 2,
    "frank": 3,
    "lfm": 4,
    "p1": 5,
    "p2": 6,
    "p3": 7,
    "p4": 8
}

PRI_type_to_id = {
    "staggered": 0,
    "stable": 1,
    "jittered": 2,
    "wobulated": 3,
    "sliding+": 4,
    "sliding-": 5
}

Scanning_type_to_id = {
    "circular_scanning": 0,
    "unidirectional_sector_scanning": 1,
    "bidirectional_sector_scanning": 2,
    "raster_scanning": 3,
    "raster_sector_scanning_centering_a_target": 4
}

def Pulse_train_labeling(waveform,frame_sample_length,start_idx,wave_param_B,fs,fc,wave_type,wave_param_L,pulse_width,train_index,pulse_index,PRI_value,PRI_type,PRI_mean,PRI_agile_ampl,emitter_id,Scanning_type,Scanning_step):
    start_time_ratio=(start_idx / frame_sample_length)-0.001
    if start_time_ratio < 0:
        start_time_ratio = 0
    length_time_ratio=(len(waveform) / frame_sample_length)+0.002
    bandwidth_ratio=wave_param_B / (fs/2)
    start_freq_ratio=fc / (fs/2)
    if wave_type in ["rect", "barker","frank", "p1", "p2", "p3", "p4"]:
        start_freq_ratio = start_freq_ratio - bandwidth_ratio / 2 -0.01
        if start_freq_ratio < 0:
            start_freq_ratio = 0
        bandwidth_ratio = bandwidth_ratio + 0.02
    if wave_type == "costas":
        costas_pulse_bandwidth = wave_param_L/pulse_width
        costas_pulse_bandwidth_ratio = costas_pulse_bandwidth / (fs/2)
        bandwidth_ratio=wave_param_B / (fs/2) + costas_pulse_bandwidth_ratio + 0.02
        start_freq_ratio=fc / (fs/2)-0.01
        if start_freq_ratio < 0:
            start_freq_ratio = 0
    x_center_ratio= start_time_ratio+ length_time_ratio/2       
    y_center_ratio= 1- (start_freq_ratio+ bandwidth_ratio/2)
    x_width_ratio= length_time_ratio
    y_width_ratio= bandwidth_ratio
    if x_center_ratio<0 or y_center_ratio<0 or x_width_ratio<0 or y_width_ratio<0:
        a=1
    train_data = {
        'type': wave_type_to_id[wave_type],
        'x_center_ratio': x_center_ratio,
        'y_center_ratio': y_center_ratio,
        'x_width_ratio': x_width_ratio,
        'y_width_ratio': y_width_ratio,
    }
    off_spring_pulses = list(range(pulse_index, pulse_index+len(PRI_value)))
    train_descriptive_words = {
        'Emitter_ID': emitter_id,
        'train_index': train_index,
        'off_spring_pulses': off_spring_pulses,
        'PRI_type': PRI_type_to_id[PRI_type],
        'PRI_mean': PRI_mean,
        'PRI_agile_ampl': PRI_agile_ampl,
        'Scanning_type': Scanning_type_to_id[Scanning_type],
        'Scanning_step': Scanning_step,
        #scanning period eklenecek scanning step yerine
        
    }
        
    return train_data,train_descriptive_words

def Pulse_labeling(pulse_index, PRI_values, frame_sample_length, start_idx, wave_param_B, fs, fc, wave_type, wave_param_L, pulse_width, amplitude, noise_power,train_index):
    pulse_start_idx=start_idx
    dtk_info=[]
    pulse_data_list=[]
    for PRI_true in PRI_values:
        one_pulse=[train_index, pulse_index, wave_type_to_id[wave_type], pulse_start_idx/fs, pulse_width, fc, wave_param_B, amplitude, noise_power] #AOA eklenecek
        dtk_info.append(one_pulse)
        pulse_start_idx=pulse_start_idx+int(PRI_true*fs)
        pulse_index+=1
    for Pulses in dtk_info:    
        start_time_ratio=(Pulses[3] / frame_sample_length*fs) 
        if start_time_ratio < 0:
            start_time_ratio = 0
        length_time_ratio=(Pulses[4] / frame_sample_length*fs) 

        bandwidth_ratio=wave_param_B / (fs/2)
        start_freq_ratio=fc / (fs/2)
        if wave_type in ["rect", "barker","frank", "p1", "p2", "p3", "p4"]:
            start_freq_ratio = start_freq_ratio - bandwidth_ratio / 2
            if start_freq_ratio < 0:
                start_freq_ratio = 0
            bandwidth_ratio = bandwidth_ratio
        if wave_type == "costas":
            costas_pulse_bandwidth = (wave_param_L-1)/pulse_width
            costas_pulse_bandwidth_ratio = costas_pulse_bandwidth / (fs/2)
            bandwidth_ratio=wave_param_B / (fs/2) + costas_pulse_bandwidth_ratio 
            start_freq_ratio=fc / (fs/2)
            if start_freq_ratio < 0:
                start_freq_ratio = 0
            
        x_center_ratio= start_time_ratio + length_time_ratio/2
        y_center_ratio= 1- (start_freq_ratio+ bandwidth_ratio/2)
        x_width_ratio= length_time_ratio
        y_width_ratio= bandwidth_ratio
        pulse_data = {
            'type': wave_type_to_id[wave_type],
            'x_center_ratio': x_center_ratio,
            'y_center_ratio': y_center_ratio,
            'x_width_ratio': x_width_ratio,
            'y_width_ratio': y_width_ratio,
        }
        pulse_data_list.append(pulse_data)
    return pulse_data_list,dtk_info

