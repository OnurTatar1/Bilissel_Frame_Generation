import os
def save_YOLO_pulse_labels_to_txt(output_directory, YOLO_pulse_labels, frame_index, snr_db):
    # Create YOLO_pulse_labels subdirectory
    labels_dir = os.path.join(output_directory, 'YOLO_pulse_labels')
    os.makedirs(labels_dir, exist_ok=True)

    save_README_txt_for_YOLO_pulse_labels_to_txt(labels_dir)
    
    # Save file with naming format: {snr_db}db_{frame_index}.txt
    filename = f'{snr_db}db_{frame_index}.txt'
    with open(os.path.join(labels_dir, filename), 'w') as f:
        for label in YOLO_pulse_labels:
            f.write(label + '\n')

def save_YOLO_tren_labelsto_txt(output_directory, YOLO_tren_labels, frame_index, snr_db):
    # Create YOLO_tren_labels subdirectory
    labels_dir = os.path.join(output_directory, 'YOLO_tren_labels')
    os.makedirs(labels_dir, exist_ok=True)

    save_README_txt_for_YOLO_tren_labels_to_txt(labels_dir)
    
    # Save file with naming format: {snr_db}db_{frame_index}.txt
    filename = f'{snr_db}db_{frame_index}.txt'
    with open(os.path.join(labels_dir, filename), 'w') as f:
        for label in YOLO_tren_labels:
            f.write(label + '\n')

def save_interleaved_dtk_labels_to_txt(output_directory, interleaved_dtk_labels, frame_index, snr_db):
    # Create interleaved_dtk_labels subdirectory
    labels_dir = os.path.join(output_directory, 'interleaved_dtk_labels')
    os.makedirs(labels_dir, exist_ok=True)

    save_README_txt_for_interleaved_dtk_labels_to_txt(labels_dir)
    
    # Save file with naming format: {snr_db}db_{frame_index}.txt
    filename = f'{snr_db}db_{frame_index}.txt'
    with open(os.path.join(labels_dir, filename), 'w') as f:
        for label in interleaved_dtk_labels:
            f.write(label + '\n')

def save_train_descriptive_words_to_txt(output_directory, train_descriptive_words, frame_index, snr_db):
    # Create train_descriptive_words subdirectory
    labels_dir = os.path.join(output_directory, 'train_descriptive_words')
    os.makedirs(labels_dir, exist_ok=True)

    save_README_txt_for_train_descriptive_words_to_txt(labels_dir)
    
    # Save file with naming format: {snr_db}db_{frame_index}.txt
    filename = f'{snr_db}db_{frame_index}.txt'
    with open(os.path.join(labels_dir, filename), 'w') as f:
        for word in train_descriptive_words:
            f.write(word + '\n')

def save_README_txt_for_YOLO_pulse_labels_to_txt(output_directory):
    readme_content = """YOLO Pulse Labels Format
    =======================

    This file contains YOLO format labels for individual pulses in the radar spectrogram.

    Column Format:
    --------------
    Each line in txt's contains the following columns for the pulses (space-separated):

    1. type
    - Waveform type ID (integer)
    - Mapping: 0=rect, 1=barker, 2=costas, 3=frank, 4=lfm, 5=p1, 6=p2, 7=p3, 8=p4

    2. x_center_ratio
    - Normalized x-center position of the pulse bounding box (0.0 to 1.0)
    - Represents the center position along the time axis

    3. y_center_ratio
    - Normalized y-center position of the pulse bounding box (0.0 to 1.0)
    - Represents the center position along the frequency axis

    4. x_width_ratio
    - Normalized width of the pulse bounding box (0.0 to 1.0)
    - Represents the width along the time axis

    5. y_width_ratio
    - Normalized height of the pulse bounding box (0.0 to 1.0)
    - Represents the height along the frequency axis

    Example:
    --------
    0 0.25 0.5 0.1 0.05
    This represents a rectangular pulse (type=0) centered at (25%, 50%) with width 10% and height 5% of the spectrogram dimensions.

    Note:
    -----
    All coordinates and dimensions are normalized ratios (0.0 to 1.0) relative to the spectrogram image dimensions.
    """
    with open(os.path.join(output_directory, 'README_YOLO_pulse_labels.txt'), 'w') as f:
        f.write(readme_content)

def save_README_txt_for_YOLO_tren_labels_to_txt(output_directory):
    readme_content = """YOLO Pulse Train Labels Format
    =======================

    This file contains YOLO format labels for pulse trains in the radar spectrogram.

    Column Format:
    --------------
    Each line in txt's contains the following columns for the trains (space-separated):

    1. type
    - Waveform type ID (integer)
    - Mapping: 0=rect, 1=barker, 2=costas, 3=frank, 4=lfm, 5=p1, 6=p2, 7=p3, 8=p4

    2. x_center_ratio
    - Normalized x-center position of the trains bounding box (0.0 to 1.0)
    - Represents the center position along the time axis

    3. y_center_ratio
    - Normalized y-center position of the trains bounding box (0.0 to 1.0)
    - Represents the center position along the frequency axis

    4. x_width_ratio
    - Normalized width of the trains bounding box (0.0 to 1.0)
    - Represents the width along the time axis

    5. y_width_ratio
    - Normalized height of the trains bounding box (0.0 to 1.0)
    - Represents the height along the frequency axis

    Example:
    --------
    0 0.25 0.5 0.1 0.05
    This represents a rectangular pulse trains (type=0) centered at (25%, 50%) with width 10% and height 5% of the spectrogram dimensions.

    Note:
    -----
    All coordinates and dimensions are normalized ratios (0.0 to 1.0) relative to the spectrogram image dimensions.
    """
    with open(os.path.join(output_directory, 'README_YOLO_tren_labels.txt'), 'w') as f:
        f.write(readme_content)

def save_README_txt_for_interleaved_dtk_labels_to_txt(output_directory):
    readme_content = """Interleaved DTK Labels Format
    =======================

    This file contains detailed pulse information in DTK (Detaylı Teknik Katalog) format.

    Column Format:
    --------------
    Each line in txt's contains the following columns for the pulses (space-separated):

    1. train_index
    - Index of the pulse train this pulse belongs to (integer)

    2. pulse_index
    - Index of the pulse within the frame (integer)

    3. wave_type
    - Waveform type (string)
    - Values: rect, barker, costas, frank, lfm, p1, p2, p3, p4

    4. pulse_start_time
    - Start time of the pulse in seconds (float)
    - Relative to the frame start

    5. pulse_width
    - Width/duration of the pulse in seconds (float)

    6. fc
    - Carrier frequency in Hz (float)

    7. bandwidth
    - Bandwidth of the pulse in Hz (float)

    8. amplitude
    - Amplitude of the pulse (float)

    9. AWGN noise_power
    - Noise power level db (float)

    Example:
    --------
    1 5 barker 0.001 2e-5 1e6 50000 5.0 -10 
    This represents pulse index 5 from train 1, type barker, starting at 1ms, 
    with width 20μs, carrier frequency 1MHz, bandwidth 50kHz, amplitude 5.0, and noise power -10 .

    Note:
    -----
    This format provides detailed technical information about each pulse for analysis and cataloging purposes.
    """
    with open(os.path.join(output_directory, 'README_interleaved_dtk_labels.txt'), 'w') as f:
        f.write(readme_content)

def save_README_txt_for_train_descriptive_words_to_txt(output_directory):
    readme_content = """Train Descriptive Words Format
    =======================

    This file contains descriptive metadata information about pulse trains.

    Format:
    ------
    Each line contains a dictionary (or structured data) with the following fields:

    1. Emitter_ID
    - Unique identifier for the emitter (integer)

    2. train_index
    - Index of the pulse train (integer)

    3. off_spring_pulses
    - List containing the range of pulse indices belonging to this train
    - Format: [first_pulse_index, last_pulse_index]

    4. PRI_type
    - Pulse Repetition Interval type (string)
    - Values: stable, staggered, jittered, wobulated, sliding+, sliding-

    5. PRI_mean
    - Mean Pulse Repetition Interval in seconds (float)

    6. PRI_agile_ampl
    - PRI agility amplitude/range parameter (float)
    - Controls the variation in PRI for agile modes

    7. Scanning_type
    - Antenna scanning type (string)
    - Describes the scanning pattern used

    8. Scanning_step
    - Antenna scanning step size (float)
    - Step size for the scanning pattern

    Example:
    --------
    {'Emitter_ID': 1, 'train_index': 5, 'off_spring_pulses': [10, 15], 
     'PRI_type': 'staggered', 'PRI_mean': 0.001, 'PRI_agile_ampl': 0.2,
     'Scanning_type': 'circular', 'Scanning_step': 10.0}
    This represents train 5 from emitter 1, containing pulses 10-15, 
    with staggered PRI (mean 1ms, agility 0.2), using circular scanning with 10 degree steps.

    Note:
    -----
    This format provides metadata and descriptive information about each pulse train 
    for tracking, analysis, and emitter identification purposes.
    """
    with open(os.path.join(output_directory, 'README_train_descriptive_words.txt'), 'w') as f:
        f.write(readme_content)