
import random
from types import NoneType
import numpy as np
from wave_generator_functions import pri_values, generate_waveform
global c
c= 299792458
class Emitter:
    def __init__(self,ID=None, wave_type_range=None, emitter_complexity=None, PW_range=None,
                 number_of_pulse_range=None, PRI_type_range=None, PRI_agile_ampl_range=None, 
                 amplitude_range=None, fc_range=None, PRI_mean_range=None,wave_param_B_range=None,start_position=None,
                 velocity=None,locked_target=None,
                 antenna_gain=None,SNR_aim=None,current_noise_power=None, Antenna_Scanning_Type=None,
                 Antenna_Scanning_step=None, Antenna_Scanning_sector_Azimuth=None,Antenna_Scanning_sector_elevation=None,
                 antenna_radiation_pattern=None, fs=None, frame_duration=None):
        # Identification & basic configuration
        self.ID = ID
        self.emitter_complexity = emitter_complexity
        self.frame_duration = frame_duration
        self.fs = fs

        # Pulse Params
        self.wave_type_range = wave_type_range
        self.PW_range = PW_range
        self.number_of_pulse_range = number_of_pulse_range
        self.wave_param_L = None
        self.wave_param_B = None
        self.fc_range = fc_range
        self.wave_param_B_range = wave_param_B_range
        self.amplitude_range = amplitude_range

        # PRI Params
        self.PRI_type_range = PRI_type_range
        self.PRI_agile_ampl_range = PRI_agile_ampl_range
        self.PRI_mean_range = PRI_mean_range

        # Motion & tracking
        self.start_position = np.array(start_position, dtype=float) if start_position is not None else None
        self.velocity = np.array(velocity, dtype=float) if velocity is not None else None
        self.locked_target = np.array(locked_target, dtype=float) if isinstance(locked_target, (list, tuple, np.ndarray)) else locked_target
        self.locked_target_position = None

        # Antenna configuration
        self.antenna_gain = antenna_gain
        self.SNR_aim = SNR_aim
        self.Antenna_Scanning_Type = Antenna_Scanning_Type
        self.Antenna_Scanning_step = Antenna_Scanning_step
        self.Antenna_Scanning_sector_Azimuth = Antenna_Scanning_sector_Azimuth
        self.Antenna_Scanning_sector_elevation = Antenna_Scanning_sector_elevation
        self.antenna_radiation_pattern = antenna_radiation_pattern

        


    def set_waveform_type(self):
        self.wave_type = random.choice(self.wave_type_range)
        return self
    
    def set_carrier_frequency(self):
        
        max_fc = self.fc_range[1] - self.wave_param_B   # Maximum f_c to avoid aliasing
        min_fc = self.fc_range[0]  # Minimum f_c to keep signal in positive frequencies
        fc = random.uniform(min_fc, max_fc)
        self.fc = fc
        return self

    def set_number_of_pulse(self):
        self.number_of_pulse = random.randint(self.number_of_pulse_range[0], self.number_of_pulse_range[1])
        return self

    def set_PRI_params_calculate_PRI_values(self):
        self.PRI_type = random.choice(self.PRI_type_range)
        self.PRI_agile_ampl = random.uniform(self.PRI_agile_ampl_range[0], self.PRI_agile_ampl_range[1])
        self.PRI_mean = random.uniform(self.PRI_mean_range[0], self.PRI_mean_range[1])
        self.PRI_values = pri_values(self)
        return self

    def set_pulse_width(self):
        min_PRI=min(self.PRI_values)
        max_PW= max(self.PW_range[1],min_PRI/1.5)
        self.pulse_width = random.uniform(self.PW_range[0], max_PW)
        return self

    def set_wave_param_L(self):
        if self.wave_type == "barker":
            self.wave_param_L = random.choice([2, 7, 11, 13])
        elif self.wave_type == "costas":
            self.wave_param_L = random.choice([7, 11, 13, 17])
        elif self.wave_type in ["frank", "p1", "p2"]:
            self.wave_param_L = random.choice([3, 4, 5, 6, 7])
        elif self.wave_type in ["p3", "p4"]:
            self.wave_param_L = random.randint(4, 50)
        else:
            self.wave_param_L = 1
        return self

    def set_wave_param_B(self):
        if self.wave_type == "costas" or self.wave_type == "LFM":
            self.wave_param_B = random.uniform(self.wave_param_B_range[0], self.wave_param_B_range[1])
        else:
            self.wave_param_B = self.wave_param_L/self.pulse_width
        return self


    def generate_waveform(self):
        self.set_waveform_type()
        self.set_number_of_pulse()
        self.set_PRI_params_calculate_PRI_values()
        self.set_pulse_width()
        self.set_wave_param_L()
        self.set_wave_param_B()
        self.set_carrier_frequency()
        self.waveform=generate_waveform(self)
        return self

    def calculate_start_index(self,frame_length):
        max_start = int(31*frame_length/32 - len(self.waveform)) 
        min_start = int(frame_length/32)
        self.start_index = random.randint(min_start, int(max_start))
        return self

    def update_position(self,time):
        self.current_position = self.start_position + self.velocity*time
        return self

    def locking_to_a_target(self, locked_emitter):
        if isinstance(self.locked_target, Emitter):
            self.locked_target_position = locked_emitter.current_position
        elif isinstance(self.locked_target, np.ndarray):
            self.locked_target_position = self.locked_target
        return self

    def power_control(self):
        wavelength = c / self.fc  # Speed of light / carrier frequency
        distance_to_target = np.linalg.norm(self.current_position - self.locked_target_position)
        path_loss_linear = (4 * np.pi * distance_to_target / wavelength) ** 4
        linear_SNR_aim = 10**(self.SNR_aim / 10)
        required_signal_power = linear_SNR_aim * self.current_noise_power * path_loss_linear / self.antenna_gain
        self.amplitude = max (np.sqrt(required_signal_power),self.amplitude_range[1])
        return self

    def antenna_scanning_gain_calculator(self):
        self.antenna_gain = self.antenna_gain_calculator(self.current_position, self.locked_target_position)
        return self
    
    def radiation_pattern_calculator(self):
        if self.Antenna_Scanning_Type == "deneme":
            elevation = np.linspace(-90, 90, 180)
            azimuth = np.linspace(-180, 180, 360)
            el_grid, az_grid = np.meshgrid(elevation, azimuth, indexing="ij")

            sigma_el = 15.0
            sigma_az = 15.0
            norm = 1.0 / (2.0 * np.pi * sigma_el * sigma_az)
            pdf = norm * np.exp(
                -0.5 * ((el_grid / sigma_el) ** 2 + (az_grid / sigma_az) ** 2)
            )

            self.antenna_radiation_pattern = np.column_stack(
                (el_grid.ravel(), az_grid.ravel(), pdf.ravel())
            )
        return self
        

    def check_parameter_limits_valid(self):
        max_waveform_length=self.number_of_pulse_range[1]*self.PRI_mean_range[1]+self.PW_range[1]
        if max_waveform_length>self.frame_duration:
            print("Waveform length is greater than frame duration")
        min_waveform_length=self.PW_range[0]*self.number_of_pulse_range[0]*self.PRI_mean_range[0]
        if min_waveform_length<0:
            print("Waveform length is less than frame duration")
        min_chip_duration=self.PW_range[0]/50
        print(f"Minimum chip duration is {min_chip_duration*self.fs} sample")

        max_frequency=self.fc_range[1]+self.wave_param_B_range[1]
        if max_frequency>self.fs/2:
            print("Maximum frequency is greater than Nyquist frequency")
        
        max_frequency=self.fc_range[1]+20/self.PW_range[0]
        if max_frequency>self.fs/2:
            print("Maximum frequency is greater than Nyquist frequency")
        return 0

