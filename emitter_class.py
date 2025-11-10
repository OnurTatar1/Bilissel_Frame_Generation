
import random
import numpy as np
global c
c= 299792458
class Emitter:
    def __init__(self,ID=None, wave_type=None, emitter_complexity=None, jamming_library=None, PRI_range=None, PW=None,
                 number_of_pulse=None, PRI_type=None, PRI_agile_ampl=None, amplitude=None, 
                 wave_param_L=None, wave_param_B=None, fc=None, fs=None, frame_duration=None,
                 number_of_pulse_range=None, PRI_agile_ampl_range=None, 
                 amplitude_range=None, fc_range=None, PRI_mean=None,band_width_range=None,start_position=None,velocity=None,locked_target=None,
                 antenna_gain=None,SNR_aim=None,current_noise_power=None, Antenna_Scanning_Type=None,
                 Antenna_Scanning_step=None, Antenna_Scanning_sector_Azimuth=None,Antenna_Scanning_sector_elevation=None):
        self.number_of_pulse_range = number_of_pulse_range
        self.PRI_agile_ampl_range = PRI_agile_ampl_range
        self.amplitude_range = amplitude_range
        self.fc_range = fc_range
        self.fs = fs
        self.frame_duration = frame_duration
        self.ID = ID
        self.wave_type = wave_type
        self.emitter_complexity = emitter_complexity
        self.jamming_library = jamming_library
        self.PRI_range = PRI_range
        self.PW = PW
        self.number_of_pulse = number_of_pulse
        self.PRI_type = PRI_type
        self.PRI_agile_ampl = PRI_agile_ampl
        self.amplitude = amplitude
        self.wave_param_L = wave_param_L
        self.wave_param_B = wave_param_B
        self.fc = fc
        self.fs = fs
        self.frame_duration = frame_duration
        self.PRI_mean = PRI_mean
        self.band_width_range = band_width_range
        self.start_position = start_position
        self.velocity = velocity
        self.locked_target = locked_target
        self.locked_target_position = None
        self.antenna_gain = antenna_gain
        self.SNR_aim = SNR_aim
        self.Antenna_Scanning_Type = Antenna_Scanning_Type
        self.Antenna_Scanning_step = Antenna_Scanning_step
        self.Antenna_Scanning_sector_Azimuth = Antenna_Scanning_sector_Azimuth
        self.Antenna_Scanning_sector_elevation = Antenna_Scanning_sector_elevation
        self.antenna_radiation_pattern 


    def select_waveform_parameters(self):
        self.wave_type = random.choice(self.wave_type)
        self.pulse_width = random.uniform(self.PW[0], self.PW[1])
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

        if self.wave_type in ["rect", "barker", "p3", "p4"]:
            self.wave_param_B = self.wave_param_L / self.pulse_width
        elif self.wave_type in ["frank", "p1", "p2"]:
            self.wave_param_B = self.wave_param_L * self.wave_param_L / self.pulse_width
        elif self.wave_type == "costas":
            self.wave_param_B = random.uniform(self.band_width_range[0], self.band_width_range[1])
        else:
            self.wave_param_B = random.uniform(self.band_width_range[0], self.band_width_range[1])
        return self
    
    def get_carrier_frequency_range(self):
        max_fc = self.fc_range[1] - self.wave_param_B   # Maximum f_c to avoid aliasing
        min_fc = self.fc_range[0]  # Minimum f_c to keep signal in positive frequencies
        fc = random.uniform(min_fc, max_fc)
        self.fc = fc
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
# Example usage:
if __name__ == "__main__":
    # Example 1: Create emitter with custom range parameters

    my_emitter = Emitter(
        ID=1,
        wave_type=['rect', 'p2'], 
        emitter_complexity=5,  # Fixed integer value
        jamming_library='barrage', 
        PRI_range=[1e-3, 1e-4], 
        PW=[1e-5, 5e-5],
        number_of_pulse_range=[3, 6],
        PRI_agile_ampl_range=[0.1, 0.5],
        amplitude_range=[0.5, 1.5],
        fc_range=[50e3, 500e3],
        fs=2e7,
        frame_duration=0.2,
        band_width_range=[1e6, 10e6]
    )
   