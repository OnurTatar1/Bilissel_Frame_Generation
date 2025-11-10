import numpy as np

class Pulse:
    def __init__(self, born_time, pulse_width, f_c, wave_type, AoA, Emitter_ID, Amplitude, received_by=None):
        self.born_time = born_time
        self.pulse_width = pulse_width
        self.f_c = f_c
        self.wave_type = wave_type
        self.AoA = AoA
        self.Emitter_ID = Emitter_ID
        self.Amplitude = Amplitude
        self.received_by = received_by if received_by is not None else np.array([])

# Example usage:
if __name__ == "__main__":
    # Example 1: Create a pulse with all parameters
    pulse1 = Pulse(
        born_time=0.1,
        pulse_width=1e-5,
        f_c=100e3,
        wave_type='barker',
        AoA=45.0,
        Emitter_ID=1,
        Amplitude=1.0,
        received_by=np.array([1, 2, 3])  # Array of receiver IDs
    )
    
