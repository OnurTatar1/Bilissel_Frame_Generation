import numpy as np

class PulseTrain:
    def __init__(self, emitter_id, pulse_list=None):
        """
        Initialize a PulseTrain instance.
        
        Parameters:
        - emitter_id (int): ID of the emitter that generated this pulse train
        - pulse_list (list): List of Pulse objects in this train
        """
        self.emitter_id = emitter_id
        self.pulse_list = pulse_list if pulse_list is not None else []
    
    def add_pulse(self, pulse):
        self.pulse_list.append(pulse)

# Example usage:
if __name__ == "__main__":
    from pulse import Pulse

    pulse1 = Pulse(
        born_time=0.1,
        pulse_width=1e-5,
        f_c=100e3,
        wave_type='barker',
        AoA=45.0,
        Emitter_ID=1,
        Amplitude=1.0,
        received_by=np.array([1, 2])
    )
    
    pulse2 = Pulse(
        born_time=0.2,
        pulse_width=1e-5,
        f_c=100e3,
        wave_type='barker',
        AoA=45.0,
        Emitter_ID=1,
        Amplitude=1.0,
        received_by=np.array([1, 2])
    )
    
    pulse3 = Pulse(
        born_time=0.3,
        pulse_width=1e-5,
        f_c=100e3,
        wave_type='barker',
        AoA=45.0,
        Emitter_ID=1,
        Amplitude=1.0,
        received_by=np.array([1, 2])
    )

    train1 = PulseTrain(emitter_id=1, pulse_list=[pulse1, pulse2, pulse3])
    

    train2 = PulseTrain(emitter_id=2)
    
    # Add pulses one by one
    for i in range(3):
        new_pulse = Pulse(
            born_time=0.1 + i * 0.1,
            pulse_width=2e-5,
            f_c=200e3 + i * 50e3,
            wave_type='costas',
            AoA=30.0 + i * 10.0,
            Emitter_ID=2,
            Amplitude=2.0,
            received_by=np.array([3, 4])
        )
        train2.add_pulse(new_pulse)
    