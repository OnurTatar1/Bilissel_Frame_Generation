from emitter_class import Emitter
import numpy as np
global fs
fs=2e7
global frame_duration
frame_duration=10e-3
def assign_emitters():
    """
    Create a cognitive environment with 5 emitters of different complexities.
    
    Returns:
    - list: List of Emitter objects with varying complexity levels
    """
    
    # Emitter 1: Complexity 3
    emitter_1 = Emitter(
        ID=0,
        wave_type=['barker', 'costas', 'frank','lfm','p1','p2','p3','p4','rect'],
        emitter_complexity=3,
        jamming_library=['barrage','sweep','spot'],
        PRI_range=[1e-3, 5e-4],
        PW=[1e-5, 3e-5],
        number_of_pulse=5,
        PRI_type=['staggered', 'stable', 'jittered', 'wobulated', 'sliding+', 'sliding-'],
        PRI_agile_ampl=[0.1, 0.7],
        amplitude=8.0,
        fc_range=[50e3, 200e3],
        fs=fs,
        frame_duration=frame_duration,
        number_of_pulse_range=[3, 7],
        PRI_agile_ampl_range=[0.2, 0.6],
        amplitude_range=[5.0, 12.0],
        fc_range=[50e3, 200e3],
        band_width_range=[1e6, 10e6],
        start_position=[3e3, 3e3, 0],
        velocity=[0, 0]

    )

    # Emitter 2: Complexity 2   (uçak acqusition)
    emitter_2 = Emitter(
        ID=1,
        wave_type=['rect', 'barker', 'frank', 'lfm'],
        emitter_complexity=2,
        jamming_library=[],
        PRI_range=[8e-4, 3e-4],
        PW=[2e-5, 4e-5],
        number_of_pulse=4,
        PRI_type=['stable', 'staggered', 'wobulated', 'sliding+'],
        PRI_agile_ampl=[0.2, 0.5],
        amplitude=6.0,
        wave_param_L=5, 
        wave_param_B=8e5,
        fc_range=[50e3, 200e3],
        fs=fs,
        frame_duration=frame_duration,
        number_of_pulse_range=[2, 5],
        PRI_agile_ampl_range=[0.1, 0.4],
        amplitude_range=[3.0, 9.0],
        fc_range=[80e3, 250e3],
        band_width_range=[1e6, 10e6],
        start_position=[3e3, 3e3*np.sqrt(3), 500],
        velocity=[0, 0]
    )

    # Emitter 3: Complexity 2  (uçak acqusition)
    emitter_3 = Emitter(
        ID=2,
        wave_type=['rect', 'p1', 'p2', 'costas'],
        emitter_complexity=2,
        jamming_library=[],
        PRI_range=[6e-4, 2e-4],
        PW=[1.5e-5, 3.5e-5],
        number_of_pulse=3,
        PRI_type=['stable', 'staggered', 'wobulated', 'sliding+'],
        PRI_agile_ampl=[0.1, 0.4],
        amplitude=7.5,
        wave_param_L=6,
        wave_param_B=9e5,
        fc_range=[50e3, 200e3],
        fs=fs,
        frame_duration=frame_duration,
        number_of_pulse_range=[2, 6],
        PRI_agile_ampl_range=[0.15, 0.5],
        amplitude_range=[4.0, 10.0],
        fc_range=[100e3, 300e3],
        band_width_range=[1e6, 10e6],
        start_position=[3e3*np.sqrt(3), 3e3, 1000],
        velocity=[0, 0]
    )

    # Emitter 4: Complexity 1  (early warning- Gemi)
    emitter_4 = Emitter(
        ID=3,
        wave_type=['frank','barker','rect'],
        emitter_complexity=1,
        jamming_library=[],
        PRI_range=[1.2e-3, 4e-4],   
        PW=[3e-5, 5e-5],
        number_of_pulse=2,
        PRI_type=['stable', 'staggered'],
        PRI_agile_ampl=[0.3, 0.4],
        amplitude=4.0,
        wave_param_L=1,
        wave_param_B=5e5,
        fc_range=[50e3, 200e3],
        fs=fs,
        frame_duration=frame_duration,
        number_of_pulse_range=[1, 3],
        PRI_agile_ampl_range=[0.05, 0.2],
        amplitude_range=[0.5, 15],
        fc_range=[40e3, 150e3],
        band_width_range=[1e6, 10e6],
        start_position=[3e3*(np.sqrt(3)+2), 3e3, 0],
        velocity=[0, 0]
    )

    # Emitter 5: Complexity 1  (early warning Gemi)
    emitter_5 = Emitter(
        ID=4,
        wave_type=['lfm','rect','p2'],  
        emitter_complexity=1,
        jamming_library=[],
        PRI_range=[9e-4, 3e-4],
        PW=[2.5e-5, 4.5e-5],
        number_of_pulse=2,
        PRI_type=['stable', 'sliding+'],
        PRI_agile_ampl=[0.15, 0.25],
        amplitude=5.0,
        wave_param_L=7,
        wave_param_B=6e5,
        fc_range=[50e3, 200e3],
        fs=fs,
        frame_duration=frame_duration,
        number_of_pulse_range=[1, 4],
        PRI_agile_ampl_range=[0.1, 10],
        amplitude_range=[0.5, 1.5],
        fc_range=[60e3, 180e3],
        band_width_range=[1e6, 10e6],
        start_position=[3e3, 3e3*(np.sqrt(3)+2), 0],
        velocity=[0, 0]
    )


    emitters = [emitter_1, emitter_2, emitter_3, emitter_4, emitter_5]
    emitters[0].locked_target= emitters[1]
    emitters[1].locked_target= emitters[3]
    emitters[2].locked_target= emitters[4]
    emitters[3].locked_target= [10e3, 2e3, 0]
    emitters[4].locked_target= [10e3, 10e3, 0]
    return emitters

