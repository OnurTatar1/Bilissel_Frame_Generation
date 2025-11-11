from emitter_class import Emitter
import numpy as np
def assign_emitters(fs,frame_duration):
    """
    Create a cognitive environment with 5 emitters of different complexities.
    
    Returns:
    - list: List of Emitter objects with varying complexity levels
    """
    
    # Emitter 1: Complexity 3
    emitter_1 = Emitter(
        # Identification & basic configuration
        emitter_complexity=3,
        frame_duration=frame_duration,
        fs=fs,
        ID=0,

        # Pulse Params
        amplitude_range=[5.0, 12.0],
        fc_range=[50e3, 7e6],
        number_of_pulse_range=[5, 5],
        PW_range=[1e-5, 3e-5],
        wave_param_B_range=[1e5, 2.8e6],
        wave_type_range=['barker', 'costas', 'frank', 'lfm', 'p1', 'p2', 'p3', 'p4', 'rect'],

        # PRI Params
        PRI_agile_ampl_range=[0.2, 0.6],
        PRI_mean_range=[1e-3, 5e-4],
        PRI_type_range=['staggered', 'stable', 'jittered', 'wobulated', 'sliding+', 'sliding-'],

        # Motion & tracking
        start_position=[3e3, 3e3, 0],
        velocity=[0, 0, 0],
    )

    # Emitter 2: Complexity 2   (uçak acqusition)
    emitter_2 = Emitter(
        # Identification & basic configuration
        emitter_complexity=2,
        frame_duration=frame_duration,
        fs=fs,
        ID=1,

        # Pulse Params
        amplitude_range=[3.0, 9.0],
        fc_range=[80e3, 250e3],
        number_of_pulse_range=[2, 5],
        PW_range=[2e-5, 4e-5],
        wave_param_B_range=[1e4, 1e5],
        wave_type_range=['rect', 'barker', 'frank', 'lfm'],

        # PRI Params
        PRI_agile_ampl_range=[0.1, 0.4],
        PRI_mean_range=[8e-4, 3e-4],
        PRI_type_range=['stable', 'staggered', 'wobulated', 'sliding+'],
        

        # Motion & tracking
        start_position=[3e3, 3e3 * np.sqrt(3), 500],
        velocity=[0, 0, 0],
    )

    # Emitter 3: Complexity 2  (uçak acqusition)
    emitter_3 = Emitter(
        # Identification & basic configuration
        emitter_complexity=2,
        frame_duration=frame_duration,
        fs=fs,
        ID=2,

        # Pulse Params
        amplitude_range=[4.0, 10.0],
        fc_range=[4e6, 6e6],
        number_of_pulse_range=[2, 6],
        PW_range=[1.5e-5, 3.5e-5],
        wave_param_B_range=[9e5, 9e5],
        wave_type_range=['rect', 'p1', 'p2', 'costas'],

        # PRI Params
        PRI_agile_ampl_range=[0.15, 0.5],
        PRI_mean_range=[6e-4, 2e-4],
        PRI_type_range=['stable', 'staggered', 'wobulated', 'sliding+'],
        
        # Motion & tracking
        start_position=[3e3 * np.sqrt(3), 3e3, 1000],
        velocity=[0, 0, 0],
    )

    # Emitter 4: Complexity 1  (early warning- Gemi)
    emitter_4 = Emitter(
        # Identification & basic configuration
        emitter_complexity=1,
        frame_duration=frame_duration,
        fs=fs,
        ID=3,

        # Pulse Params
        amplitude_range=[0.5, 15],
        fc_range=[4e6, 5e6],
        number_of_pulse_range=[1, 4],
        PW_range=[3e-5, 5e-5],
        wave_param_B_range=[5e5, 5e5],
        wave_type_range=['frank', 'barker', 'rect'],

        # PRI Params
        PRI_agile_ampl_range=[0.05, 0.2],
        PRI_mean_range=[1.2e-3, 4e-4],
        PRI_type_range=['stable', 'staggered'],
        
        # Motion & tracking
        start_position=[3e3 * (np.sqrt(3) + 2), 3e3, 0],
        velocity=[0, 0, 0],
    )

    # Emitter 5: Complexity 1  (early warning Gemi)
    emitter_5 = Emitter(
        # Identification & basic configuration
        emitter_complexity=1,
        frame_duration=frame_duration,
        fs=fs,
        ID=4,

        # Pulse Params
        amplitude_range=[0.5, 1.5],
        fc_range=[3e6, 7e6],
        number_of_pulse_range=[1, 4],
        PW_range=[2.5e-5, 4.5e-5],
        wave_param_B_range=[6e5, 7e5],
        wave_type_range=['lfm', 'rect', 'p2'],

        # PRI Params
        PRI_agile_ampl_range=[0.1, 0.5],
        PRI_mean_range=[9e-4, 3e-4],
        PRI_type_range=['stable', 'sliding+'],

        # Motion & tracking
        start_position=[3e3, 3e3 * (np.sqrt(3) + 2), 0],
        velocity=[0, 0, 0],
    )
    emitters = [emitter_1, emitter_2, emitter_3, emitter_4, emitter_5]
    emitters[0].locked_target= [0, 0, 0]
    print("Emitter 1 parameter validation:")
    emitters[0].check_parameter_limits_valid()

    emitters[1].locked_target= [0, 0, 0]
    print("Emitter 2 parameter validation:")
    emitters[1].check_parameter_limits_valid()

    emitters[2].locked_target= [0, 0, 0]
    print("Emitter 3 parameter validation:")
    emitters[2].check_parameter_limits_valid()
    
    emitters[3].locked_target= [0, 0, 0]
    print("Emitter 4 parameter validation:")
    emitters[3].check_parameter_limits_valid()

    emitters[4].locked_target= [0, 0, 0]
    print("Emitter 5 parameter validation:")
    emitters[4].check_parameter_limits_valid()
    return emitters

