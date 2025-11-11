import numpy as np
import matplotlib.pyplot as plt

# Environment to clutter model mapping
ENVIRONMENT_MODELS = {
    # Sea environments
    'open_sea': 'sirp',
    'coastal_waters': 'weibull',
    'rough_sea': 'lognormal',
    
    # Land environments
    'agricultural': 'gamma',
    'forest': 'k_distribution',
    'urban': 'rayleigh',
    'mountain': 'k_distribution',
    'desert': 'gamma',
    'grassland': 'gamma'
}

# Clutter RCS (Radar Cross Section) per unit area for different environments
# Units: m²/m² (dimensionless, represents reflectivity)
CLUTTER_RCS_PER_UNIT_AREA = {
    'open_sea': 0.001,      # Low reflectivity for calm sea
    'coastal_waters': 0.01,  # Higher reflectivity due to waves
    'rough_sea': 0.1,       # High reflectivity for rough sea
    
    'agricultural': 0.05,    # Moderate reflectivity for farmland
    'forest': 0.1,          # High reflectivity for trees
    'urban': 0.5,           # Very high reflectivity for buildings
    'mountain': 0.2,        # High reflectivity for rocky terrain
    'desert': 0.02,         # Low reflectivity for sand
    'grassland': 0.03       # Low reflectivity for grass
}

def calculate_clutter_power(radar_params, range_km, environment):
    """
    Calculate clutter power based on radar parameters and environment.
    Explicitly uses carrier frequency, bandwidth, and wavelength.
    
    Parameters:
    radar_params : dict
        Dictionary containing radar parameters:
        - frequency: Carrier frequency in Hz
        - bandwidth: Bandwidth in Hz
        - tx_power: Transmitter power in Watts
        - tx_gain: Transmitter antenna gain in dB
        - rx_gain: Receiver antenna gain in dB
        - beamwidth_az: Azimuth beamwidth in degrees
        - beamwidth_el: Elevation beamwidth in degrees
        - grazing_angle: Grazing angle in degrees
    range_km : float
        Range to clutter in kilometers
    environment : str
        Environment type (e.g., 'open_sea', 'urban', 'forest')
    
    Returns:
    float
        Clutter power in Watts
    """
    # Extract radar parameters
    frequency = radar_params['frequency']
    bandwidth = radar_params['bandwidth']
    tx_power = radar_params['tx_power']
    tx_gain_linear = 10**(radar_params['tx_gain']/10)
    rx_gain_linear = 10**(radar_params['rx_gain']/10)
    beamwidth_az = np.radians(radar_params['beamwidth_az'])
    beamwidth_el = np.radians(radar_params['beamwidth_el'])
    grazing_angle = np.radians(radar_params['grazing_angle'])
    
    # Calculate wavelength from carrier frequency
    c = 3e8
    wavelength = c / frequency
    
    # Calculate range in meters
    range_m = range_km * 1000
    
    # Calculate range resolution using bandwidth
    range_resolution = c / (2 * bandwidth)
    
    # Calculate clutter area (range cell area)
    # Clutter area = range resolution × azimuth beamwidth × range / sin(grazing_angle)
    clutter_area = range_resolution * beamwidth_az * range_m / np.sin(grazing_angle)
    
    # Get clutter RCS per unit area for the environment
    sigma0 = CLUTTER_RCS_PER_UNIT_AREA.get(environment, 0.01)
    
    # Calculate total clutter RCS
    clutter_rcs = sigma0 * clutter_area
    
    # Calculate clutter power using radar equation (explicitly using wavelength)
    # P_clutter = (P_tx * G_tx * G_rx * λ² * σ_clutter) / ((4π)³ * R⁴)
    clutter_power = (tx_power * tx_gain_linear * rx_gain_linear * 
                    wavelength**2 * clutter_rcs) / ((4 * np.pi)**3 * range_m**4)
    
    return clutter_power

def get_clutter_model_with_radar_params(environment, n_samples, radar_params, range_km, **params):
    """
    Generate clutter based on radar parameters and environment.
    
    Parameters:
    environment : str
        Type of environment (e.g., 'agricultural', 'urban', 'open_sea')
    n_samples : int
        Number of samples to generate
    radar_params : dict
        Dictionary containing radar parameters (see calculate_clutter_power)
    range_km : float
        Range to clutter in kilometers
    **params : dict
        Additional parameters for the specific clutter model
    
    Returns:
    array-like
        Clutter samples with proper power scaling
    """
    if environment not in ENVIRONMENT_MODELS:
        raise ValueError(f"Unknown environment: {environment}. Available environments: {list(ENVIRONMENT_MODELS.keys())}")
    
    # Generate base clutter using the appropriate model
    model_type = ENVIRONMENT_MODELS[environment]
    
    if model_type == 'sirp':
        base_clutter = generate_sirp_clutter(n_samples, 
                                           params.get('shape_param', 1.5),
                                           params.get('scale_param', 0.5))
    
    elif model_type == 'weibull':
        base_clutter = generate_weibull_clutter(n_samples,
                                              params.get('shape_param', 1.2),
                                              params.get('scale_param', 0.8))
    
    elif model_type == 'lognormal':
        base_clutter = generate_lognormal_clutter(n_samples,
                                                params.get('mean_param', 0.0),
                                                params.get('sigma_param', 0.5))
    
    elif model_type == 'k_distribution':
        base_clutter = generate_k_distribution_clutter(n_samples,
                                                     params.get('shape_param', 2.0),
                                                     params.get('scale_param', 1.0))
    
    elif model_type == 'gamma':
        base_clutter = generate_gamma_clutter(n_samples,
                                            params.get('shape_param', 2.5),
                                            params.get('scale_param', 0.8))
    
    elif model_type == 'rayleigh':
        base_clutter = generate_rayleigh_clutter(n_samples,
                                               params.get('scale_param', 1.0))
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Calculate desired clutter power based on radar parameters
    desired_clutter_power = calculate_clutter_power(radar_params, range_km, environment)
    
    # Calculate current clutter power
    current_clutter_power = np.mean(np.abs(base_clutter)**2)
    
    # Scale clutter to match desired power
    if current_clutter_power > 0:
        scale_factor = np.sqrt(desired_clutter_power / current_clutter_power)
        scaled_clutter = base_clutter * scale_factor
    else:
        scaled_clutter = base_clutter
    
    return scaled_clutter

def get_clutter_model(environment, n_samples, **params):
    """
    Generate clutter based on the specified environment.
    (Legacy function - kept for backward compatibility)
    
    Parameters:
    environment : str
        Type of environment (e.g., 'agricultural', 'urban', 'open_sea')
    n_samples : int
        Number of samples to generate
    **params : dict
        Additional parameters for the specific clutter model
    
    Returns:
    array-like
        Clutter samples for the specified environment
    """
    if environment not in ENVIRONMENT_MODELS:
        raise ValueError(f"Unknown environment: {environment}. Available environments: {list(ENVIRONMENT_MODELS.keys())}")
    
    model_type = ENVIRONMENT_MODELS[environment]
    
    if model_type == 'sirp':
        return generate_sirp_clutter(n_samples, 
                                   params.get('shape_param', 1.5),
                                   params.get('scale_param', 0.5))
    
    elif model_type == 'weibull':
        return generate_weibull_clutter(n_samples,
                                      params.get('shape_param', 1.2),
                                      params.get('scale_param', 0.8))
    
    elif model_type == 'lognormal':
        return generate_lognormal_clutter(n_samples,
                                        params.get('mean_param', 0.0),
                                        params.get('sigma_param', 0.5))
    
    elif model_type == 'k_distribution':
        return generate_k_distribution_clutter(n_samples,
                                             params.get('shape_param', 2.0),
                                             params.get('scale_param', 1.0))
    
    elif model_type == 'gamma':
        return generate_gamma_clutter(n_samples,
                                    params.get('shape_param', 2.5),
                                    params.get('scale_param', 0.8))
    
    elif model_type == 'rayleigh':
        return generate_rayleigh_clutter(n_samples,
                                       params.get('scale_param', 1.0))
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def generate_sirp_clutter(n_samples, shape_param, scale_param):
    """
    Generate sea clutter using the SIRP model.
    
    Parameters:
    n_samples : int
        Number of samples to generate
    shape_param : float
        Shape parameter (ν) of the gamma distribution
    scale_param : float
        Scale parameter (b) of the gamma distribution
    
    Returns:
    array-like
        SIRP clutter samples
    """
    # Generate the texture component (amplitude)
    texture = np.random.gamma(shape_param, scale_param, n_samples)
    
    # Generate the speckle component (phase)
    speckle_real = np.random.normal(0, 1, n_samples)
    speckle_imag = np.random.normal(0, 1, n_samples)
    speckle = speckle_real + 1j * speckle_imag
    
    # Combine texture and speckle to form SIRP clutter
    clutter = np.sqrt(texture) * speckle
    return clutter

def generate_weibull_clutter(n_samples, shape_param, scale_param):
    """
    Generate sea clutter using the Weibull distribution.
    
    Parameters:
    n_samples : int
        Number of samples to generate
    shape_param : float
        Shape parameter (k) of the Weibull distribution
    scale_param : float
        Scale parameter (λ) of the Weibull distribution
    
    Returns:
    array-like
        Weibull clutter samples
    """
    # Generate Weibull distributed amplitude
    amplitude = scale_param * np.random.weibull(shape_param, n_samples)
    
    # Generate random phase
    phase = 2 * np.pi * np.random.rand(n_samples)
    
    # Combine amplitude and phase
    clutter = amplitude * np.exp(1j * phase)
    
    return clutter

def generate_lognormal_clutter(n_samples, mean_param, sigma_param):
    """
    Generate sea clutter using the Log-normal distribution.
    
    Parameters:
    n_samples : int
        Number of samples to generate
    mean_param : float
        Mean parameter (μ) of the log-normal distribution
    sigma_param : float
        Standard deviation parameter (σ) of the log-normal distribution
    
    Returns:
    array-like
        Log-normal clutter samples
    """
    # Generate log-normal distributed amplitude
    amplitude = np.random.lognormal(mean_param, sigma_param, n_samples)
    
    # Generate random phase
    phase = 2 * np.pi * np.random.rand(n_samples)
    
    # Combine amplitude and phase
    clutter = amplitude * np.exp(1j * phase)
    
    return clutter

def generate_k_distribution_clutter(n_samples, shape_param, scale_param):
    """
    Generate land clutter using the K-distribution.
    Good for modeling heterogeneous terrain and forest clutter.
    
    Parameters:
    n_samples : int
        Number of samples to generate
    shape_param : float
        Shape parameter (ν) of the K-distribution
    scale_param : float
        Scale parameter (b) of the K-distribution
    
    Returns:
    array-like
        K-distribution clutter samples
    """
    # Generate the texture component (amplitude)
    texture = np.random.gamma(shape_param, scale_param, n_samples)
    
    # Generate the speckle component (phase)
    speckle_real = np.random.normal(0, 1, n_samples)
    speckle_imag = np.random.normal(0, 1, n_samples)
    speckle = speckle_real + 1j * speckle_imag
    
    # Combine texture and speckle to form K-distribution clutter
    clutter = np.sqrt(texture) * speckle
    
    return clutter

def generate_gamma_clutter(n_samples, shape_param, scale_param):
    """
    Generate land clutter using the Gamma distribution.
    Good for modeling homogeneous terrain and agricultural areas.
    
    Parameters:
    n_samples : int
        Number of samples to generate
    shape_param : float
        Shape parameter (k) of the Gamma distribution
    scale_param : float
        Scale parameter (θ) of the Gamma distribution
    
    Returns:
    array-like
        Gamma clutter samples
    """
    # Generate Gamma distributed amplitude
    amplitude = np.random.gamma(shape_param, scale_param, n_samples)
    
    # Generate random phase
    phase = 2 * np.pi * np.random.rand(n_samples)
    
    # Combine amplitude and phase
    clutter = amplitude * np.exp(1j * phase)
    
    return clutter

def generate_rayleigh_clutter(n_samples, scale_param):
    """
    Generate land clutter using the Rayleigh distribution.
    Good for modeling urban areas and man-made structures.
    
    Parameters:
    n_samples : int
        Number of samples to generate
    scale_param : float
        Scale parameter (σ) of the Rayleigh distribution
    
    Returns:
    array-like
        Rayleigh clutter samples
    """
    # Generate Rayleigh distributed amplitude
    amplitude = scale_param * np.sqrt(-2 * np.log(np.random.rand(n_samples)))
    
    # Generate random phase
    phase = 2 * np.pi * np.random.rand(n_samples)
    
    # Combine amplitude and phase
    clutter = amplitude * np.exp(1j * phase)
    
    return clutter 