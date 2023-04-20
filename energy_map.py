import numpy as np
from energy_functions import *

def compute_vertical_cumulative_energy_map(img, energy_type='magnitude', k=3):
    
    if energy_type == 'magnitude':
        energy = compute_magnitude(img, k=k)
    elif energy_type == 'laplacian':
        energy = compute_laplacian(img, k=k)
        
    # Compute the cumulative minimum energy map using dynamic programming
    h, w = energy.shape[:2]
    cumulative_energy_map = np.zeros_like(energy)
    cumulative_energy_map[0] = energy[0]
    for i in range(1, h):
        for j in range(w):
            if j == 0:
                cumulative_energy_map[i, j] = energy[i, j] + min(cumulative_energy_map[i-1, j], cumulative_energy_map[i-1, j+1])
            elif j == w-1:
                cumulative_energy_map[i, j] = energy[i, j] + min(cumulative_energy_map[i-1, j], cumulative_energy_map[i-1, j-1])
            else:
                cumulative_energy_map[i, j] = energy[i, j] + min(cumulative_energy_map[i-1, j-1], cumulative_energy_map[i-1, j], cumulative_energy_map[i-1, j+1])
    
    return cumulative_energy_map


def compute_horizontal_cumulative_energy_map(img, energy_type='magnitude', k=3):
    
    img = np.rot90(img, k=1)
    cumulative_energy_map = compute_vertical_cumulative_energy_map(img, energy_type=energy_type, k=k)
    return np.rot90(cumulative_energy_map, k=3)
    