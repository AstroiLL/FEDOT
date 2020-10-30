import random
import pytest

import numpy as np
from utilities.ts_gapfilling import ModelGapFiller
from sklearn.metrics import mean_squared_error

def get_synthetic_data(length: int = 2000, gap_size: int = 100, gap_value: float = -100.0):
    sinusoidal_data = np.linspace(-6 * np.pi, 6 * np.pi, length)
    sinusoidal_data = np.sin(sinusoidal_data)
    random_noise = np.random.normal(loc=0.0, scale=0.1, size=length)

    # Combining a sine wave and random noise
    synthetic_data = sinusoidal_data + random_noise

    random_value = random.randint(400, length - 400)
    real_values = np.array(synthetic_data[random_value: (random_value + gap_size)])
    synthetic_data[random_value: (random_value + gap_size)] = gap_value
    return synthetic_data, real_values

def test_gapfilling_inverse_ridge_correct():
    arr_with_gaps, real_values = get_synthetic_data(length=1000, gap_size=40, gap_value=-100.0)

    # Find all gap indices in the array
    id_gaps = np.argwhere(arr_with_gaps == -100.0)
    standard_deviation = np.std(real_values)

    gapfiller = ModelGapFiller(gap_value=-100.0)
    without_gap = gapfiller.inverse_ridge(arr_with_gaps, max_window_size=100)
    predicted_values = without_gap[id_gaps]

    rmse_test = (mean_squared_error(real_values, predicted_values)) ** 0.5
    assert rmse_test < standard_deviation