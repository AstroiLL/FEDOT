import random

import matplotlib.pyplot as plt
import numpy as np

from utilities.ts_gapfilling import SimpleGapFiller, ModelGapFiller


def generate_synthetic_data(length: int = 2500, gap_size: int = 100,
                            gap_value: float = -100.0):
    """
    The function generates a synthetic one-dimensional array with omissions

    :param length: the length of the array (should be more than 1000)
    :param gap_size: number of elements in the gap
    :param gap_value: value, which identify gap elements in array
    :return: an array with gaps
    """

    sinusoidal_data = np.linspace(-6 * np.pi, 6 * np.pi, length)
    sinusoidal_data = np.sin(sinusoidal_data)
    random_noise = np.random.normal(loc=0.0, scale=0.1, size=length)

    # Combining a sine wave and random noise
    synthetic_data = sinusoidal_data + random_noise

    random_value = random.randint(500, length - 500)
    synthetic_data[random_value: (random_value + gap_size)] = gap_value
    return synthetic_data


# Example of applying the algorithm
if __name__ == '__main__':
    # Get synthetic time series
    tmp_data = generate_synthetic_data()

    # Filling in gaps
    gapfiller = ModelGapFiller(gap_value=-100.0)
    without_gap_arr = gapfiller.inverse_ridge(tmp_data, max_window_size=400)

    simple_gapfill = SimpleGapFiller(gap_value=-100.0)
    without_gap_arr_poly = \
        simple_gapfill.local_poly_approximation(tmp_data, 4, 150)

    masked_array = np.ma.masked_where(tmp_data == -100.0, tmp_data)
    plt.plot(without_gap_arr_poly, c='orange',
             alpha=0.5, label='Polynomial approximation')
    plt.plot(without_gap_arr, c='red',
             alpha=0.5, label='Inverse ridge')
    plt.plot(masked_array, c='blue', alpha=1.0, label='Actual values')
    plt.legend()
    plt.grid()
    plt.show()
