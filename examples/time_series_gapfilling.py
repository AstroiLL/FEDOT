import random

import matplotlib.pyplot as plt
import numpy as np

from utilities.gapfilling import SimpleGapfiller, AdvancedGapfiller


def simulated_data(length: int = 2000, gap_size: int = 100, gap_value: float = -100.0):
    """
    The function generates a synthetic one-dimensional array with omissions

    :param length: the length of the array (should be more than 500)
    :param gap_size: number of elements in the gap
    :param gap_value: value, which identify gap elements in array
    :return: an array with gaps
    """

    sinusoidal_data = np.linspace(-5 * np.pi, 5 * np.pi, length)
    sinusoidal_data = np.sin(sinusoidal_data)
    simulated_noise = np.random.normal(loc=0.0, scale=0.1, size=length)

    # Combining a sine wave and random noise
    simulated_data = sinusoidal_data + simulated_noise

    random_value = random.randint(500, length - (gap_size + 1))
    simulated_data[random_value: (random_value + gap_size)] = gap_value
    return simulated_data


# Example of applying the algorithm
if __name__ == '__main__':
    # Get synthetic time series
    tmp_data = simulated_data()

    # Filling in gaps
    Gapfiller = SimpleGapfiller(gap_value=-100.0)
    withoutgap_arr = Gapfiller.composite_fill_gaps(tmp_data, max_window_size=500)

    SimpleGapfill = AdvancedGapfiller(gap_value=-100.0)
    withoutgap_arr_poly = SimpleGapfill.local_poly_approximation(tmp_data, 4, 150)

    plt.plot(withoutgap_arr, c='blue', alpha=0.5)
    plt.plot(withoutgap_arr_poly, c='orange', alpha=0.4)
    plt.grid()
    plt.show()
