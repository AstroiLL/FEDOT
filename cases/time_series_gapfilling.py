import numpy as np
import pandas as pd

from utilities.ts_gapfilling import AdvancedGapfiller
import numpy as np
import pandas as pd

from utilities.ts_gapfilling import AdvancedGapfiller


def check_metrics():
    pass


def plot_result()
    pass


# Example of applying the algorithm
if __name__ == '__main__':
    # Load dataframe
    dataframe = pd.read_csv('./data/gapfilling/TS_temperature_gapfilling.csv')
    dataframe['Date'] = pd.to_datetime(dataframe['Date'])

    Gapfiller = AdvancedGapfiller(gap_value=-100.0)
    withoutgap_arr = Gapfiller.inverse_ridge(np.array(dataframe['With_gap']), max_window_size=200)
