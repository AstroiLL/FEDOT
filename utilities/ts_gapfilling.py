import numpy as np
from scipy import interpolate

from core.composer.node import PrimaryNode, SecondaryNode
from core.composer.ts_chain import TsForecastingChain
from core.models.data import InputData
from core.repository.dataset_types import DataTypesEnum
from core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams


class SimpleGapFiller:
    """
    Base class used for filling in the gaps in time series with simple methods.
    Methods from the SimpleGapFiller class can be used for comparison with more complex models in class ModelGapFiller

    :param gap_value: value, which identify gap elements in array
    """

    def __init__(self, gap_value: float = -100.0):
        self.gap_value = gap_value

    def _parse_gap_ids(self, gap_list: list) -> list:
        """
        Method allows to parse source array with gaps indexes

        :param gap_list: array with indexes of gaps in array
        :return: a list with separated gaps in continuous intervals
        """

        new_gap_list = []
        local_gaps = []
        for index, gap in enumerate(gap_list):
            if index == 0:
                local_gaps.append(gap)
            else:
                prev_gap = gap_list[index - 1]
                if gap - prev_gap > 1:
                    # There is a "gap" between gaps
                    new_gap_list.append(local_gaps)

                    local_gaps = []
                    local_gaps.append(gap)
                else:
                    local_gaps.append(gap)
        new_gap_list.append(local_gaps)

        return new_gap_list

    def linear_interpolation(self, input_data):
        """
        Method allows to restore missing values in an array using linear interpolation

        :param input_data: array with gaps
        :return: array without gaps
        """

        try:
            output_data = np.array(input_data)
        except Exception:
            raise ValueError('input data should be one-dimensional array')

        # The indices of the known elements
        non_nan = np.ravel(np.argwhere(output_data != self.gap_value))
        # All known elements in the array
        masked_array = output_data[non_nan]
        f_interploate = interpolate.interp1d(non_nan, masked_array)
        x = np.arange(0, len(output_data))
        output_data = f_interploate(x)
        return output_data

    def local_poly_approximation(self, input_data, degree: int = 2, n_neighbors: int = 5):
        """
        Method allows to restore missing values in an array using Savitzky-Golay filter

        :param input_data: array with gaps
        :param degree: degree of a polynomial function
        :param n_neighbors: the number of neighboring known elements of the time series that the approximation is based on
        :return: array without gaps
        """

        try:
            output_data = np.array(input_data)
        except Exception:
            raise ValueError('input data should be one-dimensional array')

        i_gaps = np.ravel(np.argwhere(output_data == self.gap_value))

        # Iterately fill in the gaps in the time series
        for gap_index in i_gaps:
            # Indexes of known elements (updated at each iteration)
            i_known = np.argwhere(output_data != self.gap_value)
            i_known = np.ravel(i_known)

            # Based on the indexes we calculate how far from the gap the known values are located
            id_distances = np.abs(i_known - gap_index)

            # Now we know the indices of the smallest values in the array, so sort indexes
            sorted_idx = np.argsort(id_distances)
            nearest_values = []
            nearest_indices = []
            for i in sorted_idx[:n_neighbors]:
                time_index = i_known[i]
                nearest_values.append(output_data[time_index])
                nearest_indices.append(time_index)
            nearest_values = np.array(nearest_values)
            nearest_indices = np.array(nearest_indices)

            local_coefs = np.polyfit(nearest_indices, nearest_values, degree)
            est_value = np.polyval(local_coefs, gap_index)
            output_data[gap_index] = est_value

        return output_data

    def batch_poly_approximation(self, input_data, degree: int = 3, n_neighbors: int = 10):
        """
        Method allows to restore missing values in an array using batch polynomial approximations.
        Approximation is applied not for individual omissions, but for intervals of omitted values

        :param input_data: array with gaps
        :param degree: degree of a polynomial function
        :param n_neighbors: the number of neighboring known elements of time series that the approximation is based on
        :return: array without gaps
        """

        try:
            output_data = np.array(input_data)
        except Exception:
            raise ValueError('input data should be one-dimensional array')

        # Gap indices
        gap_list = np.ravel(np.argwhere(output_data == self.gap_value))
        new_gap_list = self._parse_gap_ids(gap_list)

        # Iterately fill in the gaps in the time series
        for gap in new_gap_list:
            # Find the center point of the gap
            center_index = int((gap[0] + gap[-1]) / 2)

            # Indexes of known elements (updated at each iteration)
            i_known = np.argwhere(output_data != self.gap_value)
            i_known = np.ravel(i_known)

            # Based on the indexes we calculate how far from the gap the known values are located
            id_distances = np.abs(i_known - center_index)

            # Now we know the indices of the smallest values in the array, so sort indexes
            sorted_idx = np.argsort(id_distances)

            # Nearest known values to the gap
            nearest_values = []
            # And their indexes
            nearest_indices = []
            for i in sorted_idx[:n_neighbors]:
                # Getting the index value for the series - output_data
                time_index = i_known[i]
                # Using this index, we get the value of each of the "neighbors"
                nearest_values.append(output_data[time_index])
                nearest_indices.append(time_index)
            nearest_values = np.array(nearest_values)
            nearest_indices = np.array(nearest_indices)

            # Local approximation by an n-th degree polynomial
            local_coefs = np.polyfit(nearest_indices, nearest_values, degree)

            # Estimate our interval according to the selected coefficients
            est_value = np.polyval(local_coefs, gap)
            output_data[gap] = est_value

        return output_data


class ModelGapFiller(SimpleGapFiller):
    """
    Class used for filling in the gaps in time series

    :param gap_value: value, which mask gap elements in array
    """

    def inverse_ridge(self, input_data, max_window_size: int = 50):
        """
        Method fills in the gaps in the input array

        :param input_data: data with gaps to filling in the gaps in it
        :param max_window_size: window length
        :return: array without gaps
        """

        try:
            output_data = np.array(input_data)
        except Exception:
            raise ValueError('input data should be one-dimensional array')

        # Gap indices
        gap_list = np.ravel(np.argwhere(output_data == self.gap_value))
        new_gap_list = self._parse_gap_ids(gap_list)

        # Iterately fill in the gaps in the time series
        for index, gap in enumerate(new_gap_list):

            preds = []
            weights = []
            # Two predictions are generated for each gap - forward and backward
            for prediction in ['direct', 'inverse']:

                # The entire time series is used for training until the gap
                if prediction == 'direct':
                    timeseries_train_part = output_data[:gap[0]]
                elif prediction == 'inverse':
                    if index == len(new_gap_list) - 1:
                        timeseries_train_part = output_data[(gap[-1] + 1):]
                    else:
                        next_gap = new_gap_list[index + 1]
                        timeseries_train_part = output_data[(gap[-1] + 1):next_gap[0]]
                    timeseries_train_part = np.flip(timeseries_train_part)

                # Adaptive prediction interval length
                len_gap = len(gap)
                forecast_length = len_gap

                task = Task(TaskTypesEnum.ts_forecasting,
                            TsForecastingParams(forecast_length=forecast_length,
                                                max_window_size=max_window_size,
                                                return_all_steps=True,
                                                make_future_prediction=True))

                input_data = InputData(idx=np.arange(0, len(timeseries_train_part)),
                                       features=None,
                                       target=timeseries_train_part,
                                       task=task,
                                       data_type=DataTypesEnum.ts)

                # Making predictions for the missing part in the time series
                chain = TsForecastingChain(PrimaryNode('ridge'))
                chain.fit_from_scratch(input_data)

                # "Test data" for making prediction for a specific length
                test_data = InputData(idx=np.arange(0, len_gap),
                                      features=None,
                                      target=None,
                                      task=task,
                                      data_type=DataTypesEnum.ts)

                predicted_values = chain.forecast(initial_data=input_data, supplementary_data=test_data).predict

                if prediction == 'direct':
                    weights.append(np.arange(len_gap, 0, -1))
                    preds.append(predicted_values)
                elif prediction == 'inverse':
                    predicted_values = np.flip(predicted_values)
                    weights.append(np.arange(1, (len_gap + 1), 1))
                    preds.append(predicted_values)

            preds = np.array(preds)
            weights = np.array(weights)
            result = np.average(preds, axis=0, weights=weights)

            # Replace gaps in an array with predicted values
            output_data[gap] = result

        return output_data

    def composite_fill_gaps(self, input_data, max_window_size: int = 50):

        def get_composite_chain():
            node_first = PrimaryNode('trend_data_model')
            node_second = PrimaryNode('residual_data_model')
            node_trend_model = SecondaryNode('linear', nodes_from=[node_first])
            node_residual_model = SecondaryNode('linear', nodes_from=[node_second])

            node_final = SecondaryNode('linear', nodes_from=[node_trend_model, node_residual_model])
            chain = TsForecastingChain(node_final)
            return (chain)

        try:
            output_data = np.array(input_data)
        except Exception:
            raise ValueError('input data should be one-dimensional array')

        # Gap indices
        gap_list = np.ravel(np.argwhere(output_data == self.gap_value))
        new_gap_list = self._parse_gap_ids(gap_list)

        # Iterately fill in the gaps in the time series
        for index, gap in enumerate(new_gap_list):
            # The entire time series is used for training until the gap
            timeseries_train_part = output_data[:gap[0]]

            # Adaptive prediction interval length
            len_gap = len(gap)
            forecast_length = len_gap

            # specify the task to solve
            task_to_solve = Task(TaskTypesEnum.ts_forecasting,
                                 TsForecastingParams(forecast_length=forecast_length,
                                                     max_window_size=max_window_size,
                                                     return_all_steps=True,
                                                     make_future_prediction=True))

            train_data = InputData(idx=np.arange(0, len(timeseries_train_part)),
                                   features=None,
                                   target=timeseries_train_part,
                                   task=task_to_solve,
                                   data_type=DataTypesEnum.ts)

            # "Test data" for making prediction for a specific length
            # Problem when using target array in data, if None is set, an error occurs
            test_data = InputData(idx=np.arange(0, len_gap),
                                  features=None,
                                  target=np.ones(len_gap),
                                  task=task_to_solve,
                                  data_type=DataTypesEnum.ts)

            # Chain for the task of filling in gaps
            ref_chain = get_composite_chain()
            ref_chain.fit_from_scratch(train_data)
            predicted = ref_chain.forecast(initial_data=train_data, supplementary_data=test_data).predict

            # Replace gaps in an array with predicted values
            output_data[gap] = predicted
            break
        return output_data
