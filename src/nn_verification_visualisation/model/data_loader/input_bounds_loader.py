from typing import Dict, List, Iterable

from nn_verification_visualisation.model.data.network_verification_config import NetworkVerificationConfig
from utils.result import Result, Failure, Success

import csv
from nn_verification_visualisation.utils.result import Result
from nn_verification_visualisation.utils.singleton import SingletonMeta

from nn_verification_visualisation.model.data.neural_network import NeuralNetwork
from nn_verification_visualisation.model.data.input_bounds import InputBounds

class InputBoundsLoader(metaclass=SingletonMeta):
    def load_input_bounds(self, file_path: str, network_config: NetworkVerificationConfig) -> Result[InputBounds]:
        ending = file_path.split('.')[-1]
        is_csv = (ending == 'csv')
        is_vnnlib = (ending == 'vnnlib')

        if not (is_csv or is_vnnlib):
            return Failure(ValueError(ending + ' is not supported. Please use a .csv or a .vnnlib file.'))

        if network_config is None or len(network_config.activation_values) < 1:
            return Failure(ValueError("Invalid network passed"))

        input_count = network_config.activation_values[0]

        if is_csv:
            try:
                return self.__parse_csv(file_path, input_count)
            except BaseException as e:
                return Failure(e)
        elif is_vnnlib:
            return Failure(NotImplementedError("vnnlib is not yet supported."))

    def __get_input_count(self, network_config: NetworkVerificationConfig) -> int:
        return network_config.bounds[0]

    def __parse_csv(self, file_path: str, input_count: int) -> Result[InputBounds]:
        rows = []
        try:
            with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)

                try:
                    header = next(reader)
                except StopIteration:
                    return Failure(ValueError(f"{file_path} is empty."))
                for row in reader:
                    rows.append([cell.strip() for cell in row])
        except OSError as e:
            return Failure(e)

        field_count = len(header)

        # checking whether the field count is valid
        if field_count not in (2, 3):
            return Failure(ValueError(file_path + 'needs to be consistently organized in two or three columns.'))

        # checking whether the fields match the format of the rows
        if any([len(row) != field_count for row in rows]):
            return Failure(ValueError(f"{file_path} must have {field_count} columns in every row."))

        if len(rows) != input_count:
            return Failure(ValueError(
                f"{file_path} has {str(len(rows))} rows. It needs the same number of inputs as the network ({input_count})"))

        bounds : Dict[int, tuple[float, float]] = {}
        enumeration : List[tuple[int, List[str]]] = []
        # allow custom ordering of bounds
        if field_count == 3:
            parsed_rows = []
            for i, row in enumerate(rows):
                try:
                    position = int(row[0])
                except ValueError:
                    return Failure(ValueError(f"Index at row {i + 2} is not an integer."))

                parsed_rows.append((position, row[1:]))

            indices = [p[0] for p in parsed_rows]
            if sorted(indices) != list(range(input_count)):
                return Failure(
                    ValueError("Every index in the csv file has to appear exactly once and be in range 0..N-1."))
            enumeration = parsed_rows
        else:
            enumeration = [(i, row) for i, row in enumerate(rows)]

        for i, row in enumeration:
            try:
                lower_bound, upper_bound = (float(row[0]), float(row[1]))
            except ValueError:
                return Failure(ValueError(f"Item at value {i} contains an argument that is not an integer."))

            if lower_bound > upper_bound:
                return Failure(ValueError(f"Lower bound {lower_bound} is greater than upper bound {upper_bound} for item {i}"))

            bounds[i] = (lower_bound, upper_bound)

        input_bounds = InputBounds(bounds)

        return Success(input_bounds)