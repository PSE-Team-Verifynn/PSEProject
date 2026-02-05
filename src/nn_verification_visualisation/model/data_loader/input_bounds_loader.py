import re
from logging import Logger
from pathlib import Path
from typing import Dict, List, Iterable, Any, Tuple
from venv import logger

from nn_verification_visualisation.model.data.network_verification_config import NetworkVerificationConfig
from nn_verification_visualisation.utils.result import Failure, Success

import csv
from nn_verification_visualisation.utils.result import Result
from nn_verification_visualisation.utils.singleton import SingletonMeta

class InputBoundsLoader(metaclass=SingletonMeta):
    logger = Logger(__name__)
    def load_input_bounds(self, file_path: str, network_config: NetworkVerificationConfig) -> Result[Dict[int, tuple[float, float]]]:
        ending = file_path.split('.')[-1]
        is_csv = (ending == 'csv')
        is_vnnlib = (ending == 'vnnlib')

        if not (is_csv or is_vnnlib):
            logger.error(ending + ' is not supported. Please use a .csv or a .vnnlib file.')
            return Failure(ValueError(ending + ' is not supported. Please use a .csv or a .vnnlib file.'))

        if network_config is None or len(network_config.layers_dimensions) < 1:
            return Failure(ValueError("Invalid network passed"))

        input_count = network_config.layers_dimensions[0]

        try:
            if is_csv:
                return self.__parse_csv(file_path, input_count)
            return self.__parse_vnnlib(file_path, input_count)
        except BaseException as e:
            return Failure(e)

    def __get_input_count(self, network_config: NetworkVerificationConfig) -> int:
        return network_config.bounds[0]

    def __parse_csv(self, file_path: str, input_count: int) -> Result[Dict[int, tuple[float, float]]]:
        rows = []
        try:
            with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)

                try:
                    header = next(reader)
                except StopIteration:
                    logger.error(f"{file_path} is empty.")
                    return Failure(ValueError(f"{file_path} is empty."))
                for row in reader:
                    rows.append([cell.strip() for cell in row])
        except OSError as e:
            return Failure(e)

        field_count = len(header)

        # checking whether the field count is valid
        if field_count not in (2, 3):
            logger.error(file_path + 'needs to be consistently organized in two or three columns.')
            return Failure(ValueError(file_path + 'needs to be consistently organized in two or three columns.'))

        # checking whether the fields match the format of the rows
        if any([len(row) != field_count for row in rows]):
            logger.error(f"{file_path} must have {field_count} columns in every row.")
            return Failure(ValueError(f"{file_path} must have {field_count} columns in every row."))

        if len(rows) != input_count:
            logger.error(f"{file_path} has {str(len(rows))} rows. It needs the same number of inputs as the network ({input_count})")
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
                    logger.error(f"Index at row {i + 2} is not an integer.")
                    return Failure(ValueError(f"Index at row {i + 2} is not an integer."))

                parsed_rows.append((position, row[1:]))

            indices = [p[0] for p in parsed_rows]
            if sorted(indices) != list(range(input_count)):
                logger.error("Every index in the csv file has to appear exactly once and be in range 0..N-1.")
                return Failure(
                    ValueError("Every index in the csv file has to appear exactly once and be in range 0..N-1."))
            enumeration = parsed_rows
        else:
            enumeration = [(i, row) for i, row in enumerate(rows)]

        for i, row in enumeration:
            try:
                lower_bound, upper_bound = (float(row[0]), float(row[1]))
            except ValueError:
                logger.error(f"Item at value {i} contains an argument that is not an integer.")
                return Failure(ValueError(f"Item at value {i} contains an argument that is not an integer."))

            if lower_bound > upper_bound:
                logger.error(f"Lower bound {lower_bound} is greater than upper bound {upper_bound} for item {i}")
                return Failure(ValueError(f"Lower bound {lower_bound} is greater than upper bound {upper_bound} for item {i}"))

            bounds[i] = (lower_bound, upper_bound)


        return Success(bounds)

    """
    Extract only input bounds (X_i) from vnnlib
    
    Structure:
        - assert with (and...) / (or...) / atoms (<= X_0 0.5), (>= X_1 -0.2), (= X_2 0.0)
        - if OR gives multiple regions, return bounding box (min lo, max hi), cause InputBounds can save only one rectangle
    """
    def __parse_vnnlib(self, file_path: str, input_count: int) -> Result[Dict[int, tuple[float, float]]]:
        text = Path(file_path).read_text(encoding="utf-8", errors="replace")
        bounds_dict = self.__extract_input_bounds_from_vnnlib(text, input_count)
        return Success(bounds_dict)

    """
    Parser:
     - tokenize
     - build nested lists
     - extract form asserts constraints for X_i
    """
    def __extract_input_bounds_from_vnnlib(self, text: str, input_count: int) -> Dict[int, tuple[float, float]]:
        #delete comments
        text = re.sub(r";[^\n]*", "", text)
        #tokens: (,),atom
        tokens = re.findall(r"\(|\)|[^\s()]+", text)

        top: List[Any] = []
        stack: List[List[Any]] = []

        for t in tokens:
            if t == "(":
                stack.append([])
            elif t == ")":
                if not stack:
                    logger.error("Invalid expression.")
                    raise ValueError("Invalid expression.")
                expr = stack.pop()
                if stack:
                    stack[-1].append(expr)
                else:
                    top.append(expr)
            else:
                if stack:
                    stack[-1].append(t)

        if stack:
            logger.error("Invalid expression.")
            raise ValueError("Invalid expression.")

        """X_0 or X0"""
        def is_x(sym: Any) -> bool:
            return isinstance(sym, str) and re.fullmatch(r"X_?\d+", sym) is not None

        """X_12 -> 12, X12 -> 12"""
        def x_idx(sym : str) -> int:
            return int(sym.split("_")[-1]) if "_" in sym else int(sym[1:])

        """Atom to float"""
        def to_float(v: Any) -> float | None:
            if not isinstance(v, str):
                return None
            try:
                return float(v)
            except ValueError:
                return None

        region = Dict[int, Tuple[float | None, float | None]]

        """Constraints conjunction"""
        def merge(r1 : region, r2 : region) -> region | None:
            outer = dict(r1)
            for j, (lo2, hi2) in r2.items():
                lo1, hi1 = outer.get(j, (None, None))

                low = lo1
                if lo2 is not None:
                    low = lo2 if low is None else max(low, lo2)
                high = hi1
                if hi2 is not None:
                    high = hi2 if high is None else min(high, hi2)

                if low is not None and high is not None and low > high:
                    return None

                outer[j] = (low, high)

            return outer
        """
        Atomic comparison:
         - (<= X_0 0.5) -> X_0 <= 0.5
         - (<= 0.5 X_0) -> X_0 >= 0.5
         - (= X_0 1.2) -> X_0 == 1.2
        """
        def atomic(exp: Any) -> region | None:
            if not (isinstance(exp, list) and len(exp) == 3):
                return None

            op, a, b = exp
            if op not in ("<=", "<", ">=", ">", "="):
                return None

            a_is_x = is_x(a)
            b_is_x = is_x(b)
            a_num = to_float(a)
            b_num = to_float(b)

            if a_is_x and b_num is not None:
                j = x_idx(a)
                if op in ("<=", "<"):
                    return {j: (None, b_num)}
                if op in (">=", ">"):
                    return {j: (b_num, None)}
                return {j: (b_num, b_num)}

            if b_is_x and a_num is not None:
                j = x_idx(b)
                if op in ("<=", "<"):
                    return {j: (a_num, None)}
                if op in (">=", ">"):
                    return {j: (None, a_num)}
                return {j: (a_num, a_num)}

            return None

        """Ignor asserts for Y"""
        def contains_x(exp: Any) -> bool:
            if is_x(exp):
                return True
            if isinstance(exp, list):
                return any(contains_x(e) for e in exp)
            return False

        """Returns regions list:
            - and: multiply regions
            - or: combine regions
            - atom: one region 
        """
        def regions(exp:Any) -> List[region]:
            if not contains_x(exp):
                return [{}]
            if not isinstance(exp, list) or not exp:
                return [{}]

            head = exp[0]

            if head == "and":
                regs: List[region] = [{}]
                for child in exp[1:]:
                    child_regs = regions(child)
                    new_regs: List[region] = []
                    for reg in regs:
                        for cr in child_regs:
                            me = merge(reg, cr)
                            if me is not None:
                                new_regs.append(me)
                    regs = new_regs
                    if not regs:
                        return []
                return regs

            if head == "or":
                outer: List[region] = []
                for child in exp[1:]:
                    outer.extend(regions(child))
                return outer if outer else [{}]

            ab = atomic(exp)
            return [ab] if ab is not None else [{}]

        #main constraints build
        overall: List[region] = [{}]
        found_any = False

        for expr in top:
            if isinstance(expr, list) and expr and expr[0] == "assert":
                body = expr[1] if len(expr) > 1 else []
                if not contains_x(body):
                    continue

                found_any = True
                rs = regions(body)
                if not rs:
                    logger.error("Invalid input constraints in vnnlib file")
                    raise ValueError("Invalid input constraints in vnnlib file")

                new_overall: List[region] = []
                for r in overall:
                    for rr in rs:
                        m = merge(r, rr)
                        if m is not None:
                            new_overall.append(m)

                overall = new_overall
                if not overall:
                    logger.error("Invalid input constraints in vnnlib file")
                    raise ValueError("Invalid input constraints in vnnlib file")

        if not found_any:
            logger.error("No input specs for X_i found in vnnlib file")
            raise ValueError("No input specs for X_i found in vnnlib file")

        #bounding box for regions if there was OR
        out: Dict[int, Tuple[float, float]] = {}

        for i in range(input_count):
            lows: List[float] = []
            highs: List[float] = []

            for r in overall:
                lo, hi = r.get(i, (None, None))
                if lo is None or hi is None:
                    logger.error(f"Missing bounds for X_{i}")
                    raise ValueError(f"Missing bounds for X_{i}")
                lows.append(lo)
                highs.append(hi)

            out[i] = (min(lows), max(highs))

        return out

