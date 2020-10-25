import datetime
import warnings
from time import time
from typing import Any, Callable, List, Union

from tabulate import tabulate

from .errors import MeasurementIntervalNotSetError


class InferenceReport:
    "Provides Model Agnostic inference reporting for ML model"

    def __init__(
        self,
        model: Callable,
        inputs: Any,
        loop: bool = True,
        n_seconds: Union[int, float, None] = None,
        n_iterations: int = None,
    ):
        self.model = model
        self.input = inputs
        self.loop = loop

        if not n_iterations and not n_seconds:
            raise MeasurementIntervalNotSetError(
                "You have not specified either `n_seconds` or `n_iterations`. Please specify a valid measurement interval."
            )

        if n_iterations and n_seconds:
            warnings.warn(
                "You have set both `n_seconds` and `n_iterations` only one can be specified per instance. Defaulting to seconds"
            )
            self.n_seconds = n_seconds
            self.n_iterations = None
        else:
            self.n_seconds = n_seconds
            self.n_iterations = n_iterations

    def run(self, print_report: bool = True) -> dict:
        iterations = 0
        runs: List[datetime.timedelta] = []
        total_time_taken = 0

        if self.n_seconds:
            stop = self.n_seconds
            while total_time_taken < stop:
                start = datetime.datetime.now()
                self.model(self.input)
                end = datetime.datetime.now()
                run = end - start
                runs.append(run)
                iterations += 1
                total_time_taken += run.total_seconds()
        else:
            while iterations < self.n_iterations:
                start = datetime.datetime.now()
                self.model(self.input)
                end = datetime.datetime.now()
                run = end - start
                runs.append(run)
                iterations += 1
                total_time_taken += run.total_seconds()

        self.iterations = iterations
        self.runs = runs
        self.total_time_taken = total_time_taken

        if print_report:
            self.report()

    def report(self):
        print(self.total_time_taken)
        table = [
            [
                self.iterations,
                self.total_time_taken,
                self._max_run(self.runs),
                self._min_run(self.runs),
            ]
        ]
        print(
            tabulate(
                table,
                headers=[
                    "Iterations",
                    "Total Time Taken (Seconds)",
                    "Max Run (Millseconds)",
                    "Min Run (Milliseconds)",
                ],
            )
        )

    def _max_run(self, runs: list) -> float:
        return max(runs).total_seconds() * 1000

    def _min_run(self, runs: list) -> float:
        return min(runs).total_seconds() * 1000
