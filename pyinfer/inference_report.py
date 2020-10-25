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
        inputs: Union[List, Any],
        loop: bool = True,
        n_seconds: Union[int, float, None] = None,
        n_iterations: int = None,
    ):
        self.model = model
        self.input = inputs

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

        if self.n_iterations and self.n_seconds:
            warnings.warn("You have specified both n_seconds and n_iterations. ")

    def run(self, print_report: bool = True) -> dict:
        inference_iter = 0
        runs: List[datetime.timedelta] = []

        if self.n_seconds:
            stop = datetime.datetime.now() + datetime.timedelta(seconds=self.n_seconds)
            while datetime.datetime.now() < stop:
                start = datetime.datetime.now()
                self.model(self.input)
                end = datetime.datetime.now()
                runs.append(end - start)
                inference_iter += 1
        else:
            while inference_iter <= self.n_iterations:
                start = datetime.datetime.now()
                self.model(self.input)
                end = datetime.datetime.now()
                runs.append(end - start)
                inference_iter += 1

        results = {"iterations": inference_iter, "runs": runs}
        if print_report:
            self.report(results)

    def report(self, results: dict):
        table = [[]]
        print(tabulate(table, headers=results.keys()))
        headers = ""
        pass
