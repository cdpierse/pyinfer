import datetime
import multiprocessing
import signal
import statistics
import warnings
from time import time
from typing import Any, Callable, List, Union

from tabulate import tabulate

from .errors import MeasurementIntervalNotSetError


class TimeoutException(Exception):  # Custom exception class
    pass


def timeout_handler(signum, frame):  # Custom signal handler
    raise TimeoutException


class InferenceReport:
    "Provides Model Agnostic inference reporting for ML model"

    def __init__(
        self,
        model: Callable,
        inputs: Any,
        n_seconds: Union[int, float, None] = None,
        n_iterations: int = None,
        exit_on_inputs_exhausted: bool = False,
        inference_timeout_seconds: Union[int, float, None] = None,
    ):
        self.model = model
        self.inputs = inputs
        self.exit_on_inputs_exhausted = exit_on_inputs_exhausted
        self.inference_timeout_seconds = inference_timeout_seconds

        if not n_iterations and not n_seconds:
            s = "You have not specified either `n_seconds` or `n_iterations`."
            s += " Please specify a valid measurement interval."
            raise MeasurementIntervalNotSetError(s)

        if n_iterations and n_seconds:
            s = f"You have set both `n_seconds={n_seconds}` and `n_iterations={n_iterations}` "
            s += f"only one can be specified per instance. Defaulting to measurement interval to `seconds={n_seconds}``"

            warnings.warn(s)
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
                if self.inference_timeout_seconds:
                    run, completed = self._run_model_thread(
                        self.inference_timeout_seconds
                    )
                else:
                    run, completed = self._run_model_thread(
                        self.n_seconds - total_time_taken
                    )
                runs.append(run)
                iterations += completed
                total_time_taken += run.total_seconds()
        else:
            while iterations < self.n_iterations:
                start = datetime.datetime.now()
                self.model(self.inputs)
                end = datetime.datetime.now()
                run = end - start
                runs.append(run)
                iterations += 1
                total_time_taken += run.total_seconds()

        self.iterations = iterations
        self.runs = [run.total_seconds() for run in runs]
        self.total_time_taken = total_time_taken

        if print_report:
            self.report()

    def report(self):
        table = [
            [
                "Model 1",
                self.iterations,
                self.total_time_taken,
                self._max_run(self.runs),
                self._min_run(self.runs),
                self._stdev(self.runs),
            ]
        ]
        print(
            tabulate(
                table,
                headers=[
                    "Completed",
                    "Time Taken (Seconds)",
                    "Max Run (ms)",
                    "Min Run (ms)",
                    "StDev",
                ],
            )
        )

    def _run_model_thread(self, timeout):

        signal.signal(signal.SIGALRM, timeout_handler)

        # Start the timer. Once {timeout} seconds are over, a SIGALRM signal is sent.
        signal.setitimer(signal.ITIMER_REAL, timeout)
        start = datetime.datetime.now()

        # This try/except loop ensures that
        #   you'll catch TimeoutException when it's sent.
        try:
            self.model(self.inputs)  # our potentially long function
            end = datetime.datetime.now()
            return end - start, 1
        except TimeoutException:
            print("took too long")
            end = datetime.datetime.now()
            return end - start, 0

    def _max_run(self, runs: list) -> float:
        return max(runs) * 1000

    def _min_run(self, runs: list) -> float:
        return min(runs) * 1000

    def _stdev(self, runs: list) -> float:
        return statistics.stdev(runs)


class MultiInferenceReport:
    def __init__(self, models: List[Callable], inputs: List[Any]):
        pass
