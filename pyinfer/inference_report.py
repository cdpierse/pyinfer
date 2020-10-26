import signal
import statistics
import warnings
from contextlib import contextmanager
from time import perf_counter, time
from typing import Any, Callable, List, Union

from tabulate import tabulate

from .errors import MeasurementIntervalNotSetError, ModelIsNotCallableError


class InferenceReport:
    "Provides Model Agnostic inference reporting for ML model"

    def __init__(
        self,
        model: Callable,
        inputs: Any,
        n_seconds: Union[int, float, None] = None,
        n_iterations: int = None,
        exit_on_inputs_exhausted: bool = False,
        infer_timeout: Union[int, float, None] = None,
    ):
        if not isinstance(model, Callable):
            raise ModelIsNotCallableError(
                "The model provided is not callable. Please provide a model that has a method call."
            )
        self.model = model
        self.inputs = inputs
        self.exit_on_inputs_exhausted = exit_on_inputs_exhausted
        self.infer_timeout = infer_timeout

        if not n_iterations and not n_seconds:
            s = "You have not specified either `n_seconds` or `n_iterations`."
            s += " Please specify a valid measurement interval."
            raise MeasurementIntervalNotSetError(s)

        if n_iterations and n_seconds:
            s = f"You have set both `n_seconds={n_seconds}` and `n_iterations={n_iterations}` "
            s += f"only one can be specified per instance. Defaulting measurement interval to `seconds={n_seconds}``"

            warnings.warn(s)
            self.n_seconds = n_seconds
            self.n_iterations = None
        else:
            self.n_seconds = n_seconds
            self.n_iterations = n_iterations

        self.terminated = False

    @contextmanager
    def timeout(self, duration):
        def timeout_handler(signum, frame):
            self.terminated = True

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(duration)
        yield
        signal.alarm(0)

    def run(self, print_report: bool = True) -> dict:
        iterations = 0
        runs: List[datetime.timedelta] = []
        total_time_taken = 0
        failed = 0
        completed = 0

        if self.n_seconds:
            with self.timeout(self.n_seconds):
                while not self.terminated:
                    start = perf_counter()
                    self.model(self.inputs)
                    end = perf_counter()
                    run = end - start
                    if self.infer_timeout:
                        if run >= self.infer_timeout:
                            failed += 1
                        else:
                            completed += 1
                    else:
                        completed += 1
                    runs.append(run)

                    total_time_taken += run
        else:
            while iterations < self.n_iterations:
                start = perf_counter()
                self.model(self.inputs)
                end = perf_counter()
                run = end - start
                runs.append(run)
                iterations += 1
                total_time_taken += run

        self.iterations = completed + failed
        self.runs = runs
        self.total_time_taken = total_time_taken
        self.failed = failed

        if print_report:
            self.report()

    def report(self):
        table = [
            [
                "Model 1",
                self.iterations,
                self.failed,
                self.total_time_taken,
                round(self.iterations / self.total_time_taken,2),
                self._max_run(self.runs),
                self._min_run(self.runs),
                self._stdev(self.runs),
                self._mean_run(self.runs),
                self._median_run(self.runs),
            ]
        ]
        print(
            tabulate(
                table,
                headers=[
                    "Completed",
                    "Failed",
                    "Time Taken (sec)",
                    "Infer Per Sec",
                    "Max Run (ms)",
                    "Min Run (ms)",
                    "Stdev (ms)",
                    "Mean (ms)",
                    "Median (ms)",
                ],
                tablefmt="fancy_grid",
            )
        )

    def plot(self):
        # will plot the runs on a graph
        pass

    def _max_run(self, runs: list) -> float:
        return max(runs) * 1000

    def _min_run(self, runs: list) -> float:
        return min(runs) * 1000

    def _stdev(self, runs: list) -> float:
        return statistics.stdev(runs) * 1000

    def _mean_run(self, runs: list) -> float:
        return statistics.mean(runs) * 1000

    def _median_run(self, runs: list) -> float:
        return statistics.median(runs) * 1000


class MultiInferenceReport:
    def __init__(self, models: List[Callable], inputs: List[Any]):
        pass
