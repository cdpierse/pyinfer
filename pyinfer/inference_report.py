import multiprocessing as mp
import signal
import statistics
import warnings
from contextlib import contextmanager
from time import perf_counter, perf_counter_ns, sleep, time
from typing import Any, Callable, List, Union

import psutil
from tabulate import tabulate

from .errors import (
    MatplotlibNotInstalledError,
    MeasurementIntervalNotSetError,
    ModelIsNotCallableError,
    NamesNotEqualsModelsLengthError,
)


class InferenceReport:
    "Provides Model Agnostic inference reporting for ML model"

    def __init__(
        self,
        model: Callable,
        inputs: Any,
        n_seconds: Union[int, float, None] = None,
        n_iterations: int = None,
        exit_on_inputs_exhausted: bool = False,
        infer_failure_point: Union[int, float, None] = None,
        model_name: str = None,
        drop_stats: List[str] = None,
    ):
        if not isinstance(model, Callable):
            raise ModelIsNotCallableError(
                "The model provided is not callable. Please provide a model that has a method call."
            )
        self.model = model
        self.inputs = inputs
        self.exit_on_inputs_exhausted = exit_on_inputs_exhausted
        self.infer_failure_point = infer_failure_point
        self.drop_stats = drop_stats

        self.runs = []

        if model_name:
            self.model_name = str(model_name)
        else:
            self.model_name = "Model"

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
                    start = perf_counter_ns() * 1e-9
                    self.model(self.inputs)
                    end = perf_counter_ns() * 1e-9
                    run = end - start
                    if self.infer_failure_point:
                        if run >= self.infer_failure_point:
                            failed += 1
                        else:
                            completed += 1
                    else:
                        completed += 1
                    runs.append(run)

                    total_time_taken += run
        else:
            while iterations < self.n_iterations:
                start = perf_counter_ns() * 1e-9
                self.model(self.inputs)
                end = perf_counter()
                end = perf_counter_ns() * 1e-9
                run = end - start
                if self.infer_failure_point:
                    if run >= self.infer_failure_point:
                        failed += 1
                    else:
                        completed += 1
                else:
                    completed += 1
                runs.append(round(run, 4))
                iterations += 1
                total_time_taken += run

        self.runs = runs
        results_dict = self._make_results_dict(
            completed, failed, self.runs, total_time_taken
        )

        if print_report:
            self.report(results_dict)
        return results_dict

    def _make_results_dict(
        self,
        completed: int,
        failed: int,
        runs: List[float],
        total_time_taken: float,
    ) -> dict:
        total = completed + failed
        return {
            "Model": self.model_name,
            "Success": completed,
            "Fail": failed,
            "Took": total_time_taken,
            "Infer(p/sec)": round(total / total_time_taken, 2),
            "MaxRun(ms)": self._max_run(runs),
            "MinRun(ms)": self._min_run(runs),
            "Std(ms)": self._stdev(runs),
            "Mean(ms)": self._mean_run(runs),
            "Median(ms)": self._median_run(runs),
            "IQR(ms)": self._iqr(runs),
            "Cores(L)": psutil.cpu_count(logical=True),
            "Cores(LM": psutil.cpu_count(logical=False),
        }

    @staticmethod
    def _cpu_monitor(target_function):
        worker_process = mp.Process(target=target_function)
        worker_process.start()
        p = psutil.Process(worker_process.pid)

        cpu_usage_pc = []
        while worker_process.is_alive():
            cpu_usage_pc.append(p.cpu_percent())
            sleep(0.01)

        worker_process.join()
        return cpu_usage_pc

    def report(self, results_dict: dict):
        print(
            tabulate(
                [results_dict.values()],
                headers=results_dict.keys(),
                tablefmt="fancy_grid",
                numalign="right",
            )
        )

    def plot(self, show: bool = True, save: str = None):
        try:
            import matplotlib
            import matplotlib.pyplot as plt
        except:
            s = "InferenceReport.plot() requires matplotlib to be installed."
            s += 'To use this method please install by running: pip install "pyinfer[plotting]" '
            raise MatplotlibNotInstalledError(s)

        if self.runs:
            t = list(range(0, len(self.runs)))
            ms_runs = [(run * 1000) for run in self.runs]
            fig, ax = plt.subplots()
            ax.plot(t, ms_runs, marker="o")
            ax.set(xlabel="run", ylabel="run time (ms)")
            ax.grid()
            if show:
                plt.show()
        else:
            raise ValueError(
                "self.runs is not yet set, please run the report before plotting."
            )

    def _max_run(self, runs: list) -> float:
        return max(runs) * 1000

    def _min_run(self, runs: list) -> float:
        return min(runs) * 1000

    def _stdev(self, runs: list) -> float:
        return statistics.stdev(runs) * 1000

    def _iqr(self, runs: list) -> float:
        quartiles = statistics.quantiles(runs, n=4)
        return (quartiles[2] - quartiles[0]) * 1000

    def _mean_run(self, runs: list) -> float:
        return statistics.mean(runs) * 1000

    def _median_run(self, runs: list) -> float:
        return statistics.median(runs) * 1000


class MultiInferenceReport:
    def __init__(
        self,
        models: List[Callable],
        inputs: List[Any],
        n_seconds: Union[int, float, None] = None,
        n_iterations: int = None,
        exit_on_inputs_exhausted: bool = False,
        infer_failure_point: Union[int, float, None] = None,
        model_names: List[str] = None,
    ):

        for i, model in enumerate(models):
            if not isinstance(model, Callable):
                raise ModelIsNotCallableError(
                    f"The model at index {i} is not callable. Please provide a model that has a method call."
                )

        self.models = models
        self.inputs = inputs
        self.exit_on_inputs_exhausted = exit_on_inputs_exhausted
        self.infer_failure_point = infer_failure_point
        self.models_runs = []

        if model_names:
            if len(model_names) != len(self.models):
                s = f"Length of model_names is {len(model_names)}, does not equal number of models provided {len(self.models)}. "
                s += "Please ensure that these lengths are equal if you want to set custom model names. "
                s += "Otherwise you can leave model_names as None."
                raise NamesNotEqualsModelsLengthError(s)
            else:
                self.model_names = model_names
            pass
        else:
            self.model_names = ["Model " + str(i) for i, _ in enumerate(self.models)]
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

    def run(self, print_report: bool = False):
        results = []
        for i, (model, _input) in enumerate(zip(self.models, self.inputs)):
            report = InferenceReport(
                model=model,
                inputs=_input,
                n_seconds=self.n_seconds,
                n_iterations=self.n_iterations,
                infer_failure_point=self.infer_failure_point,
                model_name=self.model_names[i],
            )
            results.append(report.run(print_report=False))
            self.models_runs.append(report.runs)

        return results

    def report(self, results_list: List[dict]):
        print(
            tabulate(
                [results_dict.values() for results_dict in results_list],
                headers=results_list[0].keys(),
                tablefmt="fancy_grid",
                numalign="right",
            )
        )

    def plot(self, show: bool = True):
        try:
            import matplotlib
            import matplotlib.pyplot as plt
        except:
            s = "InferenceReport.plot() requires matplotlib to be installed."
            s += 'To use this method please install by running: pip install "pyinfer[plotting]" '
            raise MatplotlibNotInstalledError(s)

        if self.models_runs:
            fig, ax = plt.subplots()
            ax.set(xlabel="run", ylabel="run time (ms)")
            ax.grid()
            for i, runs in enumerate(self.models_runs):
                t = list(range(0, len(runs)))
                ms_runs = [(run * 1000) for run in runs]
                plt.plot(t, ms_runs, marker="o", label=self.model_names[i])
                plt.legend()
        else:
            raise ValueError(
                "self.models_runs is not yet set, please run the report before plotting."
            )

        if show:
            plt.show()
