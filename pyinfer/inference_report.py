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


def quantiles(data, *, n=4, method="exclusive"):
    """
    Monkey patched quantiles function from statistics package
    present in python >= 3.8 as quantiles() isn't available for python 3.7.x
    """
    if n < 1:
        raise StatisticsError("n must be at least 1")
    data = sorted(data)
    ld = len(data)
    if ld < 2:
        raise StatisticsError("must have at least two data points")
    if method == "inclusive":
        m = ld - 1
        result = []
        for i in range(1, n):
            j = i * m // n
            delta = i * m - j * n
            interpolated = (data[j] * (n - delta) + data[j + 1] * delta) / n
            result.append(interpolated)
        return result
    if method == "exclusive":
        m = ld + 1
        result = []
        for i in range(1, n):
            j = i * m // n  # rescale i to m/n
            j = 1 if j < 1 else ld - 1 if j > ld - 1 else j  # clamp to 1 .. ld-1
            delta = i * m - j * n  # exact integer math
            interpolated = (data[j - 1] * (n - delta) + data[j] * delta) / n
            result.append(interpolated)
        return result
    raise ValueError(f"Unknown method: {method!r}")


class InferenceReport:
    "A model agnostic report of inference related stats for any callable model"

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
        """
        Args:
            model (Callable): The callable method or function for the model.

            inputs (Any): The input(s) parameters the model receives.

            n_seconds (Union[int, float, None], optional): Number of seconds to run model inferences.
                If this is `None` it is expected that `n_iterations` will be set. Defaults to None.

            n_iterations (int, optional): Number of iterations to run model inferences for.
                If this is `None` it is expected that `n_seconds` will be set. Defaults to None.

            exit_on_inputs_exhausted (bool, optional): If inputs are a iterable of inputs exit
                on completion. This feature is not yet implemented. Defaults to False.

            infer_failure_point (Union[int, float, None], optional): Time in seconds (int or float)
                at which an inference is to be considered a failure in the reporting stats.
                Defaults to None.

            model_name (str, optional): The name to give to the model for the report.
                Defaults to None.

            drop_stats (List[str], optional): List of keys to drop from the report.
                Defaults to None.

        Raises:
            ModelIsNotCallableError: Will raise if the model provided is not callable.
            MeasurementIntervalNotSetError: Will raise if neither `n_seconds` or
                `n_iterations` are set.
        """
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
    def _timeout(self, duration: int):
        "Creates signal to terminate execution once `duration` seconds have passed"

        def timeout_handler(signum, frame):
            self.terminated = True

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(duration)
        yield
        signal.alarm(0)

    def run(self, print_report: bool = True) -> dict:
        """
        Runs the inference report for `self.model` with input(s) `self.inputs`

        Args:
            print_report (bool, optional): If true a table representation of the report will be
            printed to console. Defaults to True.

        Returns:
            dict: A dictionary containing all the report stats created during the run.
        """
        iterations = 0
        runs: List[datetime.timedelta] = []
        total_time_taken = 0
        failed = 0
        completed = 0

        if self.n_seconds:
            with self._timeout(self.n_seconds):
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
        "Creates dict of all the stats using info collected from run"
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
            "Cores(P)": psutil.cpu_count(logical=False),
        }

    @staticmethod
    def _cpu_monitor(target_function):
        """
        Monitors cpu usage of a `target_function`. Not currently in use
        As psutil requires root access when running a worker process.
        """
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
        """
        Prints a report to console based on the values
        found in `results_dict`

        Args:
            results_dict (dict): Dictionary containing compiled stats from a run.
        """
        if self.drop_stats:
            for drop_key in self.drop_stats:
                results_dict.pop(drop_key, None)
        print(
            tabulate(
                [results_dict.values()],
                headers=results_dict.keys(),
                tablefmt="fancy_grid",
                numalign="right",
            )
        )

    def plot(self, show: bool = True, save_location: str = None):
        """
        Creates a simple plot of `self.runs`. Plots run number
        on the x-axis and run time in milliseconds on the y-axis.

        Args:
            show (bool, optional): Whether to show the plot after calling method. Defaults to True.

            save_location (str, optional): Location to save plot at. If None the plot will not
                be saved. Defaults to None.

        Raises:
            MatplotlibNotInstalledError: Raise if matplotlib is not installed in python environment.
            ValueError: Raise if the runs have not yet been calculated but `plot` is called.
        """
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

            if save_location:
                plt.savefig(save_location)
        else:
            raise ValueError(
                "self.runs is not yet set, please run the report before plotting."
            )

    def _max_run(self, runs: list) -> float:
        "Returns max run in milliseconds from `runs`"
        return max(runs) * 1000

    def _min_run(self, runs: list) -> float:
        "Returns min run in milliseconds from `runs`"
        return min(runs) * 1000

    def _stdev(self, runs: list) -> float:
        "Returns standard deviation in milliseconds from `runs`"
        if len(runs) >= 2:
            return statistics.stdev(runs) * 1000
        else:
            return None

    def _iqr(self, runs: list) -> float:
        "Returns interquartile range in milliseconds from `runs`"
        if len(runs) >= 2:
            quartiles = quantiles(runs, n=4)
            return (quartiles[2] - quartiles[0]) * 1000
        else:
            return None

    def _mean_run(self, runs: list) -> float:
        "Returns mean run time in milliseconds from `runs`"
        return statistics.mean(runs) * 1000

    def _median_run(self, runs: list) -> float:
        "Returns median run time in milliseconds from `runs`"
        return statistics.median(runs) * 1000


class MultiInferenceReport:
    "A model agnostic report of inference related stats for any list of callable models"

    def __init__(
        self,
        models: List[Callable],
        inputs: List[Any],
        n_seconds: Union[int, float, None] = None,
        n_iterations: int = None,
        exit_on_inputs_exhausted: bool = False,
        infer_failure_point: Union[int, float, None] = None,
        model_names: List[str] = None,
        drop_stats: List[str] = None,
    ):
        """
        Args:
            models (List[Callable]): A list of the callable methods or functions for the models.

            inputs (List[Any]): The input(s) parameters each of the models receives. If only one
                input is given then it is assumed each model takes the same shape/type of input and
                that input will be passed to each model.

            n_seconds (Union[int, float, None], optional): Number of seconds to run model inferences.
                If this is `None` it is expected that `n_iterations` will be set. Defaults to None.

            n_iterations (int, optional): Number of iterations to run model inferences for.
                If this is `None` it is expected that `n_seconds` will be set. Defaults to None.

            exit_on_inputs_exhausted (bool, optional): If inputs are a iterable of inputs exit
                on completion. This feature is not yet implemented. Defaults to False.

            infer_failure_point (Union[int, float, None], optional): Time in seconds (int or float)
                at which an inference. is to be considered a failure in the reporting stats.
                Defaults to None.

            model_names (List[str], optional): The names to give to the models for the report. Must
                be the same length as number of models provided. Defaults to None.

            drop_stats (List[str], optional): List of keys to drop from the report.
                Defaults to None.

        Raises:
            ModelIsNotCallableError: Will raise if the model provided is not callable.

            NamesNotEqualsModelsLengthError: Will raise if the number of models names
                does not match the number of model callables provided.

            MeasurementIntervalNotSetError: Will raise if neither `n_seconds` or
                `n_iterations` are set.
        """

        for i, model in enumerate(models):
            if not isinstance(model, Callable):
                raise ModelIsNotCallableError(
                    f"The model at index {i} is not callable. Please provide a model that has a method call."
                )

        self.models = models
        if not isinstance(inputs, list):
            self.inputs = [inputs]
        else:
            self.inputs = inputs

        if len(self.inputs) != len(self.models):
            self.inputs = [self.inputs[0]] * len(self.models)

        self.exit_on_inputs_exhausted = exit_on_inputs_exhausted
        self.infer_failure_point = infer_failure_point
        self.models_runs = []
        self.drop_stats = drop_stats

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

    def run(self, print_report: bool = True) -> List[dict]:
        """
        Runs the multi inference report for `self.models` with input(s) `self.inputs`

        Args:
            print_report (bool, optional): If true a table representation of the report will be
                printed to console. Defaults to True.

        Returns:
            List[dict]: A list of dictionaries containing all the report stats created during the run
                for each model callable.
        """
        results = []
        for i, (model, _input) in enumerate(zip(self.models, self.inputs)):
            report = InferenceReport(
                model=model,
                inputs=_input,
                n_seconds=self.n_seconds,
                n_iterations=self.n_iterations,
                infer_failure_point=self.infer_failure_point,
                model_name=self.model_names[i],
                drop_stats=self.drop_stats,
            )
            results.append(report.run(print_report=False))
            self.models_runs.append(report.runs)

        if print_report:
            self.report(results)

        return results

    def report(self, results_list: List[dict]):
        """
        Prints a report to console based on the values
        found in `results_list`

        Args:
            results_list (dict): A list of dictionaries containing compiled stats from the runs.
        """
        print(
            tabulate(
                [results_dict.values() for results_dict in results_list],
                headers=results_list[0].keys(),
                tablefmt="fancy_grid",
                numalign="right",
            )
        )

    def plot(self, show: bool = True, save_location: str = None):
        """
        Creates a simple plot of `self.models_runs`. For each run it
        plots run number on the x-axis and run time in milliseconds on the y-axis.

        Args:
            show (bool, optional): Whether to show the plot after calling method. Defaults to True.

            save_location (str, optional): Location to save plot at. If None the plot will not
                be saved. Defaults to None.

        Raises:
            MatplotlibNotInstalledError: Raise if matplotlib is not installed in python environment.
            ValueError: Raise if the model_runs have not yet been calculated but `plot` is called.
        """
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
            if save_location:
                plt.savefig(save_location)
        else:
            raise ValueError(
                "self.models_runs is not yet set, please run the report before plotting."
            )

        if show:
            plt.show()
