import sys
import time
from unittest import mock
from unittest.mock import patch

import matplotlib.pyplot as plt
import pytest
from pyinfer import InferenceReport, MultiInferenceReport
from pyinfer.errors import (
    MatplotlibNotInstalledError,
    MeasurementIntervalNotSetError,
    ModelIsNotCallableError,
    NamesNotEqualsModelsLengthError,
)
from tabulate import tabulate


class MockModel:
    def __init__(self, delay: float = 0.01):
        self.delay = delay

    def infer_callable(self, input):
        time.sleep(self.delay)
        return input * pow(input, 2)


EXPECTED_METRICS = [
    "Model",
    "Success",
    "Fail",
    "Took",
    "Infer(p/sec)",
    "MaxRun(ms)",
    "MinRun(ms)",
    "Std(ms)",
    "Mean(ms)",
    "Median(ms)",
    "IQR(ms)",
    "Cores(L)",
    "Cores(P)",
]


def test_multi_init_iterations():
    model = MockModel()
    model1 = MockModel()
    multi_report = MultiInferenceReport(
        [model.infer_callable, model1.infer_callable], 1, n_iterations=1
    )

    assert multi_report.n_iterations == 1
    assert multi_report.n_seconds == None


def test_multi_init_seconds():
    model = MockModel()
    model1 = MockModel()
    multi_report = MultiInferenceReport(
        [model.infer_callable, model1.infer_callable], 1, n_seconds=10
    )

    assert multi_report.n_iterations == None
    assert multi_report.n_seconds == 10


def test_multi_init_model_not_callable():
    model = MockModel()
    model1 = MockModel()
    with pytest.raises(ModelIsNotCallableError):
        multi_report = MultiInferenceReport(
            [model1, model.infer_callable], 1, n_seconds=10
        )


def test_multi_init_model_not_callable_both_models():
    model = MockModel()
    model1 = MockModel()
    with pytest.raises(ModelIsNotCallableError):
        multi_report = MultiInferenceReport([model1, model], 1, n_seconds=10)


def test_multi_init_list_inputs_len_less_than_models_len():
    model = MockModel()
    model1 = MockModel()
    multi_report = MultiInferenceReport(
        [model.infer_callable, model1.infer_callable], [1], n_iterations=1
    )

    assert multi_report.inputs == [1, 1]


def test_multi_init_list_inputs_len_greater_than_models_len():
    model = MockModel()
    model1 = MockModel()
    multi_report = MultiInferenceReport(
        [model.infer_callable, model1.infer_callable], [3, 7, 8], n_iterations=1
    )

    assert multi_report.inputs == [3, 3]


def test_multi_init_no_names_given():
    model = MockModel()
    model1 = MockModel()
    multi_report = MultiInferenceReport(
        [model.infer_callable, model1.infer_callable], [3], n_iterations=1
    )

    assert multi_report.model_names == ["Model 0", "Model 1"]


def test_multi_init_names_given():
    model = MockModel()
    model1 = MockModel()
    multi_report = MultiInferenceReport(
        [model.infer_callable, model1.infer_callable],
        [3],
        n_iterations=1,
        model_names=["MyModel1", "MyModel2"],
    )

    assert multi_report.model_names == ["MyModel1", "MyModel2"]


def test_multi_init_names_given_less_than_number_of_models():
    model = MockModel()
    model1 = MockModel()
    with pytest.raises(NamesNotEqualsModelsLengthError):
        multi_report = MultiInferenceReport(
            [model.infer_callable, model1.infer_callable],
            [3],
            n_iterations=1,
            model_names=["MyModel1"],
        )


def test_multi_init_names_given_greater_than_number_of_models():
    model = MockModel()
    model1 = MockModel()
    with pytest.raises(NamesNotEqualsModelsLengthError):
        multi_report = MultiInferenceReport(
            [model.infer_callable, model1.infer_callable],
            [3],
            n_iterations=1,
            model_names=["MyModel1", "MyModel2", "MyModel3"],
        )


def test_multi_two_intervals_given_defaults_to_seconds():
    model = MockModel()
    model1 = MockModel()

    multi_report = MultiInferenceReport(
        [model.infer_callable, model1.infer_callable], [3], n_iterations=1, n_seconds=1
    )

    assert multi_report.n_iterations == None
    assert multi_report.n_seconds == 1


def test_multi_no_intervals_given_raises_error():
    model = MockModel()
    model1 = MockModel()

    with pytest.raises(MeasurementIntervalNotSetError):
        multi_report = MultiInferenceReport(
            [model.infer_callable, model1.infer_callable], [3]
        )


def test_multi_run_seconds():
    model = MockModel()
    model1 = MockModel()

    multi_report = MultiInferenceReport(
        [model.infer_callable, model1.infer_callable], [3], n_seconds=1
    )
    results = multi_report.run(print_report=False)
    assert isinstance(results, list)
    for result in results:
        for expected in EXPECTED_METRICS:
            assert expected in result.keys()


def test_multi_run_iterations():
    model = MockModel()
    model1 = MockModel()

    multi_report = MultiInferenceReport(
        [model.infer_callable, model1.infer_callable], [3], n_iterations=1
    )
    results = multi_report.run(print_report=False)
    assert isinstance(results, list)
    for result in results:
        for expected in EXPECTED_METRICS:
            assert expected in result.keys()


def test_multi_run_print_report():
    model = MockModel()
    model1 = MockModel()

    multi_report = MultiInferenceReport(
        [model.infer_callable, model1.infer_callable], [3], n_iterations=1
    )
    results = multi_report.run()
    assert isinstance(results, list)
    for result in results:
        for expected in EXPECTED_METRICS:
            assert expected in result.keys()


def test_multi_plot():
    model = MockModel()
    model1 = MockModel()

    multi_report = MultiInferenceReport(
        [model.infer_callable, model1.infer_callable], [3], n_iterations=1
    )
    results = multi_report.run(print_report=False)
    multi_report.plot(show=False)


def test_multi_report_not_run_raises_value_error():
    model = MockModel()
    model1 = MockModel()

    multi_report = MultiInferenceReport(
        [model.infer_callable, model1.infer_callable], [3], n_iterations=1
    )
    with pytest.raises(ValueError):
        multi_report.plot(show=False)


@patch.object(plt, "show")
def test_multi_plot_show(plt):
    model = MockModel()
    model1 = MockModel()

    multi_report = MultiInferenceReport(
        [model.infer_callable, model1.infer_callable], [3], n_iterations=1
    )
    results = multi_report.run(print_report=False)
    multi_report.plot(show=True)


@patch.object(plt, "show")
@patch.object(plt, "savefig")
def test_multi_plot_show(plt, plt1):
    model = MockModel()
    model1 = MockModel()

    multi_report = MultiInferenceReport(
        [model.infer_callable, model1.infer_callable], [3], n_iterations=1
    )
    results = multi_report.run(print_report=False)
    multi_report.plot(show=True, save_location="test.jpg")


def test_multi_plot_matplotlib_not_installed():
    model = MockModel()
    model1 = MockModel()

    multi_report = MultiInferenceReport(
        [model.infer_callable, model1.infer_callable], [3], n_iterations=1
    )
    results = multi_report.run(print_report=False)
    with patch.dict(sys.modules, {"matplotlib": None}):
        with pytest.raises(MatplotlibNotInstalledError):
            multi_report.plot()
