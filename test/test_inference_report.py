import time
from unittest import mock

import pytest
from pyinfer import InferenceReport
from pyinfer.errors import MeasurementIntervalNotSetError, ModelIsNotCallableError


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


def test_inference_report_init_iterations():
    model = MockModel()
    report = InferenceReport(
        model.infer_callable,
        1,
        n_iterations=1,
    )
    assert report.n_iterations == 1
    assert report.n_seconds == None


def test_inference_report_init_seconds():
    model = MockModel()
    report = InferenceReport(
        model.infer_callable,
        1,
        n_seconds=10,
    )
    assert report.n_iterations == None
    assert report.n_seconds == 10


def test_inference_report_init_model_not_callable():
    model = MockModel()
    with pytest.raises(ModelIsNotCallableError):
        report = InferenceReport(model, 1, n_iterations=1)


def test_inference_report_init_model_name_given():
    model = MockModel()
    name = "my_test_model"
    report = InferenceReport(model.infer_callable, 1, n_iterations=1, model_name=name)
    assert report.model_name == name


def test_inference_report_init_model_name_not_given():
    model = MockModel()
    name = "my_test_model"
    report = InferenceReport(model.infer_callable, 1, n_iterations=1)
    assert report.model_name == "Model"


def test_inference_report_init_no_interval_given():
    model = MockModel()
    with pytest.raises(MeasurementIntervalNotSetError):
        report = InferenceReport(model.infer_callable, 1)


def test_inference_report_init_two_interval_given_defaults_to_seconds():
    model = MockModel()

    report = InferenceReport(model.infer_callable, 1, n_iterations=10, n_seconds=1)
    assert report.n_iterations == None
    assert report.n_seconds == 1


def test_inference_report_run_seconds():
    model = MockModel()

    report = InferenceReport(model.infer_callable, 1, n_seconds=1)
    results = report.run(print_report=False)
    assert isinstance(results, dict)
    for expected in EXPECTED_METRICS:
        assert expected in results.keys()


def test_inference_report_run_long_report_period_takes_more_time():
    model = MockModel()

    report = InferenceReport(model.infer_callable, 1, n_seconds=1)
    report1 = InferenceReport(model.infer_callable, 1, n_seconds=2)

    results = report.run(print_report=False)
    results1 = report1.run(print_report=False)

    assert results1["Took"] > results["Took"]


def test_inference_report_run_seconds_failure_point():
    model = MockModel(delay=0.5)

    report = InferenceReport(
        model.infer_callable, 1, n_seconds=1, infer_failure_point=0.51
    )
    results = report.run(print_report=False)
    assert results["Fail"] == 1


def test_inference_report_run_seconds_failure_point():
    model = MockModel(delay=0.51)

    report = InferenceReport(
        model.infer_callable, 1, n_seconds=1, infer_failure_point=0.50
    )
    results = report.run(print_report=False)
    assert results["Fail"] >= 1


def test_inference_report_run_iterations():
    model = MockModel()

    report = InferenceReport(model.infer_callable, 1, n_iterations=30)
    results = report.run(print_report=False)
    assert isinstance(results, dict)
    for expected in EXPECTED_METRICS:
        assert expected in results.keys()
