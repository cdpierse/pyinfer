from unittest import mock

import pytest
from pyinfer import InferenceReport
from pyinfer.errors import MeasurementIntervalNotSetError, ModelIsNotCallableError


class MockModel:
    def infer_callable(self, input):
        time.sleep(0.1)
        return input * pow(input, 2)


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
