from unittest import mock

import pytest

from .pyinfer import InferenceReport


class MockModel:
    def infer_callable(self, input):
        time.sleep(0.1)
        return input * pow(input, 2)


# def test_inference_report_init():
#     model = MockModel()
#     report = InferenceReport(model, 8, n_seconds=1)


def test_base():
    assert 1 == 1
