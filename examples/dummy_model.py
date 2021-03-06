import time

from pyinfer import InferenceReport


class DummyModel:
    def infer(self, input):
        time.sleep(0.1)
        return input * pow(input, 2)


dummy = DummyModel()
report = InferenceReport(
    model=dummy.infer,
    inputs=7,
    n_iterations=2,
    infer_failure_point=2,
)
report.run()
