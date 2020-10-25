class MeasurementIntervalNotSetError(RuntimeError):
    "Raises when user does not specify the number of seconds or the number of iterations to run for the reporting period"


class ModelIsNotCallableError(RuntimeError):
    "Raise when the object provided as a model is not a python `Callable` object"
