class MeasurementIntervalNotSetError(RuntimeError):
    "Raises when user does not specify the number of seconds or the number of iterations to run for the reporting period"


class ModelIsNotCallableError(RuntimeError):
    "Raise when the object provided as a model is not a python `Callable` object"


class NamesNotEqualsModelsLengthError(RuntimeError):
    "Raise when the number of names given for a multi report does not equal the number of models"


class MatplotlibNotInstalledError(RuntimeError):
    "Raise when user attempts to use a plotting function that depends on Matplotlib"
