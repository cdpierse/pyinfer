import math
import random
import statistics
import time

import psutil
from transformers import pipeline

from pyinfer import InferenceReport, MultiInferenceReport


def my_model(input: int) -> int:
    time.sleep(0.01)
    return input * pow(input, 2)


pipe = pipeline(task="sentiment-analysis")


class MockModel:
    def run(self, input):
        time.sleep(random.choice([0.041, 0.012, 0.067, 0.036, 0.001, 0.002]))
        return (input ** 2 / 3) ** math.pi


mock_model = MockModel()
mock_model1 = MockModel()
_input = "The light released from around the first massive black holes in the universe is so intense that it is able to reach telescopes across the entire expanse of the universe. Incredibly, the light from the most distant black holes (or quasars) has been traveling to us for more than 13 billion light years. However, we do not know how these monster black holes formed. The light released from around the first massive black holes in the universe is so intense that it is able to reach telescopes across the entire expanse of the universe. Incredibly, the light from the most distant black holes (or quasars) has been traveling to us for more than 13 billion light years. However, we do not know how these monster black holes formed. New research led by researchers from Georgia Institute of Technology, Dublin City University, Michigan State University, the University of California at San Diego, the San Diego Supercomputer Center and IBM, provides a new and extremely promising avenue for solving this cosmic riddle. The team showed that when galaxies assemble extremely rapidly -- and sometimes violently -- that can lead to the formation of very massive black holes. In these rare galaxies, normal star formation is disrupted and black hole formation takes over. The new study finds that massive black holes form in dense starless regions that are growing rapidly, turning upside down the long-accepted belief that massive black hole formation was limited to regions bombarded by the powerful radiation of nearby galaxies. Conclusions of the study, reported on January 23rd in the journal Nature and supported by funding from the National Science Foundation, the European Union and NASA, also finds that massive black holes are much more common in the universe than previously thought. The key criteria for determining where massive black holes formed during the universe's infancy relates to the rapid growth of pre-galactic gas clouds that are the forerunners of all present-day galaxies, meaning that most supermassive black holes have a common origin forming in this newly discovered scenario, said John Wise, an associate professor in the Center for Relativistic Astrophysics at Georgia Tech and the paper's corresponding author. Dark matter collapses into halos that are the gravitational glue for all galaxies. Early rapid growth of these halos prevented the formation of stars that would have competed with black holes for gaseous matter flowing into the area."

# report = InferenceReport(
#     model=pipe, inputs=_input, n_seconds=30, infer_failure_point=0.480
# )
# report.run()
# report.plot(report.runs)


multi_report = MultiInferenceReport(
    [pipe, pipe],
    [_input, "this is a short sentence"],
    n_iterations=30,
    infer_failure_point=1,
    model_names=["SA long", "SA short"],
)
results_list = multi_report.run()
multi_report.plot()
