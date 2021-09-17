from typing import List, Optional

from .generative.bleu import IDBLEU, OODBLEU
from .generative.compositionality import SyntacticCompositionality
from .generative.confidence import GenerationConfidence
from .discriminative.accuracy import OODAccuracy, IDAccuracy
from .discriminative.consistency import PredictionConsistency
from .discriminative.discriminative_estimator import DiscriminativeEstimator
from .discriminative.distributional import DistributionalCoherence
from .discriminative.overconfidence import Overconfidence

classification_estimators = [
    PredictionConsistency,
    DistributionalCoherence,
    Overconfidence,
    IDAccuracy,
    OODAccuracy
]

translation_estimators = [
    GenerationConfidence,
    SyntacticCompositionality,
    IDBLEU,
    OODBLEU
]


def get_callbacks(task: str,
                  custom_subset: Optional[List[DiscriminativeEstimator]] = None,
                  **kwargs):
    if custom_subset is None:
        if task == "classification":
            estimators = [Estimator(**kwargs) for Estimator in classification_estimators]
        elif task == "translation":
            estimators = [Estimator(**kwargs) for Estimator in translation_estimators]
        else:
            raise ValueError(task)
    else:
        estimators = custom_subset

    return [estimator.get_callback() for estimator in estimators]
