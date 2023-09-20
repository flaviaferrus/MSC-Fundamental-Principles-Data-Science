import numpy as np
from sklearn.metrics import accuracy_score
from advanced import Logger
from typing import Dict

logger = Logger()


def score_computation(prediction: Dict[str, np.ndarray],
                      observation: Dict[str, np.ndarray],
                      metric: str = 'accuracy') -> None:
    logger.debug(f"Validating the model's score using {metric}")
    scores = {}
    for k in observation.keys():
        if metric == 'accuracy':
            scores[k] = accuracy_score(observation[k], prediction[k])
        else:
            raise ValueError("Unrecognized metric to evaluate: ", metric)

        logger.debug(f"\t {k} score {str(scores[k])}")


# def validate() -> None:
#     ml = Model()
#     obs_dict: Dict[str, np.ndarray] = {'train': ml.y_train.values}
#     pred_dict: Dict[str, np.ndarray] = {'train': ml.predict(ml.x_train)}
#     obs_dict['test'] = ml.y_test.values
#     pred_dict['test'] = ml.predict(ml.x_test)
#
#     score_computation(obs_dict, pred_dict)
