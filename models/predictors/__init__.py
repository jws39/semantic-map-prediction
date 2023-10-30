""" Prediction models
"""

from models.networks import get_mp_network
from .map_predictor_model import MapPredictorAM


def get_predictor(options):
    print('--------------------------with_am--------------------------')
    return MapPredictorAM(segmentation_model=get_mp_network(options),
                                  map_loss_scale=options.map_loss_scale)
                                  # with_img_segm=options.with_img_segm)



