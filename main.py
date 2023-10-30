
""" Entry point for training
"""

from train_options import TrainOptions
from mp_trainer import MP_TrainerAM
from tester import SemMapTester
import multiprocessing as mp


if __name__ == '__main__':

    mp.set_start_method('forkserver', force=True)
    options = TrainOptions().parse_args()

    if options.is_train:
        trainer = MP_TrainerAM(options)
        trainer.train()

    else:
        if options.sem_map_test:
            tester = SemMapTester(options)
            tester.test_semantic_map()

