import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataloader import HabitatDataOffline
from datasets.dataloader import HabitatDataOfflineMPv2

from models.predictors import get_predictor
from models.img_segmentation import get_img_segmentor_from_options
import datasets.util.utils as utils
import datasets.util.viz_utils as viz_utils
import datasets.util.map_utils as map_utils
import test_utils as tutils
from models.semantic_grid import SemanticGrid
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import metrics
import json
import cv2


class SemMapTester(object):
    """ Implements testing for prediction models
    """
    def __init__(self, options):
        self.options = options
        print("options:")
        for k in self.options.__dict__.keys():
            print(k, self.options.__dict__[k])

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.test_ds = HabitatDataOfflineMPv2(options, config_file=options.config_test_file)

        ensemble_exp = os.listdir(self.options.ensemble_dir) # ensemble_dir should be a dir that holds multiple experiments
        ensemble_exp.sort() # in case the models are numbered put them in order
        N = len(ensemble_exp) # number of models in the ensemble
        self.models_dict = {} # keys are the ids of the models in the ensemble
        for n in range(self.options.ensemble_size):
            # self.models_dict[n] = {'predictor_model': get_predictor_from_options(self.options)}

            self.models_dict[n] = {'predictor_model': get_predictor(self.options)}
            self.models_dict[n] = {k:v.to(self.device) for k,v in self.models_dict[n].items()}

            # Needed only for models trained with multi-gpu setting
            self.models_dict[n]['predictor_model'] = nn.DataParallel(self.models_dict[n]['predictor_model'])

            checkpoint_dir = self.options.ensemble_dir + "/" + ensemble_exp[n]
            print('checkpoint_dir', checkpoint_dir)

            latest_checkpoint = tutils.get_latest_model(save_dir=checkpoint_dir)
            print("Model", n, "loading checkpoint", latest_checkpoint)
            self.models_dict[n] = tutils.load_model(models=self.models_dict[n], checkpoint_file=latest_checkpoint)
            self.models_dict[n]["predictor_model"].eval()

        self.spatial_classes = {0:"void", 1:"occupied", 2:"free"}
        self.object_classes = {0:"void", 17:"floor", 15:'wall', 3:"table", 4:"cushion", 13:"counter", 1:"chair", 5:"sofa", 6:"bed"}

        # initialize res dicts
        self.results_spatial = {}
        self.results_objects = {}
        for object_ in list(self.object_classes.values()):
            self.results_objects[object_] = {}
        self.results_objects['objects_all'] = {}
        self.results_objects['original_result'] = {}

        for spatial in list(self.spatial_classes.values()):
            self.results_spatial[spatial] = {}
        self.results_spatial['spatial_all'] = {}

    def test_semantic_map(self):
        test_data_loader = DataLoader(self.test_ds,
                                batch_size=self.options.test_batch_size,
                                num_workers=self.options.num_workers,
                                pin_memory=self.options.pin_memory,
                                shuffle=self.options.shuffle_test)
        batch = None
        self.options.test_iters = len(test_data_loader) # the length of dataloader depends on the batch size
        object_labels = list(range(self.options.n_object_classes))
        spatial_labels = list(range(self.options.n_spatial_classes))
        overall_confusion_matrix_objects, overall_confusion_matrix_spatial = None, None

        for tstep, batch in enumerate(tqdm(test_data_loader,
                                           desc='Testing',
                                           total=self.options.test_iters)):
            # ep_name = batch['name']

            batch = {k: v.to(self.device) for k, v in batch.items() if k != 'name'}

            with torch.no_grad():

                gt_crops_spatial = batch['gt_grid_crops_spatial'].cpu()  # B x T x 1 x cH x cW
                gt_crops_objects = batch['gt_grid_crops_objects'].cpu()  # B x T x 1 x cH x cW

                ensemble_object_maps, ensemble_spatial_maps = [], []
                N = len(self.models_dict) # numbers of models in the ensemble
                for n in range(self.options.ensemble_size):
                    pred_output = self.models_dict[n]['predictor_model'](batch)
                    ensemble_object_maps.append(pred_output['pred_maps_objects'].clone())

                ensemble_object_maps = torch.stack(ensemble_object_maps)  # N x B x T x C x cH x cW

                # Getting the mean predictions from the ensemble
                pred_maps_objects = torch.mean(ensemble_object_maps, dim=0)  # B x T x C x cH x cW


                # Decide label for each location based on predition probs
                pred_labels_objects = torch.argmax(pred_maps_objects.cpu(), dim=2, keepdim=True) # B x T x 1 x cH x cW

                current_confusion_matrix_objects = confusion_matrix(y_true=gt_crops_objects.flatten(), y_pred=pred_labels_objects.flatten(), labels=object_labels)
                current_confusion_matrix_objects = torch.tensor(current_confusion_matrix_objects)

                if overall_confusion_matrix_objects is None:
                    overall_confusion_matrix_objects = current_confusion_matrix_objects
                else:
                    overall_confusion_matrix_objects += current_confusion_matrix_objects


        mAcc_obj = metrics.overall_pixel_accuracy(overall_confusion_matrix_objects)
        class_mAcc_obj, per_class_Acc = metrics.per_class_pixel_accuracy(overall_confusion_matrix_objects)
        mIoU_obj, per_class_IoU = metrics.jaccard_index(overall_confusion_matrix_objects)
        mF1_obj, per_class_F1 = metrics.F1_Score(overall_confusion_matrix_objects)

        self.results_objects['original_result'] = {
            'mean_interesction_over_union_objects': mIoU_obj.numpy().tolist(),
            'mean_f1_score_objects': mF1_obj.numpy().tolist(),
            'overall_pixel_accuracy_objects': mAcc_obj.numpy().tolist(),
            'per_class_pixel_accuracy_objects': class_mAcc_obj.numpy().tolist(),
            'per_class_IoU': per_class_IoU.numpy().tolist(),
            'per_class_mAcc_obj': per_class_Acc.numpy().tolist(),
            'per_class_F1': per_class_F1.numpy().tolist()}

        print("\nSemantic prediction results:")
        classes = list(self.object_classes.keys())
        classes.sort()
        per_class_Acc = per_class_Acc[classes]
        per_class_IoU = per_class_IoU[classes]
        per_class_F1 = per_class_F1[classes]
        for i in range(len(classes)):
            lbl = classes[i]
            self.results_objects[self.object_classes[lbl]] = {'Acc': per_class_Acc[i].item(),
                                                              'IoU': per_class_IoU[i].item(),
                                                              'F1': per_class_F1[i].item()}
            print("Class:", self.object_classes[lbl], "Acc:", per_class_Acc[i], "IoU:", per_class_IoU[i], "F1:", per_class_F1[i])
        mean_per_class_Acc = torch.mean(per_class_Acc)
        mean_per_class_IoU = torch.mean(per_class_IoU)
        mean_per_class_F1 = torch.mean(per_class_F1)
        print("mAcc:", mean_per_class_Acc, "mIoU:", mean_per_class_IoU, "mF1:", mean_per_class_F1)
        self.results_objects['objects_all']['mAcc'] = mean_per_class_Acc.item()
        self.results_objects['objects_all']['mIoU'] = mean_per_class_IoU.item()
        self.results_objects['objects_all']['mF1'] = mean_per_class_F1.item()

        res = {
            # **self.results_spatial,
            **self.results_objects}
        with open(self.options.log_dir+'/sem_map_results.json', 'w') as outfile:
            json.dump(res, outfile, indent=4)

        # save the confusion matrices
        filepath = self.options.log_dir+'/confusion_matrices.npz'
        np.savez_compressed(filepath,
                            # overall_confusion_matrix_spatial=overall_confusion_matrix_spatial,
                            overall_confusion_matrix_objects=overall_confusion_matrix_objects)

        print()
        print(overall_confusion_matrix_objects)
