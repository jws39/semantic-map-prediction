import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.networks import get_mp_network
import test_utils as tutils
from sklearn.utils.class_weight import compute_class_weight


class MapPredictorAM(nn.Module):
    # def __init__(self, segmentation_model, map_loss_scale, with_img_segm):
    def __init__(self, segmentation_model, map_loss_scale):

        super(MapPredictorAM, self).__init__()
        self._segmentation_model = segmentation_model
        self._map_loss_scale = map_loss_scale
        # self.with_img_segm = with_img_segm

        # self.cel_loss_spatial = nn.CrossEntropyLoss()
        # original code no index ignored
        self.cel_loss_objects = nn.CrossEntropyLoss()
        self.am_loss = nn.BCELoss()

    def forward(self, batch, is_train=True):

        step_ego_crops = batch['step_ego_grid_crops_spatial']
        B, T, _, cH, cW = step_ego_crops.shape  # batch, sequence length, _, crop height, crop width

        pred_maps_raw_objects, pred_am = self._segmentation_model(batch['step_ego_grid_27'])

        # if self.with_img_segm:
            # for my models (sscnav, sscnav_light, cfnet, cmanet)
            # for data_v2
            # pred_maps_raw_objects, pred_am = self._segmentation_model(batch['step_ego_grid_27'])

            # pred_maps_raw_spatial, pred_maps_raw_objects = self._segmentation_model(step_ego_crops,
            #                                                                         batch['pred_ego_crops_sseg'])
        # else:
        #     pred_maps_raw_spatial, pred_maps_raw_objects = self._segmentation_model(step_ego_crops)

        # number of classes for each case
        # spatial_C = pred_maps_raw_spatial.shape[1]
        objects_C = pred_maps_raw_objects.shape[1]

        # Get a prob distribution over the labels
        # pred_maps_raw_spatial = pred_maps_raw_spatial.view(B, T, spatial_C, cH, cW)
        # pred_maps_spatial = F.softmax(pred_maps_raw_spatial, dim=2)
        pred_maps_raw_objects = pred_maps_raw_objects.view(B, T, objects_C, cH, cW)
        pred_maps_objects = F.softmax(pred_maps_raw_objects, dim=2)

        output = {
                  # 'pred_maps_raw_spatial': pred_maps_raw_spatial,
                  'pred_maps_raw_objects': pred_maps_raw_objects,
                  'pred_am': pred_am,
                  # 'pred_maps_spatial': pred_maps_spatial,
                  'pred_maps_objects': pred_maps_objects}

        return output

    def construct_ideal_affinity_matrix(self, label, label_size):
        num_classes = 27
        scaled_labels = F.interpolate(
            label.float(), size=label_size, mode="nearest")

        scaled_labels = scaled_labels.squeeze_().long()
        one_hot_labels = F.one_hot(scaled_labels, num_classes)
        one_hot_labels = one_hot_labels.view(
            one_hot_labels.size(0), -1, num_classes).float()
        ideal_affinity_matrix = torch.bmm(one_hot_labels,
                                          one_hot_labels.permute(0, 2, 1))
        return ideal_affinity_matrix

    ### balancing the weight 0504
    def loss_cel(self, batch, pred_outputs):

        # pred_maps_raw_spatial = pred_outputs['pred_maps_raw_spatial']
        # B, T, spatial_C, cH, cW = pred_maps_raw_spatial.shape
        # objects_C = pred_maps_raw_objects.shape[2]
        # gt_crops_spatial, gt_crops_objects = batch['gt_grid_crops_spatial'], batch['gt_grid_crops_objects']
        # pred_map_loss_spatial = self.cel_loss_spatial(input=pred_maps_raw_spatial.view(B * T, spatial_C, cH, cW),
        #                                               target=gt_crops_spatial.view(B * T, cH, cW))
        # pred_map_err_spatial = pred_map_loss_spatial.clone().detach()

        pred_maps_raw_objects = pred_outputs['pred_maps_raw_objects']
        B, T, objects_C, cH, cW = pred_maps_raw_objects.shape
        gt_crops_objects = batch['gt_grid_crops_objects']

        pred_map_loss_objects = self.cel_loss_objects(input=pred_maps_raw_objects.view(B * T, objects_C, cH, cW),
                                                      target=gt_crops_objects.view(B * T, cH, cW))
        pred_map_err_objects = pred_map_loss_objects.clone().detach()

        # label_size = (32, 32)
        label_size = (64, 64)

        gt_am = self.construct_ideal_affinity_matrix(batch['gt_grid_crops_objects'].view(B * T, 1, cH, cW), label_size)
        am_loss = self.am_loss(torch.sigmoid(pred_outputs['pred_am']), gt_am)
        am_err = am_loss.clone().detach()

        output = {}
        # output['pred_map_err_spatial'] = pred_map_err_spatial
        # output['pred_map_loss_spatial'] = self._map_loss_scale * pred_map_loss_spatial

        output['pred_map_err_objects'] = pred_map_err_objects
        output['pred_map_loss_objects'] = self._map_loss_scale * pred_map_loss_objects

        output['pred_am_err'] = am_err
        # output['pred_obj_loss'] = self._map_loss_scale * obj_loss
        output['pred_am_loss'] = am_loss

        # For wacv paper, loss test weight
        # output['pred_map_loss_objects'] = 0.1 * output['pred_map_loss_objects'] + 0.9 * output['pred_am_loss']
        output['pred_map_loss_objects'] = 0.3 * output['pred_map_loss_objects'] + 0.7 * output['pred_am_loss']
        # output['pred_map_loss_objects'] = 0.5 * output['pred_map_loss_objects'] + 0.5 * output['pred_am_loss']
        # output['pred_map_loss_objects'] = 0.7 * output['pred_map_loss_objects'] + 0.3 * output['pred_am_loss']
        # output['pred_map_loss_objects'] = 0.9 * output['pred_map_loss_objects'] + 0.1 * output['pred_am_loss']


        return output

    ### origional code without balancing the weight 0504
    def loss_cel_no_balance_weight(self, batch, pred_outputs):

        # pred_maps_raw_spatial = pred_outputs['pred_maps_raw_spatial']
        # B, T, spatial_C, cH, cW = pred_maps_raw_spatial.shape
        # objects_C = pred_maps_raw_objects.shape[2]
        # gt_crops_spatial, gt_crops_objects = batch['gt_grid_crops_spatial'], batch['gt_grid_crops_objects']
        # pred_map_loss_spatial = self.cel_loss_spatial(input=pred_maps_raw_spatial.view(B * T, spatial_C, cH, cW),
        #                                               target=gt_crops_spatial.view(B * T, cH, cW))
        # pred_map_err_spatial = pred_map_loss_spatial.clone().detach()

        pred_maps_raw_objects = pred_outputs['pred_maps_raw_objects']
        B, T, objects_C, cH, cW = pred_maps_raw_objects.shape
        gt_crops_objects = batch['gt_grid_crops_objects']

        pred_map_loss_objects = self.cel_loss_objects(input=pred_maps_raw_objects.view(B * T, objects_C, cH, cW),
                                                      target=gt_crops_objects.view(B * T, cH, cW))
        pred_map_err_objects = pred_map_loss_objects.clone().detach()

        # label_size = (32, 32)
        label_size = (64, 64)

        gt_am = self.construct_ideal_affinity_matrix(batch['gt_grid_crops_objects'].view(B * T, 1, cH, cW), label_size)
        am_loss = self.am_loss(torch.sigmoid(pred_outputs['pred_am']), gt_am)
        am_err = am_loss.clone().detach()

        output = {}
        # output['pred_map_err_spatial'] = pred_map_err_spatial
        # output['pred_map_loss_spatial'] = self._map_loss_scale * pred_map_loss_spatial

        output['pred_map_err_objects'] = pred_map_err_objects
        output['pred_map_loss_objects'] = self._map_loss_scale * pred_map_loss_objects

        output['pred_am_err'] = am_err
        # output['pred_obj_loss'] = self._map_loss_scale * obj_loss
        output['pred_am_loss'] = am_loss

        output['pred_map_loss_objects'] = 0.3 * output['pred_map_loss_objects'] + 0.7 * output['pred_am_loss']
        # output['pred_map_loss_objects'] = 0.5 * output['pred_map_loss_objects'] + 0.5 * output['pred_am_loss']
        # output['pred_map_loss_objects'] = 0.1 * output['pred_map_loss_objects'] + 0.9 * output['pred_am_loss']

        return output

