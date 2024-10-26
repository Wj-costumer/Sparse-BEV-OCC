import torch
import queue
import numpy as np
from mmcv.runner import get_dist_info
from mmcv.runner.fp16_utils import cast_tensor_type
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from .utils import pad_multiple, GpuPhotoMetricDistortion, GridMask
from mmdet3d.models import builder
import copy

@DETECTORS.register_module()
class SparseOcc(MVXTwoStageDetector):
    def __init__(self,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 pts_occ_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 data_aug=None,
                 stop_prev_grad=None,
                 use_mask_camera=False,
                 **kwargs):

        super(SparseOcc, self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)

        self.use_mask_camera = use_mask_camera
        self.fp16_enabled = False
        self.data_aug = data_aug
        self.stop_prev_grad = stop_prev_grad
        self.color_aug = GpuPhotoMetricDistortion()
        self.grid_mask = GridMask(ratio=0.5, prob=0.7)
        self.use_grid_mask = True
        self.memory = {}
        self.queue = queue.Queue()

        self.pts_occ_head = builder.build_head(pts_occ_head)
        
        # freeze occ head model parameters
        for name, param in self.named_parameters():
            if ("img" in name) or ("occ" in name):
                param.requires_grad = False

        
    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_img_feat(self, img):
        if self.use_grid_mask:
            img = self.grid_mask(img)
            
        img_feats = self.img_backbone(img)

        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())

        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        return img_feats

    @auto_fp16(apply_to=('img'))
    def extract_feat(self, img, img_metas=None):
        """Extract features from images and points."""
        if len(img.shape) == 6:
            img = img.flatten(1, 2)  # [B, TN, C, H, W]

        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)
        img = img.float()

        if self.data_aug is not None:
            if 'img_color_aug' in self.data_aug and self.data_aug['img_color_aug'] and self.training:
                img = self.color_aug(img)

            if 'img_norm_cfg' in self.data_aug:
                img_norm_cfg = self.data_aug['img_norm_cfg']

                norm_mean = torch.tensor(img_norm_cfg['mean'], device=img.device)
                norm_std = torch.tensor(img_norm_cfg['std'], device=img.device)

                if img_norm_cfg['to_rgb']:
                    img = img[:, [2, 1, 0], :, :]  # BGR to RGB

                img = img - norm_mean.reshape(1, 3, 1, 1)
                img = img / norm_std.reshape(1, 3, 1, 1)

            for b in range(B):
                img_shape = (img.shape[2], img.shape[3], img.shape[1])
                img_metas[b]['img_shape'] = [img_shape for _ in range(N)]
                img_metas[b]['ori_shape'] = [img_shape for _ in range(N)]

            if 'img_pad_cfg' in self.data_aug:
                img_pad_cfg = self.data_aug['img_pad_cfg']
                img = pad_multiple(img, img_metas, size_divisor=img_pad_cfg['size_divisor'])
                H, W = img.shape[-2:]

        input_shape = img.shape[-2:]
        # update real input shape of each single img
        for img_meta in img_metas:
            img_meta.update(input_shape=input_shape)

        img_feats = self.extract_img_feat(img)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))

        return img_feats_reshaped

    def forward_pts_train(self, 
                          mlvl_feats, 
                          gt_bboxes_3d,
                          gt_labels_3d,
                          voxel_semantics, 
                          voxel_instances, 
                          instance_class_ids,
                          mask_camera, 
                          img_metas):
        """
        voxel_semantics: [bs, 200, 200, 16], value in range [0, num_cls - 1]
        voxel_instances: [bs, 200, 200, 16], value in range [0, num_obj - 1]
        instance_class_ids: [[bs0_num_obj], [bs1_num_obj], ...], value in range [0, num_cls - 1]
        """
        mlvl_feats_copy = [copy.deepcopy(mlvl_feats[i].detach()) for i in range(len(mlvl_feats))]
        img_metas_copy = copy.deepcopy(img_metas)
      
        occ_outs = self.pts_occ_head(mlvl_feats, img_metas)
        det_outs = self.pts_bbox_head(mlvl_feats_copy, img_metas_copy)
        det_loss_inputs = [gt_bboxes_3d, gt_labels_3d, det_outs]
        occ_loss_inputs = [voxel_semantics, voxel_instances, instance_class_ids, occ_outs]
        det_loss = self.pts_bbox_head.loss(*det_loss_inputs)
        occ_loss = self.pts_occ_head.loss(*occ_loss_inputs)
        det_loss.update(occ_loss)
        loss_dict = det_loss
        return loss_dict

    def forward(self, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    @force_fp32(apply_to=('img'))
    def forward_train(self, 
                      img_metas=None, 
                      img=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None, 
                      voxel_semantics=None, 
                      voxel_instances=None, 
                      instance_class_ids=None, 
                      mask_camera=None, 
                      **kwargs):
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        for i in range(len(img_metas)):
            img_metas[i]['gt_bboxes_3d'] = gt_bboxes_3d[i]
            img_metas[i]['gt_labels_3d'] = gt_labels_3d[i]

        return self.forward_pts_train(img_feats, gt_bboxes_3d, gt_labels_3d, voxel_semantics, voxel_instances, instance_class_ids, mask_camera, img_metas)

    def forward_test(self, img_metas, img=None, **kwargs):
        bbox_output, occ_output = self.simple_test(img_metas, img)

        sem_pred = occ_output['sem_pred'].cpu().numpy().astype(np.uint8)
        occ_loc = occ_output['occ_loc'].cpu().numpy().astype(np.uint8)

        batch_size = sem_pred.shape[0]

        if 'pano_inst' and 'pano_sem' in occ_output:
            # important: uint8 is not enough for pano_pred
            pano_inst = occ_output['pano_inst'].cpu().numpy().astype(np.int16)
            pano_sem = occ_output['pano_sem'].cpu().numpy().astype(np.uint8)
            outputs = [{
                    'sem_pred': sem_pred[b:b+1],
                    'pano_inst': pano_inst[b:b+1],
                    'pano_sem': pano_sem[b:b+1],
                    'occ_loc': occ_loc[b:b+1],
                    'pts_bbox': bbox_output[b:b+1]
                } for b in range(batch_size)]
            
            return outputs
        else:
            outputs = [{
                'sem_pred': sem_pred[b:b+1],
                'occ_loc': occ_loc[b:b+1],
                'pts_bbox': bbox_output[b:b+1]
            } for b in range(batch_size)]
        return outputs

    def simple_test_pts(self, x, img_metas, rescale=False):
        # x and img_metas have been changed in occ module or det module
        img_metas_copy = copy.deepcopy(img_metas)
        x_copy = copy.deepcopy(x)
        occ_outs = self.pts_occ_head(x, img_metas)
        det_outs = self.pts_bbox_head(x_copy, img_metas_copy)
        bbox_list = self.pts_bbox_head.get_bboxes(det_outs, img_metas_copy[0], rescale=rescale)

        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        
        occ_results = self.pts_occ_head.merge_occ_pred(occ_outs)
        return bbox_results, occ_results

    def simple_test(self, img_metas, img=None, rescale=False):
        world_size = get_dist_info()[1]
        if world_size == 1:  # online
            return self.simple_test_online(img_metas, img, rescale)
        else:  # offline
            return self.simple_test_offline(img_metas, img, rescale)

    def simple_test_offline(self, img_metas, img=None, rescale=False):
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        
        # bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_results, occ_results = self.simple_test_pts(img_feats, img_metas, rescale=rescale)
        return bbox_results, occ_results

    def simple_test_online(self, img_metas, img=None, rescale=False):

        self.fp16_enabled = False
        assert len(img_metas) == 1  # batch_size = 1
        
        B, N, C, H, W = img.shape
        
        img = img.reshape(B, N//6, 6, C, H, W)

        img_filenames = img_metas[0]['filename']
        img = img.repeat(1, 8, 1, 1, 1, 1)    
        
        num_frames = len(img_filenames) // 6
        # assert num_frames == img.shape[1]

        img_shape = (H, W, C)
        img_metas[0]['img_shape'] = [img_shape for _ in range(len(img_filenames))]
        img_metas[0]['ori_shape'] = [img_shape for _ in range(len(img_filenames))]
        img_metas[0]['pad_shape'] = [img_shape for _ in range(len(img_filenames))]

        img_feats_list, img_metas_list = [], []

        # extract feature frame by frame
        for i in range(num_frames):
            img_indices = list(np.arange(i * 6, (i + 1) * 6))

            img_metas_curr = [{}]
            for k in img_metas[0].keys():
                if isinstance(img_metas[0][k], list):
                    img_metas_curr[0][k] = [img_metas[0][k][i] for i in img_indices]

            if img_filenames[img_indices[0]] in self.memory:
                # found in memory
                img_feats_curr = self.memory[img_filenames[img_indices[0]]]
            else:
                try:
                    img_feats_curr = self.extract_feat(img[:, i], img_metas_curr)
                    # breakpoint()
                except:
                    breakpoint()
                self.memory[img_filenames[img_indices[0]]] = img_feats_curr
                self.queue.put(img_filenames[img_indices[0]])
                while self.queue.qsize() > 16:  # avoid OOM
                    pop_key = self.queue.get()
                    self.memory.pop(pop_key)

            img_feats_list.append(img_feats_curr)
            img_metas_list.append(img_metas_curr)

        # reorganize
        feat_levels = len(img_feats_list[0])
        img_feats_reorganized = []
        for j in range(feat_levels):
            feat_l = torch.cat([img_feats_list[i][j] for i in range(len(img_feats_list))], dim=0)
            feat_l = feat_l.flatten(0, 1)[None, ...]
            img_feats_reorganized.append(feat_l)

        img_metas_reorganized = img_metas_list[0]
        for i in range(1, len(img_metas_list)):
            for k, v in img_metas_list[i][0].items():
                if isinstance(v, list):
                    img_metas_reorganized[0][k].extend(v)

        img_feats = img_feats_reorganized
        img_metas = img_metas_reorganized
        img_feats = cast_tensor_type(img_feats, torch.half, torch.float32)

        # run detector
        bbox_results, occ_results = self.simple_test_pts(img_feats, img_metas, rescale=rescale)

        return bbox_results, occ_results