import os
import mmcv
import glob
import torch
import numpy as np
from tqdm import tqdm
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
from nuscenes.eval.common.utils import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix
from torch.utils.data import DataLoader
from models.utils import sparse2dense
from .ray_metrics import main_rayiou, main_raypq
from .ego_pose_dataset import EgoPoseDataset
from configs.r50_nuimg_704x256_8f import occ_class_names as occ3d_class_names
from configs.r50_nuimg_704x256_8f_openocc import occ_class_names as openocc_class_names

@DATASETS.register_module()
class NuSceneOcc(NuScenesDataset):    
    def __init__(self, occ_gt_root, *args, **kwargs):
        super().__init__(filter_empty_gt=True, *args, **kwargs) # change filter_empty_gt
        self.occ_gt_root = occ_gt_root
        self.data_infos = self.load_annotations(self.ann_file)

        self.token2scene = {}
        for gt_path in glob.glob(os.path.join(self.occ_gt_root, '*/*/*.npz')):
            token = gt_path.split('/')[-2]
            scene_name = gt_path.split('/')[-3]
            self.token2scene[token] = scene_name

        for i in range(len(self.data_infos)):
            scene_name = self.token2scene[self.data_infos[i]['token']]
            self.data_infos[i]['scene_name'] = scene_name

    def collect_sweeps(self, index, into_past=150, into_future=0):
        all_sweeps_prev = []
        curr_index = index
        while len(all_sweeps_prev) < into_past:
            curr_sweeps = self.data_infos[curr_index]['sweeps']
            if len(curr_sweeps) == 0:
                break
            all_sweeps_prev.extend(curr_sweeps)
            all_sweeps_prev.append(self.data_infos[curr_index - 1]['cams'])
            curr_index = curr_index - 1
        
        all_sweeps_next = []
        curr_index = index + 1
        while len(all_sweeps_next) < into_future:
            if curr_index >= len(self.data_infos):
                break
            curr_sweeps = self.data_infos[curr_index]['sweeps']
            all_sweeps_next.extend(curr_sweeps[::-1])
            all_sweeps_next.append(self.data_infos[curr_index]['cams'])
            curr_index = curr_index + 1

        return all_sweeps_prev, all_sweeps_next

    def get_data_info(self, index):
        info = self.data_infos[index]
        sweeps_prev, sweeps_next = self.collect_sweeps(index)

        ego2global_translation = info['ego2global_translation']
        ego2global_rotation = info['ego2global_rotation']
        lidar2ego_translation = info['lidar2ego_translation']
        lidar2ego_rotation = info['lidar2ego_rotation']
        ego2global_rotation_mat = Quaternion(ego2global_rotation).rotation_matrix
        lidar2ego_rotation_mat = Quaternion(lidar2ego_rotation).rotation_matrix

        input_dict = dict(
            sample_idx=info['token'],
            sweeps={'prev': sweeps_prev, 'next': sweeps_next},
            timestamp=info['timestamp'] / 1e6,
            ego2global_translation=ego2global_translation,
            ego2global_rotation=ego2global_rotation_mat,
            lidar2ego_translation=lidar2ego_translation,
            lidar2ego_rotation=lidar2ego_rotation_mat,
        )

        ego2lidar = transform_matrix(lidar2ego_translation, Quaternion(lidar2ego_rotation), inverse=True)
        input_dict['ego2lidar'] = [ego2lidar for _ in range(6)]
        input_dict['occ_path'] = os.path.join(self.occ_gt_root, info['scene_name'], info['token'], 'labels.npz')

        if self.modality['use_camera']:
            img_paths = []
            img_timestamps = []
            lidar2img_rts = []

            for _, cam_info in info['cams'].items():
                img_paths.append(os.path.relpath(cam_info['data_path']))
                img_timestamps.append(cam_info['timestamp'] / 1e6)

                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T

                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

            input_dict.update(dict(
                img_filename=img_paths,
                img_timestamp=img_timestamps,
                lidar2img=lidar2img_rts,
            ))
            
        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict

    def evaluate_occ(self, occ_results, runner=None, show_dir=None, **eval_kwargs):
        occ_gts, occ_preds, inst_gts, inst_preds, lidar_origins = [], [], [], [], []
        print('\nStarting Evaluation...')

        sample_tokens = [info['token'] for info in self.data_infos]

        for batch in DataLoader(EgoPoseDataset(self.data_infos), num_workers=8):
            token = batch[0][0]
            output_origin = batch[1]
            
            data_id = sample_tokens.index(token)
            info = self.data_infos[data_id]

            occ_path = os.path.join(self.occ_gt_root, info['scene_name'], info['token'], 'labels.npz')
            occ_gt = np.load(occ_path, allow_pickle=True)
            gt_semantics = occ_gt['semantics']

            occ_pred = occ_results[data_id]
            sem_pred = torch.from_numpy(occ_pred['sem_pred'])  # [B, N]
            occ_loc = torch.from_numpy(occ_pred['occ_loc'].astype(np.int64))  # [B, N, 3]
            
            data_type = self.occ_gt_root.split('/')[-1]
            if data_type == 'occ3d' or data_type == 'occ3d_panoptic':
                occ_class_names = occ3d_class_names
            elif data_type == 'openocc_v2':
                occ_class_names = openocc_class_names
            else:
                raise ValueError
            free_id = len(occ_class_names) - 1
            
            occ_size = list(gt_semantics.shape)
            sem_pred, _ = sparse2dense(occ_loc, sem_pred, dense_shape=occ_size, empty_value=free_id)
            sem_pred = sem_pred.squeeze(0).numpy()

            if 'pano_inst' in occ_pred.keys():
                pano_inst = torch.from_numpy(occ_pred['pano_inst'])
                pano_sem = torch.from_numpy(occ_pred['pano_sem'])

                pano_inst, _ = sparse2dense(occ_loc, pano_inst, dense_shape=occ_size, empty_value=0)
                pano_sem, _ = sparse2dense(occ_loc, pano_sem, dense_shape=occ_size, empty_value=free_id)
                pano_inst = pano_inst.squeeze(0).numpy()
                pano_sem = pano_sem.squeeze(0).numpy()
                sem_pred = pano_sem

                gt_instances = occ_gt['instances']
                inst_gts.append(gt_instances)
                inst_preds.append(pano_inst)

            lidar_origins.append(output_origin)
            occ_gts.append(gt_semantics)
            occ_preds.append(sem_pred)
        
        if len(inst_preds) > 0:
            results = main_raypq(occ_preds, occ_gts, inst_preds, inst_gts, lidar_origins, occ_class_names=occ_class_names)
            results.update(main_rayiou(occ_preds, occ_gts, lidar_origins, occ_class_names=occ_class_names))
            return results
        else:
            return main_rayiou(occ_preds, occ_gts, lidar_origins, occ_class_names=occ_class_names)

    def format_occ_results(self, occ_results, submission_prefix, **kwargs):
        if submission_prefix is not None:
            mmcv.mkdir_or_exist(submission_prefix)

        for index, occ_pred in enumerate(tqdm(occ_results)):
            info = self.data_infos[index]
            sample_token = info['token']
            save_path = os.path.join(submission_prefix, '{}.npz'.format(sample_token))
            np.savez_compressed(save_path, occ_pred.astype(np.uint8))
        
        print('\nFinished.')
    
    def get_det_results(self, results):
        det_results = [dict() for _ in range(len(results))]
        for i, result in enumerate(results):
            det_results[i]['pts_bbox'] = result['pts_bbox'][0]
        return det_results # [{'pts_bbox': {'boxes_3d'：..., 'scores_3d': ..., 'labels_3d': ...}, ...}]
    
    def format_results(self, results, jsonfile_prefix=None):
        import tempfile
        from os import path as osp
        results = self.get_det_results(results)
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        # currently the output prediction results could be in two formats
        # 1. list of dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...)
        # 2. list of dict('pts_bbox' or 'img_bbox':
        #     dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...))
        # this is a workaround to enable evaluation of both formats on nuScenes
        # refer to https://github.com/open-mmlab/mmdetection3d/issues/449
        if not ('pts_bbox' in results[0] or 'img_bbox' in results[0]):
            result_files = self._format_bbox(results, jsonfile_prefix)
        else:
            # should take the inner dict out of 'pts_bbox' or 'img_bbox' dict
            result_files = dict()
            for name in results[0]:
                print(f'\nFormating bboxes of {name}')
                results_ = [out[name] for out in results]
                tmp_file_ = osp.join(jsonfile_prefix, name)
                result_files.update(
                    {name: self._format_bbox(results_, tmp_file_)})
        return result_files, tmp_dir
        
        
    def evaluate(self,
                results,
                metric='bbox',
                logger=None,
                jsonfile_prefix=None,
                result_names=['pts_bbox'],
                show=False,
                out_dir=None,
                pipeline=None):
            """Evaluation in nuScenes protocol.

            Args:
                results (list[dict]): Testing results of the dataset.
                metric (str | list[str], optional): Metrics to be evaluated.
                    Default: 'bbox'.
                logger (logging.Logger | str, optional): Logger used for printing
                    related information during evaluation. Default: None.
                jsonfile_prefix (str, optional): The prefix of json files including
                    the file path and the prefix of filename, e.g., "a/b/prefix".
                    If not specified, a temp file will be created. Default: None.
                show (bool, optional): Whether to visualize.
                    Default: False.
                out_dir (str, optional): Path to save the visualization results.
                    Default: None.
                pipeline (list[dict], optional): raw data loading for showing.
                    Default: None.

            Returns:
                dict[str, float]: Results of each evaluation metric.
            """
            print('Evaluating occupancy prediction results ...')
            self.evaluate_occ(results, runner=None, show_dir=None)
            
            result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

            if isinstance(result_files, dict):
                results_dict = dict()
                for name in result_names:
                    print('Evaluating bboxes of {}'.format(name))
                    ret_dict = self._evaluate_single(result_files[name])
                results_dict.update(ret_dict)
            elif isinstance(result_files, str):
                results_dict = self._evaluate_single(result_files)

            if tmp_dir is not None:
                tmp_dir.cleanup()

            if show or out_dir:
                self.show(results, out_dir, show=show, pipeline=pipeline)
            return results_dict