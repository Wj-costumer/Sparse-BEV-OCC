dataset_type = 'NuSceneOcc'
dataset_root = 'data/nuscenes/'
occ_gt_root = 'data/nuscenes/occ3d'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-40, -40, -1.0, 40, 40, 5.4]
occ_size = [200, 200, 16]
voxel_size = [0.2, 0.2, 6.4]

img_norm_cfg = dict(
    mean=[123.675, 116.280, 103.530],
    std=[58.395, 57.120, 57.375],
    to_rgb=True
)

# For nuScenes we usually do 10-class detection
det_class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

occ_class_names = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free'
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False
)

# occ arch config
_dim_ = 256
_num_points_ = 4
_num_groups_ = 4
_num_layers_ = 2
_num_frames_ = 8
_num_queries_ = 100
_topk_training_ = [4000, 16000, 64000]
_topk_testing_ = [2000, 8000, 32000]

# det arch config
embed_dims = 256
num_layers = 6
num_query = 900
num_frames = 8
num_levels = 4
num_points = 4

model = dict(
    type='SparseOcc',
    data_aug=dict(
        img_color_aug=True,  # Move some augmentations to GPU
        img_norm_cfg=img_norm_cfg,
        img_pad_cfg=dict(size_divisor=32)),
    use_mask_camera=False,
    stop_prev_grad=0,
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        with_cp=True),
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=_dim_,
        num_outs=4),
    pts_bbox_head=dict(
        type='SparseBEVHead',
        num_classes=10,
        in_channels=embed_dims,
        num_query=num_query,
        query_denoising=True,
        query_denoising_groups=10,
        code_size=10,
        code_weights=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        sync_cls_avg_factor=True,
        transformer=dict(
            type='SparseBEVTransformer',
            embed_dims=embed_dims,
            num_frames=num_frames,
            num_points=num_points,
            num_layers=num_layers,
            num_levels=num_levels,
            num_classes=10,
            code_size=10,
            pc_range=point_cloud_range),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            score_threshold=0.05,
            num_classes=10),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=embed_dims // 2,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    pts_occ_head=dict(
        type='SparseOccHead',
        class_names=occ_class_names,
        embed_dims=_dim_,
        occ_size=occ_size,
        pc_range=point_cloud_range,
        transformer=dict(
            type='SparseOccTransformer',
            embed_dims=_dim_,
            num_layers=_num_layers_,
            num_frames=_num_frames_,
            num_points=_num_points_,
            num_groups=_num_groups_,
            num_queries=_num_queries_,
            num_levels=4,
            num_classes=len(occ_class_names),
            pc_range=point_cloud_range,
            occ_size=occ_size,
            topk_training=_topk_training_,
            topk_testing=_topk_testing_),
        loss_cfgs=dict(
            loss_mask2former=dict(
                type='Mask2FormerLoss',
                num_classes=len(occ_class_names),
                no_class_weight=0.1,
                loss_cls_weight=2.0,
                loss_mask_weight=5.0,
                loss_dice_weight=5.0,
            ),
            loss_geo_scal=dict(
                type='GeoScalLoss',
                num_classes=len(occ_class_names),
                loss_weight=1.0
            ),
            loss_sem_scal=dict(
                type='SemScalLoss',
                num_classes=len(occ_class_names),
                loss_weight=1.0
            )
        ),
    ),
    train_cfg=dict(pts=dict(
        grid_size=[400, 400, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0),
        )
    ))
)

ida_aug_conf = {
    'resize_lim': (0.38, 0.55),
    'final_dim': (256, 704),
    'bot_pct_lim': (0.0, 0.0),
    'rot_lim': (0.0, 0.0),
    'H': 900, 'W': 1600,
    'rand_flip': True,
}

bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5
)

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False, color_type='color'),
    dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=_num_frames_ - 1),
    dict(type='BEVAug', bda_aug_conf=bda_aug_conf, classes=det_class_names, is_train=True),
    dict(type='LoadOccGTFromFile', num_classes=len(occ_class_names)),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=det_class_names),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=True),
    dict(type='GlobalRotScaleTransImage', rot_range=[-0.3925, 0.3925], scale_ratio_range=[0.95, 1.05]),
    dict(type='DefaultFormatBundle3D', class_names=det_class_names),
    dict(type='Collect3D', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d', 'voxel_semantics', 'voxel_instances', 'instance_class_ids'],  # other keys: 'mask_camera'
         meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape', 'lidar2img', 'img_timestamp', 'ego2lidar'))
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False, color_type='color'),
    dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=_num_frames_ - 1, test_mode=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=det_class_names),
    dict(type='BEVAug', bda_aug_conf=bda_aug_conf, classes=det_class_names, is_train=False),
    dict(type='LoadOccGTFromFile', num_classes=len(occ_class_names)),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=False),
    dict(type='DefaultFormatBundle3D', class_names=det_class_names),
    dict(type='Collect3D', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d', 'voxel_semantics', 'voxel_instances', 'instance_class_ids'],
         meta_keys=('filename', 'box_type_3d', 'ori_shape', 'img_shape', 'pad_shape', 'lidar2img', 'img_timestamp', 'ego2lidar'))
]

data = dict(
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=dataset_root,
        occ_gt_root=occ_gt_root,
        ann_file=dataset_root + 'nuscenes_infos_train_sweep.pkl',
        pipeline=train_pipeline,
        classes=det_class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        box_type_3d='LiDAR'
    ),
    val=dict(
        type=dataset_type,
        data_root=dataset_root,
        occ_gt_root=occ_gt_root,
        ann_file=dataset_root + 'nuscenes_infos_val_mini_sweep.pkl',
        pipeline=test_pipeline,
        classes=det_class_names,
        modality=input_modality,
        test_mode=False
    ),
    test=dict(
        type=dataset_type,
        data_root=dataset_root,
        occ_gt_root=occ_gt_root,
        ann_file=dataset_root + 'nuscenes_infos_test_sweep.pkl',
        pipeline=test_pipeline,
        classes=det_class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'
    ),
)

optimizer = dict(
    type='AdamW',
    lr=5e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
            'sampling_offset': dict(lr_mult=0.1),
        }),
    weight_decay=0.01
)
optimizer_config = dict(
    type='Fp16OptimizerHook',
    loss_scale=512.0,
    grad_clip=dict(max_norm=35, norm_type=2)
)

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    by_epoch=True,
    step=[22, 24],
    gamma=0.2
)

total_epochs = 24
batch_size = 16

# load pretrained weights
load_from = './bev_occ_model.pth' # '../SparseBEV/ckpts/r50_nuimg_704x256.pth'
revise_keys = None # [('backbone', 'img_backbone')]

# resume the last training
resume_from = None

# checkpointing: save model per interval, and the max number of models to be saved
checkpoint_config = dict(interval=1, max_keep_ckpts=total_epochs)

# logging
log_config = dict(
    interval=1,
    hooks=[
        dict(type='MyTextLoggerHook', interval=1, reset_flag=True),
        dict(type='MyTensorboardLoggerHook', interval=500, reset_flag=True)
    ]
)

eval_interval = 0

# evaluation: train eval_interval times then eval model
eval_config = dict(interval=eval_interval, pipeline=test_pipeline)

# other flags
debug = False