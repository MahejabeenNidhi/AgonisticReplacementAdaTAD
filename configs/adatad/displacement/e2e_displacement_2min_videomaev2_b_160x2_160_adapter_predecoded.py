_base_ = [
    "../../_base_/datasets/displacement/e2e_resize_160_1x160x160.py",
    "../../_base_/models/actionformer.py",
]

resize_length = 160
scale_factor = 2
num_frames = resize_length * scale_factor  # 320

# VideoMAEv2-Base: tubelet_size=2, num_frames=16
chunk_num = num_frames // 16  # = 20

# Path to pre-decoded frames
frames_dir = "data/displacement/predecoded_frames"

dataset = dict(
    train=dict(
        resize_length=resize_length,
        empty_clip_ratio=0.0,
        pipeline=[
            dict(type="PreparePredecodedFrames", frames_dir=frames_dir, num_frames=num_frames, fps=20.0),
            dict(type="LoadFrames", num_clips=1, method="resize", scale_factor=scale_factor),
            dict(type="DecodePredecodedFrames", filename_tmpl="frame_{:06d}.jpg", temporal_jitter=1),
            dict(type="CodecSimulationJitter",
                 noise_std_range=(0.5, 2.5),
                 channel_shift_range=1.5,
                 jpeg_resample_prob=0.0,
                 temporal_smooth_prob=0.1,
                 temporal_smooth_alpha=(0.02, 0.08)),
            dict(type="mmaction.Resize", scale=(-1, 182)),
            dict(type="mmaction.RandomResizedCrop"),
            dict(type="mmaction.Resize", scale=(160, 160), keep_ratio=False),
            dict(type="mmaction.Flip", flip_ratio=0.5),
            dict(type="mmaction.ImgAug", transforms="default"),
            dict(type="mmaction.ColorJitter"),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    val=dict(
        resize_length=resize_length,
        empty_clip_ratio=0.0,
        pipeline=[
            dict(type="PreparePredecodedFrames", frames_dir=frames_dir, num_frames=num_frames, fps=20.0),
            dict(type="LoadFrames", num_clips=1, method="resize", scale_factor=scale_factor),
            dict(type="DecodePredecodedFrames", filename_tmpl="frame_{:06d}.jpg"),
            dict(type="mmaction.Resize", scale=(-1, 160)),
            dict(type="mmaction.CenterCrop", crop_size=160),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    test=dict(
        resize_length=resize_length,
        pipeline=[
            dict(type="PreparePredecodedFrames", frames_dir=frames_dir, num_frames=num_frames, fps=20.0),
            dict(type="LoadFrames", num_clips=1, method="resize", scale_factor=scale_factor),
            dict(type="DecodePredecodedFrames", filename_tmpl="frame_{:06d}.jpg"),
            dict(type="mmaction.Resize", scale=(-1, 160)),
            dict(type="mmaction.CenterCrop", crop_size=160),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs"]),
            dict(type="Collect", inputs="imgs", keys=["masks"]),
        ],
    ),
)

model = dict(
    backbone=dict(
        type="mmaction.Recognizer3D",
        backbone=dict(
            type="VisionTransformerAdapter",
            img_size=224,
            patch_size=16,
            embed_dims=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            num_frames=16,
            tubelet_size=2,
            drop_path_rate=0.1,
            norm_cfg=dict(type="LN", eps=1e-6),
            return_feat_map=True,
            with_cp=True,
            total_frames=num_frames,
            adapter_index=list(range(12)),
            adapter_mlp_ratio=0.25,
        ),
        data_preprocessor=dict(
            type="mmaction.ActionDataPreprocessor",
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            format_shape="NCTHW",
        ),
        custom=dict(
            pretrain="/media/D2/public/mae/OpenTAD/pretrained/DisplacementFinetuned_videomaev2base_wAug.pth",
            pre_processing_pipeline=[
                dict(
                    type="Rearrange",
                    keys=["frames"],
                    ops="b n c (t1 t) h w -> (b t1) n c t h w",
                    t1=chunk_num,
                ),
            ],
            post_processing_pipeline=[
                dict(
                    type="Reduce",
                    keys=["feats"],
                    ops="b n c t h w -> b c t",
                    reduction="mean",
                ),
                dict(
                    type="Rearrange",
                    keys=["feats"],
                    ops="(b t1) c t -> b c (t1 t)",
                    t1=chunk_num,
                ),
                dict(type="Interpolate", keys=["feats"], size=resize_length),
            ],
            norm_eval=False,
            freeze_backbone=False,
        ),
    ),
    projection=dict(
        in_channels=768,
        out_channels=256,
        arch=(2, 2, 5),
        conv_cfg=dict(kernel_size=3, proj_pdrop=0.0),
        norm_cfg=dict(type="LN"),
        attn_cfg=dict(n_head=4, n_mha_win_size=-1),
        path_pdrop=0.1,
        use_abs_pe=True,
        max_seq_len=resize_length,
    ),
    neck=dict(in_channels=256, out_channels=256, num_levels=6),
    rpn_head=dict(
        num_classes=1,
        in_channels=256,
        feat_channels=256,
        num_convs=2,
        loss_normalizer=16,
        loss_normalizer_momentum=0.9,
        label_smoothing=0.0,
        loss_weight=1.0,
        center_sample="radius",
        center_sample_radius=1.5,
        cls_prior_prob=0.05,
        prior_generator=dict(
            type="PointGenerator",
            strides=[1, 2, 4, 8, 16, 32],
            regression_range=[
                (0, 4),
                (4, 8),
                (8, 16),
                (16, 32),
                (32, 64),
                (64, 10000),
            ],
        ),
        loss=dict(
            cls_loss=dict(type="FocalLoss", alpha=0.5, gamma=2.0),
            reg_loss=dict(type="DIOULoss"),
        ),
    ),
)

solver = dict(
    train=dict(batch_size=4, num_workers=4),
    val=dict(batch_size=4, num_workers=4),
    test=dict(batch_size=4, num_workers=4),
    clip_grad_norm=1,
    amp=False,
    fp16_compress=False,
    static_graph=True,
    ema=True,
)

optimizer = dict(
    type="AdamW",
    lr=5e-4,
    weight_decay=0.05,
    paramwise=True,
    backbone=dict(
        lr=0,
        weight_decay=0,
        custom=[dict(name="adapter", lr=2e-4, weight_decay=0.05)],
        exclude=["backbone"],
    ),
)

scheduler = dict(
    type="LinearWarmupCosineAnnealingLR",
    warmup_epoch=10,
    max_epoch=100,
)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)

post_processing = dict(
    nms=dict(
        use_soft_nms=True,
        sigma=0.5,
        max_seg_num=200,
        min_score=0.01,
        multiclass=False,
        voting_thresh=0.7,
    ),
    external_cls=["displacement"],
    save_dict=False,
)

workflow = dict(
    logging_interval=50,
    checkpoint_interval=10,
    val_loss_interval=1,
    val_eval_interval=4,
    val_start_epoch=10,
)

work_dir = "exps/displacement/adatad/e2e_videomaev2_b_160x2_160_adapter_2min_predecoded"
