net = dict(
    type='Detector',
)

backbone = dict(
    type='MobileNet',
    net='MobileNetV2',
    pretrained=True,
    out_conv=False,
)
featuremap_out_channel = 1280
# featuremap_out_channel = 128
featuremap_out_stride = 32 
# featuremap_out_stride = 16 

num_points = 72
max_lanes = 2
# sample_y = range(710, 150, -10)
sample_y = range(400, 300, -20)

heads = dict(type='LaneATT',
        anchors_freq_path='.cache/tusimple_anchors_freq.pt',
        topk_anchors=1000)

train_parameters = dict(
    conf_threshold=None,
    nms_thres=15.,
    nms_topk=3000
)
test_parameters = dict(
    conf_threshold=0.2,
    nms_thres=45,
    nms_topk=max_lanes
)

optimizer = dict(
  type = 'Adam',
  lr = 0.0003,
)

epochs = 100
batch_size = 8
total_iter = (3616 // batch_size) * epochs
scheduler = dict(
    type = 'CosineAnnealingLR',
    T_max = total_iter
)

eval_ep = 1
save_ep = 1

#ori_img_w=640
#ori_img_h=480
ori_img_w=1280
ori_img_h=720
img_w=640 
img_h=360
cut_height=0

train_process = [
    dict(type='GenerateLaneLine', 
        transforms = (
            dict(
                name = 'Affine',
                parameters = dict(
                    translate_px = dict(
                        x = (-25, 25),
                        y = (-10, 10)
                    ),
                    rotate=(-6, 6),
                    scale=(0.85, 1.15)
                )
            ),
            dict(
                name = 'HorizontalFlip',
                parameters = dict(
                    p=0.5
                ),
            )
        ), 
        wh = (img_w, img_h),
    ),
    dict(type='ToTensor', keys=['img', 'lane_line']),
] 

val_process = [
    dict(type='GenerateLaneLine'),
    dict(type='ToTensor', keys=['img']),
] 

dataset_path = './data/TUSimpleMars/TuSimple2'
test_json_file = 'data/TUSimpleMars/TuSimple2/test_label.json'
dataset_type = 'TuSimple'

dataset = dict(
    train=dict(
        type=dataset_type,
        data_root=dataset_path,
        split='trainval',
        processes=train_process,
    ),
    val=dict(
        type=dataset_type,
        data_root=dataset_path,
        split='test',
        processes=val_process,
    ),
    test=dict(
        type=dataset_type,
        data_root=dataset_path,
        split='test',
        processes=val_process,
    )
)


workers = 12
log_interval = 100
seed=0
lr_update_by_epoch = False
