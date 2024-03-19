_base_=[
        "../_base_/six_gray_sim_data.py",
        "../_base_/davis.py",
        "../_base_/default_runtime.py"
        ]
resize_h,resize_w = 64,64
cr = 8 

train_pipeline = [ 
    dict(type='RandomResize'),
    dict(type='RandomCrop',crop_h=resize_h,crop_w=resize_w,random_size=True),
    dict(type='Flip', direction='horizontal',flip_ratio=0.5,),
    dict(type='Flip', direction='diagonal',flip_ratio=0.5,),
    dict(type='Resize', resize_h=resize_h,resize_w=resize_w),
]
train_data = dict(
    mask_path = None,
    mask_shape = (resize_h,resize_w,cr),
    pipeline = train_pipeline
)
test_data = dict(
    mask_path="test_datasets/mask/random_mask.mat"#mask.mat"
)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
)

model = dict(
    type='RNN_ViT_SCI',
    color_channels=1,
    num_filters=64, 
    num_iterations=5
)

eval=dict(
    flag=True,
    interval=1
)

checkpoints=None#"checkpoints/dun_rnnvit/dun_rnnvit.pth"