import os
config = {
    'output_folder_name' : 'train_davit_b_face_age_gender_race_emotion_attribute_pose_landmarks_visibility_DaVinciFace',
    'backbone_name' : 'davit_b',
    'compile_model' : True, # whether to compile the model with torch.compile. If True, it will attempt to compile the model and print a message if successful. If False, it will skip compilation.
    'pretrained_backbone_path' : os.path.join('data', 'models', 'pretrained models', 'davit_b_adaface_Glint360k', 'backbone.pth'),
    'pretrained_face_recognition_subnet_path' : os.path.join('data', 'models', 'pretrained models', 'davit_b_adaface_Glint360k', 'recognition_subnet.pth'),
    'pretrained_margin_head_path' : os.path.join('data', 'models', 'pretrained models', 'davit_b_adaface_Glint360k', 'margin_head.pth'),
    'head_type' : 'adaface',
    'embedding_dim' : 512, # for face recognition
    'transformer_embedding_dim' : 256, # for the dual attention decoder 
    'num_decoder_layers' : 4, # for the dual attention decoder
    'num_heads': 8, # for the dual attention decoder
    'mlp_ratio' : 8.0, # for the dual attention decoder
    'decoder_type' : 'base', # can be either 'base' or 'channel_cross_attention'
    'margin' : 0.4,
    'scale' : 64,
    'h' : 0.333,
    't_alpha' : 0.05, # So that the exponential moving average is stable.
    'optimizer' : 'adamw',
    'learning_rate' : 5e-4,
    'backbone_lr_multiplier' : 1.0/10.0,
    'weight_decay' : 0.05,
    'gradient_clip_val' : 1.0,
    'scheduler' : 'cosine', # can be cosine, multistep, or cosine_restart
    'cosine_restart_start' : 15, # for cosine restart scheduler
    'cosine_restart_mult' : 10, # for cosine restart scheduler
    'max_epochs' : 36,
    'epoch_size' : 2_000_000,
    'start_factor' : 1e-3, # for linear warmup with cosine annealing
    'min_lr' : 5e-7,
    'step_scheduler_milestones' : [10, 25, 70], # for multistep learning rate scheduler
    'warmup_epochs' : 10,
    'freeze_backbone_epochs' : 1,
    'task_weights' : {
        'face_recognition' : 1e-2,
        'emotion_recognition' : 1e-2,
        'age_estimation' : 1e-2,
        'gender_recognition' : 1e-2,
        'race_recognition' : 1e-2,
        'attribute_recognition' : 1e-2,
        'head_pose_estimation' : 1000,
        'landmark_98_detection' : 1,
        'landmark_68_detection' : 1,
        'landmark_29_detection' : 1,
        'visibility_detection' : 1e-2,
    },
    'loss_weighing_type' : 'uncertainty', # can be either 'dynamic_weight_average', 'uncertainty', or 'custom_weights'
    'batch_size' : 64,
    'accumulate_grad_batches' : 8,
    'precision' : 'bf16-mixed',
    'num_workers' : 12,
}