import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import datasets as db
import eval, numpy as np
from torchvision.transforms import v2
from torch.optim import lr_scheduler
from augmentation import get_task_augmentation_transforms
from tqdm import tqdm
from multitask.DaVinciFace.model import DaVinciFace
from utility_scripts.loss_weighing import DynamicWeightAverage, UncertaintyLoss
from matplotlib import pyplot as plt
from utility_scripts.losses import dldl_loss, AsymmetricLossOptimized, GeodesicLoss, SoftWingLoss, FocalLoss
from utility_scripts.pose_estimation_utilities import *
from configs.DaVinciFace.train_davit_t_face_age_gender_race_emotion_attribute_pose_landmarks_visibility_DaVinciFace_channel_cross_attention_glint360k_pretrained_4_layer_decoder import config

torch.set_float32_matmul_precision('medium')

def main(**kwargs): 

    use_validation = True # toggle validation datasets on or off. 
    return_name = False # Whether to return the name of the dataset or not.
    output_folder_name = kwargs.get('output_folder_name')
    compile_model = kwargs.get('compile_model')
    checkpoint_path = os.path.join('checkpoints', output_folder_name, 'model.pth')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # train and test transforms
    image_size = (224, 224)
    transforms = get_task_augmentation_transforms(image_size=image_size,)
    
    train_face_rec_transform = transforms['face_recognition']
    emotion_recognition_transform = transforms['emotion_recognition']
    age_gender_race_transform = transforms['age_gender_race_recognition']
    attribute_recognition_transform = transforms['attribute_recognition']
    head_pose_transform = transforms['head_pose_estimation']
    landmarks_transform = transforms['landmark_detection']
    visibility_transform = transforms['visibility_prediction']

    test_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize(image_size),
        v2.ToDtype(torch.float32, scale = True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    attribute_test_transform = v2.Compose([
        v2.ToPILImage(),
        # v2.Pad((20, 0, 20, 0), padding_mode='edge'), 
        v2.Resize(image_size, antialias=True),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale = True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Data Preparation
    
    # training
    glint360k = db.Glint360k(transform=train_face_rec_transform, return_name = return_name)
    num_classes = glint360k.number_of_classes()
    # ms1mv2= db.MS1MV2(transform = train_face_rec_transform, return_name = return_name)
    # num_classes = ms1mv2.number_of_classes()
    celeba = db.CelebA(transform = attribute_recognition_transform, subset = 'combined')
    attribute_pos_weight = celeba.get_attribute_weights().to(device)

    cofw_db = db.COFW(transform = visibility_transform, subset = 'train')
    cofw_visibility_pos_weights = cofw_db.get_pos_weight().to(device)


    train_db_dict = {
        'face_recognition' : [glint360k],
        'emotion_recognition' : [
            db.FERPlus(transform = emotion_recognition_transform, subset = 'train'),
            db.AffectNet(transform = emotion_recognition_transform, subset = 'train'), 
            db.RAFDB(transform = emotion_recognition_transform, subset = 'train')
        ],
        'age_gender_race_recognition' : [
            db.MORPH(transform = age_gender_race_transform, subset = 'combined'),
            db.FairFace(transform = age_gender_race_transform, subset = 'train'),
            db.UTKFace(transform = age_gender_race_transform, subset = 'train'),
            db.IMDB_WIKI(transform = age_gender_race_transform)
        ],
        'attribute_recognition' : [celeba],
        'head_pose_estimation' : [db.W300LP(transform = head_pose_transform)],
        'landmark_98_detection' : [db.WFLW(transform = landmarks_transform)],
        'landmark_68_detection' : [db.W300(transform = landmarks_transform)],
        'landmark_29_detection_and_visibility' : [cofw_db],
    }
    
    batch_size = kwargs.get('batch_size')
    accumulate_grad_batches = kwargs.get('accumulate_grad_batches')
    num_workers = kwargs.get('num_workers')

    train_loader = db.get_balanced_loader(
        train_db_dict,
        batch_size = batch_size, 
        num_workers = num_workers,
        epoch_size = int(kwargs.get('epoch_size')),
    )


    if use_validation:
        # validation
        fairface_test_db = torch.utils.data.DataLoader( # for race recognition and gender recognition
            dataset = db.FairFace(transform = test_transform, subset = 'test'),
            batch_size = 32,
            shuffle = False,
            num_workers = 2,
            pin_memory = True,
        )

        rafdb_validation_db = torch.utils.data.DataLoader( # for emotion recognition
            dataset = db.RAFDB(transform = test_transform, subset = 'test'),
            batch_size = 32,
            shuffle = False,
            num_workers = 2,
            pin_memory = True,
        )

        morph_test_db = torch.utils.data.DataLoader( # for age estimation and gender recognition 
            dataset = db.MORPH(transform = test_transform, subset = 'test'),
            batch_size = 32,
            shuffle = False,
            num_workers = 2,
            pin_memory = True,
        )

        utkface_test_db = torch.utils.data.DataLoader( # for age estimation and gender recognition 
            dataset = db.UTKFace(transform = test_transform, subset = 'test'),
            batch_size = 32,
            shuffle = False,
            num_workers = 2,
            pin_memory = True,
        )

        celeba_test_db = torch.utils.data.DataLoader( # for attribute recognition
            dataset = db.CelebA(transform = attribute_test_transform, subset = 'test'),
            batch_size = 32,
            shuffle = False,
            num_workers = 2,
            pin_memory = True,
        )

        biwi_validation_db = torch.utils.data.DataLoader( # for head pose
            dataset = db.BIWI(transform = test_transform),
            batch_size = 32,
            shuffle = False,
            num_workers = 2,
            pin_memory = True,
        )

        w300_validation_db_full = torch.utils.data.DataLoader( # for landmark detection
            dataset = db.W300(transform = test_transform, subset = 'full'),
            batch_size = 32,
            shuffle = False,
            num_workers = 2,
            pin_memory = True,
        )

        w300_validation_db_common = torch.utils.data.DataLoader( # for landmark detection
            dataset = db.W300(transform = test_transform, subset = 'common'),
            batch_size = 32,
            shuffle = False,
            num_workers = 2,
            pin_memory = True,
        )

        w300_validation_db_challenge = torch.utils.data.DataLoader( # for landmark detection
            dataset = db.W300(transform = test_transform, subset = 'challenge'),
            batch_size = 32,
            shuffle = False,
            num_workers = 2,
            pin_memory = True,
        )

        wflw_validation_db = torch.utils.data.DataLoader( # for landmark 98 detection
            dataset = db.WFLW(transform = test_transform, subset = 'test'),
            batch_size = 32,
            shuffle = False,
            num_workers = 2,
            pin_memory = True,
        )

        cofw_validation_db = torch.utils.data.DataLoader( # for landmark 29 detection and visibility
            dataset = db.COFW(transform = test_transform, subset = 'test'),
            batch_size = 32,
            shuffle = False,
            num_workers = 2,
            pin_memory = True,
        )


        print('fairface length: ', len(fairface_test_db) * 64)
        print('affectnet length: ', len(rafdb_validation_db) * 64)
        print('morph length: ', len(morph_test_db) * 64)
        print('biwi length: ', len(biwi_validation_db) * 64)
        print('w300 full length: ', len(w300_validation_db_full) * 64)
        print('w300 common length: ', len(w300_validation_db_common) * 64)
        print('w300 challenge length: ', len(w300_validation_db_challenge) * 64)
        print('wflw length: ', len(wflw_validation_db) * 64)
        print('cofw length: ', len(cofw_validation_db) * 64)


    # Training setup
    model = DaVinciFace(
        num_classes = num_classes,
        **kwargs
    )

    model.to(device)


    epochs = kwargs.get('max_epochs')
    optimizer_name = kwargs.get('optimizer')
    weight_decay = kwargs.get('weight_decay')
    lr_scheduler_name = kwargs.get('scheduler')
    milestones = kwargs.get('step_scheduler_milestones')
    cosine_restart_start = kwargs.get('cosine_restart_start')
    cosine_restart_mult = kwargs.get('cosine_restart_mult')
    start_factor = kwargs.get('start_factor')
    min_lr = kwargs.get('min_lr')
    warmup_epochs = kwargs.get('warmup_epochs')
    freeze_backbone_epochs = kwargs.get('freeze_backbone_epochs')
    learning_rate = kwargs.get('learning_rate')
    task_weights = kwargs.get('task_weights')
    loss_weighing_type = kwargs.get('loss_weighing_type')



    # Prepare the backbone, face recognition subnet, and margin head for freezing, and count the number of parameters of each part
    margin_head_params = []
    face_rec_params = []
    backbone_params = []
    other_params = []
    backbone_params_count = 0
    face_rec_subnet_parameters_count = 0
    margin_head_params_count = 0
    emotion_rec_parameters_count = 0
    age_estimation_parameters_count = 0
    gender_rec_parameters_count = 0
    race_rec_parameters_count = 0
    attribute_rec_parameters_count = 0
    head_pose_parameters_count = 0
    fusion_parameter_count = 0
    decoder_parameter_count = 0
    landmarks_98_detection_subnet_parameters_count = 0
    landmarks_68_detection_subnet_parameters_count = 0
    landmarks_29_detection_subnet_parameters_count = 0
    visibility_detection_subnet_parameters_count = 0

    
    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
            backbone_params_count += param.numel()

        elif 'face_recognition_embedding' in name:
            face_rec_params.append(param)
            face_rec_subnet_parameters_count += param.numel()

        elif 'margin_head' in name:
            margin_head_params.append(param)
            margin_head_params_count += param.numel()

        elif 'emotion_recognition' in name:
            other_params.append(param)
            emotion_rec_parameters_count += param.numel()

        elif 'age_estimation' in name:
            other_params.append(param)
            age_estimation_parameters_count += param.numel()

        elif 'gender' in name:
            other_params.append(param)
            gender_rec_parameters_count += param.numel()

        elif 'race_recognition' in name:
            other_params.append(param)
            race_rec_parameters_count += param.numel()

        elif 'attribute' in name:
            other_params.append(param)
            attribute_rec_parameters_count += param.numel()

        elif "head_pose_estimation" in name:
            other_params.append(param)
            head_pose_parameters_count += param.numel()
        
        elif "multiscale_projector" in name:
            other_params.append(param)
            fusion_parameter_count += param.numel()
        
        elif "landmarks_98" in name:
            other_params.append(param)
            landmarks_98_detection_subnet_parameters_count += param.numel()
        
        elif "landmarks_68" in name:
            other_params.append(param)
            landmarks_68_detection_subnet_parameters_count += param.numel()
        
        elif "landmarks_29" in name:
            other_params.append(param)
            landmarks_29_detection_subnet_parameters_count += param.numel()
        
        elif "visibility" in name:
            other_params.append(param)
            visibility_detection_subnet_parameters_count += param.numel()

        elif "decoder" in name:
            other_params.append(param)
            decoder_parameter_count += param.numel()

        else:
            other_params.append(param)
    
    assert len(backbone_params) != 0, 'No backbone parameters found!'
    assert len(face_rec_params) != 0, 'No face recognition subnet parameters found!'
    assert len(margin_head_params) != 0, 'No margin head parameters found!'
    assert len(other_params) != 0, 'No subnet parameters found!'


    print(f'Backbone parameters: {backbone_params_count} <'.ljust(75, '='))
    print(f'Face recognition subnet parameters: {face_rec_subnet_parameters_count} <'.ljust(75, '='))
    print(f'Margin head parameters:  {margin_head_params_count} <'.ljust(75, '='))
    print(f'Emotion recognition parameters: {emotion_rec_parameters_count} <'.ljust(75, '='))
    print(f'Age estimation parameters: {age_estimation_parameters_count} <'.ljust(75, '='))
    print(f'Gender recognition parameters: {gender_rec_parameters_count} <'.ljust(75, '='))
    print(f'Race recognition parameters: {race_rec_parameters_count} <'.ljust(75, '='))
    print(f'Attribute recognition parameters: {attribute_rec_parameters_count} <'.ljust(75, '='))
    print(f'Head pose estimation parameters: {head_pose_parameters_count} <'.ljust(75, '='))
    print(f'Feature fusion parameters: {fusion_parameter_count} <'.ljust(75, '='))
    print(f'Landmark 98 detection subnet parameters: {landmarks_98_detection_subnet_parameters_count} <'.ljust(75, '='))
    print(f'Landmark 68 detection subnet parameters: {landmarks_68_detection_subnet_parameters_count} <'.ljust(75, '='))
    print(f'Landmark 29 detection subnet parameters: {landmarks_29_detection_subnet_parameters_count} <'.ljust(75, '='))
    print(f'Visibility detection subnet parameters: {visibility_detection_subnet_parameters_count} <'.ljust(75, '='))
    print(f'Decoder parameters: {decoder_parameter_count} <'.ljust(75, '='))
    print(f'Total (excluding the margin head parameters): {backbone_params_count + 
                    face_rec_subnet_parameters_count + 
                    emotion_rec_parameters_count + 
                    age_estimation_parameters_count + 
                    gender_rec_parameters_count + 
                    race_rec_parameters_count + 
                    attribute_rec_parameters_count + 
                    head_pose_parameters_count + 
                    fusion_parameter_count +
                    landmarks_98_detection_subnet_parameters_count +
                    landmarks_68_detection_subnet_parameters_count +
                    landmarks_29_detection_subnet_parameters_count +
                    visibility_detection_subnet_parameters_count +
                    decoder_parameter_count} <'.ljust(75, '='))
    
    if loss_weighing_type == 'dwa': 
        loss_weighing = DynamicWeightAverage(num_tasks = 11)
        losses_weights = loss_weighing.get_weights(avg_losses_current_epoch=None)
    elif loss_weighing_type == 'uncertainty':
        loss_weighing = UncertaintyLoss(num_tasks = 11)
        losses_weights = loss_weighing.get_weights()
    else:
        losses_weights = list(task_weights.values())
        loss_weighing = None

    losses_weights_history = [losses_weights]

    # Create parameter groups with different learning rates
    if loss_weighing_type == 'uncertainty':
        param_groups = [
            {'params': backbone_params, 'lr': learning_rate * float(kwargs.get('backbone_lr_multiplier'))},
            {'params': face_rec_params, 'lr': learning_rate},
            {'params': margin_head_params, 'lr': learning_rate},
            {'params': other_params, 'lr': learning_rate},
            {'params': loss_weighing.parameters(), 'lr': learning_rate}
        ]
    else:
        param_groups = [
            {'params': backbone_params, 'lr': learning_rate * float(kwargs.get('backbone_lr_multiplier'))},
            {'params': face_rec_params, 'lr': learning_rate},
            {'params': margin_head_params, 'lr': learning_rate},
            {'params': other_params, 'lr': learning_rate},
        ]


    if loss_weighing_type == 'uncertainty':
        loss_weighing.to(device)

    if optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(param_groups, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(param_groups, lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    if lr_scheduler_name == 'cosine':
        linear = lr_scheduler.LinearLR(optimizer, start_factor=start_factor, end_factor=1.0, total_iters=warmup_epochs)
        cosine = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min = min_lr)
        scheduler = lr_scheduler.SequentialLR(optimizer, [linear, cosine], milestones=[warmup_epochs])
    elif lr_scheduler_name == 'multistep':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones)
    elif lr_scheduler_name == 'cosine_restart':
        linear = lr_scheduler.LinearLR(optimizer, start_factor=start_factor, end_factor=1.0, total_iters=warmup_epochs)
        cosine_restart = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = cosine_restart_start, T_mult = cosine_restart_mult, eta_min = min_lr)
        scheduler = lr_scheduler.SequentialLR(optimizer, [linear, cosine_restart], milestones=[warmup_epochs])
    else:
        raise ValueError(f"Unsupported scheduler: {lr_scheduler_name}")


    precision_mode = kwargs.get('precision')

    if precision_mode == 'bf16-mixed':
        amp_dtype = torch.bfloat16
        use_scaler = False # no need to use scaler since bf16-mixed has enough range
    elif precision_mode == '32':
        amp_dtype = torch.float32
        use_scaler = False
    else:
        amp_dtype = torch.float16
        use_scaler = True
    print(f'Using amp dtype: {amp_dtype}')

    scaler = torch.amp.GradScaler(device = device, enabled = use_scaler)


    # asymmetric loss for attribute recognition
    asymmetric_loss = AsymmetricLossOptimized()

    # Soft Wing loss for landmark detection
    softwing_loss = SoftWingLoss()

    # Geodesic Loss
    geodesic_loss = GeodesicLoss()



    # load the previous checkpoint if it exists:
    if checkpoint_path is not None and os.path.exists(checkpoint_path) and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only = False, map_location = device)
        model_state_dict = {k.replace('_orig_mod.', ''): v for k, v in model_state_dict.items()}
        model.load_state_dict(model_state_dict, strict = True)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        losses_weights_history = checkpoint['losses_weights_history']
        
        if loss_weighing_type == 'uncertainty' and 'loss_weighing' in checkpoint:
            loss_weighing.load_state_dict(checkpoint['loss_weighing'])
        elif loss_weighing_type == 'dwa' and 'loss_weighing' in checkpoint:
            loss_weighing.avg_losses_previous_epoch = checkpoint['loss_weighing']

        
        losses_weights = checkpoint['losses_weights']

        print(f"Resuming training from epoch {start_epoch}")
    else:
        start_epoch = 0

    if compile_model:
        print('Compiling the model for faster inference...')
        model = torch.compile(model)

    # Training loop
    for epoch in range(start_epoch, epochs):
        model.train() # Put here to ensure the epoch starts with the model in training mode
        
        if epoch < freeze_backbone_epochs:
            print(f"🔒 Epoch {epoch}: Backbone + face recognition subnet + margin head are FROZEN (Warmup Phase)") 

            model.backbone.eval()
            model.face_recognition_embedding_subnet.eval()
            model.margin_head.eval()

            for param in backbone_params:
                param.requires_grad = False
            
            for param in face_rec_params:
                param.requires_grad = False
            
            for param in margin_head_params:
                param.requires_grad = False

            for param in other_params: # just to make sure other parameters are trainable during the warmup phase
                param.requires_grad = True

        elif epoch == freeze_backbone_epochs:
            print(f"🔓 Epoch {epoch}: Unfreezing! (Finetuning Phase)")

            for param in backbone_params:
                param.requires_grad = True
            
            for param in face_rec_params:
                param.requires_grad = True
            
            for param in margin_head_params:
                param.requires_grad = True
            
            

        running_face_rec_loss = 0.0
        running_emotion_rec_loss = 0.0
        running_age_estimation_loss = 0.0
        running_gender_rec_loss = 0.0
        running_race_rec_loss = 0.0
        running_attribute_rec_loss = 0.0
        running_head_pose_loss = 0
        running_landmark_98_detection_loss = 0
        running_landmark_68_detection_loss = 0
        running_landmark_29_detection_loss = 0
        running_visibility_detection_loss = 0



        num_face_rec_samples = 0
        num_emotion_samples = 0
        num_age_samples = 0
        num_gender_samples = 0
        num_race_samples = 0
        num_attribute_samples = 0
        num_head_pose_samples = 0
        num_landmark_98_detection_samples = 0
        num_landmark_68_detection_samples = 0
        num_landmark_29_detection_samples = 0
        num_visibility_detection_samples = 0

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}")
        optimizer.zero_grad(set_to_none = True) # zero the gradient before each epoch starts

        for batch_counter, batch in progress_bar:
            
            images = batch[0]
            labels = batch[1]
            images = images.to(device, non_blocking=True) # The labels are in a dict format, which cannot be moved to cuda. Task labels will be extracted below and converted to cuda one by one.

            
            # Initialize the loss tensors at the start
            # We use torch.tensor(0.0) so we can use .item() later safely
            face_rec_loss = torch.tensor(0.0, device=device)
            emotion_loss = torch.tensor(0.0, device=device)
            age_loss = torch.tensor(0.0, device=device)
            gender_loss = torch.tensor(0.0, device=device)
            race_loss = torch.tensor(0.0, device=device)
            attribute_loss = torch.tensor(0.0, device=device)
            head_pose_loss = torch.tensor(0.0, device=device)
            landmark_98_detection_loss = torch.tensor(0.0, device = device)
            landmark_68_detection_loss = torch.tensor(0.0, device = device)
            landmark_29_detection_loss = torch.tensor(0.0, device = device)
            visibility_detection_loss = torch.tensor(0.0, device = device)


            # forward pass
            with torch.amp.autocast(device_type = device, dtype = amp_dtype, enabled = (precision_mode != '32')):

                (normalized_embedding, embedding_norm), \
                emotion_output, age_output, \
                gender_output, race_output, \
                attribute_output, head_pose_output, \
                landmarks_98_output, landmarks_68_output, \
                landmarks_29_output, visibility_output = model(images)

                # Face recognition loss
                face_recognition_labels = labels['face_recognition'].to(device, non_blocking=True)
                face_recognition_mask = face_recognition_labels != -1

                if face_recognition_mask.sum() > 1: # Set a minimum of 2 samples per batch.
                    valid_embeddings = normalized_embedding[face_recognition_mask]
                    valid_embedding_norms = embedding_norm[face_recognition_mask]
                    valid_labels = face_recognition_labels[face_recognition_mask]

                    face_rec_logits = model.get_face_recognition_logits(valid_embeddings, valid_embedding_norms, valid_labels)
                    face_rec_loss = F.cross_entropy(face_rec_logits, valid_labels)
                    num_face_rec_samples += face_recognition_mask.sum().item()

                
                # Emotion recognition
                T = 1
                ALPHA = 0 # controls the ratio of soft label integration. Setting it to zero means zero soft labels integration (zero knowledge distillation)
                emotion_labels = labels['emotion'].to(device, non_blocking=True)
                smooth_emotion_labels = labels['smooth_emotion'].to(device, non_blocking = True)
                smooth_emotion_is_logit = labels['smooth_emotion_is_logit'].to(device, non_blocking = True)
                emotion_mask = emotion_labels != -1
                
                if emotion_mask.sum() > 1:  
                    if ALPHA != 0:
                        crossentorpy_loss = F.cross_entropy(emotion_output[emotion_mask], emotion_labels[emotion_mask])  
                        soft_student_log_probs = F.log_softmax(emotion_output[emotion_mask] / T, dim=1)
                        
                        valid_smooth_labels = smooth_emotion_labels[emotion_mask]
                        valid_is_logit = smooth_emotion_is_logit[emotion_mask]
                        
                        # Initialize an empty tensor for the teacher probabilities
                        soft_teacher_probs = torch.zeros_like(valid_smooth_labels)
                        
                        # 1. Apply Softmax and Temperature ONLY to POSTER++ logits (AffectNet/RAFDB)
                        if valid_is_logit.any():
                            soft_teacher_probs[valid_is_logit] = F.softmax(valid_smooth_labels[valid_is_logit] / T, dim=1)
                            
                        # 2. Directly copy the human probabilities (FERPlus) without Temperature
                        if (~valid_is_logit).any():
                            soft_teacher_probs[~valid_is_logit] = valid_smooth_labels[~valid_is_logit]

                        kl_loss = F.kl_div(soft_student_log_probs, soft_teacher_probs, reduction='batchmean') * (T ** 2)
                        emotion_loss = (1.0 - ALPHA) * crossentorpy_loss + ALPHA * kl_loss
                    elif ALPHA == 0: # if no smooth lables are blended with the hard labels, use label smoothing.
                        emotion_loss = F.cross_entropy(
                            emotion_output[emotion_mask], 
                            emotion_labels[emotion_mask], 
                            label_smoothing = 0.1
                        ) 

                    num_emotion_samples += emotion_mask.sum().item()
                


                # Age estimation
                age_labels = labels['age'].to(device, non_blocking=True)
                age_mask = age_labels != -1
                if age_mask.sum() > 1:  # Set a minimum of 2 samples per batch.
                    age_loss = dldl_loss(age_output[age_mask], age_labels[age_mask])
                    num_age_samples += age_mask.sum().item()



                # Gender recognition
                gender_labels = labels['gender'].to(device, non_blocking=True)
                gender_mask = gender_labels != -1
                if gender_mask.sum() > 1: # Set a minimum of 2 samples per batch.
                    gender_loss = F.binary_cross_entropy_with_logits(gender_output[gender_mask], gender_labels[gender_mask].view(-1, 1).float())
                    num_gender_samples += gender_mask.sum().item()
                

                
                # Race recognition
                race_labels = labels['race'].to(device, non_blocking=True)
                race_mask = race_labels != -1
                if race_mask.sum() > 1:  # Set a minimum of 2 samples per batch
                    race_loss = F.cross_entropy(race_output[race_mask], race_labels[race_mask], label_smoothing = 0.1)                                                                                                                                                  
                    num_race_samples += race_mask.sum().item()

                
                # Attribute recognition
                attribute_labels = labels['attributes'].to(device, non_blocking=True)
                # CelebA returns either 0 or 1 for valid sample, where 0 = the attribute doesn't exist, 1 = attribute exists. The default label for dummy attributes is -1,
                # so check if the first attribute from all the samples is not -1 to see which one is a valid label. 
                attibute_mask = attribute_labels[:, 0] != -1
                if attibute_mask.sum() > 1:  # Set a minimum of 2 samples per batch.
                    # # Alternative 1: Asymmetric Loss
                    # attribute_loss = asymmetric_loss(
                    #     attribute_output[attibute_mask], 
                    #     attribute_labels[attibute_mask].view(-1, 40).float()
                    # ) 
                    
                    # Alternative 2: Binary Cross Entropy (BCE)
                    raw_loss = F.binary_cross_entropy_with_logits(
                        attribute_output[attibute_mask], 
                        attribute_labels[attibute_mask].view(-1, 40).float(),
                        reduction='none' # <--- get raw loss (without reduction)
                    )
                    # Mean over attributes (dim=1)
                    loss_per_image = raw_loss.mean(dim=1) 
                    
                    # mean over the batch
                    attribute_loss = loss_per_image.mean()
                    num_attribute_samples += attibute_mask.sum().item()
                
                # head pose estimation
                head_pose_labels = labels['pose'].to(device, non_blocking=True)
                head_pose_mask = head_pose_labels[:, 0] != -999
                
                if head_pose_mask.sum() > 1:                      
                    head_pose_matrix = compute_rotation_matrix_from_ortho6d(head_pose_output[head_pose_mask])
                    head_pose_label_matrix = euler_angles_to_matrix(head_pose_labels[head_pose_mask])
                    head_pose_loss = geodesic_loss(head_pose_matrix, head_pose_label_matrix)
                    
                    num_head_pose_samples += head_pose_mask.sum().item()
                

                # Landmark 98 detection
                landmark_98_detection_labels = labels['landmarks_98'].to(device, non_blocking=True)
                landmark_98_valid_mask = labels['valid_mask_98'].to(device, non_blocking = True)
                landmark_98_detection_mask = landmark_98_detection_labels[:, 0] != -1
                if landmark_98_detection_mask.sum() > 1:  # Set a minimum of 2 samples per batch
                    landmark_98_detection_loss = softwing_loss(landmarks_98_output[landmark_98_detection_mask], landmark_98_detection_labels[landmark_98_detection_mask], landmark_98_valid_mask[landmark_98_detection_mask])
                    num_landmark_98_detection_samples += landmark_98_detection_mask.sum().item()

                # Landmark 68 detection
                landmark_68_detection_labels = labels['landmarks_68'].to(device, non_blocking=True)
                landmark_68_valid_mask = labels['valid_mask_68'].to(device, non_blocking = True)
                landmark_68_detection_mask = landmark_68_detection_labels[:, 0] != -1
                if landmark_68_detection_mask.sum() > 1:  # Set a minimum of 2 samples per batch
                    landmark_68_detection_loss = softwing_loss(landmarks_68_output[landmark_68_detection_mask], landmark_68_detection_labels[landmark_68_detection_mask], landmark_68_valid_mask[landmark_68_detection_mask])
                    num_landmark_68_detection_samples += landmark_68_detection_mask.sum().item()
                
                # Landmark 29 detection
                landmark_29_detection_labels = labels['landmarks_29'].to(device, non_blocking=True)
                landmark_29_detection_mask = landmark_29_detection_labels[:, 0] != -1
                if landmark_29_detection_mask.sum() > 1:  # Set a minimum of 2 samples per batch
                    landmark_29_detection_loss = softwing_loss(landmarks_29_output[landmark_29_detection_mask], landmark_29_detection_labels[landmark_29_detection_mask])
                    num_landmark_29_detection_samples += landmark_29_detection_mask.sum().item()
                
                # Visibility detection  
                visibility_detection_labels = labels['visibility'].to(device, non_blocking=True)
                visibility_detection_mask = visibility_detection_labels[:, 0] != -1
                if visibility_detection_mask.sum() > 1:  # Set a minimum of 2 samples per batch
                    visibility_detection_loss = F.binary_cross_entropy_with_logits(
                        visibility_output[visibility_detection_mask], 
                        visibility_detection_labels[visibility_detection_mask].view(-1, 29).float(),
                        pos_weight = cofw_visibility_pos_weights,
                    )
                    num_visibility_detection_samples += visibility_detection_mask.sum().item()

                losses = [
                    face_rec_loss, 
                    emotion_loss, 
                    age_loss, 
                    gender_loss, 
                    race_loss, 
                    attribute_loss, 
                    head_pose_loss,
                    landmark_98_detection_loss,
                    landmark_68_detection_loss,
                    landmark_29_detection_loss,
                    visibility_detection_loss
                ]

                valid_task_mask = [loss.item() > 0.0 for loss in losses]
                if loss_weighing_type == 'uncertainty':
                    total_loss = loss_weighing(losses, valid_mask=valid_task_mask)
                    losses_weights = loss_weighing.get_weights()

                elif loss_weighing_type == 'dwa':
                    total_loss = 0
                    for i in range(len(losses_weights)):
                        if valid_task_mask[i]: 
                            total_loss += losses_weights[i] * losses[i]

                else:
                    face_rec_loss = face_rec_loss * task_weights['face_recognition']
                    emotion_loss = emotion_loss * task_weights['emotion_recognition']
                    age_loss = age_loss * task_weights['age_estimation']
                    gender_loss = gender_loss * task_weights['gender_recognition']
                    race_loss = race_loss * task_weights['race_recognition']
                    attribute_loss = attribute_loss * task_weights['attribute_recognition']
                    head_pose_loss = head_pose_loss * task_weights['head_pose_estimation']
                    landmark_98_detection_loss = landmark_98_detection_loss * task_weights['landmark_98_detection']
                    landmark_68_detection_loss = landmark_68_detection_loss * task_weights['landmark_68_detection']
                    landmark_29_detection_loss = landmark_29_detection_loss * task_weights['landmark_29_detection']
                    visibility_detection_loss = visibility_detection_loss * task_weights['visibility_detection']

                    total_loss = face_rec_loss + emotion_loss + age_loss + gender_loss + race_loss + attribute_loss + head_pose_loss + landmark_98_detection_loss + landmark_68_detection_loss + landmark_29_detection_loss + visibility_detection_loss
                    
            
 
            # backward pass
            if isinstance(total_loss, torch.Tensor):
                loss_for_backward = total_loss / accumulate_grad_batches
                scaler.scale(loss_for_backward).backward()

            if (batch_counter + 1) % accumulate_grad_batches == 0 or (batch_counter + 1) == len(train_loader): # gradient accumulation
                # gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=kwargs.get('gradient_clip_val'))

                # update weights
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none = True)

            
            running_face_rec_loss += face_rec_loss.item() * face_recognition_mask.sum().item()
            running_emotion_rec_loss += emotion_loss.item() * emotion_mask.sum().item()
            running_age_estimation_loss += age_loss.item() * age_mask.sum().item()
            running_gender_rec_loss += gender_loss.item() * gender_mask.sum().item()
            running_race_rec_loss += race_loss.item() * race_mask.sum().item()
            running_attribute_rec_loss += attribute_loss.item() * attibute_mask.sum().item()
            running_head_pose_loss += head_pose_loss.item() * head_pose_mask.sum().item()
            running_landmark_98_detection_loss += landmark_98_detection_loss.item() * landmark_98_detection_mask.sum().item()
            running_landmark_68_detection_loss += landmark_68_detection_loss.item() * landmark_68_detection_mask.sum().item()
            running_landmark_29_detection_loss += landmark_29_detection_loss.item() * landmark_29_detection_mask.sum().item()
            running_visibility_detection_loss += visibility_detection_loss.item() * visibility_detection_mask.sum().item()

            

            progress_bar.set_postfix(ordered_dict={
                'loss' : f"{total_loss.item():.4f}",
                'fr' : f"{face_rec_loss.item():.4f}",
                'em' : f"{emotion_loss.item():.4f}",
                'age' : f"{age_loss.item():.4f}",
                'gen' : f"{gender_loss.item():.4f}",
                'race' : f"{race_loss.item():.4f}",
                'attr' : f"{attribute_loss.item():.4f}",
                'pose' : f"{head_pose_loss.item():.4f}",
                'lnd98' : f"{landmark_98_detection_loss.item():.4f}",
                'lnd68' : f"{landmark_68_detection_loss.item():.4f}",
                'lnd29' : f"{landmark_29_detection_loss.item():.4f}",
                'vis' : f"{visibility_detection_loss.item():.4f}",
                'lr' : f"{optimizer.param_groups[3]['lr']:.7f}",
                'lr_back' : f'{optimizer.param_groups[0]['lr']:.7f}',
            })

        scheduler.step()

        epoch_fr_loss = running_face_rec_loss / num_face_rec_samples if num_face_rec_samples > 0 else 0
        epoch_em_loss = running_emotion_rec_loss / num_emotion_samples if num_emotion_samples > 0 else 0
        epoch_age_loss = running_age_estimation_loss / num_age_samples if num_age_samples > 0 else 0
        epoch_gen_loss = running_gender_rec_loss / num_gender_samples if num_gender_samples > 0 else 0
        epoch_race_loss = running_race_rec_loss / num_race_samples if num_race_samples > 0 else 0
        epoch_attr_loss = running_attribute_rec_loss / num_attribute_samples if num_attribute_samples > 0 else 0
        epoch_head_pose_loss = running_head_pose_loss / num_head_pose_samples if num_head_pose_samples > 0 else 0
        epoch_landmark_98_detection_loss = running_landmark_98_detection_loss / num_landmark_98_detection_samples if num_landmark_98_detection_samples > 0 else 0
        epoch_landmark_68_detection_loss = running_landmark_68_detection_loss / num_landmark_68_detection_samples if num_landmark_68_detection_samples > 0 else 0
        epoch_landmark_29_detection_loss = running_landmark_29_detection_loss / num_landmark_29_detection_samples if num_landmark_29_detection_samples > 0 else 0
        epoch_visibility_detection_loss = running_visibility_detection_loss / num_visibility_detection_samples if num_visibility_detection_samples > 0 else 0

        epoch_losses = np.array([epoch_fr_loss, epoch_em_loss, epoch_age_loss, epoch_gen_loss, epoch_race_loss, epoch_attr_loss, epoch_head_pose_loss, epoch_landmark_98_detection_loss, epoch_landmark_68_detection_loss, epoch_landmark_29_detection_loss, epoch_visibility_detection_loss])
        
        if loss_weighing_type == 'uncertainty':
            losses_weights = loss_weighing.get_weights()
        elif loss_weighing_type == 'dwa':
            losses_weights = loss_weighing.get_weights(epoch_losses)
        else:
            losses_weights = list(task_weights.values())

        losses_weights_history.append(losses_weights)
        epoch_loss = epoch_fr_loss + epoch_em_loss + epoch_age_loss + epoch_gen_loss + epoch_race_loss + epoch_attr_loss + epoch_head_pose_loss + epoch_landmark_98_detection_loss + epoch_landmark_68_detection_loss + epoch_landmark_29_detection_loss + epoch_visibility_detection_loss



        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Total Loss: {epoch_loss:.4f}")
        print(f"  Face Recognition Loss: {epoch_fr_loss:.4f}")
        print(f"  Emotion Loss: {epoch_em_loss:.4f}")
        print(f"  Age Loss: {epoch_age_loss:.4f}")
        print(f"  Gender Loss: {epoch_gen_loss:.4f}")
        print(f"  Race Loss: {epoch_race_loss:.4f}")
        print(f"  Attribute Loss: {epoch_attr_loss:.4f}")
        print(f"  Head Pose Loss: {epoch_head_pose_loss:.4f}")
        print(f"  Landmark 98 Detection Loss: {epoch_landmark_98_detection_loss:.4f}")
        print(f"  Landmark 68 Detection Loss: {epoch_landmark_68_detection_loss:.4f}")
        print(f"  Landmark 29 Detection Loss: {epoch_landmark_29_detection_loss:.4f}")
        print(f"  Visibility Detection Loss: {epoch_visibility_detection_loss:.4f}")
        task_names = ['Face Recognition', 'Emotion Recognition', 'Age Estimation', 'Gender Recognition', 'Race Recognition', 'Attribute Recognition', 'Head Pose Estimation', 'Landmark 98 Detection', 'Landmark 68 Detection', 'Landmark 29 Detection', 'Visibility Prediction']
        for task_idx in range(len(losses_weights)):
            print(f"  {task_names[task_idx]} Loss Weight: {losses_weights[task_idx]:.4f}")
        print()


        # Validation
        if use_validation:
            model.eval()
            print(' Validating '.center(100, '='))
            with torch.no_grad():
                # Face Recognition
                face_rec_model = nn.Sequential(
                    model.backbone,
                    model.face_recognition_embedding_subnet
                )
                metrics = eval.evaluate_face_recognition(face_rec_model, datasets_to_test = ['CPLFW', 'CALFW'], image_size = image_size[0])

                for key, db_metrics in metrics.items():
                    accuracy, _, _, f1_score, _, _, _, _ = db_metrics
                    print(f'Accuracy for {key} = {accuracy}.')

                # Emotion Recognition
                emotion_accuracy, _, _, _, _= eval.evaluate_emotion(model = model, dataloader = rafdb_validation_db)
                print(f'Accuracy for RAFDB (emotion recognition) = {emotion_accuracy}.')

                # Age estimation
                age_mae, _, _ = eval.evaluate_age(model = model, dataloader = morph_test_db)
                print(f'MAE for MORPH (age estimation) = {age_mae}.')

                age_mae, _, _ = eval.evaluate_age(model = model, dataloader = utkface_test_db)
                print(f'MAE for UTKFace (age estimation) = {age_mae}.')

                # Gender recognition
                gender_accuracy, _, _, _, _, _, _, _, _ = eval.evaluate_gender(model = model, dataloader = morph_test_db)
                print(f'Accuracy for MORPH (gender recognition) = {gender_accuracy}.')

                gender_accuracy, _, _, _, _, _, _, _, _ = eval.evaluate_gender(model = model, dataloader = fairface_test_db)
                print(f'Accuracy for FairFace (gender recognition) = {gender_accuracy}.')

                # Race recognition
                race_accuracy, _, _, _, _ = eval.evaluate_race(model = model, dataloader = fairface_test_db, device = device)
                print(f'Accuracy for FairFace (race recognition) = {race_accuracy}.')

                # Attribute recognition
                thresholds = np.full(40, 0.5) # default threshold for all attributes is 0.5

                element_wise_accuracy, f1, precision, recall, thresholds_to_use = eval.evaluate_attributes(model = model, dataloader = celeba_test_db, thresholds = thresholds)
                print(f'Accuracy for CelebA (attribute recognition) test = {element_wise_accuracy}.')
                print(f'F1 score for CelebA (attribute recognition) test = {f1}')


                # Head pose estimation
                pose_mae, _, _, _ = eval.evaluate_head_pose(model = model, dataloader = biwi_validation_db)
                print(f'MAE for BIWI (head pose estimation) = {pose_mae}.')

                # Landmark detection 68
                pixel_mae_full, nme_full = eval.evaluate_landmarks_68(model = model, dataloader = w300_validation_db_full, image_size = image_size[0])
                print(f'300W Full: Pixel MAE: {pixel_mae_full}, NME: {nme_full}  (W300 Landmark detection)')

                pixel_mae_common, nme_common = eval.evaluate_landmarks_68(model = model, dataloader = w300_validation_db_common, image_size = image_size[0])
                print(f'300W Common: Pixel MAE: {pixel_mae_common}, NME: {nme_common}  (W300 Landmark detection)')

                pixel_mae_challenge, nme_challenge = eval.evaluate_landmarks_68(model = model, dataloader = w300_validation_db_challenge, image_size = image_size[0])
                print(f'300W Challenge: Pixel MAE: {pixel_mae_challenge}, NME: {nme_challenge}  (W300 Landmark detection)')


                # Landmark detection 98
                pixel_mae_98, nme_98 = eval.evaluate_landmarks_98(model = model, dataloader = wflw_validation_db, image_size = image_size[0])
                print(f'WFLW: Pixel MAE: {pixel_mae_98}, NME: {nme_98}  (WFLW Landmark 98 detection)')

                # Landmark detection 29
                pixel_mae_29, nme_29 = eval.evaluate_landmarks_29(model = model, dataloader = cofw_validation_db, image_size = image_size[0])
                print(f'COFW: Pixel MAE: {pixel_mae_29}, NME: {nme_29}  (COFW Landmark 29 detection)')

                # Visibility detection
                acc_50, f1_50, prec_50, rec_50, cm_50, recall_at_80_prec, best_threshold = eval.evaluate_visibility_29(model = model, dataloader = cofw_validation_db)
                print(f'Visibility Prediction (COFW): Accuracy: {acc_50}, F1: {f1_50}, Precision: {prec_50}, Recall: {rec_50}, Recall at 80% Precision: {recall_at_80_prec}, Best Threshold: {best_threshold}')
        
        checkpoint = { 
            'epoch' : epoch,
            'model_state_dict' : model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
            'scheduler_state_dict' : scheduler.state_dict(),
            'loss' : epoch_loss,
            'losses_weights_history' : losses_weights_history,
            'losses_weights' : losses_weights,
            'attribute_thresholds' : thresholds_to_use
        }

        if loss_weighing_type == 'uncertainty':
            checkpoint['loss_weighing'] = loss_weighing.state_dict()
        elif loss_weighing_type == 'dwa':
            checkpoint['loss_weighing'] = loss_weighing.avg_losses_previous_epoch
        

        # Create the checkpoint file if it doesn't exist.
        if not os.path.exists(checkpoint_path):
            os.makedirs(os.path.split(checkpoint_path)[0], exist_ok = True) # create directory
        
        torch.save(
            checkpoint, 
            checkpoint_path
        )

    # Save the final model
    output_model_path = os.path.join('data', 'models', output_folder_name)
    os.makedirs(output_model_path, exist_ok = True)
    torch.save(
        model.state_dict(), 
        os.path.join(output_model_path, 'model.pth')
    )

    # Plot dynamic weight average history
    task_names = ['Face Recognition', 'Emotion Recognition', 'Age Estimation', 'Gender Recognition', 'Race Recognition', 'Attribute Recognition', 'Head Pose Estimation', 'Landmark 98 Detection', 'Landmark 68 Detection', 'Landmark 29 Detection', 'Visibility Prediction']
    plt.figure(figsize=(10, 5), dpi=600)
    plt.title('The history of the loss weights per task.')
    
    weight_history = np.array(losses_weights_history)
    
    # Create an X-axis array (e.g., Epoch 1, 2, 3...)
    epochs_x = range(1, weight_history.shape[0] + 1)

    for i in range(weight_history.shape[1]):
        # Plot X (Epochs) vs Y (Weights for task i)
        plt.plot(epochs_x, weight_history[:, i], label=task_names[i])
        
    plt.legend(loc='best')
    plt.xlabel('Epoch')
    plt.ylabel('Task weight')
    
    # Ensure directory exists before saving
    os.makedirs(os.path.join('data', 'figures', 'MultiTask Training', output_folder_name), exist_ok=True)
    
    plt.savefig(
        os.path.join('data', 'figures', 'MultiTask Training', output_folder_name, 'Task_loss_weights_history.png')
    )
    plt.close()

            
if __name__ == '__main__':
    for key, value in config.items():
        print(f'{key}: {value}')
    main(**config)