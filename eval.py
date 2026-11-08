import numpy as np
import torch
from torchvision.transforms import v2
import torch.nn.functional as F
import datasets
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.model_selection import KFold
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, precision_recall_curve
from utility_scripts.pose_estimation_utilities import *


##################################

#   MultiTask testing and validation utilities

#################################

def evaluate_emotion(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device = 'cuda'):
    """
    Evaluates Emotion Recognition (Multi-class Classification).
    """
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating Emotion", leave=False):
            images, labels = batch
            images = images.to(device)
            targets = labels['emotion'].to(device)
            
            # Forward pass - Extract only emotion output
            emotion_out = model(images)[1] # Emotion is the second output
            
            preds = torch.argmax(emotion_out, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)
    
    return accuracy, f1, precision, recall, cm


def evaluate_age(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device='cuda', min_age=0, max_age=101):
    """
    Evaluates Age Estimation (DLDL approach).
    Returns: Mean Absolute Error (MAE) in years.
    """
    model.to(device)
    model.eval()
    true_ages = []
    predicted_ages_list = []
    
    # Pre-create the age vector (0, 1, ..., 101) once to save time
    age_values = torch.arange(min_age, max_age + 1, dtype=torch.float32, device=device).view(1, -1)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating Age", leave=False):
            images, labels = batch
            images = images.to(device)

            targets = labels['age'].to(device).float()
            
            age_logits = model(images)[2] # Age is the third output
            
            probs = F.softmax(age_logits, dim=1)
            predicted_ages = torch.sum(probs * age_values, dim=1) 
            
            true_ages.extend(targets.cpu().numpy())
            predicted_ages_list.extend(predicted_ages.cpu().numpy())
    
    
    true_ages = np.array(true_ages)
    predicted_ages_list = np.array(predicted_ages_list)

    mae = np.mean(np.abs(true_ages - predicted_ages_list))

    return mae, true_ages, predicted_ages_list



def evaluate_gender(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device = 'cuda'):
    """
    Evaluates Gender Recognition (Binary Classification).
    """
    model.to(device)
    model.eval()
    true_labels = []
    predicted_labels = []
    probs_list = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating Gender", leave=False):
            images, labels = batch
            images = images.to(device)
            targets = labels['gender'].to(device)
            
            # Forward pass - Extract only gender output
            gender_out = model(images)[3] # Gender is the fourth output
            
            # Accuracy (Sigmoid > 0.5)
            probs = torch.sigmoid(gender_out)
            preds = (probs > 0.5).long()

            true_labels.extend(targets.cpu().numpy())
            predicted_labels.extend(preds.cpu().numpy())
            probs_list.extend(probs.cpu().numpy())

    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    probs_list = np.array(probs_list)

    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    cm = confusion_matrix(true_labels, predicted_labels)
    fpr, tpr, thresholds = roc_curve(true_labels, probs_list)
    auc_score = roc_auc_score(true_labels, probs_list)
    return accuracy, f1, precision, recall, cm, fpr, tpr, auc_score, thresholds


def evaluate_race(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device = 'cuda'):
    """
    Evaluates Race Recognition (Multi-class Classification).
    """
    model.to(device)
    model.eval()
    predicted_labels = []
    actual_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating Race", leave=False):
            images, labels = batch
            images = images.to(device)
            targets = labels['race'].to(device)
            
            race_out = model(images)[4] # Race is the fifth output
            
            preds = torch.argmax(race_out, dim=1)

            predicted_labels.extend(preds.cpu().numpy())
            actual_labels.extend(targets.cpu().numpy())

    predicted_labels = np.array(predicted_labels)
    actual_labels = np.array(actual_labels)

    accuracy = accuracy_score(actual_labels, predicted_labels)
    f1 = f1_score(actual_labels, predicted_labels, average='weighted')
    precision = precision_score(actual_labels, predicted_labels, average='weighted')
    recall = recall_score(actual_labels, predicted_labels, average='weighted')
    cm = confusion_matrix(actual_labels, predicted_labels)

    return accuracy, f1, precision, recall, cm


def evaluate_attributes(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device='cuda', thresholds=None):
    """
    Evaluates Attribute Recognition (Multi-label Classification).
    If thresholds is None, it calculates the optimal threshold per attribute to maximize F1.
    """
    model.to(device)
    model.eval()
    
    all_probs = []
    actual_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating Attributes", leave=False):
            images, labels = batch
            images = images.to(device)
            
            # targets is the Tensor (B, 40)
            targets = labels['attributes'].to(device)

            attribute_out = model(images)[5] # Attributes is the sixth output
            
            # Save raw probabilities instead of hard thresholding at 0.5
            predictions = torch.sigmoid(attribute_out)

            actual_labels.extend(targets.cpu().numpy())
            all_probs.extend(predictions.cpu().numpy())

    actual_labels = np.array(actual_labels)
    all_probs = np.array(all_probs)
    num_attributes = actual_labels.shape[1]

    # --- ADAPTIVE THRESHOLD CALCULATION ---
    best_thresholds = np.zeros(num_attributes)
    
    if thresholds is None:
        for i in range(num_attributes):
            # Calculate Precision-Recall curve for this specific attribute
            precisions, recalls, threshs = precision_recall_curve(actual_labels[:, i], all_probs[:, i])
            
            # Calculate F1 scores across all possible thresholds
            # Add 1e-8 to prevent division by zero
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
            
            # Find the index of the maximum F1 score
            best_idx = np.argmax(f1_scores)
            
            # Scikit-learn's thresholds array is 1 element shorter than precisions/recalls
            best_thresholds[i] = threshs[best_idx] if best_idx < len(threshs) else 0.5
            
        thresholds_to_use = best_thresholds
    else:
        thresholds_to_use = thresholds

    # --- APPLY THRESHOLDS ---
    predicted_labels = (all_probs > thresholds_to_use).astype(int)
    
    # Element-wise Accuracy
    element_wise_accuracy = (actual_labels == predicted_labels).mean()

    # Calculate final metrics using the optimized predictions
    f1 = f1_score(actual_labels, predicted_labels, average='micro')
    precision = precision_score(actual_labels, predicted_labels, average='micro')
    recall = recall_score(actual_labels, predicted_labels, average='micro')
    
    # Return the thresholds so they can be saved for inference deployment
    return element_wise_accuracy, f1, precision, recall, thresholds_to_use


def evaluate_head_pose(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device='cuda'):
    """
    Evaluates Head Pose Estimation (Regression).
    Returns Mean Absolute Error (MAE) for Roll, Pitch, and Yaw in Degrees.
    """
    model.to(device)
    model.eval()
    
    # Accumulators for errors
    roll_errors = []
    pitch_errors = []
    yaw_errors = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating Head Pose", leave=False):
            images, labels = batch
            images = images.to(device)
            
            # Ground Truth (Degrees)
            pose_labels = labels['pose'].to(device)
            predictions = model(images)[6] # Pose is the seventh output
            pred_matrices = compute_rotation_matrix_from_ortho6d(predictions) 
            
            pred_degrees = matrix_to_euler_angles(pred_matrices)
            
            errors = torch.abs(pred_degrees - pose_labels)
            
            # Store errors (CPU)
            roll_errors.extend(errors[:, 0].cpu().numpy())
            pitch_errors.extend(errors[:, 1].cpu().numpy())
            yaw_errors.extend(errors[:, 2].cpu().numpy())

    # Calculate Mean Errors
    mae_roll = np.mean(roll_errors)
    mae_pitch = np.mean(pitch_errors)
    mae_yaw = np.mean(yaw_errors)
    
    # Average MAE across all axes
    mae_mean = np.mean([mae_roll, mae_pitch, mae_yaw])
    
    return mae_mean, mae_roll, mae_pitch, mae_yaw

def evaluate_landmarks_98(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device='cuda', image_size=112.0):
    """
    Evaluates Landmark Detection (Regression) using standard SOTA metrics.
    Returns: Mean Pixel Error, and Normalized Mean Error (NME)
    """
    model.to(device)
    model.eval()
    
    all_pixel_errors = []
    all_nme_errors = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating Landmarks", leave=False):
            images, labels = batch
            images = images.to(device)
            gt_landmarks = labels['landmarks_98'].to(device)
            
            landmarks_98_output = model(images)[7] # Landmarks 98 is the eighth output
            
            # 1. Reshape from (Batch, 136) to (Batch, 98, 2) to pair X and Y coordinates
            pred_pts = landmarks_98_output.view(-1, 98, 2) * image_size
            gt_pts = gt_landmarks.view(-1, 98, 2) * image_size
            
            # 2. Calculate Euclidean Distance per point: sqrt((x1-x2)^2 + (y1-y2)^2)
            # Shape becomes (Batch, 98)
            point_errors = torch.norm(pred_pts - gt_pts, dim=2) 
            
            # 3. Calculate mean pixel error per image: (Batch,)
            mean_pixel_error_per_image = point_errors.mean(dim=1)
            all_pixel_errors.extend(mean_pixel_error_per_image.cpu().numpy())
            
            # 4. Calculate NME (Normalized Mean Error) using Inter-ocular distance
            # Index 60: Left eye outer corner, Index 72: Right eye outer corner
            left_eye_corners = gt_pts[:, 60, :]
            right_eye_corners = gt_pts[:, 72, :]
            
            # Calculate distance between the two eyes: (Batch,)
            inter_ocular_distances = torch.norm(left_eye_corners - right_eye_corners, dim=1)
            
            # Normalize the image's pixel error by its specific eye distance
            nme_per_image = mean_pixel_error_per_image / inter_ocular_distances
            all_nme_errors.extend(nme_per_image.cpu().numpy())

    final_mean_pixel_error = np.mean(all_pixel_errors)
    final_nme = np.mean(all_nme_errors) * 100 # Often reported as a percentage
    
    return final_mean_pixel_error, final_nme


def evaluate_landmarks_68(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device='cuda', image_size=112.0):
    """
    Evaluates Landmark Detection (Regression) using standard SOTA metrics.
    Returns: Mean Pixel Error, and Normalized Mean Error (NME)
    """
    model.to(device)
    model.eval()
    
    all_pixel_errors = []
    all_nme_errors = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating Landmarks", leave=False):
            images, labels = batch
            images = images.to(device)
            gt_landmarks = labels['landmarks_68'].to(device)
            
            landmarks_68_output = model(images)[8] # Landmarks 68 is the ninth output
            
            # 1. Reshape from (Batch, 136) to (Batch, 68, 2) to pair X and Y coordinates
            pred_pts = landmarks_68_output.view(-1, 68, 2) * image_size
            gt_pts = gt_landmarks.view(-1, 68, 2) * image_size
            
            # 2. Calculate Euclidean Distance per point: sqrt((x1-x2)^2 + (y1-y2)^2)
            # Shape becomes (Batch, 68)
            point_errors = torch.norm(pred_pts - gt_pts, dim=2) 
            
            # 3. Calculate mean pixel error per image: (Batch,)
            mean_pixel_error_per_image = point_errors.mean(dim=1)
            all_pixel_errors.extend(mean_pixel_error_per_image.cpu().numpy())
            
            # 4. Calculate NME (Normalized Mean Error) using Inter-ocular distance
            # Index 36: Left eye outer corner, Index 45: Right eye outer corner
            left_eye_corners = gt_pts[:, 36, :]
            right_eye_corners = gt_pts[:, 45, :]
            
            # Calculate distance between the two eyes: (Batch,)
            inter_ocular_distances = torch.norm(left_eye_corners - right_eye_corners, dim=1)
            
            # Normalize the image's pixel error by its specific eye distance
            nme_per_image = mean_pixel_error_per_image / inter_ocular_distances
            all_nme_errors.extend(nme_per_image.cpu().numpy())

    final_mean_pixel_error = np.mean(all_pixel_errors)
    final_nme = np.mean(all_nme_errors) * 100 # Often reported as a percentage
    
    return final_mean_pixel_error, final_nme


def evaluate_landmarks_29(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device='cuda', image_size=112.0):
    """
    Evaluates Landmark Detection (Regression) using standard SOTA metrics.
    Returns: Mean Pixel Error, and Normalized Mean Error (NME)
    """
    model.to(device)
    model.eval()
    
    all_pixel_errors = []
    all_nme_errors = []    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating Landmarks", leave=False):
            images, labels = batch
            images = images.to(device)
            gt_landmarks = labels['landmarks_29'].to(device)
            
            landmark_29_output = model(images)[9] # Landmark 29 is the tenth output
            
            # 1. Reshape from (Batch, 136) to (Batch, 29, 2) to pair X and Y coordinates
            pred_pts = landmark_29_output.view(-1, 29, 2) * image_size
            gt_pts = gt_landmarks.view(-1, 29, 2) * image_size
            
            # 2. Calculate Euclidean Distance per point: sqrt((x1-x2)^2 + (y1-y2)^2)
            # Shape becomes (Batch, 29)
            point_errors = torch.norm(pred_pts - gt_pts, dim=2) 
            
            # 3. Calculate mean pixel error per image: (Batch,)
            mean_pixel_error_per_image = point_errors.mean(dim=1)
            all_pixel_errors.extend(mean_pixel_error_per_image.cpu().numpy())
            
            # 4. Calculate NME (Normalized Mean Error) using Inter-ocular distance
            # Index 8: Left eye outer corner, Index 9: Right eye outer corner
            left_eye_corners = gt_pts[:, 8, :]
            right_eye_corners = gt_pts[:, 9, :]
            
            # Calculate distance between the two eyes: (Batch,)
            inter_ocular_distances = torch.norm(left_eye_corners - right_eye_corners, dim=1)
            
            # Normalize the image's pixel error by its specific eye distance
            nme_per_image = mean_pixel_error_per_image / inter_ocular_distances
            all_nme_errors.extend(nme_per_image.cpu().numpy())

    final_mean_pixel_error = np.mean(all_pixel_errors)
    final_nme = np.mean(all_nme_errors) * 100 # Often reported as a percentage
    
    return final_mean_pixel_error, final_nme




def evaluate_visibility_29(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device='cuda'):
    """
    Evaluates COFW 29-point Landmark Visibility.
    Calculates standard metrics at a 0.5 threshold, and Recall @ 80% Precision.
    """
    model.to(device)
    model.eval()
    
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating Visibility", leave=False):
            images, labels = batch
            images = images.to(device)
            targets = labels['visibility'].to(device) # Shape: (Batch, 29)
            
            *_, visibility_output = model(images)
            
            probs = torch.sigmoid(visibility_output)

            all_probs.extend(probs.cpu().float().numpy().flatten())
            all_targets.extend(targets.cpu().float().numpy().flatten())

    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)

    # 1. Standard Metrics (at the default 0.5 threshold)
    all_preds_50 = (all_probs > 0.5).astype(int)
    
    acc_50 = accuracy_score(all_targets, all_preds_50)
    f1_50 = f1_score(all_targets, all_preds_50, average='binary')
    prec_50 = precision_score(all_targets, all_preds_50, average='binary')
    rec_50 = recall_score(all_targets, all_preds_50, average='binary')
    cm_50 = confusion_matrix(all_targets, all_preds_50)

    # 2. SOTA Metric: Recall @ 80% Precision
    # precision_recall_curve tests every single possible threshold automatically
    precisions, recalls, thresholds = precision_recall_curve(all_targets, all_probs)
    
    # Find all indices where the precision is >= 80%
    target_precision = 0.80
    valid_indices = np.where(precisions >= target_precision)[0]
    
    if len(valid_indices) > 0:
        # We want the lowest possible threshold that gives >= 80% precision
        # because a lower threshold maximizes our Recall.
        best_idx = valid_indices[0] 
        recall_at_80_prec = recalls[best_idx]
        
        # Scikit-learn's thresholds array is 1 element shorter than precisions/recalls
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 1.0
    else:
        # If the network is so bad it literally never hits 80% precision
        recall_at_80_prec = 0.0
        best_threshold = 1.0

    return acc_50, f1_50, prec_50, rec_50, cm_50, recall_at_80_prec, best_threshold




def evaluate_bias_fairface(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device='cuda'):
    """
    Evaluates Bias and Fairness on the FairFace dataset.
    Calculates Gender and Age accuracy discrepancies between White and Non-White cohorts
    using ground-truth racial labels.
    """
    model.to(device)
    model.eval()
    
    min_age = 0
    max_age = 101
    age_values = torch.arange(min_age, max_age + 1, dtype=torch.float32, device=device).view(1, -1)
    
    # FairFace uses 9 age bins: 0-2, 3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+
    # These are the upper limits for bucketizing.
    age_boundaries = torch.tensor([2, 9, 19, 29, 39, 49, 59, 69], dtype=torch.float32, device=device)

    # Accumulators
    white_gender_correct, white_age_correct, white_total = 0, 0, 0
    nonwhite_gender_correct, nonwhite_age_correct, nonwhite_total = 0, 0, 0

    # Assuming '0' is the index for White in your 5-class race mapping
    WHITE_CLASS_IDX = 0 

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating FairFace Bias", leave=False):
            images, labels = batch
            images = images.to(device)
            
            # Extract ground-truth demographic groups
            age_groups = labels['age_group'].to(device) # The 0-8 categorical bins
            genders = labels['gender'].to(device)
            races = labels['race'].to(device)
            
            # Forward Pass
            _, _, age_logits, gender_output, race_output, *_ = model(images)

            # 1. Process Gender Prediction
            gender_pred = (torch.sigmoid(gender_output) > 0.5).long().squeeze()
            
            # 2. Process Age Prediction (Continuous -> Categorical Bin)
            age_pred_continuous = torch.sum(F.softmax(age_logits, dim=1) * age_values, dim=1)
            age_bin_preds = torch.bucketize(age_pred_continuous, age_boundaries, right=True)

            # 3. Create Racial Masks (Based on GROUND TRUTH, not predictions)
            is_white = (races == WHITE_CLASS_IDX)
            is_nonwhite = (races != WHITE_CLASS_IDX)

            # 4. Calculate Correct Hits for this batch
            correct_gender = (gender_pred == genders)
            correct_age = (age_bin_preds == age_groups)

            # 5. Accumulate for White Cohort
            white_gender_correct += correct_gender[is_white].sum().item()
            white_age_correct += correct_age[is_white].sum().item()
            white_total += is_white.sum().item()

            # 6. Accumulate for Non-White Cohort
            nonwhite_gender_correct += correct_gender[is_nonwhite].sum().item()
            nonwhite_age_correct += correct_age[is_nonwhite].sum().item()
            nonwhite_total += is_nonwhite.sum().item()

    # --- Final Metric Calculation ---
    
    # Prevent division by zero just in case testing with a tiny batch
    white_total = max(white_total, 1)
    nonwhite_total = max(nonwhite_total, 1)

    white_gender_acc = (white_gender_correct / white_total) * 100
    nonwhite_gender_acc = (nonwhite_gender_correct / nonwhite_total) * 100
    gender_discrepancy = abs(white_gender_acc - nonwhite_gender_acc)

    white_age_acc = (white_age_correct / white_total) * 100
    nonwhite_age_acc = (nonwhite_age_correct / nonwhite_total) * 100
    age_discrepancy = abs(white_age_acc - nonwhite_age_acc)

    print("\n" + "="*50)
    print(" FAIRFACE BIAS AND FAIRNESS EVALUATION")
    print("="*50)
    print(f"White Gender Acc:     {white_gender_acc:.2f}%")
    print(f"Non-White Gender Acc: {nonwhite_gender_acc:.2f}%")
    print(f"Gender Discrepancy:   {gender_discrepancy:.2f}")
    print("-" * 50)
    print(f"White Age Acc:        {white_age_acc:.2f}%")
    print(f"Non-White Age Acc:    {nonwhite_age_acc:.2f}%")
    print(f"Age Discrepancy:      {age_discrepancy:.2f}")
    print("="*50 + "\n")

    return {
        'white_gender_acc': white_gender_acc,
        'nonwhite_gender_acc': nonwhite_gender_acc,
        'gender_discrepancy': gender_discrepancy,
        'white_age_acc': white_age_acc,
        'nonwhite_age_acc': nonwhite_age_acc,
        'age_discrepancy': age_discrepancy
    }




            
            

##################################

#   Face recognition testing utilities

##################################


def evaluate_face_recognition(backbone, datasets_to_test = ['LFW', 'CPLFW', 'CALFW', 'CFP-FP', 'CFP-FF', 'AgeDB30', 'VGG2FP'], batch_size = 128, image_size = 224):
    """
    Evaluates a face recognition backbone on standard benchmarks using 10-fold cross-validation.

    This function computes squared L2 distance scores on various test datasets
    and then calculates the average accuracy and other metrics following the
    standard face recognition evaluation protocol (e.g., LFW).

    Args:
        backbone : The trained face recognition model (embedding extractor).

    Returns:
        dict: A dictionary where keys are dataset names (e.g., 'LFW') and values
              are tuples containing:
              (mean_accuracy, mean_precision, mean_recall, mean_f1_score,
               global_auc_score, global_fpr, global_tpr, global_roc_thresholds).
              The primary metric reported in papers is 'mean_accuracy'.
    """
    distances_results = get_face_recognition_distances_from_backbone(backbone, datasets_to_test, batch_size = batch_size, image_size = image_size)
    metrics_results = get_metrics_from_distances(distances_results)
    return metrics_results

# --- Feature Extraction and Distance Calculation ---

def get_face_recognition_distances_from_backbone(backbone: torch.nn.Module,
                                                 datasets_to_test = ['LFW', 'CPLFW', 'CALFW', 'CFP-FP', 'CFP-FF', 'AgeDB30', 'VGG2FP'],
                                                 use_tta = True,
                                                 batch_size = 512,
                                                 image_size = 224,
                                                 use_gpu = True,
                                                 test_in_bgr = False):
    """
    Computes squared L2 distance scores for image pairs from specified face recognition datasets.

    Extracts embeddings using the provided backbone, applies optional Test-Time Augmentation (TTA),
    and calculates the squared Euclidean distance between embeddings for each pair.

    Args:
        backbone (torch.nn.Module): The trained face recognition model.
        datasets_to_test (list, optional): List of dataset names.
            Defaults to ['LFW', 'CPLFW', 'CALFW', 'CFP-FP', 'CFP-FF'].
        use_tta (bool, optional): If True, averages embeddings from original and horizontally flipped images.
                                  Defaults to True.
        batch_size (int, optional): Batch size for processing images. Defaults to 512.
        image_size (int, optional): Size of the input images. Defaults to 224.
        use_gpu (bool): Whether to use GPU for processing. Defaults to True.


    Returns:
        dict: A dictionary where keys are dataset names and values are tuples
              (distances, labels), where 'distances' is a NumPy array of squared L2
              distances and 'labels' is a NumPy array of ground-truth labels (1=same, 0=different).
    """
    if use_gpu:
        backbone.to('cuda')
    backbone.eval()

    image_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale = True),
        v2.Resize((image_size, image_size)),
        v2.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5]) # To [-1, 1]
    ])

    test_datasets_map = {
        'LFW': datasets.LFW,
        'CPLFW': datasets.CPLFW,
        'CALFW': datasets.CALFW,
        'CFP-FP': datasets.CFPFP,
        'CFP-FF': datasets.CFPFF,
        'AgeDB30' : datasets.AgeDB30,
        'VGG2FP' : datasets.VGG2FP
    }
    test_datasets = []
    loaded_dataset_names = []
    for name in datasets_to_test:
        if name in test_datasets_map:
            test_datasets.append(test_datasets_map[name](transform=image_transform))
            loaded_dataset_names.append(name)
        else:
            print(f"Warning: Dataset '{name}' not recognized and will be skipped.")
            continue

    results = {}

    for i, dataset in enumerate(test_datasets):
        dataset_name = loaded_dataset_names[i]
        print(f"Processing dataset: {dataset_name}...")
        predicted_dist_list = []
        actual_list = []

        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=4,
                                             pin_memory=True)

        with torch.no_grad():
            for (image1_batch, image2_batch), label_batch in loader:
                
                if test_in_bgr:
                    #rgb to bgr
                    image1_batch = torch.flip(image1_batch, dims = [1])
                    image2_batch = torch.flip(image2_batch, dims = [1])

                if use_gpu:
                    image1_batch = image1_batch.to('cuda', non_blocking=True)
                    image2_batch = image2_batch.to('cuda', non_blocking=True)

                if use_tta:
                    image1_flipped = torch.flip(image1_batch, dims=[3])
                    image2_flipped = torch.flip(image2_batch, dims=[3])

                    embeddings1_original, norm1_original = backbone(image1_batch)
                    embeddings2_original, norm2_original = backbone(image2_batch)
                    embeddings1_flipped, norm1_flipped = backbone(image1_flipped)
                    embeddings2_flipped, norm2_flipped = backbone(image2_flipped)

                    #unnormalize
                    embeddings1_original = embeddings1_original * norm1_original
                    embeddings2_original = embeddings2_original * norm2_original
                    embeddings1_flipped = embeddings1_flipped * norm1_flipped
                    embeddings2_flipped = embeddings2_flipped * norm2_flipped

                    # sum
                    embeddings1 = (embeddings1_original + embeddings1_flipped)
                    embeddings2 = (embeddings2_original + embeddings2_flipped)

                    #normalize
                    embeddings1 = F.normalize(embeddings1, p=2, dim=1)
                    embeddings2 = F.normalize(embeddings2, p=2, dim=1)

                    embeddings1 = embeddings1.cpu().numpy()
                    embeddings2 = embeddings2.cpu().numpy()

                else: # No TTA
                    embeddings1, norm1 = backbone(image1_batch)
                    embeddings2, norm2 = backbone(image2_batch)

                    embeddings1 = embeddings1.cpu().numpy()
                    embeddings2 = embeddings2.cpu().numpy()


                diff = embeddings1 - embeddings2
                dist = np.sum(np.square(diff), axis=1)

                predicted_dist_list.extend(dist)
                actual_list.extend(label_batch.cpu().numpy())

        results[dataset_name] = (np.array(predicted_dist_list), np.array(actual_list))
        print(f"Finished processing {dataset_name}.")

    return results

# --- Metric Calculation Utilities ---

def get_metrics_from_distances(results: dict):
    """
    Calculates face recognition metrics using 10-fold cross-validation on distance scores.

    Follows the standard LFW protocol: finds the best accuracy threshold on 9 folds,
    tests on the remaining fold, and averages the test accuracy over all 10 folds.
    Also calculates average precision, recall, F1, and global ROC/AUC.

    Args:
        results (dict): Dictionary from `get_face_recognition_distances_from_backbone`.
                        Keys are dataset names, values are (distances, labels) tuples.

    Returns:
        dict: A dictionary where keys are dataset names and values are tuples:
              (mean_accuracy, mean_precision, mean_recall, mean_f1_score,
               global_auc_score, global_fpr, global_tpr, global_roc_thresholds).
    """
    dataset_metrics = {}
    for key, (predicted_dist, actual_labels) in results.items():

        kf = KFold(n_splits=10, shuffle=False)
        fold_metrics = []

        for fold_idx, (train_indices, test_indices) in enumerate(kf.split(predicted_dist)):
            fold_best_threshold, _ = find_best_threshold_by_accuracy(
                predicted_dist[train_indices], actual_labels[train_indices]
            )

            accuracy, precision, recall, f1_score = compute_metrics_at_threshold(
                predicted_dist[test_indices], actual_labels[test_indices], fold_best_threshold
            )

            fold_metrics.append([accuracy, precision, recall, f1_score])

        mean_metrics = np.mean(fold_metrics, axis=0)
        mean_accuracy, mean_precision, mean_recall, mean_f1_score = mean_metrics

        fpr, tpr, roc_thresholds = roc_curve(actual_labels, -predicted_dist)
        auc_score = auc(fpr, tpr)

        dataset_metrics[key] = (mean_accuracy, mean_precision, mean_recall, mean_f1_score,
                                auc_score, fpr, tpr, roc_thresholds)

    return dataset_metrics


def compute_metrics_at_threshold(distances, labels, threshold, eta=1e-11):
    """
    Computes accuracy, precision, recall, and F1-score for a given distance threshold.

    Args:
        distances (np.ndarray): Array of squared L2 distance scores.
        labels (np.ndarray): Array of ground-truth labels (1 for same ID, 0 for different).
        threshold (float): Decision threshold. Distances < threshold are predicted as positive (same ID).
        eta (float, optional): Small epsilon to prevent division by zero. Defaults to 1e-11.

    Returns:
        tuple: (accuracy, precision, recall, f1_score).
    """
    distances = np.array(distances)
    labels = np.array(labels)

    predictions = (distances < threshold).astype(int)

    tp = np.sum((predictions == 1) & (labels == 1))
    fp = np.sum((predictions == 1) & (labels == 0))
    tn = np.sum((predictions == 0) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))
    
    total_samples = len(labels)
    accuracy = (tp + tn) / (total_samples + eta)
    precision = tp / (tp + fp + eta)
    recall = tp / (tp + fn + eta)
    f1_score = 2 * precision * recall / (precision + recall + eta)

    return accuracy, precision, recall, f1_score


def find_best_threshold_by_accuracy(distances, labels, num_thresholds=1000):
    """
    Finds the decision threshold that maximizes accuracy on the given distances and labels.

    Iterates through potential thresholds in the range [0, 4] (valid range for
    squared L2 distance between normalized vectors) and returns the one yielding
    the highest accuracy.

    Args:
        distances (np.ndarray): Array of squared L2 distance scores.
        labels (np.ndarray): Array of ground-truth labels (1=same, 0=different).
        num_thresholds (int, optional): Number of threshold values to test.
            Defaults to 1000.

    Returns:
        tuple: (best_threshold, best_accuracy). Returns (0.0, 0.0) if inputs are empty or invalid.
    """
    if len(distances) == 0 or len(labels) == 0:
        return 0.0, 0.0 # Handle empty input case

    thresholds = np.linspace(0.0, 4.0, num_thresholds) # the maximum squared distance between two normalized vectors is 4, so the range is from 0 to 4

    best_threshold = 0.0
    best_accuracy = 0.0

    for threshold in thresholds:
        accuracy, _, _, _ = compute_metrics_at_threshold(distances, labels, threshold)

        if accuracy >= best_accuracy: # Use >= to prefer higher thresholds in case of tie
            best_accuracy = accuracy
            best_threshold = threshold

    return best_threshold, best_accuracy