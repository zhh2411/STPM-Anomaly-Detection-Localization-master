import cv2
import torch.nn.functional as F
import numpy as np
# from scipy.ndimage import gaussian_filter


def cal_loss(fs_list, ft_list):
    t_loss = 0
    N = len(fs_list)
    for i in range(N):
        fs = fs_list[i]
        ft = ft_list[i]
        _, _, h, w = fs.shape
        fs_norm = F.normalize(fs, p=2)
        ft_norm = F.normalize(ft, p=2)
        f_loss = 0.5 * (ft_norm - fs_norm)**2
        f_loss = f_loss.sum() / (h*w)
        t_loss += f_loss

    return t_loss / N


def cal_anomaly_maps(fs_list, ft_list, out_size, is_numpy=False):
    """
    Unified function to calculate anomaly maps that works with either:
    - PyTorch tensors (from original model)
    - NumPy arrays (from ONNX model)
    
    Args:
        fs_list: List of student features (either torch tensors or numpy arrays)
        ft_list: List of teacher features (either torch tensors or numpy arrays)
        out_size: Target output size
        is_numpy: Boolean flag to indicate if inputs are numpy arrays (True) or PyTorch tensors (False)
    
    Returns:
        Anomaly maps (numpy array)
    """
    
    if not is_numpy:        
        anomaly_map = 0
        for i in range(len(ft_list)):
            fs = fs_list[i]
            ft = ft_list[i]
            fs_norm = F.normalize(fs, p=2)
            ft_norm = F.normalize(ft, p=2)
            _, _, h, w = fs.shape
            a_map = (0.5 * (ft_norm - fs_norm)**2) / (h*w)
            a_map = a_map.sum(1, keepdim=True)
            a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=False)
            anomaly_map += a_map
        
        # Convert to numpy for further processing
        anomaly_map = anomaly_map.squeeze().cpu().numpy()
        
    else:       
        anomaly_map = 0
        batch_size = fs_list[0].shape[0]
        
        for i in range(len(ft_list)):
            fs = fs_list[i]
            ft = ft_list[i]
            
            # Normalize features
            fs_norm = fs / (np.sqrt(np.sum(fs**2, axis=1, keepdims=True)) + 1e-10)
            ft_norm = ft / (np.sqrt(np.sum(ft**2, axis=1, keepdims=True)) + 1e-10)
            
            _, h, w = fs.shape[1:]
            
            # Compute squared difference and normalize
            a_map = (0.5 * (ft_norm - fs_norm)**2) / (h*w)
            
            # Sum across channel dimension
            a_map = np.sum(a_map, axis=1, keepdims=True)
            
            # Resize to target size using OpenCV
            resized_maps = np.zeros((batch_size, 1, out_size, out_size))
            for b in range(batch_size):
                # Extract the map for this batch item
                single_map = a_map[b, 0]
                
                # Use OpenCV for resizing
                resized = cv2.resize(single_map, (out_size, out_size), 
                                    interpolation=cv2.INTER_LINEAR)
                
                resized_maps[b, 0] = resized
            
            anomaly_map += resized_maps
        
        # Remove channel dimension
        anomaly_map = anomaly_map.squeeze(axis=1)
    
    # Apply Gaussian filter to each image in the batch
    for i in range(anomaly_map.shape[0]):
        # anomaly_map[i] = gaussian_filter(anomaly_map[i], sigma=4)
        anomaly_map[i] = cv2.GaussianBlur(anomaly_map[i], ksize=(0, 0), sigmaX=4, sigmaY=4)
    
    return anomaly_map

