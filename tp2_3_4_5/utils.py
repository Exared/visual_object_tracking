import numpy as np
from scipy.spatial.distance import cosine
import torch
import cv2

class KalmanFilter:
    def __init__(self, dt, u_x, u_y, std_acc, x_std_meas, y_std_meas):
        self.dt = dt
        self.u = np.array([[u_x], [u_y]])
        self.state = np.array([[0], [0], [0], [0]])
        self.A = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

        self.B = np.array([[0.5 * dt**2, 0],
                           [0, 0.5 * dt**2],
                           [dt, 0],
                           [0, dt]])

        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])

        self.Q = np.array([[0.25 * dt**4, 0, 0.5 * dt**3, 0],
                           [0, 0.25 * dt**4, 0, 0.5 * dt**3],
                           [0.5 * dt**3, 0, dt**2, 0],
                           [0, 0.5 * dt**3, 0, dt**2]]) * std_acc**2

        self.R = np.array([[x_std_meas**2, 0],
                           [0, y_std_meas**2]])

        self.P = np.eye(self.A.shape[0])

    def predict(self):
        self.state = np.dot(self.A, self.state) + np.dot(self.B, self.u)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.state
    
    def update(self, z):
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.state += np.dot(K, (z - np.dot(self.H, self.state)))
        self.P = self.P - np.dot(K, np.dot(S, K.T))
        return self.state

def read_file(filename):
    """
    Read tracking data from a file.
    filename: Path to the file containing tracking data.
    Returns an array of tracking data.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    strtok = [line.strip().split(',') for line in lines]
    to_int = []
    for i, line in enumerate(strtok):
        to_int.append([int(float(x)) for x in line])
    return np.array(to_int)

def compute_similarity(box1, box2):
    """
    Compute the similarity (Intersection over Union) between two bounding boxes. 
    box1: The first bounding box as a list or numpy array [x1, y1, x2, y2].
    box2: The second bounding box as a list or numpy array [x1, y1, x2, y2].
    Returns the IoU score as a float.
    """
    x_inter1 = max(box1[0], box2[0])
    y_inter1 = max(box1[1], box2[1])
    x_inter2 = min(box1[2], box2[2])
    y_inter2 = min(box1[3], box2[3])

    intersection_area = max(0, x_inter2 - x_inter1) * max(0, y_inter2 - y_inter1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    if union_area == 0:
        return 0
    return intersection_area / union_area

def compute_similarity_optimized(box1, box2):
    """
    Compute the similarity (Intersection over Union) between two bounding boxes. Optimized to exit early if no overlap.
    
    box1: The first bounding box as a list or numpy array [x1, y1, x2, y2].
    box2: The second bounding box as a list or numpy array [x1, y1, x2, y2].
    
    Returns the IoU score as a float.
    """
    x_inter1 = max(box1[0], box2[0])
    y_inter1 = max(box1[1], box2[1])
    x_inter2 = min(box1[2], box2[2])
    y_inter2 = min(box1[3], box2[3])

    # OPTIMIZATION : No need to compute the intersection area if there is no overlap
    if x_inter1 >= x_inter2 or y_inter1 >= y_inter2:
        return 0

    intersection_area = (x_inter2 - x_inter1) * (y_inter2 - y_inter1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    if union_area == 0:
        return 0
    return intersection_area / union_area


def compute_advanced_similarity(box1, box2, features1, features2, histogram1, histogram2):
    """
    Compute a combined similarity score using IoU, deep features, and color histograms.
    box1: The first bounding box as a list or numpy array [x1, y1, x2, y2].
    box2: The second bounding box as a list or numpy array [x1, y1, x2, y2].
    features1: Deep features of the first box as a numpy array.
    features2: Deep features of the second box as a numpy array.
    histogram1: Color histogram of the first box as a numpy array.
    histogram2: Color histogram of the second box as a numpy array.
    Returns the combined similarity score as a float.
    """
    iou = compute_similarity(box1, box2)
    feature_similarity = 1 - cosine(features1.flatten(), features2.flatten())
    histogram_similarity = np.corrcoef(histogram1, histogram2)[0, 1]
    # Combine these metrics
    combined_similarity_metric = 0.5 * iou + 0.3 * feature_similarity + 0.2 * histogram_similarity
    if np.isnan(combined_similarity_metric):
        combined_similarity_metric = 0
    return combined_similarity_metric

def compute_advanced_similarity_optimized(box1, box2, features1, features2, histogram1, histogram2):
    """
    Compute a combined similarity score using IoU, deep features, and color histograms. Optimized to exit early if no overlap.
    box1: The first bounding box as a list or numpy array [x1, y1, x2, y2].
    box2: The second bounding box as a list or numpy array [x1, y1, x2, y2].
    features1: Deep features of the first box as a numpy array.
    features2: Deep features of the second box as a numpy array.
    histogram1: Color histogram of the first box as a numpy array.
    histogram2: Color histogram of the second box as a numpy array.
    Returns the combined similarity score as a float.
    """
    iou = compute_similarity_optimized(box1, box2)
    
    # OPTIMIZATION : No need to compute the similarity metrics if there is no overlap
    if iou == 0:
        return 0

    feature_similarity = 1 - cosine(features1.flatten(), features2.flatten())
    histogram_similarity = np.corrcoef(histogram1, histogram2)[0, 1]
    combined_similarity_metric = 0.5 * iou + 0.3 * feature_similarity + 0.2 * histogram_similarity
    if np.isnan(combined_similarity_metric):
        combined_similarity_metric = 0
    return combined_similarity_metric

def compute_jaccard(data1, data2):
    """
    Compute the Jaccard matrix (IoU) for sets of bounding boxes.
    data1: Array of bounding boxes for the tracks.
    data2: Array of bounding boxes for the detection.
    Returns a matrix of IoU scores.
    """
    mat = np.zeros((len(data1), len(data2)))
    for i, d1 in enumerate(data1):
        for j, d2 in enumerate(data2):
            mat[i][j] = compute_similarity([d1[2], d1[3], d1[2] + d1[4], d1[3] + d1[5]], [d2[2], d2[3], d2[2] + d2[4], d2[3] + d2[5]])
    return mat

def compute_jaccard_optimized(data1, data2):
    """
    Compute the Jaccard matrix (IoU) for sets of bounding boxes. Optimized to exit early if no overlap. 
    data1: Array of bounding boxes for the tracks.
    data2: Array of bounding boxes for the detection.
    Returns a matrix of IoU scores.
    """
    mat = np.zeros((len(data1), len(data2)))
    for i, d1 in enumerate(data1):
        for j, d2 in enumerate(data2):
            mat[i][j] = compute_similarity_optimized([d1[2], d1[3], d1[2] + d1[4], d1[3] + d1[5]], [d2[2], d2[3], d2[2] + d2[4], d2[3] + d2[5]])
    return mat

def compute_advanced_jaccard(data1, data2, features1, features2, histograms1, histograms2):
    """
    Compute a matrix of combined similarity scores using IoU, deep features, and color histograms for sets of bounding boxes.
    data1: Array of bounding boxes for the tracks.
    data2: Array of bounding boxes for the detection.
    features1: Array of deep features for the tracks.
    features2: Array of deep features for the detection.
    histograms1: Array of color histograms for the tracks.
    histograms2: Array of color histograms for the detection.
    Returns a matrix of combined similarity scores.
    """
    mat = np.zeros((len(data1), len(data2)))
    for i, d1 in enumerate(data1):
        for j, d2 in enumerate(data2):
            mat[i][j] = compute_advanced_similarity([d1[2], d1[3], d1[2] + d1[4], d1[3] + d1[5]], [d2[2], d2[3], d2[2] + d2[4], d2[3] + d2[5]], features1[i], features2[j], histograms1[i], histograms2[j])
    return mat

def compute_advanced_jaccard_optimized(data1, data2, features1, features2, histograms1, histograms2):
    """
    Compute a matrix of combined similarity scores using IoU, deep features, and color histograms for sets of bounding boxes. Optimized to exit early if no overlap.
    data1: Array of bounding boxes for the tracks.
    data2: Array of bounding boxes for the detection.
    features1: Array of deep features for the tracks.
    features2: Array of deep features for the detection.
    histograms1: Array of color histograms for the tracks.
    histograms2: Array of color histograms for the detection.
    Returns a matrix of combined similarity scores.
    """
    mat = np.zeros((len(data1), len(data2)))
    for i, d1 in enumerate(data1):
        for j, d2 in enumerate(data2):
            mat[i][j] = compute_advanced_similarity_optimized([d1[2], d1[3], d1[2] + d1[4], d1[3] + d1[5]], [d2[2], d2[3], d2[2] + d2[4], d2[3] + d2[5]], features1[i], features2[j], histograms1[i], histograms2[j])
    return mat

def save_tracking_results(filename, tracks, frame_number):
    """
    Save tracking results to a file.
    filename: The path to the file where results should be saved.
    tracks: The tracking data to be saved.
    frame_number: The current frame number.
    """
    with open(filename, 'a') as file:
        for track in tracks:
            bbox = track['bbox'][2:6]
            line = f"{frame_number},{track['id']},"
            line += ",".join(map(str, map(int, bbox))) + ",1,-1,-1,-1\n"
            file.write(line)

def get_center(bbox):
    """
    Get the center of a bounding box.
    bbox: The bounding box as a list or numpy array [x, y, width, height].
    Returns the center of the bounding box as a numpy array [[x_center], [y_center]].
    """
    x = bbox[0] + bbox[2]/2
    y = bbox[1] + bbox[3]/2
    return np.array([[x], [y]])


def extract_features(image, bbox, transform, model):
    """
    Extract deep features and color histogram from a bounding box in an image.
    image: The image from which features are to be extracted.
    bbox: The bounding box as a list or numpy array [x, y, width, height].
    transform: The transformation to apply to the cropped image before feature extraction.
    model: The model to use for deep feature extraction.
    Returns deep features and color histogram as numpy arrays.
    """
    x_min, y_min, width, height = bbox
    crop_img = image[y_min:y_min+height, x_min:x_min+width]
    tensor_img = transform(crop_img)
    with torch.no_grad():
        deep_features = model(tensor_img.unsqueeze(0)) 
    histogram = [cv2.calcHist([crop_img], [i], None, [256], [0, 256]).flatten() for i in range(3)]
    color_histogram = np.concatenate(histogram)
    return deep_features, color_histogram