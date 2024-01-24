import numpy as np
from scipy.spatial.distance import cosine
import torch
import cv2

class KalmanFilter:
    def __init__(self, dt, u_x, u_y, std_acc, x_std_meas, y_std_meas):
        self.dt = dt

        # Control input (acceleration in x and y)
        self.u = np.array([[u_x], [u_y]])

        # State Matrix [x, y, vx, vy]
        self.state = np.array([[0], [0], [0], [0]])

        # System model matrices
        self.A = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

        self.B = np.array([[0.5 * dt**2, 0],
                           [0, 0.5 * dt**2],
                           [dt, 0],
                           [0, dt]])

        # Measurement mapping matrix
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])

        # Process noise covariance
        self.Q = np.array([[0.25 * dt**4, 0, 0.5 * dt**3, 0],
                           [0, 0.25 * dt**4, 0, 0.5 * dt**3],
                           [0.5 * dt**3, 0, dt**2, 0],
                           [0, 0.5 * dt**3, 0, dt**2]]) * std_acc**2

        # Measurement noise covariance
        self.R = np.array([[x_std_meas**2, 0],
                           [0, y_std_meas**2]])

        # Prediction error covariance
        self.P = np.eye(self.A.shape[0])

    def predict(self):

        # Update time state
        self.state = np.dot(self.A, self.state) + np.dot(self.B, self.u)

        # Calculate error covariance
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

        return self.state
    
    def update(self, z):
    
        # Calculate Kalman Gain
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
    
        # Update state estimate
        self.state += np.dot(K, (z - np.dot(self.H, self.state)))
    
        # Update error covariance
        self.P = self.P - np.dot(K, np.dot(S, K.T))
    
        return self.state

def read_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    strtok = [line.strip().split(',') for line in lines]
    #convert to int and add index
    to_int = []
    for i, line in enumerate(strtok):
        to_int.append([int(float(x)) for x in line])
    return np.array(to_int)

def compute_similarity(box1, box2):
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

def compute_advanced_similarity(box1, box2, features1, features2, histogram1, histogram2):
    iou = compute_similarity(box1, box2)
    feature_similarity = 1 - cosine(features1.flatten(), features2.flatten())
    histogram_similarity = np.corrcoef(histogram1, histogram2)[0, 1]
    # Combine these metrics
    combined_similarity_metric = 0.5 * iou + 0.3 * feature_similarity + 0.2 * histogram_similarity
    if np.isnan(combined_similarity_metric):
        combined_similarity_metric = 0
    return combined_similarity_metric

def compute_jaccard(data1, data2):
    mat = np.zeros((len(data1), len(data2)))
    for i, d1 in enumerate(data1):
        for j, d2 in enumerate(data2):
            mat[i][j] = compute_similarity([d1[2], d1[3], d1[2] + d1[4], d1[3] + d1[5]], [d2[2], d2[3], d2[2] + d2[4], d2[3] + d2[5]])
    return mat

def compute_advanced_jaccard(data1, data2, features1, features2, histograms1, histograms2):
    mat = np.zeros((len(data1), len(data2)))
    for i, d1 in enumerate(data1):
        for j, d2 in enumerate(data2):
            mat[i][j] = compute_advanced_similarity([d1[2], d1[3], d1[2] + d1[4], d1[3] + d1[5]], [d2[2], d2[3], d2[2] + d2[4], d2[3] + d2[5]], features1[i], features2[j], histograms1[i], histograms2[j])
    return mat

def save_tracking_results(filename, tracks, frame_number):
    with open(filename, 'a') as file:
        for track in tracks:
            bbox = track['bbox'][2:6]
            line = f"{frame_number},{track['id']},"
            line += ",".join(map(str, map(int, bbox))) + ",1,-1,-1,-1\n"
            file.write(line)

def get_center(bbox):
    x = bbox[0] + bbox[2]/2
    y = bbox[1] + bbox[3]/2
    return np.array([[x], [y]])


def extract_features(image, bbox, transform, model):
    # Crop the image patch from the bounding box
    x_min, y_min, width, height = bbox
    crop_img = image[y_min:y_min+height, x_min:x_min+width]

    # Transform the image patch for the model
    tensor_img = transform(crop_img)

    # Extract deep features
    with torch.no_grad():
        deep_features = model(tensor_img.unsqueeze(0))  # Add batch dimension

    # Compute color histogram (optional)
    # Flatten the channels and compute the histogram with 256 bins per channel
    histogram = [cv2.calcHist([crop_img], [i], None, [256], [0, 256]).flatten() for i in range(3)]
    color_histogram = np.concatenate(histogram)

    return deep_features, color_histogram