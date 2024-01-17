import numpy as np
import os
import cv2
import time
from scipy.optimize import linear_sum_assignment

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

def compute_jaccard(data1, data2):
    mat = np.zeros((len(data1), len(data2)))
    for i, d1 in enumerate(data1):
        for j, d2 in enumerate(data2):
            mat[i][j] = compute_similarity([d1[2], d1[3], d1[2] + d1[4], d1[3] + d1[5]], [d2[2], d2[3], d2[2] + d2[4], d2[3] + d2[5]])
    return mat

def save_tracking_results(filename, tracks, frame_number):
    with open(filename, 'a') as file:
        for track in tracks:
            bbox = track['bbox'][2:6]
            line = f"{frame_number},{track['id']},"
            line += ",".join(map(str, map(int, bbox))) + ",1,-1,-1,-1\n"
            file.write(line)



data = read_file('data/tp2/det/det.txt')

matchthreshold = 0.5
tracks = []
track_id = 0
previous_time = 0

for frame_number in range(1, 525):
    current_time = time.time()
    fps = 1 / (current_time - previous_time) if previous_time else 0
    previous_time = current_time
    filename = "data/tp2/img1/{:06d}.jpg".format(frame_number)
    img = cv2.imread(filename)

    detections = data[data[:, 0] == frame_number]
    if frame_number == 1:
        for det in detections:
            tracks.append({'id': track_id, 'bbox': det})
            track_id += 1
    else:
        current_track = [track["bbox"] for track in tracks]
        iou_matrix = compute_jaccard(current_track, detections)
        track_indices, detection_indices = linear_sum_assignment(-iou_matrix)  # Negative for maximization

        used_detections = set(detection_indices)
        unmatched_tracks = set(range(len(tracks))) - set(track_indices)
        for track_idx, detection_idx in zip(track_indices, detection_indices):
            if iou_matrix[track_idx, detection_idx] >= matchthreshold:
                tracks[track_idx]['bbox'] = detections[detection_idx]
            else:
                unmatched_tracks.add(track_idx)

        # Delete unmatched tracks and create new tracks for unmatched detections
        tracks = [track for idx, track in enumerate(tracks) if idx not in unmatched_tracks]
        for d_idx, det in enumerate(detections):
            if d_idx not in used_detections:
                tracks.append({'id': track_id, 'bbox': det})
                track_id += 1

    # Visualization
    for track in tracks:
        bbox = track['bbox'][2:6]
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (0, 255, 0), 2)
        cv2.putText(img, str(track['id']), (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        # Update track history
        if 'history' not in track:
            track['history'] = []
        track['history'].append((int(bbox[0] + bbox[2]/2), int(bbox[1] + bbox[3]/2)))

        # Draw trajectory
        if len(track['history']) > 1:
            for i in range(1, len(track['history'])):
                cv2.line(img, track['history'][i - 1], track['history'][i], (0, 0, 255), 2)

    save_tracking_results('TP2_data/res.txt', tracks, frame_number)
    cv2.putText(img, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Tracking", img)
    cv2.waitKey(1)

cv2.destroyAllWindows()
    
