import numpy as np
import os
import cv2
import time
from scipy.optimize import linear_sum_assignment
from utils import read_file, compute_jaccard, save_tracking_results

data = read_file('data/tp2/det/det.txt')

matchthreshold = 0.5
tracks = []
track_id = 1
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

    save_tracking_results('tracking_result_output/tp3_track_output.txt', tracks, frame_number)
    cv2.putText(img, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Tracking", img)
    cv2.waitKey(1)

cv2.destroyAllWindows()
    
