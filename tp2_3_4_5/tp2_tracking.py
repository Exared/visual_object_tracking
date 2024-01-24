import numpy as np
import os
import cv2
import time
from utils import read_file, compute_jaccard

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

        used_detections = set()
        tracks_to_delete = []

        # Update existing tracks
        for t_idx, track in enumerate(tracks):
            # Set IoU of used detections to -1 (or 0) for current track
            current_iou_values = iou_matrix[t_idx].copy()
            for used_idx in used_detections:
                current_iou_values[used_idx] = -1

            # Now find the best match excluding already used detections
            best_idx = np.argmax(current_iou_values)

            if iou_matrix[t_idx][best_idx] >= matchthreshold and best_idx not in used_detections:
                track['bbox'] = detections[best_idx]
                used_detections.add(best_idx)
            else:
                # Mark track for deletion if no matching detection
                tracks_to_delete.append(t_idx)

        # Delete tracks without matching detections
        for t_idx in sorted(tracks_to_delete, reverse=True):
            del tracks[t_idx]

        # Create new tracks for unmatched detections
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

    cv2.putText(img, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Tracking", img)
    cv2.waitKey(1)

cv2.destroyAllWindows()
    
