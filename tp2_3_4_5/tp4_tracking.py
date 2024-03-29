import cv2
import numpy as np
import time
from scipy.optimize import linear_sum_assignment
from utils import KalmanFilter, read_file, compute_jaccard, save_tracking_results, get_center

data = read_file('data/tp2/det/det.txt')

matchthreshold = 0.5
tracks = []
track_id = 0
previous_time = 0
Karman = KalmanFilter(dt=0.1, u_x=1, u_y=1, std_acc=1, x_std_meas=0.1, y_std_meas=0.1)

lower_fps = np.inf
higher_fps = 0
average_fps = 0

for frame_number in range(1, 525):
    current_time = time.time()
    fps = 1 / (current_time - previous_time) if previous_time else 0
    if fps < lower_fps and frame_number > 2: # Ignore first frames
        lower_fps = fps
    if fps > higher_fps:
        higher_fps = fps
    average_fps += fps
    previous_time = current_time
    filename = "data/tp2/img1/{:06d}.jpg".format(frame_number)
    img = cv2.imread(filename)

    detections = data[data[:, 0] == frame_number]
    if frame_number == 1:
        for det in detections:
            kf = KalmanFilter(dt=0.1, u_x=0, u_y=0, std_acc=1, x_std_meas=0.1, y_std_meas=0.1)
            kf.state[:2] = get_center(det[2:6])  # Initialize state with the center of bbox
            tracks.append({'id': track_id, 'bbox': det, 'kf': kf})
            track_id += 1
    else:
        for track in tracks:
            track['kf'].predict()
        
        current_track = [track["bbox"] for track in tracks]
        iou_matrix = compute_jaccard(current_track, detections)
        track_indices, detection_indices = linear_sum_assignment(-iou_matrix)  # Negative for maximization

        used_detections = set(detection_indices)
        unmatched_tracks = set(range(len(tracks))) - set(track_indices)
        for track_idx, detection_idx in zip(track_indices, detection_indices):
            if iou_matrix[track_idx, detection_idx] >= matchthreshold:
                track = tracks[track_idx]
                det = detections[detection_idx]
                track['bbox'] = det
                # Update step for Kalman filter
                z = get_center(det[2:6])
                track['kf'].update(z)
            else:
                unmatched_tracks.add(track_idx)

        # Delete unmatched tracks and create new tracks for unmatched detections
        tracks = [track for idx, track in enumerate(tracks) if idx not in unmatched_tracks]
        for d_idx, det in enumerate(detections):
            if d_idx not in used_detections:
                kf = KalmanFilter(dt=0.1, u_x=0, u_y=0, std_acc=1, x_std_meas=0.1, y_std_meas=0.1)
                kf.state[:2] = get_center(det[2:6])  # Initialize state with the center of bbox
                tracks.append({'id': track_id, 'bbox': det, 'kf': kf})
                track_id += 1

    # Visualization
    for track in tracks:
        # Get estimated state from Kalman filter
        estimated_state = track['kf'].state
        x_est, y_est = estimated_state[0, 0], estimated_state[1, 0]
        w_est, h_est = track['bbox'][4], track['bbox'][5]

        # Calculate the top left corner of the estimated bbox
        top_left_x_est = int(x_est - w_est / 2)
        top_left_y_est = int(y_est - h_est / 2)

        # Draw estimated bounding box
        cv2.rectangle(img, (top_left_x_est, top_left_y_est), (top_left_x_est + w_est, top_left_y_est + h_est), (255, 0, 0), 2)

        # Draw actual detection bounding box
        bbox = track['bbox'][2:6]
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (0, 0, 255), 2)
        cv2.putText(img, str(track['id']), (top_left_x_est, top_left_y_est - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)


    save_tracking_results('tracking_result_output/ADL-Rundle-6.txt', tracks, frame_number)
    cv2.putText(img, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Tracking", img)
    cv2.waitKey(1)

cv2.destroyAllWindows()
    
print(f"Lower FPS: {lower_fps}")
print(f"Higher FPS: {higher_fps}")
print(f"Average FPS: {average_fps/524}")