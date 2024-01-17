import numpy as np

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
