import numpy as np
import scipy.linalg


__all__ = [ "KalmanFilter3D" ]

"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class KalmanFilter3D:
    """Kalman filter for modeling spatial state in image space

    Here are the states kalman filter trying to maintain

        np.ndarray([ x, y, z, a, h, vx, vy, vz, va, vh ])

    Target explanation:
        x: bounding box center postion along x axis in image space
        y: bounding box center postion along y axis in image space
        z: bounding box estimated depth value
        a: aspect ratio of bounding box width over bounding box height
        h: bounding box height
        v*: respective velocities of (x, y, z, a, h)

    The motion model is a constant velocity model. The bounding box location
    (x, y, z, a, h) is taken as direct observation.

    Attributes:
        _pred_mat (ndarray): a (10, 10) matrix for predicting the next state
        _project_mat (ndarray): a (5, 10) matrix for projecting state vector from
            state space to observation space.
        _std_position (float): uncertainty of the (x, y, z, a, h)
        _std_velocity (float): uncertainty of the (vx, vy, vz, va, vh)
    """
    def __init__(self):
        n_dim = 5
        delta_time = 1.

        # Dimension of prediction matrix: (10, 10)
        self._pred_mat = np.eye(2*n_dim, 2*n_dim)
        for i in range(n_dim):
            self._pred_mat[i, n_dim+i] = delta_time

        # Dimension of projection matrix: (5, 10)
        self._project_mat = np.eye(n_dim, 2*n_dim)

        # Uncertainty of the state (x, y, z, a, h) => (pixel, pixel, meter, fraction, pixel)
        self._std_pixel = 0.05          # For (x, y, _, _, _) (dynamic)
        self._std_depth = 0.30          # For (_, _, z, _, _) (constant)
        self._std_ratio = 0.01          # For (_, _, _, a, _) (constant)
        self._std_height = 0.01         # For (_, _, _, _, h) (dynamic)
        self._std_pixel_vel = 0.05      # For changing speed  (dynamic)
        self._std_depth_vel = 0.0001     # For changing speed  (constant)
        self._std_ratio_vel = 0.0001    # For changing speed  (constant)
        self._std_height_vel = 0.05     # For changing speed  (dynamic)

    def initiate(self, observation):
        """Initialize state of kalman filter

        Args:
            observation (ndarray): a (5,) vector representing (x, y, z, a, h)

        Returns:
            mean (ndarray): intialized state vector of shape (10,)
            covariance (ndarray): initialized uncertainty matrix of shape (10x10)
        """
        # mean vector (state vector)
        mean_pos = observation
        mean_vel = np.zeros_like(mean_pos)
        mean = np.concatenate([mean_pos, mean_vel])

        # covariance matrix (uncertainty matrix)
        std = [
            # State
            self._std_pixel * observation[4],
            self._std_pixel * observation[4],
            self._std_depth,
            self._std_ratio,
            self._std_height * observation[4],
            # Velocity
            self._std_pixel_vel * observation[4],
            self._std_pixel_vel * observation[4],
            self._std_depth_vel,
            self._std_ratio_vel,
            self._std_height_vel * observation[4]]
        covariance = np.diag(np.square(std))

        return mean, covariance

    def predict(self, mean, covariance):
        """Predict the next state given the previous state (mean & covariance)

        Args:
            mean (ndarray): previous state vector of shape (10,)
            covariance (ndarray): previous uncertainty matrix of shape (10, 10)

        Returns:
            mean (ndarray): predicted state vector of shape (10,)
            covariance (ndarray): predicted uncertainty matrix of shape (10, 10)
        """
        # Noise for covariance matrix (noise from the "world", unknown factors)
        std_state = [
            0.3 * self._std_pixel * mean[4],
            0.3 * self._std_pixel * mean[4],
            0.3 * self._std_depth,
            self._std_ratio,
            0.3 * self._std_height * mean[4]]
        std_vel = [
            0.1 * self._std_pixel_vel * mean[4],
            0.1 * self._std_pixel_vel * mean[4],
            0.1 * self._std_depth_vel,
            0.1 * self._std_ratio_vel,
            0.1 * self._std_height_vel * mean[4]]

        world_noise = np.diag(np.square(np.concatenate([std_state, std_vel])))

        # Update mean vector and covariance matrix
        mean = np.dot(self._pred_mat, mean)
        covariance = np.linalg.multi_dot((
            self._pred_mat, covariance, self._pred_mat.T)) + world_noise

        return mean, covariance

    def _project(self, mean, covariance):
        """Project state vector and uncertainty matrix to the observation space

        Args:
            mean (ndarray): predicted state vector of shape (10,)
            covariance (ndarray): predicted uncertainty matrix of shape (10, 10)

        Returns:
            mean (ndarray): projected state vector of shape (10,)
            covariance (ndarray): projected uncertainty matrix of shape (10, 10)
        """
        # Noise for projected covariance matrix
        std_state = [
            self._std_pixel * mean[4],
            self._std_pixel * mean[4],
            self._std_depth,
            self._std_ratio,
            self._std_height * mean[4]]
        project_noise = np.diag(np.square(std_state))

        # Projected mean and covariance matrix
        mean = np.dot(self._project_mat, mean)
        covariance = np.linalg.multi_dot((
            self._project_mat, covariance, self._project_mat.T)) + project_noise

        return mean, covariance

    def update(self, mean, covariance, observation):
        """Refine the predicted state with the observed data

        Args:
            mean (ndarray): predicted state vector of shape (8,)
            covariacne (ndarray): predicted uncertainty matrix of shape (8, 8)
            observation (ndarray): observed data of shape (4,)

        Returns:
            mean (ndarray): refined state vector of shape (8,)
            covariacne (ndarray): refined uncertainty matrix of shape (8, 8)
        """
        # Project mean & covariance so that they are in the same space as observation
        project_mean, project_covariance = self._project(mean, covariance)

        # Calculate kalman gain
        chol_factor, lower = scipy.linalg.cho_factor(
                                    project_covariance,
                                    lower=True,
                                    check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
                                    (chol_factor, lower),
                                    np.dot(covariance, self._project_mat.T).T,
                                    check_finite=False).T

        # Update mean and covariance with observation
        mean = mean + np.dot(observation-project_mean, kalman_gain.T)
        covariance = covariance - np.linalg.multi_dot((
            kalman_gain, project_covariance, kalman_gain.T))

        return mean, covariance
