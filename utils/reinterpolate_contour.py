import numpy as np


def reinterpolate_contour(contour, point_distance=1, num_of_points=None):

    contour = np.concatenate((contour, contour[[0], :]), axis=0)

    distances = np.sqrt(np.sum(np.diff(contour, axis=0)**2, axis=1))
    cumulative_length = np.insert(np.cumsum(distances), 0, 0)  # Add 0 at the start

    total_length = cumulative_length[-1]

    if num_of_points is None:
      num_of_points = int(np.floor(total_length / point_distance)) + 1
    target_lengths = np.linspace(0, total_length, num_of_points)

    new_contour = np.empty((num_of_points, 2))
    for dim in range(2):  # Interpolate x and y independently
        new_contour[:, dim] = np.interp(target_lengths, cumulative_length, contour[:, dim])

    new_contour = new_contour[:-1,:]

    return new_contour