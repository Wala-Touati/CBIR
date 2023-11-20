import cv2
import numpy as np

def harris_corners(image, gray, blockSize, ksize, k, noise=False):

    if noise:
        awgn = np.random.normal(0, 5, size=gray.shape).astype(np.uint8)
    else:
        awgn = 0

    # compute Harris Corner responses
    R = cv2.cornerHarris(gray + awgn, blockSize, ksize, k)

    # dilate
    R = cv2.dilate(R, None)

    # threshold image and convert to uint8
    ret, dst = cv2.threshold(R, 0.05*R.max(), 255, 0)
    dst = np.uint8(dst)

    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray, np.float32(centroids), (5,5), (-1,-1), criteria)

    return corners


def create_feature_vector(image, keypoint_method, blockSize=None, ksize=None, k=None, noise=None):
    """
    Create a feature vector for an image based on keypoints.

    Parameters:
    - image: Input image.
    - keypoint_method: String, either 'harris' or 'sift', indicating the method used to detect keypoints.
    - blockSize: Size of the neighborhood for Harris corner detection (only for 'harris' method).
    - ksize: Aperture parameter for the Sobel operator in Harris corner detection (only for 'harris' method).
    - k: Harris detector free parameter in the equation (only for 'harris' method).
    - noise: Flag indicating whether to add noise to the image (only for 'harris' method).

    Returns:
    - feature_vector: Concatenated feature vector based on descriptors.
    - image_out: Image with detected keypoints (for visualization).
    """

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initialize the SIFT detector
    sift = cv2.SIFT_create()

    if keypoint_method == 'harris':
        # For Harris keypoints, use the provided keypoints
        keypoints = harris_corners(image, gray, blockSize, ksize, k, noise)
    elif keypoint_method == 'sift':
        # For SIFT keypoints, detect keypoints using SIFT detector
        _, keypoints = sift.detectAndCompute(gray, None)
    else:
        raise ValueError("Invalid keypoint_method. Use 'harris' or 'sift'.")

    # Calculate SIFT descriptors for each keypoint
    sift_descriptors = []
    for keypoint in keypoints:
        x, y = np.round(keypoint.pt).astype(int)

        # Extract SIFT descriptor
        _, descriptor = sift.compute(gray, [keypoint])
        sift_descriptors.append(descriptor.flatten())

    # Concatenate the SIFT descriptors to create a feature vector for the image
    feature_vector = np.concatenate(sift_descriptors)

    # Draw detected keypoints on the image for visualization
    image_out = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return feature_vector, image_out

class ImageSimilarityCalculator:
    def __init__(self, num_images):
        self.num_images = num_images
        self.similarity_matrix = np.zeros((num_images, num_images))

    def calculate_similarity_matrix(self, images, keypoint_method, blockSize=None, ksize=None, k=None, noise=None):
        feature_vectors = []

        for i in range(self.num_images):
            feature_vector, _ = create_feature_vector(images[i], keypoint_method, blockSize, ksize, k, noise)
            feature_vectors.append(feature_vector)

        feature_vectors = np.array(feature_vectors)
        
        for i in range(self.num_images):
            for j in range(self.num_images):
                if i != j:
                    similarity_count = self.calculate_similarity_count(feature_vectors[i], feature_vectors[j])
                    self.similarity_matrix[i, j] = similarity_count
                    self.similarity_matrix[j, i] = similarity_count

        normalized_matrix = self.normalize_matrix(self.similarity_matrix)

        np.savetxt("similarity_matrix.txt", normalized_matrix, fmt="%.4f", delimiter="\t")

    def calculate_similarity_count(self, vector_a, vector_b):
        # Step (a): Calculate the set of L2 sift-to-sift distances in a matrix
        distances_matrix = np.linalg.norm(vector_a.reshape(-1, 128) - vector_b.reshape(-1, 128), axis=1)
        # Step (b): Find the most similar sift in vector_b for each sift in vector_a
        min_indices_a_to_b = np.argmin(distances_matrix)
        # Find the most similar sift in vector_a for each sift in vector_b
        min_indices_b_to_a = np.argmin(distances_matrix.reshape(-1, len(vector_a)), axis=1)
        # Step (c): Keep only reciprocal couples (if Ai -> Bj and Bj -> Ai)
        reciprocal_couples = sum(min_indices_a_to_b[min_indices_b_to_a] == np.arange(len(vector_a)))
        # The number of reciprocal couples is the similarity count
        similarity_count = reciprocal_couples

        return similarity_count

    def normalize_matrix(self, matrix):
        # Step (4): Normalize the matrix
        # For each row, divide each element by the minimum value in that row
        normalized_matrix = matrix / np.min(matrix, axis=1, keepdims=True)

        return normalized_matrix

    def get_most_similar_images(self, image_index):
        # Step (5): Get the most similar images for a given image
        # Retrieve the similarity scores for the specified image index
        similarity_scores = self.similarity_matrix[image_index]
        # Sort indices based on similarity scores in descending order
        sorted_indices = np.argsort(similarity_scores)[::-1]
        # Exclude the image itself from the list of most similar images
        most_similar_indices = sorted_indices[1:4]  # Replace 4 with the desired number of similar images

        return most_similar_indices
'''
# Example usage:
num_images = 5  # Replace with the actual number of images
images = [cv2.imread(f'image_{i}.jpg') for i in range(num_images)]  # Replace with your actual images

calculator = ImageSimilarityCalculator(num_images)
calculator.calculate_similarity_matrix(images, 'sift')
most_similar_images = calculator.get_most_similar_images(0)
print("Most similar images:", most_similar_images)
'''