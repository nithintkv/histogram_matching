import cv2
import numpy as np
import matplotlib.pyplot as plt


def main():
    def get_histogram(image):
        histogram = np.zeros(256)
        for height in range(image.shape[0]):
            for width in range(image.shape[1]):
                histogram[image[height][width]] += 1
        return histogram

    def total_pixels(hist):
        pixels_count = 0
        for pixels in hist:
            pixels_count += pixels
        return pixels_count

    def cumulative_probability(cumulative_hist):
        cum_probability = cumulative_hist.copy()
        for i in np.arange(1, 256):
            cum_probability[i] = cum_probability[i - 1] + cum_probability[i]
        return cum_probability

    original_image = cv2.imread('image1.jpg', 0)
    target_image = cv2.imread('image2.jpg', 0)

    hist_original = get_histogram(original_image)
    hist_target = get_histogram(target_image)

    total_pixels_original = total_pixels(hist_original)
    total_pixels_target = total_pixels(hist_target)

    original_cumulative_hist = np.true_divide(hist_original, total_pixels_original)
    target_cumulative_hist = np.true_divide(hist_target, total_pixels_target)

    original_cumulative_prob = cumulative_probability(original_cumulative_hist)
    target_cumulative_prob = cumulative_probability(target_cumulative_hist)

    k = 256
    j_array = np.zeros(k)
    for x in np.arange(k):
        j = 0
        while True:
            j_array[x] = j
            j += 1
            if j > k - 1 or original_cumulative_prob[x] < target_cumulative_prob[j]:
                break

    for i in range(original_image.shape[0]):
        for j in range(original_image.shape[1]):
            a = original_image[i][j]
            b = j_array[a]
            original_image[i][j] = b

    plt.figure(0)
    plt.plot(hist_original)
    plt.savefig('output/original_hist.png')
    plt.figure(1)
    plt.plot(hist_target)
    plt.savefig('output/target_hist.png')

    matched_hist = get_histogram(original_image)
    plt.figure(2)
    plt.plot(matched_hist)
    plt.savefig('output/matched_hist.png')
    cv2.imwrite('output/matched_image.jpg', original_image)


if __name__ == "__main__":
    main()
