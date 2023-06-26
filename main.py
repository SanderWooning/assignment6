import numpy as np
import matplotlib.pyplot as plt
import diplib as dip
import cv2



def segment_image(gray_image):

    # Smooth image
    smoothed_image = dip.MedianFilter(gray_image)

    # Threshold image
    thresholded_image = dip.OtsuThreshold(smoothed_image)

    # Calculate the threshold value
    threshold_value = np.max(smoothed_image[thresholded_image])

    # Small component removal
    removed_small = dip.SmallObjectsRemove(thresholded_image, 2)

    print(threshold_value)

    return removed_small


def depth_cueing(image_stack):
    num_planes = len(image_stack)
    max_intensity = num_planes - 1  # Maximum intensity value for the planes

    # Create an empty image with the same shape as the input stack
    output_image = np.zeros_like(image_stack[0])

    # Iterate over each pixel in the image stack
    for i in range(image_stack[0].shape[0]):
        for j in range(image_stack[0].shape[1]):
            intensity_sum = 0

            # Calculate the sum of intensities for each pixel across all planes
            for plane, image in enumerate(image_stack):
                intensity_sum += image[i, j] * plane

            # Calculate the final intensity value using depth cueing formula
            final_intensity = intensity_sum / max_intensity

            # Assign the intensity value to the output image
            output_image[i, j] = final_intensity

    # Normalize the output image to the range [0, 255]
    output_image_normalized = (output_image / max_intensity) * 255
    output_image_normalized = output_image_normalized.astype(np.uint8)

    # Display the output image using OpenCV
    cv2.imshow('Depth Cueing', output_image_normalized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



image = dip.ImageReadICS("images/CHROMO3D.ICS")

# A 16 stack image
image_array = np.asarray(image)

segmented_stack = segment_image(image_array)

# The 16 stack image binary masks
stack = np.asarray(segmented_stack)


depth_cueing(stack)

print(np.unique(stack))

fig, axes = plt.subplots(4, 4, figsize=(10, 10))


for i, ax in enumerate(axes.flat):
    ax.imshow(stack[i], cmap='gray')
    ax.axis('off')

plt.tight_layout()
plt.show()
