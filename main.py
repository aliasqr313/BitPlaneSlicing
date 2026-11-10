
import cv2
import numpy as np
import matplotlib.pyplot as plt

# =====================================
# 1. Read Original Image
# =====================================

image = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: Image not found!")
    exit()

print("Image loaded successfully!")
print("Image dimensions:", image.shape)

# Display original image
plt.figure(figsize=(6, 6))
plt.imshow(image, cmap="gray", vmin=0, vmax=255)
plt.title("Original Image")
plt.axis("off")
plt.show()


# =====================================
# 2. Extract 8 Bit Planes
# =====================================

bit_planes = []

for i in range(8):

    bit_plane = (image >> i) & 1

    bit_planes.append(bit_plane)

# Display all 8 bit planes
plt.figure(figsize=(12, 8))

for i in range(8):

    plt.subplot(2, 4, i + 1)

    plt.imshow(bit_planes[i] * 255, cmap="gray")

    plt.title(f"Bit Plane {i}")

    plt.axis("off")

plt.suptitle("Eight Bit Planes")
plt.tight_layout()
plt.show()


# =====================================
# 3. Reconstruct Images Without One Bit
# =====================================

plt.figure(figsize=(12, 8))

for i in range(8):

    reconstructed = np.zeros_like(image)

    for j in range(8):

        if i != j:

            reconstructed += bit_planes[j] * (2 ** j)

    plt.subplot(2, 4, i + 1)

    plt.imshow(reconstructed, cmap="gray", vmin=0, vmax=255)

    plt.title(f"Without Bit Plane {i}")

    plt.axis("off")

plt.suptitle("Reconstructed Images")
plt.tight_layout()
plt.show()

print("All operations completed successfully!")
