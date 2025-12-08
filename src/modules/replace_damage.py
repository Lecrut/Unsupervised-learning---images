#%% Imports 
import numpy as np
import matplotlib.pyplot as plt

#%% Replace damaged areas function
def replace_damage(origin_image_with_mask, repaired_image): 
    mask = origin_image_with_mask[:, :, 3]
    mask = 1 - mask

    concatenated_image = origin_image_with_mask[:, :, :3] * mask[:, :, np.newaxis] + repaired_image[:, :, :3] * (1 - mask[:, :, np.newaxis])

    return concatenated_image

#%% Replace damaged areas test
if __name__ == "__main__":
    origin_image = np.zeros((256, 256, 3))
    origin_image[:, :, 0] = 1
    damage_mask = np.zeros((256, 256, 1))
    damage_mask[50:100, 50:100, 0] = 1

    origin_image_with_mask = np.concatenate([origin_image, damage_mask], axis=2)

    repaired_image = np.copy(origin_image)
    repaired_image[:, :, 1] = 1
    repaired_image[:, :, 0] = 0

    new_img = replace_damage(origin_image_with_mask, repaired_image)

    plt.subplot(1, 3, 1)
    plt.title("Original Image with Damage Mask")
    plt.imshow(origin_image_with_mask)
    plt.subplot(1, 3, 2)
    plt.title("Repaired Image")
    plt.imshow(repaired_image)
    plt.subplot(1, 3, 3)
    plt.title("Final Image after Replacement")
    plt.imshow(new_img)
    plt.show()

