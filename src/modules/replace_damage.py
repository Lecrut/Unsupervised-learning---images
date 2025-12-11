#%% Imports 
import numpy as np
import matplotlib.pyplot as plt

#%% Replace damaged areas function
def replace_damage(origin_image_with_mask, repaired_image): 
    # Check if input is (C, H, W) - typically C is 3 or 4
    if origin_image_with_mask.shape[0] == 4:
        # (C, H, W) format
        mask = origin_image_with_mask[3, :, :]
        # Mask is 1 for damage, 0 for valid.
        # We want to keep valid parts of origin.
        # So we need a mask that is 1 for valid, 0 for damage.
        valid_mask = 1 - mask
        
        # Expand mask to (3, H, W)
        valid_mask_expanded = np.repeat(valid_mask[np.newaxis, :, :], 3, axis=0)
        damage_mask_expanded = 1 - valid_mask_expanded
        
        # Ensure repaired_image is (3, H, W)
        if repaired_image.shape[0] != 3 and repaired_image.shape[2] == 3:
             repaired_image = np.transpose(repaired_image, (2, 0, 1))
        
        concatenated_image = origin_image_with_mask[:3, :, :] * valid_mask_expanded + repaired_image[:3, :, :] * damage_mask_expanded
        
        return concatenated_image
    elif len(origin_image_with_mask.shape) == 3 and origin_image_with_mask.shape[2] == 4:
        # (H, W, C) format
        mask = origin_image_with_mask[:, :, 3]
        valid_mask = 1 - mask

        # Ensure repaired_image is (H, W, 3)
        if repaired_image.shape[2] != 3 and repaired_image.shape[0] == 3:
             repaired_image = np.transpose(repaired_image, (1, 2, 0))

        concatenated_image = origin_image_with_mask[:, :, :3] * valid_mask[:, :, np.newaxis] + repaired_image[:, :, :3] * (1 - valid_mask[:, :, np.newaxis])

        return concatenated_image
    else:
        raise ValueError(f"Input image must have 4 channels (RGBA) to extract damage mask. Got shape {origin_image_with_mask.shape}")

#%% Replace damaged areas test
if __name__ == "__main__":
    origin_image = np.zeros((256, 256, 3))
    origin_image[:, :, 0] = 1
    damage_mask = np.zeros((256, 256, 1))
    damage_mask[50:100, 50:100, 0] = 1

    origin_image_with_mask = np.concatenate([origin_image, damage_mask], axis=2)

    repaired_image = np.copy(origin_image_with_mask)
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
