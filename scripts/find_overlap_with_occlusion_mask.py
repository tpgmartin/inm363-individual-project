import numpy as np
# TODO: load images

# Load image segments
# segments = ... 

# Load occlusion results
# occlusion_img = ...

# Get occlusion mask
# occlusion_mask = (occlusion_img).astype(float)

segment_in_occlusion_mask = []
for s in range(len(segments)):
    # need to normalise to take into account uniform grey background
    # mask = (...)
    if np.mean(mask) > 0.001:

        jaccard = np.sum(occlusion_mask * mask) / np.sum((occlusion_mask + mask) > 0)

    if jaccard > 0.5:
        # append image segment name
        # segment_in_occlusion_mask.append(mask)