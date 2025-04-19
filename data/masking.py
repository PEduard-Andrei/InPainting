import numpy as np
import cv2
import random

def get_mask_coverage_range(epoch, total_epochs):
    """
    Returns appropriate masking coverage range based on training progress.
    
    Args:
        epoch: Current training epoch
        total_epochs: Total number of training epochs
    
    Returns:
        min_coverage, max_coverage as floats
    """
    # Calculate progress percentage (0 to 1)
    progress = epoch / total_epochs
    
    if progress < 0.25:
        return 0.05, 0.10  # 5-10% for initial phase
    elif progress < 0.5:
        return 0.10, 0.20  # 10-20% for early intermediate phase
    elif progress < 0.75:
        return 0.15, 0.30  # 15-30% for late intermediate phase
    else:
        return 0.20, 0.40  # 20-40% for advanced phase

def apply_random_mask(image, min_coverage, max_coverage):
    """
    Applies a random mask to the image with holes, lines, dirt, scratches, and blur.
    Controls coverage between min_coverage and max_coverage.
    
    Args:
        image: Input image
        min_coverage: Minimum percentage of image to be masked
        max_coverage: Maximum percentage of image to be masked
    
    Returns: 
        masked_image, mask, actual_coverage
    """
    h, w = image.shape[:2]
    mask = np.ones((h, w), dtype=np.uint8) * 255  # 255 = keep, 0 = mask
    
    # Keep track of masked pixels
    total_pixels = h * w
    masked_pixels = 0
    target_masked_pixels = random.uniform(min_coverage, max_coverage) * total_pixels
    
    # Adjust mask type distribution based on coverage range
    if min_coverage < 0.15:
        # Early training: simpler masks
        mask_types = ['hole', 'hole', 'line', 'dirt', 'scratch', 'blur']
    else:
        # Later training: more complex and varied masks
        mask_types = ['hole', 'line', 'dirt', 'dirt', 'scratch', 'scratch', 'blur', 'blur']
    
    # Generate masks until we reach target coverage
    while masked_pixels < target_masked_pixels:
        mask_type = random.choice(mask_types)
        
        # Implement different mask types (holes, lines, dirt, scratches, blur)
        # Scale mask sizes based on coverage range
        scale_factor = 1.0 + (min_coverage / 0.05)  # Larger masks for later phases
        
        # Create temporary mask
        temp_mask = mask.copy()
        new_masked_pixels = 0
        
        # Apply appropriate mask type
        if mask_type == 'hole':
            # Implementation for holes (similar to previous code)
            # Adjust size based on scale_factor
            width = int(random.randint(5, w // 10) * scale_factor)
            height = int(random.randint(5, h // 10) * scale_factor)
            # Rest of hole implementation...
            
        elif mask_type == 'line':
            # Implementation for lines
            thickness = int(random.randint(1, 3) * scale_factor)
            # Rest of line implementation...
            
        # Implementation for other mask types...
        
        # Check if adding this mask doesn't exceed maximum coverage
        if masked_pixels + new_masked_pixels <= max_coverage * total_pixels:
            mask = temp_mask.copy()
            masked_pixels += new_masked_pixels
    
    # Apply mask and blur to image
    masked_image = image.copy()
    masked_image[mask == 0] = 0  # Black out masked regions
    
    # Handle blur regions
    blur_regions = (mask == 128)
    if np.any(blur_regions):
        blur_kernel = int(11 + 10 * (min_coverage / 0.4))  # Larger kernel for later phases
        blurred_image = cv2.GaussianBlur(image, (blur_kernel, blur_kernel), 0)
        masked_image[blur_regions] = blurred_image[blur_regions]
    
    actual_coverage = masked_pixels / total_pixels
    return masked_image, mask, actual_coverage