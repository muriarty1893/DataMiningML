#!/usr/bin/env python3
# Utility functions for the laptop price prediction app
import numpy as np
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('utils')

def fuzzy_match_gpu(value, encoder_classes, threshold=0.7):
    """
    Fuzzy match GPU models using pattern recognition
    
    Args:
        value: The input value to match
        encoder_classes: List of available encoder classes
        threshold: Similarity threshold (0-1)
        
    Returns:
        Matched value or None
    """
    # Normalize input value
    input_value = str(value).strip().lower()
    
    # If exact match exists, return it
    for ec in encoder_classes:
        if str(ec).strip().lower() == input_value:
            return ec
    
    # Extract GPU model patterns
    # Common patterns: [BRAND] [MODEL NAME] [MODEL NUMBER] [SUFFIX]
    # E.g., "NVIDIA GeForce RTX 3060", "AMD Radeon RX 6700M", "Intel Iris Xe Graphics"
    
    # Create a normalized version for each class
    normalized_classes = {str(ec).strip().lower(): ec for ec in encoder_classes}
    
    # Extract GPU brand and model number from input
    gpu_patterns = [
        # NVIDIA pattern
        r'(?:nvidia|geforce|rtx|gtx)\s*(?:rtx|gtx)?\s*(\d{4,})\s*(?:ti|super)?',
        # AMD pattern
        r'(?:amd|radeon)\s*(?:rx|vega)?\s*(\d{3,})\s*(?:m|xt)?',
        # Intel pattern
        r'(?:intel|iris|uhd)\s*(?:iris|uhd|xe)?\s*(?:graphics)?\s*(?:(\d{3,}))?'
    ]
    
    # Extract model information from input
    model_number = None
    brand = None
    
    # Check if it's NVIDIA
    if re.search(r'nvidia|geforce|rtx|gtx', input_value):
        brand = 'nvidia'
        match = re.search(gpu_patterns[0], input_value)
        if match and match.group(1):
            model_number = match.group(1)
    
    # Check if it's AMD
    elif re.search(r'amd|radeon', input_value):
        brand = 'amd'
        match = re.search(gpu_patterns[1], input_value)
        if match and match.group(1):
            model_number = match.group(1)
    
    # Check if it's Intel
    elif re.search(r'intel|iris|uhd', input_value):
        brand = 'intel'
        match = re.search(gpu_patterns[2], input_value)
        if match and match.group(1):
            model_number = match.group(1)
    
    logger.info(f"Extracted GPU info: brand={brand}, model_number={model_number}")
    
    # If we found brand and model, look for matches
    matches = []
    if brand and model_number:
        for normalized, original in normalized_classes.items():
            if brand in normalized and model_number in normalized:
                matches.append((original, 0.9))  # High confidence match
            elif brand in normalized and re.search(rf'\d{{{len(model_number)}}}', normalized):
                # Same brand, different model number but same length
                matches.append((original, 0.6))
    
    # If no brand/model match, try just matching the brand
    if not matches and brand:
        for normalized, original in normalized_classes.items():
            if brand in normalized:
                matches.append((original, 0.5))  # Lower confidence match
    
    # Sort by confidence
    matches.sort(key=lambda x: x[1], reverse=True)
    
    # Return the best match if it's above threshold
    if matches and matches[0][1] >= threshold:
        logger.info(f"Found match for '{value}': '{matches[0][0]}' with score {matches[0][1]}")
        return matches[0][0]
    
    # If all else fails, return the most common GPU in the encoder
    # Count frequency of each class
    if encoder_classes.size > 0:
        logger.warning(f"No good match found for '{value}'. Using most frequent GPU.")
        values, counts = np.unique(encoder_classes, return_counts=True)
        most_common = values[counts.argmax()]
        return most_common
    
    return None

def fuzzy_match_cpu(value, encoder_classes, threshold=0.7):
    """
    Fuzzy match CPU models using pattern recognition
    
    Args:
        value: The input value to match
        encoder_classes: List of available encoder classes
        threshold: Similarity threshold (0-1)
        
    Returns:
        Matched value or None
    """
    # Normalize input value
    input_value = str(value).strip().lower()
    
    # If exact match exists, return it
    for ec in encoder_classes:
        if str(ec).strip().lower() == input_value:
            return ec
    
    # Extract CPU model patterns
    # Common patterns: [BRAND] [SERIES] [MODEL NUMBER] [SUFFIX]
    # E.g., "Intel Core i7-12700H", "AMD Ryzen 7 5800H"
    
    # Create a normalized version for each class
    normalized_classes = {str(ec).strip().lower(): ec for ec in encoder_classes}
    
    # Extract CPU brand, series and model number from input
    cpu_patterns = [
        # Intel pattern
        r'(?:intel|core)?\s*(?:i\d+|celeron|pentium)?\-?(\d{4,})\w*',
        # AMD pattern
        r'(?:amd|ryzen)?\s*(?:ryzen|athlon)?\s*(\d+)?\s*(\d{4,})\w*',
        # Apple pattern
        r'(?:apple|m)\s*(\d+)?(?:\s+(?:pro|max|ultra))?'
    ]
    
    # Extract model information from input
    model_number = None
    brand = None
    
    # Check if it's Intel
    if re.search(r'intel|core|i\d+|celeron|pentium', input_value):
        brand = 'intel'
        match = re.search(cpu_patterns[0], input_value)
        if match and match.group(1):
            model_number = match.group(1)
    
    # Check if it's AMD
    elif re.search(r'amd|ryzen|athlon', input_value):
        brand = 'amd'
        match = re.search(cpu_patterns[1], input_value)
        if match and (match.group(1) or match.group(2)):
            model_number = match.group(2) if match.group(2) else match.group(1)
    
    # Check if it's Apple
    elif re.search(r'apple|m\d', input_value):
        brand = 'apple'
        match = re.search(cpu_patterns[2], input_value)
        if match and match.group(1):
            model_number = match.group(1)
    
    logger.info(f"Extracted CPU info: brand={brand}, model_number={model_number}")
    
    # If we identified the model number, look for matches
    matches = []
    if model_number:
        for normalized, original in normalized_classes.items():
            if model_number in normalized:
                matches.append((original, 0.8))  # High confidence match for model number
    
    # If no model number matches, try just matching the input directly
    # Look for partial matches (more relaxed)
    if not matches and len(input_value) >= 3:
        for normalized, original in normalized_classes.items():
            if input_value in normalized or normalized in input_value:
                # Calculate string similarity as confidence score
                longer = max(len(input_value), len(normalized))
                if longer == 0:
                    continue
                similarity = (longer - abs(len(input_value) - len(normalized))) / longer
                matches.append((original, similarity * 0.7))  # Scale down the confidence
    
    # Sort by confidence
    matches.sort(key=lambda x: x[1], reverse=True)
    
    # Return the best match if it's above threshold
    if matches and matches[0][1] >= threshold:
        logger.info(f"Found match for '{value}': '{matches[0][0]}' with score {matches[0][1]}")
        return matches[0][0]
    
    # If all else fails, return the most common CPU in the encoder
    if encoder_classes.size > 0:
        logger.warning(f"No good match found for '{value}'. Using most frequent CPU.")
        values, counts = np.unique(encoder_classes, return_counts=True)
        most_common = values[counts.argmax()]
        return most_common
    
    return None
