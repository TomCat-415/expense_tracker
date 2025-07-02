"""
Enhanced OCR utilities for receipt processing.
"""
import re
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import cv2
import numpy as np
from PIL import Image
import pytesseract
from dataclasses import dataclass
import logging
from config.settings import MAX_IMAGE_SIZE_MB, ALLOWED_IMAGE_EXTENSIONS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExtractedData:
    """Container for extracted receipt data with confidence scores."""
    amount: Optional[float] = None
    amount_confidence: float = 0.0
    date: Optional[datetime] = None
    date_confidence: float = 0.0
    merchant: Optional[str] = None
    merchant_confidence: float = 0.0
    raw_text: str = ""

def is_valid_image(file_path: Path) -> bool:
    """Check if file is a supported image and not too large."""
    ext = file_path.suffix.lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        return False
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    return size_mb <= MAX_IMAGE_SIZE_MB

def detect_and_correct_orientation(image: np.ndarray) -> np.ndarray:
    """Detect and correct image orientation using Tesseract OSD."""
    try:
        # Get orientation info
        osd = pytesseract.image_to_osd(image)
        angle = int([line for line in osd.split('\n') if 'Rotate:' in line][0].split(':')[1].strip())
        logger.info(f"Detected orientation angle: {angle}")
        
        # Rotate if needed
        if angle != 0:
            logger.info(f"Rotating image by {angle} degrees")
            if angle == 90:
                return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif angle == 180:
                return cv2.rotate(image, cv2.ROTATE_180)
            elif angle == 270:
                return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        return image
    except Exception as e:
        logger.warning(f"Could not detect orientation: {e}")
        return image

def preprocess_image_adaptive(image: np.ndarray, upscale: bool = True) -> np.ndarray:
    """Advanced preprocessing with adaptive thresholding."""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Increase contrast
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=10)
    
    # Apply bilateral filter to reduce noise while preserving edges
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Apply adaptive thresholding with larger block size for better text separation
    adaptive_thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )
    
    # Optional upscale (use cubic interpolation for better quality)
    if upscale:
        scale_factor = 2.0
        height, width = adaptive_thresh.shape
        new_height, new_width = int(height * scale_factor), int(width * scale_factor)
        # Ensure even dimensions
        new_height = new_height + (new_height % 2)
        new_width = new_width + (new_width % 2)
        processed = cv2.resize(adaptive_thresh, (new_width, new_height), 
                             interpolation=cv2.INTER_CUBIC)
    else:
        processed = adaptive_thresh
    
    return processed

def preprocess_image_standard(image: np.ndarray) -> np.ndarray:
    """Standard preprocessing with Otsu thresholding."""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply Otsu thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Denoise
    denoised = cv2.medianBlur(thresh, 3)
    
    return denoised

def preprocess_image(image_path: Path, method: str = "adaptive") -> Optional[np.ndarray]:
    """Enhance image quality for better OCR results."""
    try:
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Could not load image: {image_path}")
            return None
        
        # Correct orientation
        image = detect_and_correct_orientation(image)
        
        # Apply preprocessing based on method
        if method == "adaptive":
            processed = preprocess_image_adaptive(image, upscale=True)
        else:
            processed = preprocess_image_standard(image)
        
        return processed
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return None

def extract_amount(text: str) -> Tuple[Optional[float], float]:
    """Extract amount from text with confidence score."""
    logger.info("Raw text for amount extraction:")
    logger.info(text)
    
    def parse_amount(amount_str: str) -> Optional[float]:
        """Helper function to parse amount strings with proper validation."""
        # Remove currency symbols and normalize spaces
        clean_str = amount_str.replace('¥', '').replace('\\', '').strip()
        
        # Handle OCR artifacts that split numbers
        clean_str = re.sub(r'\s+', '', clean_str)  # Remove all spaces
        clean_str = re.sub(r'[.。、]{2,}', '', clean_str)  # Remove repeated dots
        clean_str = re.sub(r'[.。、](?!\d)', '', clean_str)  # Remove dots not followed by digits
        
        # Remove any remaining special characters
        clean_str = re.sub(r'[^\d,.]', '', clean_str)
        
        try:
            # Handle both comma and dot as thousand separators
            parts = re.split(r'[,.]', clean_str)
            if len(parts) > 1:
                # If we have separators, validate the parts
                if not all(len(p) <= 3 for p in parts):
                    logger.info(f"Invalid number format in '{amount_str}'")
                    return None
            
            # Join all parts and convert to float
            amount = float(''.join(parts))
            return amount
            
        except ValueError:
            logger.info(f"Failed to parse amount '{amount_str}'")
            return None
    
    # Common patterns for Japanese and international amounts, ordered by priority
    amount_patterns = [
        # Exact Maruetsu credit card total patterns
        (r'クレ.*?金額\s*[¥\\]?\s*(\d[\d,.\s]*\d)', 0.98),  # Credit amount
        (r'(?:合計|お会計|total)\s*[¥\\]?\s*(\d[\d,.\s]*\d)', 0.95),  # Final total
        (r'(?:税込(?:合計)?|消費税等?込?(?:合計)?)\s*[¥\\]?\s*(\d[\d,.\s]*\d)', 0.9),  # Tax-included total
        (r'(?:小計|お買上げ金額)\s*[¥\\]?\s*(\d[\d,.\s]*\d)', 0.8),  # Subtotal
    ]
    
    candidates = []
    lines = text.split('\n')
    
    logger.info("Processing lines for amount extraction:")
    # First pass: collect all amount candidates with their line numbers
    for i, line in enumerate(lines):
        logger.info(f"Line {i}: {line}")
        for pattern, base_confidence in amount_patterns:
            matches = list(re.finditer(pattern, line, re.IGNORECASE))
            for match in matches:
                # Get the full match for logging
                full_match = match.group(0)
                # Get just the number part
                amount_str = match.group(1).strip()
                logger.info(f"Found match: '{full_match}' -> amount_str: '{amount_str}'")
                
                # Parse and validate the amount
                amount = parse_amount(amount_str)
                if amount is None:
                    continue
                
                # Basic validation
                if amount <= 0 or amount > 1000000:  # Ignore invalid amounts
                    logger.info(f"Ignoring amount {amount} - out of valid range")
                    continue
                
                curr_confidence = base_confidence
                
                # Increase confidence for amounts near the end of the receipt
                position_boost = min((i / len(lines)) * 0.2, 0.2)
                curr_confidence += position_boost
                
                # Boost confidence for amounts in typical receipt range
                if 1000 <= amount <= 100000:
                    curr_confidence += 0.1
                
                # Additional context-based confidence adjustments
                if re.search(r'クレジット|カード|お客様控え', line, re.IGNORECASE):
                    curr_confidence += 0.1
                
                # Penalize amounts that look like item quantities or unit prices
                if re.search(r'(?:単|個|点|円/個|\\/?個|x\s*\d+)', line, re.IGNORECASE):
                    curr_confidence -= 0.3
                    logger.info(f"Penalizing amount {amount} - looks like item price/quantity")
                
                # Penalize amounts that look like phone numbers, card numbers, etc.
                if re.search(r'(?:電話|TEL|FAX|カード.*番号|ID|No\.?)', line, re.IGNORECASE):
                    curr_confidence -= 0.5
                    logger.info(f"Penalizing amount {amount} - looks like phone/card number")
                
                # Only accept amounts that appear in the expected format
                if re.sub(r'\s+', '', amount_str) in re.sub(r'\s+', '', line):
                    candidates.append((amount, min(curr_confidence, 1.0)))
                    logger.info(f"Added candidate: amount={amount}, confidence={curr_confidence}")
                else:
                    logger.info(f"Skipping amount {amount} - doesn't match expected format in line")
    
    if not candidates:
        logger.info("No valid candidates found")
        return None, 0.0
    
    # Sort candidates by confidence
    candidates.sort(key=lambda x: x[1], reverse=True)
    logger.info("All candidates (sorted by confidence):")
    for amount, conf in candidates:
        logger.info(f"Amount: {amount}, Confidence: {conf}")
    
    # Among candidates with similar confidence (within 0.1), prefer larger amounts
    best_confidence = candidates[0][1]
    best_candidates = [c for c in candidates if abs(c[1] - best_confidence) <= 0.1]
    
    if best_candidates:
        # Return the largest amount among the high-confidence candidates
        result = max(best_candidates, key=lambda x: x[0])
        logger.info(f"Selected final amount: {result[0]} with confidence {result[1]}")
        return result
    
    logger.info(f"Falling back to highest confidence candidate: {candidates[0]}")
    return candidates[0]

def extract_date(text: str) -> Tuple[Optional[datetime], float]:
    """Extract date from text with confidence score."""
    # Common date patterns
    date_patterns = [
        (r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})', 0.9),  # YYYY-MM-DD
        (r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})', 0.8),  # DD-MM-YYYY
        (r'(\d{2})年(\d{1,2})月(\d{1,2})日', 0.9),    # Japanese format
    ]
    
    for pattern, base_confidence in date_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            try:
                if '年' in pattern:  # Japanese format
                    year = int('20' + match.group(1)) if len(match.group(1)) == 2 else int(match.group(1))
                    month = int(match.group(2))
                    day = int(match.group(3))
                else:
                    if len(match.group(1)) == 4:  # YYYY-MM-DD
                        year = int(match.group(1))
                        month = int(match.group(2))
                        day = int(match.group(3))
                    else:  # DD-MM-YYYY
                        year = int(match.group(3))
                        month = int(match.group(2))
                        day = int(match.group(1))
                
                date = datetime(year, month, day)
                
                # Adjust confidence based on date reasonableness
                if date > datetime.now():
                    base_confidence -= 0.3
                elif (datetime.now() - date).days > 90:
                    base_confidence -= 0.2
                
                return date, base_confidence
            except ValueError:
                continue
    
    return None, 0.0

def extract_merchant(text: str) -> Tuple[Optional[str], float]:
    """Extract merchant name from text with confidence score."""
    lines = text.split('\n')
    if not lines:
        return None, 0.0
    
    # Specific patterns for Maruetsu
    maruetsu_patterns = [
        r'マルエツ',
        r'まるえつ',
        r'MARUETSU',
        r'maruetsu',
        r'マー',  # Common OCR fragment for マルエツ
        r'のツ'   # Common OCR fragment for マルエツ
    ]
    
    # Check all lines for Maruetsu patterns
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check for exact Maruetsu matches
        for pattern in maruetsu_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                return 'maruetsu', 0.95
        
        # Look for partial matches with higher tolerance
        if any(p.lower() in line.lower() for p in ['マル', 'まる', 'maru']):
            return 'maruetsu', 0.8
    
    # Fallback: return first non-empty line if it's short enough to be a store name
    for line in lines[:2]:
        line = line.strip()
        if line and len(line) <= 20 and not re.search(r'^\d', line):
            return line, 0.6
    
    return None, 0.0

def extract_text_from_image(image_path: Path, lang: str = 'eng+jpn') -> ExtractedData:
    """Enhanced OCR processing with data extraction and confidence scoring."""
    try:
        # Preprocess image
        processed_image = preprocess_image(image_path, method="adaptive")
        if processed_image is None:
            return ExtractedData(raw_text="Error processing image")
        
        # Configure Tesseract parameters for better accuracy with Japanese text
        custom_config = r'--oem 1 --psm 4 -c preserve_interword_spaces=1 -c tessedit_char_blacklist=><|' 
        
        # Perform OCR with both preprocessed and original images
        processed_text = pytesseract.image_to_string(
            Image.fromarray(processed_image), 
            lang=lang,
            config=custom_config
        )
        
        logger.info("Processed image OCR text:")
        logger.info(processed_text)
        
        # Also try original image as backup
        original_image = cv2.imread(str(image_path))
        original_text = pytesseract.image_to_string(
            Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)), 
            lang=lang,
            config=custom_config
        )
        
        logger.info("Original image OCR text:")
        logger.info(original_text)
        
        # Use the text with more content
        raw_text = processed_text if len(processed_text.strip()) > len(original_text.strip()) else original_text
        
        # Extract data with confidence scores
        amount, amount_conf = extract_amount(raw_text)
        logger.info(f"Extracted amount: {amount} (confidence: {amount_conf})")
        
        date, date_conf = extract_date(raw_text)
        logger.info(f"Extracted date: {date} (confidence: {date_conf})")
        
        merchant, merchant_conf = extract_merchant(raw_text)
        logger.info(f"Extracted merchant: {merchant} (confidence: {merchant_conf})")
        
        return ExtractedData(
            amount=amount,
            amount_confidence=min(amount_conf, 1.0),  # Ensure confidence is capped at 1.0
            date=date,
            date_confidence=date_conf,
            merchant=merchant,
            merchant_confidence=merchant_conf,
            raw_text=raw_text
        )
        
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {str(e)}")
        return ExtractedData(raw_text=f"Error processing image: {str(e)}")

def batch_extract_text(image_paths: List[Path], **kwargs) -> dict:
    """Extract text from multiple images."""
    results = {}
    
    for image_path in image_paths:
        logger.info(f"Processing: {image_path}")
        text = extract_text_from_image(image_path, **kwargs)
        results[str(image_path)] = text
    
    return results

def process_receipt(image_path: str, debug: bool = False) -> ExtractedData:
    """
    Process a receipt image and extract relevant information.
    
    Args:
        image_path: Path to the receipt image
        debug: If True, print debug information
    
    Returns:
        ExtractedData object containing the extracted information
    """
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    
    # Convert string path to Path object
    path = Path(image_path)
    
    # Validate image
    if not is_valid_image(path):
        logger.error(f"Invalid image file: {path}")
        return ExtractedData()
    
    # Extract text and data
    return extract_text_from_image(path)

# Usage examples:
if __name__ == "__main__":
    # Single image
    image_path = Path("sample_image.jpg")
    text = extract_text_from_image(image_path, preprocessing_method="adaptive", multi_config=True)
    print(text)
    
    # Batch processing
    image_paths = [Path("img1.jpg"), Path("img2.png")]
    results = batch_extract_text(image_paths)
    for path, text in results.items():
        print(f"{path}: {text[:100]}...")  # First 100 chars