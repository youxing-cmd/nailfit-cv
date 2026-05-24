import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

EURO_COIN_DIAMETER_MM = 23.25
BANK_CARD_WIDTH_MM = 85.60
BANK_CARD_HEIGHT_MM = 53.98


def _detect_coin_by_contours(image: np.ndarray) -> tuple[bool, float]:
    """
    Fallback coin detection using contour circularity.
    Returns (detected, diameter_in_pixels).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)

    # Adaptive threshold to handle varying lighting
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Morphological closing to fill gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_circularity = 0.0
    best_diameter = 0.0
    min_area = 500   # ~25px radius
    max_area = 31416  # ~100px radius

    h, w = image.shape[:2]
    img_area = h * w

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        # Reject if too small relative to image (likely noise) or too large
        if area < img_area * 0.001 or area > img_area * 0.15:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter ** 2)

        if circularity > best_circularity and circularity > 0.75:
            best_circularity = circularity
            best_diameter = 2 * np.sqrt(area / np.pi)

    if best_diameter > 0:
        logger.info(
            "Coin detected by contour fallback: diameter=%.1fpx, circularity=%.3f",
            best_diameter,
            best_circularity,
        )
        return True, float(best_diameter)

    return False, 0.0


def detect_coin(image: np.ndarray) -> tuple[bool, float]:
    """
    Detect a 1 euro coin in the image and return its diameter in pixels.
    First tries HoughCircles, then falls back to contour-based detection.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100,
        param1=100,
        param2=30,
        minRadius=20,
        maxRadius=100,
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        largest = max(circles, key=lambda c: c[2])
        diameter = largest[2] * 2
        logger.info("Coin detected by HoughCircles: diameter=%.1fpx", diameter)
        return True, float(diameter)

    # Fallback to contour-based detection
    logger.warning("HoughCircles failed, trying contour fallback")
    return _detect_coin_by_contours(image)


def detect_card(image: np.ndarray) -> tuple[bool, float]:
    """
    Detect a bank card in the image and return its width (long edge) in pixels.
    Returns (detected, width_in_pixels).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_rect = None
    best_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 2000:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)
        if len(approx) == 4:
            if area > best_area:
                best_area = area
                best_rect = approx

    if best_rect is None:
        return False, 0.0

    pts = best_rect.reshape(4, 2)
    sides = []
    for i in range(4):
        p1 = pts[i]
        p2 = pts[(i + 1) % 4]
        side_len = float(np.linalg.norm(p2 - p1))
        sides.append(side_len)

    width_px = max(sides)
    aspect_ratio = max(sides) / min(sides) if min(sides) > 0 else 0

    # Validate aspect ratio is close to bank card (85.6 / 53.98 ≈ 1.586)
    if 1.3 < aspect_ratio < 1.9:
        logger.info("Card detected: width=%.1fpx, aspect_ratio=%.2f", width_px, aspect_ratio)
        return True, width_px

    logger.warning("Detected quadrilateral rejected: aspect_ratio=%.2f", aspect_ratio)
    return False, 0.0


def detect_reference(image: np.ndarray, reference_type: str = "eur_coin") -> tuple[bool, float, float]:
    """
    Detect reference object and return scale information.

    Args:
        image: Input image
        reference_type: "eur_coin" or "card"

    Returns:
        (detected, size_in_pixels, real_size_mm)
    """
    if reference_type == "card":
        detected, size_px = detect_card(image)
        return detected, size_px, BANK_CARD_WIDTH_MM

    # Default: euro coin
    detected, size_px = detect_coin(image)
    return detected, size_px, EURO_COIN_DIAMETER_MM
