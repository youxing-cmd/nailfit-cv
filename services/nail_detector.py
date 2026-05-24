import cv2
import numpy as np
import mediapipe as mp
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)
mp_hands = mp.solutions.hands

# Per-finger anatomical width-to-nail-length ratios (tip-to-DIP)
# Based on average human hand proportions
_WIDTH_RATIOS = {
    "thumb": 0.90,
    "index": 0.85,
    "middle": 0.75,
    "ring": 0.70,
    "pinky": 0.60,
}

# MediaPipe hand landmark indices for MCP joints
_MCP_INDICES = {
    "thumb": 2,   # Thumb MCP is actually index 2 (CMC=1, MCP=2)
    "index": 5,
    "middle": 9,
    "ring": 13,
    "pinky": 17,
}


def _apply_clahe(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE contrast enhancement to improve detection in poor lighting."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def _estimate_width_from_landmarks(
    hand_landmarks,
    finger_name: str,
    tip_idx: int,
    image_width: int,
    image_height: int,
) -> float:
    """
    Estimate nail width in pixels using adjacent-finger MCP spacing as a scale proxy.
    Falls back to per-finger anatomical ratio if spacing cannot be computed.
    """
    # Try to compute width from adjacent MCP spacing
    mcp_idx = _MCP_INDICES[finger_name]
    mcp = hand_landmarks.landmark[mcp_idx]

    # Find neighbor MCP for spacing reference
    neighbor_names = list(_MCP_INDICES.keys())
    finger_idx = neighbor_names.index(finger_name)

    neighbor_mcp = None
    if finger_idx > 0:
        neighbor_idx = _MCP_INDICES[neighbor_names[finger_idx - 1]]
        neighbor_mcp = hand_landmarks.landmark[neighbor_idx]
    elif finger_idx < len(neighbor_names) - 1:
        neighbor_idx = _MCP_INDICES[neighbor_names[finger_idx + 1]]
        neighbor_mcp = hand_landmarks.landmark[neighbor_idx]

    if neighbor_mcp is not None:
        spacing_px = np.sqrt(
            ((mcp.x - neighbor_mcp.x) * image_width) ** 2
            + ((mcp.y - neighbor_mcp.y) * image_height) ** 2
        )
        # Nail width is roughly 70-80% of inter-finger spacing at MCP level
        width_px = spacing_px * 0.75
        return float(width_px)

    # Fallback: anatomical ratio from nail length
    tip = hand_landmarks.landmark[tip_idx]
    dip = hand_landmarks.landmark[tip_idx - 1]
    length_px = abs(tip.y - dip.y) * image_height
    ratio = _WIDTH_RATIOS.get(finger_name, 0.70)
    return float(length_px * ratio)


def _estimate_curve_type(
    hand_landmarks,
    tip_idx: int,
) -> str:
    """
    Estimate nail curve type from finger joint geometry.
    Uses the angle at the DIP joint (PIP-DIP-TIP).
    """
    pip = hand_landmarks.landmark[tip_idx - 2]
    dip = hand_landmarks.landmark[tip_idx - 1]
    tip = hand_landmarks.landmark[tip_idx]

    # Vectors: DIP->PIP and DIP->TIP
    v1 = np.array([pip.x - dip.x, pip.y - dip.y])
    v2 = np.array([tip.x - dip.x, tip.y - dip.y])

    # Compute angle
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return "medium"

    cos_angle = np.dot(v1, v2) / (norm1 * norm2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_deg = float(np.degrees(np.arccos(cos_angle)))

    # Heuristic: sharper joint angle -> more room for curved nail growth
    if angle_deg < 160:
        return "curved"
    if angle_deg > 175:
        return "flat"
    return "medium"


def _compute_finger_confidence(
    hand_landmarks,
    tip_idx: int,
    ref_detected: bool,
) -> float:
    """
    Compute per-finger confidence based on landmark visibility and detection quality.
    """
    # Landmarks involved: PIP, DIP, TIP (and optionally MCP for width estimation)
    indices = [tip_idx - 2, tip_idx - 1, tip_idx]
    visibilities = []
    for idx in indices:
        if 0 <= idx < len(hand_landmarks.landmark):
            visibilities.append(hand_landmarks.landmark[idx].visibility)

    avg_visibility = sum(visibilities) / len(visibilities) if visibilities else 0.5

    # Base confidence
    confidence = 0.5 + (avg_visibility * 0.3)
    if ref_detected:
        confidence += 0.1

    return round(min(confidence, 0.98), 2)


def estimate_nail_size(
    image: np.ndarray,
    reference_object: str = "eur_coin",
) -> Tuple[bool, float, float, Dict[str, Any]]:
    """
    Estimate nail width and length for each finger.

    Args:
        image: Input BGR image
        reference_object: "eur_coin" or "card"

    Returns:
        (reference_detected, overall_confidence, pixel_per_mm, fingers_data)
    """
    from .reference_detector import detect_reference

    # Apply contrast enhancement
    processed_image = _apply_clahe(image)

    detected, ref_size_px, ref_size_mm = detect_reference(processed_image, reference_object)
    if not detected or ref_size_px <= 0:
        logger.warning("Reference object not detected")
        return False, 0.0, 0.0, {}

    mm_per_pixel = ref_size_mm / ref_size_px
    pixel_per_mm = 1.0 / mm_per_pixel

    with mp_hands.Hands(
        static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3
    ) as hands:
        rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if not results.multi_hand_landmarks:
            logger.warning("No hands detected in image")
            return detected, 0.0, pixel_per_mm, {}

        fingers = {}
        per_finger_confidences = []

        finger_tips = {
            "thumb": 4,
            "index": 8,
            "middle": 12,
            "ring": 16,
            "pinky": 20,
        }

        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_prefix = "r" if idx == 0 else "l"

            for name, tip_idx in finger_tips.items():
                tip = hand_landmarks.landmark[tip_idx]
                dip = hand_landmarks.landmark[tip_idx - 1]
                length_px = abs(tip.y - dip.y) * processed_image.shape[0]

                # Improved width estimation
                width_px = _estimate_width_from_landmarks(
                    hand_landmarks, name, tip_idx,
                    processed_image.shape[1], processed_image.shape[0]
                )

                # Curve type estimation
                curve_type = _estimate_curve_type(hand_landmarks, tip_idx)

                # Dynamic confidence
                finger_conf = _compute_finger_confidence(
                    hand_landmarks, tip_idx, detected
                )
                per_finger_confidences.append(finger_conf)

                finger_code = f"{hand_prefix}{list(finger_tips.keys()).index(name) + 1}"
                fingers[finger_code] = {
                    "width_mm": round(width_px * mm_per_pixel, 1),
                    "length_mm": round(length_px * mm_per_pixel, 1),
                    "curve_type": curve_type,
                    "confidence": finger_conf,
                }

        # Overall confidence: average of per-finger + bonus for completeness
        if per_finger_confidences:
            overall = sum(per_finger_confidences) / len(per_finger_confidences)
        else:
            overall = 0.5

        if len(fingers) >= 10:
            overall = min(overall + 0.05, 0.98)

        logger.info(
            "Measured %d fingers, overall_confidence=%.2f", len(fingers), overall
        )
        return detected, round(overall, 2), pixel_per_mm, fingers
