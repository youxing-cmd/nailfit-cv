from fastapi import APIRouter, UploadFile, File, Form
from pydantic import BaseModel
from typing import List
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/measure/quality", tags=["measure"])


class QualityResponse(BaseModel):
    ok: bool
    qualityScore: float
    warnings: List[str]
    canContinue: bool


@router.post("", response_model=QualityResponse)
async def check_quality(
    file: UploadFile = File(...),
    reference_object: str = Form("eur_coin"),
):
    """
    Check image quality for nail measurement.
    Returns quality score (0-1) and list of warnings.
    """
    logger.info("Quality check request: file=%s reference=%s", file.filename, reference_object)

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            logger.warning("Invalid image file: cannot decode")
            return QualityResponse(
                ok=False,
                qualityScore=0.0,
                warnings=["Invalid image file"],
                canContinue=False,
            )

        warnings = []
        h, w = image.shape[:2]

        # 1. Resolution check
        min_pixels = 1280 * 720
        if w * h < min_pixels:
            warnings.append("Image resolution too low")

        # 2. Brightness check
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        if mean_brightness < 40:
            warnings.append("Image too dark")
        elif mean_brightness > 230:
            warnings.append("Image too bright / overexposed")

        # 3. Blur check (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 100:
            warnings.append("Image is blurry")

        # 4. Hand detection (unified threshold with nail_detector: 0.3)
        try:
            import mediapipe as mp
            mp_hands = mp.solutions.hands
            with mp_hands.Hands(
                static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3
            ) as hands:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
                if not results.multi_hand_landmarks:
                    warnings.append("No hand detected")
                else:
                    hand_count = len(results.multi_hand_landmarks)
                    if hand_count < 1:
                        warnings.append("No hand detected")
        except Exception as e:
            logger.warning("Hand detection error in quality check: %s", e)
            warnings.append("Hand detection unavailable")

        # 5. Reference object detection
        from services.reference_detector import detect_reference

        ref_detected, _, _ = detect_reference(image, reference_object)
        if not ref_detected:
            label = "bank card" if reference_object == "card" else "€1 coin"
            warnings.append(f"Reference object ({label}) not detected")

        # Calculate quality score
        # Start at 1.0, deduct for each warning
        score = 1.0
        for w in warnings:
            if "too dark" in w or "too bright" in w:
                score -= 0.15
            elif "blurry" in w:
                score -= 0.2
            elif "resolution" in w:
                score -= 0.1
            elif "No hand" in w:
                score -= 0.3
            elif "Reference object" in w:
                score -= 0.15
            else:
                score -= 0.1

        score = max(0.0, min(1.0, score))

        logger.info(
            "Quality check result: score=%.2f warnings=%d canContinue=%s",
            score,
            len(warnings),
            score >= 0.5,
        )

        return QualityResponse(
            ok=len(warnings) == 0,
            qualityScore=round(score, 2),
            warnings=warnings,
            canContinue=score >= 0.5,
        )

    except Exception as e:
        logger.exception("Quality check failed")
        return QualityResponse(
            ok=False,
            qualityScore=0.0,
            warnings=["Quality check failed"],
            canContinue=False,
        )
