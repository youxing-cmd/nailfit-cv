from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from typing import Dict, Optional
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/measure", tags=["measure"])


class FingerMeasurement(BaseModel):
    width_mm: float
    length_mm: float
    curve_type: str
    confidence: float


class MeasureResponse(BaseModel):
    coin_detected: bool
    confidence: float
    pixel_per_mm: Optional[float]
    fingers: Dict[str, FingerMeasurement]


@router.post("", response_model=MeasureResponse)
async def measure(
    file: UploadFile = File(...),
    reference_object: str = Form("eur_coin"),
):
    logger.info("Measure request: file=%s reference=%s", file.filename, reference_object)

    try:
        from services.nail_detector import estimate_nail_size

        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            logger.error("Invalid image file: cannot decode")
            raise HTTPException(status_code=400, detail="Invalid image file")

        ref_detected, confidence, pixel_per_mm, fingers = estimate_nail_size(
            image, reference_object=reference_object
        )

        response_fingers = {}
        for code, data in fingers.items():
            response_fingers[code] = {
                "width_mm": data["width_mm"],
                "length_mm": data["length_mm"],
                "curve_type": data.get("curve_type", "medium"),
                "confidence": data.get("confidence", 0.82),
            }

        logger.info(
            "Measure success: ref_detected=%s fingers=%d confidence=%.2f",
            ref_detected,
            len(response_fingers),
            confidence,
        )

        return {
            "coin_detected": ref_detected,
            "confidence": confidence,
            "pixel_per_mm": pixel_per_mm if pixel_per_mm > 0 else None,
            "fingers": response_fingers,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Measurement failed")
        raise HTTPException(status_code=500, detail=f"Measurement failed: {str(e)}")
