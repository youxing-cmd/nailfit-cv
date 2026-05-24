from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional
import numpy as np
import cv2
import base64
import io
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/preview", tags=["preview"])


class PreviewResponse(BaseModel):
    preview_url: Optional[str]
    render_method: str
    message: str


@router.post("", response_model=PreviewResponse)
async def generate_preview(
    hand_image: UploadFile = File(...),
    nail_cutout: UploadFile = File(...),
):
    """
    Generate a perspective-transform preview by overlaying nail cutout
    onto the user's hand photo at detected finger tip positions.
    """
    logger.info(
        "Preview request: hand=%s nail=%s",
        hand_image.filename,
        nail_cutout.filename,
    )

    try:
        import mediapipe as mp

        # Load images
        hand_bytes = await hand_image.read()
        nail_bytes = await nail_cutout.read()

        hand_np = np.frombuffer(hand_bytes, np.uint8)
        nail_np = np.frombuffer(nail_bytes, np.uint8)

        hand_img = cv2.imdecode(hand_np, cv2.IMREAD_COLOR)
        nail_img = cv2.imdecode(nail_np, cv2.IMREAD_UNCHANGED)

        if hand_img is None or nail_img is None:
            logger.error("Invalid image file: hand=%s nail=%s", hand_img is not None, nail_img is not None)
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Detect hands
        mp_hands = mp.solutions.hands
        with mp_hands.Hands(
            static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3
        ) as hands:
            rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if not results.multi_hand_landmarks:
                logger.warning("No hand detected in preview image")
                return {
                    "preview_url": None,
                    "render_method": "perspective_overlay",
                    "message": "No hand detected in image",
                }

            # Prepare output image
            output = hand_img.copy()
            h, w = hand_img.shape[:2]

            # Finger tip landmarks: thumb=4, index=8, middle=12, ring=16, pinky=20
            tip_indices = [4, 8, 12, 16, 20]

            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                for tip_idx in tip_indices:
                    tip = hand_landmarks.landmark[tip_idx]
                    px = int(tip.x * w)
                    py = int(tip.y * h)

                    # Scale nail cutout to ~12% of image width
                    scale = int(w * 0.12)
                    resized_nail = cv2.resize(nail_img, (scale, scale))

                    # Calculate overlay bounds
                    half = scale // 2
                    y1 = max(0, py - half)
                    y2 = min(h, py + half)
                    x1 = max(0, px - half)
                    x2 = min(w, px + half)

                    if y2 <= y1 or x2 <= x1:
                        continue

                    # Adjust nail crop to fit bounds
                    nail_crop = resized_nail[: y2 - y1, : x2 - x1]

                    if nail_crop.shape[2] == 4:
                        # Alpha blending
                        alpha = nail_crop[:, :, 3:] / 255.0
                        output[y1:y2, x1:x2] = (
                            nail_crop[:, :, :3] * alpha
                            + output[y1:y2, x1:x2] * (1 - alpha)
                        ).astype(np.uint8)
                    else:
                        output[y1:y2, x1:x2] = nail_crop

            # Encode result as base64 data URL
            _, buffer = cv2.imencode(".png", output)
            b64 = base64.b64encode(buffer).decode("utf-8")
            data_url = f"data:image/png;base64,{b64}"

            logger.info("Preview generated successfully")
            return {
                "preview_url": data_url,
                "render_method": "perspective_overlay",
                "message": "Preview generated successfully",
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Preview generation failed")
        raise HTTPException(status_code=500, detail=f"Preview generation failed: {str(e)}")
