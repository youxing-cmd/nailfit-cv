from typing import List, Dict, Any, Optional


def match_finger(
    user_width: float,
    user_length: float,
    user_curve: str,
    tip_sizes: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Find the best nail tip size for a single finger.
    Returns recommendation with fit grade, delta, and risk reasons.
    """
    best = None
    best_delta = float("inf")

    for tip in tip_sizes:
        if tip.get("stock", 0) <= 0:
            continue

        delta = abs(user_width - tip["width_mm"])
        if delta < best_delta:
            best_delta = delta
            best = tip

    if best is None:
        return None

    grade = "not_recommended"
    risk_reasons = []

    if best_delta <= 0.5:
        grade = "best"
    elif best_delta <= 1.0:
        grade = "good"
    else:
        grade = "risky"
        risk_reasons.append(f"width_delta_{best_delta:.1f}mm")

    if user_curve == "curved" and best.get("curve_type") == "flat":
        if grade == "best":
            grade = "good"
        elif grade == "good":
            grade = "risky"
        risk_reasons.append("curve_mismatch")

    length_delta = (user_length - best.get("length_mm", 999)) if best.get("length_mm") else 0
    if length_delta > 0:
        risk_reasons.append("tip_shorter_than_nail")

    return {
        "tip_id": best.get("id"),
        "size_code": best["size_code"],
        "width_delta_mm": round(best_delta, 2),
        "length_delta_mm": round(length_delta, 2) if best.get("length_mm") else None,
        "fit_grade": grade,
        "risk_reasons": risk_reasons,
    }


def match_profile_to_product(
    user_fingers: Dict[str, Dict[str, Any]],
    tip_sizes: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Match a full nail profile against a product's size library.
    """
    overall_grades = []
    finger_results = {}

    for finger_code, data in user_fingers.items():
        result = match_finger(
            data.get("width_mm", 0),
            data.get("length_mm", 0),
            data.get("curve_type", "medium"),
            tip_sizes,
        )
        if result:
            finger_results[finger_code] = result
            overall_grades.append(result["fit_grade"])

    if not overall_grades:
        return {"fit_grade": "not_recommended", "confidence": 0, "fingers": {}}

    # Overall grade is the worst individual grade
    grade_priority = {"not_recommended": 0, "risky": 1, "good": 2, "best": 3}
    worst = min(overall_grades, key=lambda g: grade_priority.get(g, 0))

    return {
        "fit_grade": worst,
        "confidence": sum(d.get("confidence", 0) for d in user_fingers.values()) / len(user_fingers),
        "fingers": finger_results,
    }
