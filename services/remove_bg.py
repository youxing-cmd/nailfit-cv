import httpx


REMOVE_BG_API_URL = "https://api.remove.bg/v1.0/removebg"


async def remove_background(image_bytes: bytes, api_key: str) -> bytes:
    """
    Remove background from product image using remove.bg API.
    Returns PNG bytes with transparent background.
    """
    async with httpx.AsyncClient() as client:
        response = await client.post(
            REMOVE_BG_API_URL,
            headers={"X-Api-Key": api_key},
            files={"image_file": ("image.jpg", image_bytes, "image/jpeg")},
            data={"size": "auto"},
            timeout=30.0,
        )
        response.raise_for_status()
        return response.content
