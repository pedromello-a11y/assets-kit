import os
import io
import base64
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from PIL import Image
import numpy as np

app = FastAPI()

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.0-flash-preview-image-generation:generateContent"
)

BASE_PROMPT = (
    "cartoon illustration character, full body, front-facing, male, "
    "light skin tone, dark curly hair, brown eyes, mustache, neutral expression, "
    "MAGENTA background (#FF00FF), clean illustration style"
)

MAGENTA = (255, 0, 255)
TOLERANCE = 30


class AvatarRequest(BaseModel):
    prompt: str


def remove_magenta(image_bytes: bytes) -> bytes:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    data = np.array(img, dtype=np.int32)

    r, g, b, a = data[:, :, 0], data[:, :, 1], data[:, :, 2], data[:, :, 3]

    mask = (
        (np.abs(r - MAGENTA[0]) <= TOLERANCE)
        & (np.abs(g - MAGENTA[1]) <= TOLERANCE)
        & (np.abs(b - MAGENTA[2]) <= TOLERANCE)
    )

    data[:, :, 3] = np.where(mask, 0, a)

    result = Image.fromarray(data.astype(np.uint8), "RGBA")
    out = io.BytesIO()
    result.save(out, format="PNG")
    out.seek(0)
    return out.read()


@app.post("/generate-avatar")
async def generate_avatar(request: AvatarRequest):
    full_prompt = f"{BASE_PROMPT}\n\n{request.prompt}"

    payload = {
        "contents": [{"parts": [{"text": full_prompt}]}],
        "generationConfig": {"responseModalities": ["IMAGE", "TEXT"]},
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            GEMINI_API_URL,
            params={"key": GEMINI_API_KEY},
            json=payload,
        )

    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Gemini API error {response.status_code}: {response.text}",
        )

    body = response.json()

    image_b64 = None
    try:
        for part in body["candidates"][0]["content"]["parts"]:
            if part.get("inlineData", {}).get("mimeType", "").startswith("image/"):
                image_b64 = part["inlineData"]["data"]
                break
    except (KeyError, IndexError):
        pass

    if image_b64 is None:
        raise HTTPException(status_code=502, detail="No image returned by Gemini API")

    image_bytes = base64.b64decode(image_b64)
    transparent_png = remove_magenta(image_bytes)

    return StreamingResponse(
        io.BytesIO(transparent_png),
        media_type="image/png",
        headers={"Content-Disposition": "inline; filename=avatar.png"},
    )
