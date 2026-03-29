import os
import io
import base64
from pathlib import Path
from collections import deque

import httpx
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-image-preview")

GEMINI_API_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)

BASE_DIR = Path(__file__).resolve().parent
REFERENCE_IMAGE_PATH = BASE_DIR / "MODELO_AVATAR.png"

BASE_PROMPT_TEMPLATE = (
    Create only a front-view {item_description} as a modular PNG asset using the uploaded avatar as an exact fit template.
The {item_description} must fit the avatar perfectly at 100% scale with no resizing, no repositioning, and no manual adjustment when layered on top of the base avatar. 
Use the exact same canvas size, framing, centering, proportions, and alignment as the reference.
Show only the front of the {item_description}. 
Output only the {item_description}, isolated, with ackground must be a SINGLE FLAT SOLID MAGENTA (#FF00FF) only OUTSIDE the garment silhouette. 
No checkerboard, no fake transparency, no body, no mannequin, no full character. 
Match the exact style and outline thickness of the reference avatar.
The output must be 1000x1000 just has de reference image."
)


class AvatarRequest(BaseModel):
    item_description: str


@app.get("/")
async def root():
    return {"ok": True, "service": "assets-kit"}


@app.get("/health")
async def health():
    return {"ok": True}


def load_reference_image_base64() -> str:
    if not REFERENCE_IMAGE_PATH.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {REFERENCE_IMAGE_PATH}")

    with open(REFERENCE_IMAGE_PATH, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def remove_magenta(image_bytes: bytes) -> bytes:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")

    rgba = np.array(img, dtype=np.uint8)
    hsv = np.array(img.convert("HSV"), dtype=np.uint8)

    h = hsv[:, :, 0].astype(np.int16)
    s = hsv[:, :, 1].astype(np.int16)
    v = hsv[:, :, 2].astype(np.int16)

    r = rgba[:, :, 0].astype(np.int16)
    g = rgba[:, :, 1].astype(np.int16)
    b = rgba[:, :, 2].astype(np.int16)

    # Fundo externo roxo/magenta mais amplo
    broad_magenta_mask = (
        (h >= 185) & (h <= 235) &
        (s >= 35) &
        (v >= 35)
    )

    # Magenta quase puro, inclusive em ilhas internas
    pure_magenta_mask = (
        (np.abs(r - 255) <= 45) &
        (np.abs(g - 0) <= 55) &
        (np.abs(b - 255) <= 45)
    )

    height, width = broad_magenta_mask.shape
    visited = np.zeros((height, width), dtype=bool)
    q = deque()

    def try_add(y, x):
        if 0 <= y < height and 0 <= x < width:
            if broad_magenta_mask[y, x] and not visited[y, x]:
                visited[y, x] = True
                q.append((y, x))

    # Remove apenas o fundo externo conectado às bordas
    for x in range(width):
        try_add(0, x)
        try_add(height - 1, x)

    for y in range(height):
        try_add(y, 0)
        try_add(y, width - 1)

    directions = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1)
    ]

    while q:
        y, x = q.popleft()
        for dy, dx in directions:
            try_add(y + dy, x + dx)

    # 1) fundo externo
    rgba[visited, 3] = 0

    # 2) magenta puro restante dentro de aberturas internas
    rgba[pure_magenta_mask, 3] = 0

    out = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(out, format="PNG")
    out.seek(0)
    return out.read()


def extract_image_bytes(body: dict) -> bytes | None:
    candidates = body.get("candidates", [])
    for candidate in candidates:
        content = candidate.get("content", {})
        parts = content.get("parts", [])
        for part in parts:
            inline_data = part.get("inlineData") or part.get("inline_data")
            if not inline_data:
                continue

            mime_type = inline_data.get("mimeType") or inline_data.get("mime_type", "")
            data_b64 = inline_data.get("data")
            if mime_type.startswith("image/") and data_b64:
                return base64.b64decode(data_b64)

    return None


@app.post("/generate-avatar")
async def generate_avatar(request: AvatarRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY não configurada nas variáveis de ambiente."
        )

    try:
        reference_image_b64 = load_reference_image_base64()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    full_prompt = BASE_PROMPT_TEMPLATE.format(
        item_description=request.item_description
    )

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": reference_image_b64
                        }
                    },
                    {
                        "text": full_prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "responseModalities": ["IMAGE"],
            "imageConfig": {
                "aspectRatio": "1:1",
                "imageSize": "1K"
            }
        }
    }

    headers = {
        "x-goog-api-key": GEMINI_API_KEY,
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                GEMINI_API_URL,
                headers=headers,
                json=payload,
            )
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Timeout ao chamar a Gemini API.")
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Erro de rede na Gemini API: {str(e)}")

    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Gemini API error {response.status_code}: {response.text}",
        )

    try:
        body = response.json()
    except Exception:
        raise HTTPException(status_code=502, detail="Resposta inválida da Gemini API.")

    image_bytes = extract_image_bytes(body)
    if not image_bytes:
        raise HTTPException(
            status_code=502,
            detail=f"A Gemini respondeu, mas não retornou imagem. Resposta: {body}"
        )

    transparent_png = remove_magenta(image_bytes)

    return StreamingResponse(
        io.BytesIO(transparent_png),
        media_type="image/png",
        headers={"Content-Disposition": 'inline; filename="avatar.png"'},
    )