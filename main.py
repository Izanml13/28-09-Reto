"""
Reto Día 1 — Gemini API con Python (google-genai)
=================================================
1) Prueba de texto simple con parámetros (temperatura, top-p, max tokens)
2) Análisis de sentimiento ES (5) + EN (5) con extracción de entidades
3) Multimodal: descripción detallada de una imagen

Uso:
    pip install -r requirements.txt
    copy .env.example .env   # y pega tu GEMINI_API_KEY
    python main.py --todo
    python main.py --texto
    python main.py --sentimiento
    python main.py --imagen ./ejemplo.jpg
    python main.py --imagen-url https://ejemplo.com/foto.jpg
"""

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image, ImageDraw, ImageFont

load_dotenv()

MODELO = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

# ---------- 5 textos ES + 5 EN ----------
TEXTOS_ES = [
    "Me encanta el nuevo disco de Rosalía, es una obra maestra llena de energía.",
    "Odio el servicio de esta compañía, tardaron tres semanas y nadie responde.",
    "El restaurante está en la calle Mayor 12, la comida es correcta, sin más.",
    "Qué alegría ver a mi familia en Madrid después de dos años, fue emocionante.",
    "El vuelo de Iberia se retrasó 5 horas y perdimos la conexión, fatal.",
]
TEXTOS_EN = [
    "I absolutely love the new iPhone camera, the photos in Paris look stunning.",
    "This movie was terrible, boring plot and awful acting, waste of time.",
    "The meeting is scheduled for Monday at 10am in London, please confirm.",
    "I'm so happy with my new job at Google, the team is amazing.",
    "The delivery from Amazon arrived broken and customer support was unhelpful.",
]

PROMPT_SENTIMIENTO = """Analiza el sentimiento de cada texto.
Devuelve SOLO un JSON válido (array) con objetos:
{{"id": int, "idioma": "es"|"en", "texto": "...", "sentimiento": "positivo"|"negativo"|"neutro", "confianza": 0.0-1.0, "entidades": ["..."], "justificacion": "...corta..."}}
Textos:
"""


def get_client() -> genai.Client:
    if not API_KEY:
        print("ERROR: no hay GEMINI_API_KEY.", file=sys.stderr)
        print("  1) Ve a https://aistudio.google.com", file=sys.stderr)
        print("  2) Botón 'Get API key' > Create API key", file=sys.stderr)
        print("  3) copy .env.example .env y pega la clave", file=sys.stderr)
        print("     o: $env:GEMINI_API_KEY='tu-clave'  (PowerShell)", file=sys.stderr)
        sys.exit(1)
    return genai.Client(api_key=API_KEY)


def prueba_texto(client: genai.Client):
    print(f"\n=== 1) PRUEBA DE TEXTO (modelo={MODELO}) ===")
    print("Parámetros: temperature=0.7, top_p=0.9, max_output_tokens=300\n")
    config = types.GenerateContentConfig(
        temperature=0.7,
        top_p=0.9,
        max_output_tokens=300,
    )
    resp = client.models.generate_content(
        model=MODELO,
        contents="Explícame en 3 frases qué es la API de Gemini y para qué sirve.",
        config=config,
    )
    print(resp.text)
    print(f"\n[modelo usado: {MODELO} | tokens aprox: entrada+salida visibles en AI Studio > Usage]")


def analisis_sentimiento(client: genai.Client):
    print("\n=== 2) ANÁLISIS DE SENTIMIENTO (5 ES + 5 EN) + ENTIDADES ===")
    todos = [(i + 1, "es", t) for i, t in enumerate(TEXTOS_ES)]
    todos += [(len(TEXTOS_ES) + i + 1, "en", t) for i, t in enumerate(TEXTOS_EN)]
    bloque = "\n".join(f'{i} [{lang}] {t}' for i, lang, t in todos)

    config = types.GenerateContentConfig(
        temperature=0.2,  # bajo = más determinista, ideal para clasificar
        top_p=0.9,
        max_output_tokens=2048,
        response_mime_type="application/json",
    )
    resp = client.models.generate_content(
        model=MODELO,
        contents=PROMPT_SENTIMIENTO + bloque,
        config=config,
    )
    try:
        datos = json.loads(resp.text)
    except json.JSONDecodeError:
        print("El modelo no devolvió JSON puro. Respuesta cruda:")
        print(resp.text)
        return

    for d in datos:
        print(f"\n#{d.get('id')} [{d.get('idioma')}] {d.get('sentimiento')} "
              f"(conf: {d.get('confianza')})")
        print(f"  Texto: {d.get('texto')}")
        print(f"  Entidades: {', '.join(d.get('entidades', [])) or '—'}")
        print(f"  Por qué: {d.get('justificacion')}")

    with open("resultado_sentimiento.json", "w", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=False, indent=2)
    print("\nGuardado en resultado_sentimiento.json")


def crear_imagen_ejemplo(ruta: Path) -> Path:
    """Crea una imagen local si el usuario no tiene una (evita depender de internet)."""
    img = Image.new("RGB", (800, 500), color=(20, 30, 60))
    d = ImageDraw.Draw(img)
    d.rectangle([50, 60, 750, 440], outline=(255, 200, 60), width=4)
    d.ellipse([330, 150, 470, 290], fill=(255, 200, 60))
    d.text((90, 100), "Playa al atardecer - imagen de prueba", fill=(255, 255, 255))
    d.text((90, 350), "Barco, sol, mar y gaviotas (dibujo simple)", fill=(200, 200, 200))
    img.save(ruta)
    return ruta


def cargar_imagen(path_or_url: str | None) -> Image.Image:
    if path_or_url and path_or_url.startswith("http"):
        print(f"Descargando imagen: {path_or_url}")
        with urllib.request.urlopen(path_or_url) as r:
            tmp = Path("imagen_url.jpg")
            tmp.write_bytes(r.read())
        return Image.open(tmp)
    if path_or_url:
        p = Path(path_or_url)
        if not p.exists():
            sys.exit(f"No existe la imagen: {p}")
        return Image.open(p)
    # por defecto: ejemplo.jpg o generarla
    ejemplo = Path(__file__).parent / "ejemplo.jpg"
    if not ejemplo.exists():
        print("No se encontró ejemplo.jpg, generando una de prueba...")
        crear_imagen_ejemplo(ejemplo)
    return Image.open(ejemplo)


def descripcion_imagen(client: genai.Client, path_or_url: str | None):
    print("\n=== 3) MULTIMODAL: DESCRIPCIÓN DE IMAGEN ===")
    imagen = cargar_imagen(path_or_url)
    prompt = ("Describe esta imagen en detalle en español: "
              "objetos principales, colores, ambiente, texto visible, "
              "y sugiere 3 posibles usos (ej. marketing, accesibilidad).")
    config = types.GenerateContentConfig(
        temperature=0.6,
        top_p=0.9,
        max_output_tokens=800,
    )
    resp = client.models.generate_content(
        model=MODELO,
        contents=[prompt, imagen],
        config=config,
    )
    print(resp.text)
    with open("resultado_imagen.txt", "w", encoding="utf-8") as f:
        f.write(resp.text)
    print("\nGuardado en resultado_imagen.txt")


def main():
    ap = argparse.ArgumentParser(description="Reto Día 1 — Gemini API")
    ap.add_argument("--todo", action="store_true", help="Ejecuta texto + sentimiento + imagen")
    ap.add_argument("--texto", action="store_true", help="Solo prueba de texto")
    ap.add_argument("--sentimiento", action="store_true", help="Solo sentimiento")
    ap.add_argument("--imagen", metavar="RUTA_O_URL", nargs="?", const="",
                    help="Describe imagen (ruta local o URL). Sin valor usa ejemplo.jpg")
    args = ap.parse_args()

    if not any([args.todo, args.texto, args.sentimiento, args.imagen is not None]):
        args.todo = True  # por defecto todo

    client = get_client()
    try:
        if args.todo or args.texto:
            prueba_texto(client)
        if args.todo or args.sentimiento:
            analisis_sentimiento(client)
        if args.todo or args.imagen is not None:
            descripcion_imagen(client, args.imagen or None)
    except Exception as e:
        print(f"\nERROR de la API: {e}", file=sys.stderr)
        msg = str(e)
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
            print("Has llegado al límite gratuito (15 RPM). Espera 60 s y reintenta.", file=sys.stderr)
        if "API_KEY_INVALID" in msg or "API key not valid" in msg:
            print("La API Key no es válida. Crea una nueva en https://aistudio.google.com", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
