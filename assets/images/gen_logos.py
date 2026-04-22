#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║  ExploreSV — Logo Background Remover & Multi-Format Generator  ║
║  Elimina fondos (cuadrícula, blanco, gris) y genera formatos   ║
║  optimizados para navbar, hero, footer y favicon.              ║
╚══════════════════════════════════════════════════════════════════╝

Dependencias:
    pip install Pillow numpy rembg onnxruntime

Uso:
    python limpiar_logo_pro.py logo.jpeg
    python limpiar_logo_pro.py logo.jpeg --solo-limpiar
    python limpiar_logo_pro.py logo.jpeg --salida ./assets/images/
    python limpiar_logo_pro.py logo.jpeg --sin-ia
"""

import sys
import os
import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image


# ─────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────

FORMATOS_SALIDA = {
    "logo_navbar.png":  {"ancho": 200, "descripcion": "Barra de navegación"},
    "logo_hero.png":    {"ancho": 350, "descripcion": "Sección principal"},
    "logo_footer.png":  {"ancho": 150, "descripcion": "Pie de página"},
}

FAVICON_TAMANO = 32
FAVICON_CROP_SUPERIOR = 0.60  # % superior de la imagen para extraer solo el icono

# Umbrales para detección de cuadrícula/fondo
UMBRAL_GRIS = 15       # Tolerancia entre canales R, G, B
BRILLO_MINIMO = 150     # Intensidad mínima para considerar "fondo claro"
UMBRAL_BORDE = 5        # Píxeles de borde a limpiar adicionalmente


# ─────────────────────────────────────────────────────────────────
# FUNCIONES PRINCIPALES
# ─────────────────────────────────────────────────────────────────

def cargar_imagen(ruta: str) -> Image.Image:
    """Carga y valida la imagen de entrada."""
    ruta_path = Path(ruta)

    if not ruta_path.exists():
        print(f"❌ No se encontró el archivo: {ruta}")
        print(f"   Directorio actual: {os.getcwd()}")
        print(f"   Archivos disponibles:")
        for f in Path(".").glob("*.*"):
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
                print(f"     → {f.name}")
        sys.exit(1)

    if ruta_path.stat().st_size == 0:
        print(f"❌ El archivo está vacío: {ruta}")
        sys.exit(1)

    try:
        img = Image.open(ruta)
        img.load()  # Forzar carga completa para detectar corrupción
        print(f"📂 Imagen cargada: {ruta}")
        print(f"   Formato: {img.format} | Modo: {img.mode} | Tamaño: {img.size[0]}×{img.size[1]}px")
        return img
    except Exception as e:
        print(f"❌ Error al abrir la imagen: {e}")
        sys.exit(1)


def eliminar_fondo_por_color(img: Image.Image) -> Image.Image:
    """
    Elimina fondos blancos, grises y cuadrículas de transparencia falsa
    usando análisis de color por píxel con NumPy.
    """
    print("\n🎨 PASO 1: Eliminando fondo por análisis de color...")

    rgba = img.convert("RGBA")
    data = np.array(rgba, dtype=np.int16)  # int16 para evitar overflow en restas

    r = data[:, :, 0]
    g = data[:, :, 1]
    b = data[:, :, 2]

    # ── Detectar píxeles de fondo ──
    # Condición 1: Los canales R, G, B son muy similares (gris/blanco)
    dist_rg = np.abs(r - g)
    dist_gb = np.abs(g - b)
    dist_rb = np.abs(r - b)
    es_gris = (dist_rg < UMBRAL_GRIS) & (dist_gb < UMBRAL_GRIS) & (dist_rb < UMBRAL_GRIS)

    # Condición 2: Es un gris/blanco brillante (no oscuro)
    es_brillante = (r > BRILLO_MINIMO) & (g > BRILLO_MINIMO) & (b > BRILLO_MINIMO)

    # Condición 3: Detectar patrón de cuadrícula específico
    # Las cuadrículas suelen alternar entre blanco puro (255) y gris claro (~204/~230)
    es_blanco_puro = (r > 250) & (g > 250) & (b > 250)
    es_gris_cuadricula = (r > 195) & (r < 240) & es_gris

    # Máscara final: es fondo si cumple cualquiera de estas condiciones
    es_fondo = (es_gris & es_brillante) | es_blanco_puro | es_gris_cuadricula

    # ── Aplicar transparencia ──
    pixeles_fondo = int(np.sum(es_fondo))
    pixeles_total = data.shape[0] * data.shape[1]
    porcentaje = (pixeles_fondo / pixeles_total) * 100

    print(f"   Píxeles de fondo detectados: {pixeles_fondo:,} / {pixeles_total:,} ({porcentaje:.1f}%)")

    # Verificación de seguridad: si detectamos más del 95% como fondo, algo anda mal
    if porcentaje > 95:
        print("   ⚠️  Se detectó demasiado como fondo (>95%). Ajustando umbrales...")
        # Ser más conservador: solo eliminar blanco puro
        es_fondo = es_blanco_puro
        pixeles_fondo = int(np.sum(es_fondo))
        porcentaje = (pixeles_fondo / pixeles_total) * 100
        print(f"   Reintento: {pixeles_fondo:,} píxeles ({porcentaje:.1f}%)")

    data_out = np.array(rgba, dtype=np.uint8)
    data_out[:, :, 3][es_fondo] = 0  # Alpha = 0 (transparente)

    resultado = Image.fromarray(data_out, "RGBA")
    print("   ✅ Fondo eliminado por color")
    return resultado


def suavizar_bordes_con_ia(img: Image.Image) -> Image.Image:
    """
    Usa rembg (modelo U2Net) para suavizar los bordes del logo
    después de la limpieza por color.
    """
    print("\n🤖 PASO 2: Refinando bordes con IA (rembg)...")

    try:
        from rembg import remove, new_session
    except ImportError:
        print("   ⚠️  rembg no está instalado. Instalando...")
        try:
            import subprocess
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "--break-system-packages", "--quiet", "rembg", "onnxruntime"
            ])
            from rembg import remove, new_session
            print("   ✅ rembg instalado correctamente")
        except Exception as e:
            print(f"   ❌ No se pudo instalar rembg: {e}")
            print("   Continuando sin refinamiento de IA...")
            return img

    try:
        print("   Cargando modelo U2Net (primera vez tarda más)...")
        session = new_session("u2net")

        resultado = remove(
            img,
            session=session,
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=10,
        )
        print("   ✅ Bordes refinados con IA")
        return resultado

    except Exception as e:
        print(f"   ⚠️  Error con rembg: {e}")
        print("   Continuando con la limpieza por color solamente...")
        return img


def limpiar_bordes_residuales(img: Image.Image) -> Image.Image:
    """
    Limpia píxeles semitransparentes residuales en los bordes
    que quedan después del procesamiento.
    """
    print("\n🧹 PASO 3: Limpiando bordes residuales...")

    data = np.array(img, dtype=np.uint8)

    # Eliminar píxeles con alpha muy bajo (< 30) que son artefactos
    alpha = data[:, :, 3]
    artefactos = alpha < 30
    data[:, :, 3][artefactos] = 0

    # Eliminar halos: píxeles casi blancos con alpha medio
    r, g, b = data[:, :, 0], data[:, :, 1], data[:, :, 2]
    es_halo = (r > 230) & (g > 230) & (b > 230) & (alpha < 200) & (alpha > 0)
    data[:, :, 3][es_halo] = 0

    halos_eliminados = int(np.sum(es_halo))
    if halos_eliminados > 0:
        print(f"   Halos eliminados: {halos_eliminados:,} píxeles")

    resultado = Image.fromarray(data, "RGBA")
    print("   ✅ Bordes limpios")
    return resultado


def recortar_espacios(img: Image.Image) -> Image.Image:
    """Recorta el espacio vacío (transparente) alrededor del logo."""
    print("\n✂️  Recortando espacios vacíos...")

    bbox = img.getbbox()
    if bbox is None:
        print("   ⚠️  La imagen quedó completamente transparente. Algo falló.")
        return img

    # Agregar un pequeño margen (padding) de 5px
    margen = 5
    x1 = max(0, bbox[0] - margen)
    y1 = max(0, bbox[1] - margen)
    x2 = min(img.width, bbox[2] + margen)
    y2 = min(img.height, bbox[3] + margen)

    resultado = img.crop((x1, y1, x2, y2))
    print(f"   Antes: {img.size[0]}×{img.size[1]}px → Después: {resultado.size[0]}×{resultado.size[1]}px")
    return resultado


def generar_formatos(logo_base: Image.Image, carpeta_salida: str) -> None:
    """Genera todos los formatos de salida (navbar, hero, footer)."""
    print("\n📦 Generando formatos para el sitio web...")

    carpeta = Path(carpeta_salida)
    carpeta.mkdir(parents=True, exist_ok=True)

    ancho_base, alto_base = logo_base.size
    ratio = alto_base / ancho_base

    for nombre, config in FORMATOS_SALIDA.items():
        ancho = config["ancho"]
        alto = int(ancho * ratio)
        resized = logo_base.resize((ancho, alto), Image.Resampling.LANCZOS)

        ruta = carpeta / nombre
        resized.save(str(ruta), "PNG", optimize=True)
        tamano_kb = ruta.stat().st_size / 1024
        print(f"   ✅ {nombre:<20s} {ancho}×{alto}px  ({tamano_kb:.0f} KB)  — {config['descripcion']}")


def generar_favicon(logo_base: Image.Image, carpeta_salida: str) -> None:
    """Extrae solo el icono (parte superior) y genera favicon."""
    print("\n✨ Generando Favicon...")

    ancho, alto = logo_base.size
    carpeta = Path(carpeta_salida)

    # ── Recortar solo la parte del icono (sin texto) ──
    corte_y = int(alto * FAVICON_CROP_SUPERIOR)
    icono = logo_base.crop((0, 0, ancho, corte_y))

    # Recortar espacio vacío del icono
    bbox = icono.getbbox()
    if bbox:
        icono = icono.crop(bbox)

    # ── Hacer cuadrado con padding transparente ──
    lado = max(icono.size)
    canvas = Image.new("RGBA", (lado, lado), (0, 0, 0, 0))
    offset_x = (lado - icono.width) // 2
    offset_y = (lado - icono.height) // 2
    canvas.paste(icono, (offset_x, offset_y), icono)  # Usar icono como máscara

    # ── Guardar en múltiples tamaños ──
    # favicon.png (32×32)
    favicon_32 = canvas.resize((32, 32), Image.Resampling.LANCZOS)
    ruta_png = carpeta / "favicon.png"
    favicon_32.save(str(ruta_png), "PNG", optimize=True)
    print(f"   ✅ favicon.png       32×32px")

    # favicon.ico (multi-tamaño: 16, 32, 48)
    try:
        sizes = [(16, 16), (32, 32), (48, 48)]
        iconos = [canvas.resize(s, Image.Resampling.LANCZOS) for s in sizes]

        ruta_ico = carpeta / "favicon.ico"
        iconos[0].save(str(ruta_ico), format="ICO", sizes=sizes, append_images=iconos[1:])
        print(f"   ✅ favicon.ico       16/32/48px (multi-tamaño)")
    except Exception as e:
        print(f"   ⚠️  No se pudo crear .ico: {e}")
        print(f"   El favicon.png sigue siendo válido y funcional.")

    # favicon-192.png (para PWA / Android)
    favicon_192 = canvas.resize((192, 192), Image.Resampling.LANCZOS)
    ruta_192 = carpeta / "favicon-192.png"
    favicon_192.save(str(ruta_192), "PNG", optimize=True)
    print(f"   ✅ favicon-192.png   192×192px (PWA/Android)")


def guardar_logo_completo(logo: Image.Image, carpeta_salida: str) -> None:
    """Guarda el logo completo limpio a resolución original."""
    carpeta = Path(carpeta_salida)
    ruta = carpeta / "logo_completo.png"
    logo.save(str(ruta), "PNG", optimize=True)
    tamano_kb = ruta.stat().st_size / 1024
    print(f"\n💎 Logo completo: logo_completo.png  {logo.size[0]}×{logo.size[1]}px  ({tamano_kb:.0f} KB)")


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Elimina el fondo de un logo y genera formatos web optimizados.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python limpiar_logo_pro.py logo.jpeg
  python limpiar_logo_pro.py logo.jpeg --salida ./assets/images/
  python limpiar_logo_pro.py logo.jpeg --sin-ia
  python limpiar_logo_pro.py logo.jpeg --solo-limpiar
        """
    )
    parser.add_argument("imagen", help="Ruta de la imagen del logo (JPEG, PNG, WEBP)")
    parser.add_argument("--salida", "-o", default=".", help="Carpeta de salida (defecto: directorio actual)")
    parser.add_argument("--sin-ia", action="store_true", help="Omitir refinamiento con rembg (más rápido)")
    parser.add_argument("--solo-limpiar", action="store_true", help="Solo limpiar fondo, no generar formatos")
    return parser


def main():
    parser = crear_parser()

    # Si no hay argumentos, buscar logo.jpeg por defecto
    if len(sys.argv) == 1:
        archivos_default = ["logo.jpeg", "logo.jpg", "logo.png", "logo.webp"]
        encontrado = None
        for nombre in archivos_default:
            if Path(nombre).exists():
                encontrado = nombre
                break

        if encontrado is None:
            parser.print_help()
            print("\n💡 Coloca tu logo en esta carpeta con el nombre 'logo.jpeg' y ejecuta de nuevo.")
            sys.exit(1)

        sys.argv.append(encontrado)

    args = parser.parse_args()

    # ── Banner ──
    print("╔══════════════════════════════════════════════════╗")
    print("║   🐦 ExploreSV — Logo Background Remover        ║")
    print("╚══════════════════════════════════════════════════╝")

    inicio = time.time()

    # ── Pipeline ──
    img = cargar_imagen(args.imagen)
    logo = eliminar_fondo_por_color(img)

    if not args.sin_ia:
        logo = suavizar_bordes_con_ia(logo)

    logo = limpiar_bordes_residuales(logo)
    logo = recortar_espacios(logo)

    # ── Guardar resultados ──
    guardar_logo_completo(logo, args.salida)

    if not args.solo_limpiar:
        generar_formatos(logo, args.salida)
        generar_favicon(logo, args.salida)

    # ── Resumen ──
    duracion = time.time() - inicio
    carpeta = Path(args.salida).resolve()

    print("\n" + "═" * 50)
    print(f"🎉 ¡Proceso completado en {duracion:.1f} segundos!")
    print(f"📁 Archivos guardados en: {carpeta}")
    print("═" * 50)

    if not args.solo_limpiar:
        print("\n📋 Archivos generados:")
        print("   logo_completo.png  — Logo limpio a resolución completa")
        for nombre, config in FORMATOS_SALIDA.items():
            print(f"   {nombre:<20s} — {config['descripcion']}")
        print("   favicon.png        — Icono de pestaña del navegador")
        print("   favicon.ico        — Icono multi-tamaño (compatibilidad)")
        print("   favicon-192.png    — Icono para PWA/Android")

    print("\n💡 Tip: Usa --sin-ia para procesamiento más rápido sin rembg")
    print("💡 Tip: Usa --salida ./carpeta/ para elegir dónde guardar\n")


if __name__ == "__main__":
    main()