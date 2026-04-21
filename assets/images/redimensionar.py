from PIL import Image

img = Image.open('banner-04.webp')
img_resized = img.resize((1920, 1080), Image.Resampling.LANCZOS)
img_resized.save('banner-04-redimensionado.webp', 'WEBP', quality=90)
print("✓ Imagen redimensionada a 1920x1080px y guardada")