from urllib.parse import urlparse
from numpy import array


def is_valid_URL(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def is_valid_image(img, img_path):
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {img_path}")


def pil_img_to_np_array(img_pil):
    return array(img_pil.convert('RGB'))
