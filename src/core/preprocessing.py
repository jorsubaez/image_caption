from src.utils.helpers import is_valid_URL, pil_img_to_np_array

from urllib.request import urlopen
from urllib.error import URLError
from PIL import Image, UnidentifiedImageError
from io import BytesIO


class ImagePreprocessor:
    def __init__(self, img_path):
        self.img_path = img_path
        self._img = self._load_url_img() if is_valid_URL(img_path) else self._load_local_image()

    @property
    def img(self):
        return self._img

    def _load_url_img(self):
        try:
            with urlopen(self.img_path) as resp:
                image_bytes = resp.read()

            image_file = BytesIO(image_bytes)
            img_pil = Image.open(image_file)

        except (IOError, UnidentifiedImageError, URLError) as e:
            print(f"Error: No se pudo cargar o identificar la imagen desde la URL: {self.img_path}\nDetalle: {e}")
            raise ValueError(f"URL de imagen inválida o corrupta: {self.img_path}") from e

        np_img = pil_img_to_np_array(img_pil)
        return np_img

    def _load_local_image(self):
        try:
            img_pil = Image.open(self.img_path)

        except (IOError, UnidentifiedImageError, FileNotFoundError) as e:
            print(f"Error: No se pudo cargar o identificar la imagen local: {self.img_path}\nDetalles: {e}")
            raise ValueError(f"Ruta de imagen inválida o corrupta: {self.img_path}") from e

        np_img = pil_img_to_np_array(img_pil)
        return np_img
