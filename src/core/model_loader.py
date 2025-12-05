from transformers import BlipProcessor, BlipForConditionalGeneration
from torch import device, cuda


class ModelLoader:
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base"):
        self._device = device("cuda" if cuda.is_available() else "cpu")
        self._processor = BlipProcessor.from_pretrained(model_name)
        self._model = BlipForConditionalGeneration.from_pretrained(model_name)
        self._model.to(self._device)


model = None
