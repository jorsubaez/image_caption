from src.core.preprocessing import ImagePreprocessor

from transformers import BlipProcessor, BlipForConditionalGeneration
from torch import device
from torch.cuda import is_available


class CaptionGenerator:
    def __init__(self, img_path: str, model_name: str = "Salesforce/blip-image-captioning-base") -> None:
        self._device = device("cuda" if is_available() else "cpu")

        self._processor = BlipProcessor.from_pretrained(model_name)
        self._model = BlipForConditionalGeneration.from_pretrained(model_name)
        self._model.to(self._device)

        self.gen_kwargs = {"max_length": 20, "num_beams": 5}

        preprocessor = ImagePreprocessor(img_path)
        self.img = preprocessor.img

    def predict_step(self) -> str:
        pixel_values = self._processor(images=self.img, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self._device)

        output_ids = self._model.generate(pixel_values, **self.gen_kwargs)

        prediction = self._processor.batch_decode(output_ids, skip_special_tokens=True)
        return prediction[0]
