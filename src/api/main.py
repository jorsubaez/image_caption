from src.core.caption_generator import CaptionGenerator


def main():
    image = ""
    generator = CaptionGenerator(image)
    caption = generator.predict_step()
    print(caption)


if __name__ == '__main__':
    main()
