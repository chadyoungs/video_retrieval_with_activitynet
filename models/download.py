from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPProcessor

clip_path = "./models/clip-vit-base-patch32"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.save_pretrained(clip_path)
processor.save_pretrained(clip_path)
