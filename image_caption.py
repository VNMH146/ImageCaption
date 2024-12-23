import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch
from PIL import Image, UnidentifiedImageError
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, CLIPProcessor, CLIPModel
import unittest
import pandas as pd
import warnings
from transformers import logging
logging.set_verbosity_error()  # Only log errors

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

class ImageCaptioningViTCLIP:
    def __init__(self, vit_model_name="nlpconnect/vit-gpt2-image-captioning", clip_model_name="openai/clip-vit-base-patch32"):
        """
        Initialize the combined ViT-based image captioning system with CLIP for alignment.

        Args:
            vit_model_name (str): Pretrained Vision Encoder-Decoder model for captioning.
            clip_model_name (str): Pretrained CLIP model for image-text alignment.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize ViT-based captioning system
        self.vit_processor = ViTImageProcessor.from_pretrained(vit_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(vit_model_name)
        self.vit_model = VisionEncoderDecoderModel.from_pretrained(vit_model_name).to(self.device)

        # Initialize CLIP model for image-text alignment
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(self.device)

    def generate_vit_captions(self, image_path, max_length=75, num_beams=5, num_return_sequences=3, temperature=1.0):
        """
        Generate multiple detailed captions using the ViT-based model.

        Args:
            image_path (str): Path to the input image.
            max_length (int, optional): Maximum length of the caption. Defaults to 75.
            num_beams (int, optional): Number of beams for beam search. Defaults to 5.
            num_return_sequences (int, optional): Number of captions to generate. Defaults to 3.
            temperature (float, optional): Sampling temperature for diversity. Defaults to 1.0.

        Returns:
            list of str: Generated detailed captions for the image.
        """
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"Error: File not found at '{image_path}'. Please provide a valid path.")
            return None
        except UnidentifiedImageError:
            print(f"Error: Unable to process the file at '{image_path}'. Ensure it is a valid image.")
            return None

        # Preprocess the image
        pixel_values = self.vit_processor(images=image, return_tensors="pt").pixel_values.to(self.device)

        # Generate multiple captions
        outputs = self.vit_model.generate(
            pixel_values,
            max_length=max_length,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            early_stopping=True
        )
        captions = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return captions

    def refine_with_clip(self, image_path, candidate_captions):
        """
        Refine the generated captions using CLIP by selecting the most contextually accurate one.

        Args:
            image_path (str): Path to the input image.
            candidate_captions (list of str): Candidate captions to refine.

        Returns:
            str: The most accurate caption based on CLIP's alignment score.
        """
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"Error: File not found at '{image_path}'. Please provide a valid path.")
            return None
        except UnidentifiedImageError:
            print(f"Error: Unable to process the file at '{image_path}'. Ensure it is a valid image.")
            return None

        # Preprocess the image and text
        inputs = self.clip_processor(
            text=candidate_captions,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        # Compute image-text alignment scores
        outputs = self.clip_model(**inputs)
        logits_per_image = outputs.logits_per_image  # (1, num_captions)
        probs = logits_per_image.softmax(dim=1)  # Convert logits to probabilities

        # Select the caption with the highest probability
        best_caption_idx = probs.argmax().item()
        return candidate_captions[best_caption_idx]

    def generate_and_refine_captions(self, image_path, max_length=75, num_beams=5, num_return_sequences=5, temperature=1.0):
        """
        Generate and refine a caption for the given image using ViT and CLIP.

        Args:
            image_path (str): Path to the input image.
            max_length (int, optional): Maximum length of the ViT-generated caption. Defaults to 75.
            num_beams (int, optional): Number of beams for beam search. Defaults to 5.
            num_return_sequences (int, optional): Number of refined captions generated for CLIP scoring. Defaults to 5.
            temperature (float, optional): Sampling temperature for diversity. Defaults to 1.0.

        Returns:
            str: The most detailed and accurate caption for the image.
        """
        # Generate multiple captions with ViT
        candidate_captions = self.generate_vit_captions(image_path, max_length, num_beams, num_return_sequences, temperature)

        if not candidate_captions:
            return None

        # Use CLIP to choose the best caption
        refined_caption = self.refine_with_clip(image_path, candidate_captions)
        return refined_caption

    def generate_captions_for_folder(self, folder_path, output_file="batch_captions_results.csv", max_length=75, num_beams=5, num_return_sequences=3, temperature=1.0):
        """
        Generate captions for all images in a folder.

        Args:
            folder_path (str): Path to the folder containing images.
            output_file (str): CSV file to save the results. Defaults to "batch_captions_results.csv".
            max_length (int): Maximum length of the ViT-generated caption.
            num_beams (int): Number of beams for beam search.
            num_return_sequences (int): Number of captions generated for each image.
            temperature (float): Sampling temperature for diversity.

        Returns:
            None
        """
        results = []
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path) and file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                print(f"Processing: {file_name}")
                refined_caption = self.generate_and_refine_captions(
                    file_path,
                    max_length=max_length,
                    num_beams=num_beams,
                    num_return_sequences=num_return_sequences,
                    temperature=temperature
                )
                results.append({"Image": file_name, "Refined Caption": refined_caption})
        
        # Save results to CSV
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")

if __name__ == "__main__":
    mode = input("Choose mode ('single' or 'batch'): ").strip().lower()
    captioning_system = ImageCaptioningViTCLIP()

    if mode == 'single':
        image_path = input("Enter the image path: ").strip()
        caption = captioning_system.generate_and_refine_captions(image_path, max_length=100, num_beams=7, num_return_sequences=5, temperature=1.2)
        if caption:
            print(f"Generated Caption: {caption}")
    elif mode == 'batch':
        folder_path = input("Enter the folder path: ").strip()
        output_file = input("Enter the output CSV file name (default: batch_captions_results.csv): ").strip() or "batch_captions_results.csv"
        captioning_system.generate_captions_for_folder(folder_path, output_file=output_file)
    else:
        print("Invalid mode. Please choose 'single' or 'batch'.")
