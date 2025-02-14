import os
import gradio as gr
import pandas as pd
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import traceback
import time
from functools import lru_cache
import re

def natural_sort_key(s):
    """Trie les fichiers en respectant l'ordre des nombres."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ImageProcessingResult:
    success: bool
    message: str
    image_data: Optional[Image.Image] = None

@dataclass
class CaptionConfig:
    """Configuration for caption generation"""
    model_name: str
    task_prompt: str
    prefix: str = ""
    max_tokens: int = 256
    num_beams: int = 1
    do_sample: bool = True

class ModelManager:
    """Singleton class to manage model loading and caching"""
    _instance = None
    _model = None
    _processor = None
    _current_model_name = None

    MODELS = {
        "Florence-2 Large": "microsoft/Florence-2-large",
        "Florence-2 Base": "microsoft/Florence-2-base"
    }

    CAPTION_TYPES = {
        "Caption": "<CAPTION>",
        "Detailed Caption": "<DETAILED_CAPTION>",
        "More Detailed Caption": "<MORE_DETAILED_CAPTION>"
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance

    @classmethod
    @lru_cache(maxsize=2)
    def load_model(cls, model_name: str):
        logger.info(f"Loading model: {model_name}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_id = cls.MODELS.get(model_name)
        if model_id is None:
            raise ValueError(f"Unknown model name: {model_name}")
        try:
            model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device).eval()
            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            return model, processor
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise RuntimeError(f"Failed to load model {model_name}: {str(e)}")

    @classmethod
    def get_model_and_processor(cls, model_name: str) -> Tuple[AutoModelForCausalLM, AutoProcessor]:
        if cls._model is None or model_name != cls._current_model_name:
            logger.info(f"Loading model: {model_name}")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_id = cls.MODELS[model_name]
            try:
                cls._model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device).eval()
                cls._processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
                cls._current_model_name = model_name
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                raise RuntimeError(f"Failed to load model {model_name}: {str(e)}")
        return cls._model, cls._processor

    @classmethod
    def get_caption_prompt(cls, caption_type: str) -> str:
        return cls.CAPTION_TYPES.get(caption_type, "<CAPTION>")

class ImageLoader:
    """Handles image loading and validation"""
    VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".avif"}

    @staticmethod
    def validate_image(file_path: Path) -> ImageProcessingResult:
        try:
            if file_path.suffix.lower() not in ImageLoader.VALID_EXTENSIONS:
                return ImageProcessingResult(False, f"Unsupported format: {file_path.suffix}")
            with Image.open(file_path) as img:
                img_rgb = img.convert("RGB")
                if img_rgb.size[0] < 10 or img_rgb.size[1] < 10:
                    return ImageProcessingResult(False, "Image too small")
                return ImageProcessingResult(True, "Success", img_rgb)
        except Exception as e:
            return ImageProcessingResult(False, f"Error loading image: {str(e)}")

    @staticmethod
    def load_caption_if_exists(image_path: Path) -> str:
        """Load caption from a corresponding .txt file if it exists."""
        caption_path_with_ext = Path(f"{image_path}.txt")
        caption_path_without_ext = image_path.with_suffix(".txt")
        for caption_path in [caption_path_with_ext, caption_path_without_ext]:
            if caption_path.exists():
                try:
                    with open(caption_path, "r", encoding="utf-8") as f:
                        return f.read().strip()
                except Exception as e:
                    logger.error(f"Error loading caption for {image_path}: {str(e)}")
        return ""



    @staticmethod
    def list_images(directory: str, load_captions: bool = False) -> Tuple[Optional[pd.DataFrame], Optional[str], str]:
        try:
            dir_path = Path(directory)
            if not dir_path.exists():
                return None, None, "❌ Directory does not exist"

            image_files = sorted(
                [f for f in dir_path.iterdir() if f.suffix.lower() in ImageLoader.VALID_EXTENSIONS],
                key=lambda f: natural_sort_key(f.name)  # ✅ On applique le tri naturel ici
            )

            if not image_files:
                return None, None, "⚠️ No valid image files found"

            captions = [ImageLoader.load_caption_if_exists(f) if load_captions else "" for f in image_files]
            df = pd.DataFrame({
                "Process": [True] * len(image_files),
                "Filename": [f.name for f in image_files],
                "Caption": captions,
                "Selected": [False] * len(image_files),
                "Status": ["Loaded" if cap else "Pending" for cap in captions]
            })

            paths = "\n".join(str(f.absolute()) for f in image_files)
            loaded_count = sum(1 for cap in captions if cap)
            msg = f"✅ Found {len(image_files)} images"
            if load_captions and loaded_count > 0:
                msg += f" and loaded {loaded_count} existing captions"

            return df, paths, msg

        except Exception as e:
            logger.error(f"Error listing images: {traceback.format_exc()}")
            return None, None, f"❌ Error: {str(e)}"

class CaptionGenerator:
    """Handles caption generation with batching and error handling"""
    @staticmethod
    def generate_captions_batch(
        images: List[Image.Image],
        config: CaptionConfig,
        batch_size: int = 4
    ) -> List[str]:
        model, processor = ModelManager.get_model_and_processor(config.model_name)
        device = next(model.parameters()).device
        captions = []

        def process_batch(batch):
            try:
                with torch.no_grad():
                    inputs = processor(
                        text=[config.task_prompt] * len(batch),
                        images=batch,
                        return_tensors="pt"
                    ).to(device)
                    generated_ids = model.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        max_new_tokens=config.max_tokens,
                        do_sample=config.do_sample,
                        num_beams=config.num_beams,
                        early_stopping=True if config.num_beams > 1 else False
                    )
                    batch_captions = processor.batch_decode(generated_ids, skip_special_tokens=True)
                    return [f"{config.prefix}{caption}" for caption in batch_captions]
            except Exception as e:
                logger.error(f"Error generating captions for batch: {str(e)}")
                return ["Error: Failed to generate caption"] * len(batch)

        batches = [images[i:i + batch_size] for i in range(0, len(images), batch_size)]
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = executor.map(process_batch, batches)
        for batch_captions in results:
            captions.extend(batch_captions)
        return captions

class BatchCaptioningUI:
    """Handles the Gradio interface and UI interactions"""
    def __init__(self):
        self.model_manager = ModelManager()

    def create_interface(self):
        with gr.Blocks(css=self._get_custom_css()) as demo:
            gr.Markdown("## 🚀 Florence-2 Batch Captioning Optimizer")

            # --- Fonctions pour l'éditeur de Caption ---
            def update_caption_selector(data):
                """Met à jour la liste des captions à éditer avec un tri logique."""
                new_choices = []
                
                if data is None:
                    return gr.update(choices=[], value="")

                elif isinstance(data, dict) and "data" in data:
                    rows = data["data"]
                    headers = data.get("headers", [])
                    idx = headers.index("Filename") if "Filename" in headers else 1
                    new_choices = [str(row[idx]) for row in rows if len(row) > idx]

                elif isinstance(data, pd.DataFrame):
                    new_choices = list(data["Filename"])

                elif isinstance(data, list):
                    if len(data) > 0 and isinstance(data[0], list):
                        new_choices = [str(row[1]) for row in data if len(row) > 1]
                    elif len(data) > 0 and isinstance(data[0], dict):
                        new_choices = [str(row.get("Filename", "")) for row in data]

                # ✅ On applique le tri naturel ici
                new_choices = sorted(new_choices, key=natural_sort_key)

                # Gérer le cas où la liste est vide
                default_value = new_choices[0] if new_choices else ""

                return gr.update(choices=new_choices, value=default_value)



            def load_caption_for_edit(df, selected_filename):
                if df is None or not selected_filename:
                    return ""
                try:
                    if isinstance(df, pd.DataFrame):
                        for _, row in df.iterrows():
                            if row["Filename"] == selected_filename:
                                return row["Caption"]
                except Exception as e:
                    logger.error(f"Error loading caption for edit: {str(e)}")
                return ""

            def update_preview(text):
                return text

            def edit_caption(df, selected_filename, new_caption, image_paths):
                if df is None or not selected_filename or not image_paths:
                    return df, "⚠️ Unable to save the caption."

                try:
                    paths_list = image_paths.strip().split("\n")
                    base_path = None

                    # 🔍 Trouver le chemin complet de l'image sélectionnée
                    for path in paths_list:
                        if selected_filename in path:
                            base_path = Path(path)
                            break
                    
                    if not base_path:
                        return df, "⚠️ File no found."

                    # ✅ Mise à jour du DataFrame
                    updated = False
                    for idx, row in df.iterrows():
                        if row["Filename"] == selected_filename:
                            if row["Caption"] == new_caption.strip():
                                return df, "⚠️ No changes detected."
                            
                            df.at[idx, "Caption"] = new_caption.strip()
                            df.at[idx, "Selected"] = True
                            df.at[idx, "Status"] = "✅ Edited"
                            updated = True
                            break

                    if not updated:
                        return df, "⚠️ Image not found in the list."

                    # 🔹 Sauvegarde automatique du caption dans son fichier .txt
                    txt_path1 = base_path.with_suffix(".txt")  # ex: "image.png" → "image.txt"
                    txt_path2 = Path(f"{base_path}.txt")       # ex: "image.png" → "image.png.txt"
                    caption_path = txt_path1 if txt_path1.exists() else txt_path2 if txt_path2.exists() else txt_path1

                    with open(caption_path, "w", encoding="utf-8") as f:
                        f.write(new_caption.strip())

                    return df, f"✅ Saved caption for {selected_filename}."

                except Exception as e:
                    logger.error(f"Error when editing and saving caption: {str(e)}")
                    return df, f"❌ Error : {str(e)}"


            # --- Fonction de génération des captions ---
            def batch_caption(directory, df, image_paths, model_name, caption_type, prefix, max_tokens, num_beams, do_sample):
                if df is None or image_paths is None:
                    return df, None, "⚠️ No files to process."
                paths_list = image_paths.strip().split("\n")
                process_indices = df[df["Process"]].index
                if len(process_indices) == 0:
                    return df, image_paths, "⚠️ No files selected for processing."
                config = CaptionConfig(
                    model_name=model_name,
                    task_prompt=ModelManager.get_caption_prompt(caption_type),
                    prefix=prefix,
                    max_tokens=max_tokens,
                    num_beams=num_beams,
                    do_sample=do_sample
                )
                images = []
                valid_indices = []
                for idx in process_indices:
                    try:
                        result = ImageLoader.validate_image(Path(paths_list[idx]))
                        if result.success:
                            images.append(result.image_data)
                            valid_indices.append(idx)
                            df.at[idx, "Status"] = "Processing"
                        else:
                            df.at[idx, "Status"] = f"Error: {result.message}"
                    except Exception as e:
                        df.at[idx, "Status"] = f"Error: {str(e)}"
                if not images:
                    return df, image_paths, "❌ No valid images to process."
                try:
                    captions = CaptionGenerator.generate_captions_batch(images, config)
                    for valid_idx, caption in zip(valid_indices, captions):
                        df.at[valid_idx, "Caption"] = caption
                        df.at[valid_idx, "Status"] = "✅ Success"
                        df.at[valid_idx, "Selected"] = True
                except Exception as e:
                    logger.error(f"Caption generation error: {str(e)}")
                    return df, image_paths, f"❌ Error during caption generation: {str(e)}"
                return df, image_paths, f"✅ Generated {len(captions)} captions successfully"

            # --- Début de l'interface ---
            with gr.Row():
                with gr.Column(scale=1):
                    input_directory = gr.Textbox(label="📂 Input directory")
                    load_captions = gr.Checkbox(label="📝 Load Images and captions (if already generated)", value=False)
                    list_btn = gr.Button("📋 Load Images", variant="primary")
                    model_selector = gr.Dropdown(choices=list(ModelManager.MODELS.keys()), label="🧠 Model", value="Florence-2 Large")
                    caption_type = gr.Dropdown(choices=list(ModelManager.CAPTION_TYPES.keys()), label="✍️ Caption Type", value="More Detailed Caption")
                    prefix_input = gr.Textbox(label="🔤 Caption prefix (optional)")
                    with gr.Accordion("Advanced Options", open=False):
                        max_tokens = gr.Slider(minimum=64, maximum=512, value=256, step=32, label="Max Tokens")
                        num_beams = gr.Slider(minimum=1, maximum=5, value=1, step=1, label="Beam Search Size")
                        do_sample = gr.Checkbox(label="Use Sampling", value=True)
                with gr.Column(scale=2):
                    with gr.Row():
                        select_all_btn = gr.Button("✅ Select All")
                        deselect_all_btn = gr.Button("❌ Deselect All")
                    files_df = gr.Dataframe(
                        headers=["Process", "Filename", "Caption", "Selected", "Status"],
                        datatype=["bool", "str", "str", "bool", "str"],
                        col_count=(5, "fixed"),
                        interactive={"Process": True, "Caption": True, "Selected": True},
                        label="📝 Files & Captions",
                        height=500,
                        wrap=True
                    )
                    hidden_paths = gr.Textbox(visible=False)
                    status_text = gr.Textbox(label="📜 Status", interactive=False)
                    
                    # Définition du Dropdown en dehors du bloc d'Accordéon
                    caption_selector = gr.Dropdown(label="Select an image file to edit", choices=[], interactive=True, allow_custom_value=True)
                    
                    with gr.Accordion("Caption editor", open=True):
                        # On peut ici ajouter d'autres composants liés à l'édition, sans redéfinir caption_selector
                        caption_editor = gr.Textbox(label="Caption editor", lines=4, placeholder="Modify caption here...", interactive=True)
                        preview_caption = gr.Markdown(label="Preview Caption")
                        update_caption_btn = gr.Button("Update and save Caption", variant="primary")
                    
                    with gr.Row():
                        generate_btn = gr.Button("🚀 Generate Captions", variant="primary")
                        save_btn = gr.Button("💾 Save Selected", variant="secondary")
                    
                    generate_btn.click(
                        fn=batch_caption,
                        inputs=[input_directory, files_df, hidden_paths, model_selector, caption_type, prefix_input, max_tokens, num_beams, do_sample],
                        outputs=[files_df, hidden_paths, status_text]
                    ).then(
                        fn=update_caption_selector,
                        inputs=files_df,
                        outputs=caption_selector
                    )
                    # ... (le reste de vos callbacks)
                    
                    list_btn.click(
                        fn=ImageLoader.list_images,
                        inputs=[input_directory, load_captions],
                        outputs=[files_df, hidden_paths, status_text]
                    ).then(  # Ajoutez cette partie .then()
                        fn=update_caption_selector,
                        inputs=files_df,
                        outputs=caption_selector
                    )
                    
                    def save_captions(df, image_paths):
                        if df is None or image_paths is None:
                            return "⚠️ No captions to save."
                        paths_list = image_paths.strip().split("\n")
                        saved_count = 0
                        for idx, row in df.iterrows():
                            if row["Selected"] and row["Caption"].strip():
                                try:
                                    base_path = Path(paths_list[idx])
                                    txt_path1 = base_path.with_suffix(".txt")
                                    txt_path2 = Path(f"{base_path}.txt")
                                    caption_path = txt_path1 if txt_path1.exists() else txt_path2 if txt_path2.exists() else txt_path1
                                    with open(caption_path, "w", encoding="utf-8") as f:
                                        f.write(row["Caption"].strip())
                                    saved_count += 1
                                except Exception as e:
                                    logger.error(f"Error saving caption for {paths_list[idx]}: {str(e)}")
                        return f"✅ Saved {saved_count} captions successfully!"
                    
                    save_btn.click(
                        fn=save_captions,
                        inputs=[files_df, hidden_paths],
                        outputs=status_text
                    )
                    
                    def toggle_selection(df, select_all):
                        if df is not None:
                            df["Selected"] = select_all
                            df = df.copy()
                        return df
                    
                    select_all_btn.click(fn=lambda df: toggle_selection(df, True), inputs=[files_df], outputs=[files_df])
                    deselect_all_btn.click(fn=lambda df: toggle_selection(df, False), inputs=[files_df], outputs=[files_df])
                    
                    files_df.change(
                        fn=update_caption_selector,
                        inputs=files_df,
                        outputs=caption_selector
                    )
                    
                    caption_selector.change(
                        fn=load_caption_for_edit,
                        inputs=[files_df, caption_selector],
                        outputs=caption_editor
                    )
                    
                    caption_editor.change(
                        fn=update_preview,
                        inputs=caption_editor,
                        outputs=preview_caption
                    )
                    
                    update_caption_btn.click(
                        fn=edit_caption, 
                        inputs=[files_df, caption_selector, caption_editor, hidden_paths],  # 🔥 Ajout de `hidden_paths` (chemins des fichiers)
                        outputs=[files_df, status_text]
                    ).then(
                        fn=update_caption_selector,
                        inputs=files_df,
                        outputs=caption_selector
                    )
                    
            return demo

    @staticmethod
    def _get_custom_css():
        return """
        .generate-button { background: #2ecc71; }
        .save-button { background: #3498db; }
        .status-pending { color: #f39c12; }
        .status-success { color: #27ae60; }
        .status-error { color: #e74c3c; }
        """

def main():
    ui = BatchCaptioningUI()
    demo = ui.create_interface()
    demo.launch()

if __name__ == "__main__":
    main()
