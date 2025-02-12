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
    def get_model_and_processor(cls, model_name: str) -> Tuple[AutoModelForCausalLM, AutoProcessor]:
        if cls._model is None or model_name != cls._current_model_name:
            logger.info(f"Loading model: {model_name}")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_id = cls.MODELS[model_name]
            
            try:
                cls._model = (AutoModelForCausalLM
                    .from_pretrained(model_id, trust_remote_code=True)
                    .to(device)
                    .eval())
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
            if not file_path.suffix.lower() in ImageLoader.VALID_EXTENSIONS:
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
        
        # Option 1: Check "test (13).jpg.txt"
        caption_path_with_ext = Path(f"{image_path}.txt")
        
        # Option 2: Check "test (13).txt"
        caption_path_without_ext = image_path.with_suffix(".txt")

        # Try to load from either filename style
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
                
            image_files = [
                f for f in dir_path.iterdir()
                if f.suffix.lower() in ImageLoader.VALID_EXTENSIONS
            ]
            
            if not image_files:
                return None, None, "⚠️ No valid image files found"
            
            captions = [""] * len(image_files)
            if load_captions:
                captions = [ImageLoader.load_caption_if_exists(f) for f in image_files]
                
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
        """Generates captions for a batch of images using multi-threading for performance optimization."""
        
        model, processor = ModelManager.get_model_and_processor(config.model_name)
        device = next(model.parameters()).device
        captions = []

        def process_batch(batch):
            """Processes a single batch and returns captions."""
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
                        early_stopping=True if config.num_beams > 1 else False  # Adjust early stopping dynamically
                    )

                    batch_captions = processor.batch_decode(
                        generated_ids,
                        skip_special_tokens=True
                    )

                    # Add prefix if specified
                    return [f"{config.prefix}{caption}" for caption in batch_captions]

            except Exception as e:
                logger.error(f"Error generating captions for batch: {str(e)}")
                return ["Error: Failed to generate caption"] * len(batch)

        # Splitting images into batches
        batches = [images[i:i + batch_size] for i in range(0, len(images), batch_size)]

        # Using ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = executor.map(process_batch, batches)

        # Flatten results
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
            
            with gr.Row():
                with gr.Column(scale=1):
                    input_directory = gr.Textbox(label="📂 Input directory")
                    load_captions = gr.Checkbox(
                        label="📝 Load Images and captions (if already generated)",
                        value=False
                    )
                    list_btn = gr.Button("📋 Load Images", variant="primary")
                    
                    # Model selection
                    model_selector = gr.Dropdown(
                        choices=list(ModelManager.MODELS.keys()),
                        label="🧠 Model",
                        value="Florence-2 Large"
                    )
                    
                    # Caption type selection
                    caption_type = gr.Dropdown(
                        choices=list(ModelManager.CAPTION_TYPES.keys()),
                        label="✍️ Caption Type",
                        value="More Detailed Caption"
                    )
                    
                    prefix_input = gr.Textbox(label="🔤 Caption prefix (optional)")
                    
                    # Advanced options in a collapsible section
                    with gr.Accordion("Advanced Options", open=False):
                        max_tokens = gr.Slider(
                            minimum=64,
                            maximum=512,
                            value=256,
                            step=32,
                            label="Max Tokens"
                        )
                        num_beams = gr.Slider(
                            minimum=1,
                            maximum=5,
                            value=1,
                            step=1,
                            label="Beam Search Size"
                        )
                        do_sample = gr.Checkbox(
                            label="Use Sampling",
                            value=True
                        )
                
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
                    
                    with gr.Row():
                        generate_btn = gr.Button("🚀 Generate Captions", variant="primary")
                        save_btn = gr.Button("💾 Save Selected", variant="secondary")
                        
            def batch_caption(
                directory, df, image_paths, model_name, caption_type,
                prefix, max_tokens, num_beams, do_sample
            ):
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
                        df.at[valid_idx, "Selected"] = true
                        
                except Exception as e:
                    logger.error(f"Caption generation error: {str(e)}")
                    return df, image_paths, f"❌ Error during caption generation: {str(e)}"

                return df, image_paths, f"✅ Generated {len(captions)} captions successfully"

            # Wire up all event handlers
            list_btn.click(
                fn=ImageLoader.list_images,
                inputs=[input_directory, load_captions],
                outputs=[files_df, hidden_paths, status_text]
            )

            generate_btn.click(
                fn=batch_caption,
                inputs=[
                    input_directory, files_df, hidden_paths,
                    model_selector, caption_type, prefix_input,
                    max_tokens, num_beams, do_sample
                ],
                outputs=[files_df, hidden_paths, status_text]
            )

            def save_captions(df, image_paths):
                if df is None or image_paths is None:
                    return "⚠️ No captions to save."

                paths_list = image_paths.strip().split("\n")
                saved_count = 0
                
                for idx, row in df.iterrows():
                    if row["Selected"] and row["Caption"].strip():
                        try:
                            caption_path = f"{paths_list[idx]}.txt"
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
                """Toggle selection for all items in the DataFrame and force update."""
                if df is not None:
                    df["Selected"] = select_all  # Mise à jour interne
                    df = df.copy()  # ✅ Force le rafraîchissement
                return df

            select_all_btn.click(
                fn=lambda df: toggle_selection(df, True),
                inputs=[files_df],
                outputs=[files_df]
            )

            deselect_all_btn.click(
                fn=lambda df: toggle_selection(df, False),
                inputs=[files_df],
                outputs=[files_df]
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
