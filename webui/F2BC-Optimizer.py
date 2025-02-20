import os
import gradio as gr
import pandas as pd
from PIL import Image
import torch
from safetensors import safe_open
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
import io 
import random

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
    def fix_filenames_and_captions(directory: str) -> str:
        """
        Parcourt le dossier 'directory' et renomme les fichiers images et leurs
        fichiers .txt associés en supprimant les espaces avant l'extension.
        """
        dir_path = Path(directory)
        if not dir_path.exists() or not dir_path.is_dir():
            return "❌ Invalid directory"

        for f in dir_path.iterdir():
            # Nom d'origine
            old_name = f.name
            # Nouveau nom sans espace avant l'extension (ex : "mario(01) .png" -> "mario(01).png")
            new_name = old_name.replace(" ", "")

            # Si le nom n'a pas changé, on passe au suivant
            if new_name == old_name:
                continue

            new_path = f.with_name(new_name)

            # Vérifier qu'on n'écrase pas un fichier existant
            if new_path.exists():
                # À vous de décider comment gérer ce conflit : 
                # on peut soit passer, soit ajouter un suffixe, etc.
                continue

            # Renommer le fichier (image ou autre)
            f.rename(new_path)

            # Si c'est une image, on renomme aussi le .txt associé
            if new_path.suffix.lower() in ImageLoader.VALID_EXTENSIONS:
                # Deux formes possibles : "image.png.txt" ou "image.txt"
                old_txt_1 = dir_path / f"{old_name}.txt"       # ex: "mario(01) .png.txt"
                old_txt_2 = f.with_suffix(".txt")              # ex: "mario(01) .txt"

                new_txt_1 = dir_path / f"{new_name}.txt"       # ex: "mario(01).png.txt"
                new_txt_2 = new_path.with_suffix(".txt")       # ex: "mario(01).txt"

                # Si le premier existe, on le renomme
                if old_txt_1.exists():
                    if not new_txt_1.exists():
                        old_txt_1.rename(new_txt_1)
                # Sinon, si le second existe, on le renomme
                elif old_txt_2.exists():
                    if not new_txt_2.exists():
                        old_txt_2.rename(new_txt_2)

        return "✅ Filenames and captions fixed!"



    @staticmethod
    def list_images(directory: str, load_captions: bool = False, only_uncaptioned: bool = False) -> Tuple[Optional[pd.DataFrame], Optional[str], str]:
        try:
            if not directory.strip():
                return None, None, "❌ No directory specified (empty input)"

            dir_path = Path(directory)
            if not dir_path.exists():
                return None, None, "❌ Directory does not exist"

            image_files = sorted(
                [f for f in dir_path.iterdir() if f.suffix.lower() in ImageLoader.VALID_EXTENSIONS],
                key=lambda f: natural_sort_key(f.name)
            )

            if not image_files:
                return None, None, "⚠️ No valid image files found"

            # 1) Si only_uncaptioned est vrai, on force la lecture des captions
            if only_uncaptioned:
                load_captions = True

            # 2) Charger les légendes si nécessaire
            captions = [
                ImageLoader.load_caption_if_exists(f) if load_captions else ""
                for f in image_files
            ]

            # 3) Filtrer uniquement les images qui n'ont pas de légende (si only_uncaptioned est True)
            if only_uncaptioned:
                filtered = [(f, cap) for f, cap in zip(image_files, captions) if not cap]
                if not filtered:
                    return None, None, "⚠️ No files without captions were found"
                image_files, captions = zip(*filtered)
                image_files = list(image_files)
                captions = list(captions)

            # 4) Créer le DataFrame final
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

    @staticmethod
    def get_thumbnails(df, hidden_paths, selected_filename):
        if df is None or not hidden_paths:
            return None, gr.update(value=selected_filename)

        paths_list = hidden_paths.strip().split("\n")
        images = []
        for idx, row in df.iterrows():
            img_path = paths_list[idx]
            if os.path.exists(img_path):
                with Image.open(img_path) as img:
                    img_resized = img.resize((300, 300), Image.LANCZOS)
                    images.append(img_resized.convert("RGB"))

        # On prend la première image si dispo
        first_image = images[0] if images else None
        return first_image, gr.update(value=selected_filename)

    """Handles caption generation with batching and error handling"""
    @staticmethod
    def generate_captions_batch(
        images: List[Image.Image],
        config: CaptionConfig,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        min_new_tokens: int,
        early_stopping: bool,
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
                        temperature=temperature if config.do_sample else None,
                        top_k=top_k if config.do_sample else None,
                        top_p=top_p if config.do_sample else None,
                        repetition_penalty=repetition_penalty,
                        min_new_tokens=min_new_tokens,
                        early_stopping=early_stopping
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

    @staticmethod
    def update_gallery_selection(df, selected_filename, hidden_paths):
        if df is None or not hidden_paths or not selected_filename:
            return gr.update(value=None)  # Renvoie None si pas d'image

        paths_list = hidden_paths.strip().split("\n")
        try:
            for idx, row in df.iterrows():
                if row["Filename"] == selected_filename:
                    img_path = paths_list[idx]
                    if not os.path.exists(img_path):
                        return gr.update(value=None)

                    with Image.open(img_path) as raw_img:
                        img_resized = raw_img.resize((300, 300), Image.LANCZOS)

                        # Convertir en BytesIO si vous tenez à contourner le cache
                        img_bytes = io.BytesIO()
                        img_resized.save(img_bytes, format="PNG")
                        img_bytes.seek(0)

                        # On réouvre l'image en PIL
                        single_img = Image.open(img_bytes).convert("RGB")

                        # On renvoie UN SEUL objet PIL, pas de liste
                        return gr.update(value=single_img)

        except Exception as e:
            logger.error(f"Error loading image {selected_filename}: {str(e)}")

        # Si on n'a rien trouvé
        return gr.update(value=None)



def build_batch_captioning_ui(container):
    with container:
        gr.Markdown("## 🚀 Florence-2 Batch Captioning Optimizer")
        hidden_paths = gr.Textbox(visible=False)
        current_index_state = gr.State(value=0)  # index initial = 0

        # --- Fonctions pour l'éditeur de Caption ---
        def update_caption_selector(data, current_selection=None):

            new_choices = []

            # Gestion du cas data=None
            if data is None:
                return gr.update(choices=[], value="")

            # Selon le type de data, on extrait la colonne "Filename"
            if isinstance(data, dict) and "data" in data:
                # data["data"] est généralement une liste de listes
                rows = data["data"]
                headers = data.get("headers", [])
                idx = headers.index("Filename") if "Filename" in headers else 1
                new_choices = [str(row[idx]) for row in rows if len(row) > idx]

            elif isinstance(data, pd.DataFrame):
                # DataFrame direct
                new_choices = list(data["Filename"])

            elif isinstance(data, list):
                # data est déjà une liste de listes ou de dict
                if len(data) > 0 and isinstance(data[0], list):
                    # ex: [[True, "fichier1.jpg", "Caption1", ...], [True, "fichier2.jpg", ...], ...]
                    new_choices = [str(row[1]) for row in data if len(row) > 1]
                elif len(data) > 0 and isinstance(data[0], dict):
                    # ex: [{"Filename": "fichier1.jpg", "Caption": "..."}, ...]
                    new_choices = [str(row.get("Filename", "")) for row in data]

            # Tri naturel (ex: file2, file10, file11, etc.)
            new_choices = sorted(new_choices, key=natural_sort_key)

            # Détermine la valeur sélectionnée par défaut
            if not current_selection or current_selection not in new_choices:
                # On prend le premier élément si la sélection courante est None ou plus valide
                default_value = new_choices[0] if new_choices else ""
            else:
                default_value = current_selection

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

                # Trouver le chemin complet
                for path in paths_list:
                    if selected_filename in path:
                        base_path = Path(path)
                        break

                if not base_path:
                    return df, "⚠️ File not found."

                # On trouve l’index de la ligne correspondante dans le DataFrame
                idx_list = df.index[df["Filename"] == selected_filename].tolist()
                if not idx_list:
                    return df, "⚠️ Image not found in the list."

                idx = idx_list[0]

                # Vérifier si la légende n'a pas déjà cette valeur
                if df.at[idx, "Caption"] == new_caption.strip():
                    return df, "⚠️ No changes detected."

                # Mise à jour du DataFrame
                df.at[idx, "Caption"] = new_caption.strip()
                df.at[idx, "Selected"] = True
                df.at[idx, "Status"] = "✅ Edited"
                df = df.copy()  # Force le rafraîchissement de Gradio

                # Sauvegarde automatique du caption
                txt_path1 = base_path.with_suffix(".txt")
                txt_path2 = Path(f"{base_path}.txt")
                # On choisit un des deux chemins
                caption_path = txt_path1 if txt_path1.exists() else txt_path2 if txt_path2.exists() else txt_path1

                with open(caption_path, "w", encoding="utf-8") as f:
                    f.write(new_caption.strip())

                return df, f"✅ Saved caption for {selected_filename}."

            except Exception as e:
                logger.error(f"❌ Error when editing and saving caption: {str(e)}")
                return df, f"❌ Error : {str(e)}"


        def search_and_replace_in_captions(df, search_text, replace_text, match_case, whole_word):
            """
            Replaces all occurrences of `search_text` with `replace_text`
            in captions, with options for case and whole word replacement.
            Displays a warning if no changes have been made.
            """
            if df is None or df.empty:
                return df, "⚠️ No captions available for search and replace."

            if not search_text.strip():
                return df, "⚠️ Please enter a search term."

            if not replace_text.strip():
                return df, "⚠️ Replacement text cannot be empty."

            df = df.copy()  # Force Gradio à rafraîchir l'affichage

            # Construction de l'expression régulière en fonction des options cochées
            pattern = r'\b' + re.escape(search_text) + r'\b' if whole_word else re.escape(search_text)
            flags = 0 if match_case else re.IGNORECASE

            changes_made = 0  # Compteur de modifications

            for idx in df.index:
                caption = df.at[idx, "Caption"]
                new_caption, count = re.subn(pattern, replace_text, caption, flags=flags)  # Utilisation de re.subn pour compter les remplacements
                
                if count > 0:  # Si au moins un remplacement est fait
                    df.at[idx, "Caption"] = new_caption
                    changes_made += count

            # Génération du message de statut
            if changes_made == 0:
                message = f"⚠️ No occurrences of '{search_text}' found in captions."
                if whole_word:
                    message += " Try disabling 'Whole Word Only'."
                if not match_case:
                    message += " Try enabling 'Match Case'."
            else:
                message = f"✅ Replaced {changes_made} occurrences of '{search_text}' with '{replace_text}'."

            return df, message


        def apply_prefix_to_captions(df, prefix):
            """
            Applique un préfixe à toutes les captions déjà chargées sans nécessiter
            de régénération avec Florence-2.
            """
            if df is None or df.empty:
                return df, "⚠️ No captions loaded to apply the prefix."

            df = df.copy()  # Force Gradio à rafraîchir l'affichage

            for idx in df.index:
                if df.at[idx, "Caption"]:  # Vérifie si un caption existe déjà
                    df.at[idx, "Caption"] = f"{prefix}{df.at[idx, 'Caption']}"

            return df, "✅ Prefix applied to loaded captions."


        def delete_caption_files(df, directory):
          
            if df is None or df.empty:
                return df, "⚠️ No captions available to delete."

            if not os.path.exists(directory):
                return df, f"⚠️ The directory '{directory}' does not exist."

            deleted_count = 0
            not_found_count = 0

            for idx in df.index:
                filename = df.at[idx, "Filename"]
                base_name, _ = os.path.splitext(filename)  # Supprime l'extension (.jpg, .png, etc.)

                #  Vérifier plusieurs variantes du fichier .txt
                possible_txt_files = [
                    os.path.join(directory, f"{base_name}.txt"),       # Variante classique : test.txt
                    os.path.join(directory, f"{filename}.txt"),        # Variante avec extension complète : test.jpg.txt
                ]

                #  Ajout d'une vérification pour les fichiers avec parenthèses
                parenthesis_match = re.match(r"^(.*)\s\(\d+\)$", base_name)
                if parenthesis_match:
                    base_without_parentheses = parenthesis_match.group(1)
                    possible_txt_files.append(os.path.join(directory, f"{base_without_parentheses}.txt"))

                #  Tentative de suppression des fichiers trouvés
                file_deleted = False
                for txt_path in possible_txt_files:
                    if os.path.exists(txt_path):
                        try:
                            os.remove(txt_path)
                            deleted_count += 1
                            file_deleted = True
                            break  # Sortir après suppression
                        except Exception as e:
                            return df, f"❌ Error deleting {txt_path}: {str(e)}"

                if not file_deleted:
                    not_found_count += 1

            #  Message de statut après suppression
            if deleted_count == 0 and not_found_count > 0:
                message = f"⚠️ No caption files found for deletion. Check if filenames contain special characters."
            else:
                message = f"✅ Deleted {deleted_count} caption files."
                if not_found_count > 0:
                    message += f" ⚠️ {not_found_count} files were missing."

            return df, message


        # --- Fonction de génération des captions ---
        def batch_caption(directory, df, image_paths, model_name, caption_type, prefix, max_tokens, num_beams, do_sample, temperature, top_k, top_p, repetition_penalty, min_new_tokens, early_stopping):
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
                captions = CaptionGenerator.generate_captions_batch(
                    images, 
                    config, 
                    temperature, 
                    top_k, 
                    top_p, 
                    repetition_penalty, 
                    min_new_tokens, 
                    early_stopping
                )
                for valid_idx, caption in zip(valid_indices, captions):
                    df.at[valid_idx, "Caption"] = caption
                    df.at[valid_idx, "Status"] = "✅ Success"
                    df.at[valid_idx, "Selected"] = True
            except Exception as e:
                logger.error(f"Caption generation error: {str(e)}")
                return df, image_paths, f"❌ Error during caption generation: {str(e)}"
            return df, image_paths, f"✅ Generated {len(captions)} captions successfully"


        def go_previous(df, current_index, hidden_paths):
            if df is None or df.empty:
                return gr.update(), current_index
            rows = df["Filename"].tolist()
            new_index = max(0, current_index - 1)
            return rows[new_index], new_index

        def go_next(df, current_index, hidden_paths):
            if df is None or df.empty:
                return gr.update(), current_index
            rows = df["Filename"].tolist()
            last_index = len(rows) - 1
            new_index = min(last_index, current_index + 1)
            return rows[new_index], new_index

        def select_image(df, selected_filename, hidden_paths, old_index):
            # Trouve l'index correspondant dans le DataFrame
            idx_list = df.index[df["Filename"] == selected_filename].tolist()
            if not idx_list:
                return old_index, "", None  # On ne change rien si on ne trouve pas

            new_index = idx_list[0]
            # On charge la caption
            new_caption = load_caption_for_edit(df, selected_filename)
            new_preview = BatchCaptioningUI.update_gallery_selection(df, selected_filename, hidden_paths)
            return new_index, new_caption, new_preview


        def reset_index():
            return 0


        # --- Début de l'interface ---
        with gr.Row():
            with gr.Column(scale=1):
                input_directory = gr.Textbox(label="📂 Input directory")
                load_captions = gr.Checkbox(label="Load Images with captions (if already generated)", value=False)
                only_uncaptioned_checkbox = gr.Checkbox(label="Filter only images with no existing .txt caption", value=False)
                list_btn = gr.Button("📋 Load Images", variant="primary")
                generate_btn = gr.Button("🚀 Generate Captions", variant="primary")
                save_btn = gr.Button("💾 Save Selected Catpions", variant="secondary")
                delete_captions_btn = gr.Button("🗑️ Delete Captions Files", variant="secondary")
                rename_btn = gr.Button(" Format filenames (remove spaces) (optional)")
                model_selector = gr.Dropdown(choices=list(ModelManager.MODELS.keys()), label="🧠 Model", value="Florence-2 Base")
                caption_type = gr.Dropdown(choices=list(ModelManager.CAPTION_TYPES.keys()), label="✍️ Caption Type", value="Caption")
                prefix_input = gr.Textbox(label="🔤 Caption prefix (optional)")
                apply_prefix_btn = gr.Button("Apply Prefix", variant="secondary")


                with gr.Accordion("Tokens editor", open=False):
                    search_input = gr.Textbox(label="🔍 Search token", placeholder="Enter token to search...")
                    replace_input = gr.Textbox(label="✏️ Replace with", placeholder="Enter replacement token...")
                    match_case = gr.Checkbox(label="🔤 Match case", value=False)
                    whole_word = gr.Checkbox(label="🔍 Whole word only", value=False)
                    replace_btn = gr.Button("🔄 Search & Replace token")
                
                with gr.Accordion("Advanced Options", open=False):
                    max_tokens = gr.Slider(minimum=64, maximum=512, value=128, step=32, label="Max Tokens")
                    num_beams = gr.Slider(minimum=1, maximum=5, value=2, step=1, label="Beam Search Size")
                    do_sample = gr.Checkbox(label="Use Sampling", value=True)

                    temperature = gr.Slider(minimum=0.0, maximum=2.0, value=1.0, step=0.1, label="Temperature (creativity)", interactive=True)
                    top_k = gr.Slider(minimum=0, maximum=100, value=50, step=1, label="Top-k (token sampling limit)")
                    top_p = gr.Slider(minimum=0.0, maximum=1.0, value=1.0, step=0.05, label="Top-p (nucleus sampling)")
                    repetition_penalty = gr.Slider(minimum=1.0, maximum=3.0, value=1.0, step=0.1, label="Repetition Penalty (discourage repeats)")
                    min_new_tokens = gr.Slider(minimum=0, maximum=50, value=0, step=5, label="Min New Tokens")
                    early_stopping = gr.Checkbox(label="Early Stopping", value=False)

            with gr.Column(scale=2):
                with gr.Row():
                    select_all_btn = gr.Button("✅ Select All")
                    deselect_all_btn = gr.Button("❌ Deselect All")
                files_df = gr.Dataframe(
                    headers=["Process", "Filename", "Caption", "Selected", "Status"],
                    datatype=["bool", "str", "str", "bool", "str"],
                    col_count=(5, "fixed"),
                    interactive={"Process": True, "Caption": False, "Selected": True},
                    label="📝 Files & Captions",
                    height=500,
                    wrap=True                       
                )
                status_text = gr.Textbox(label="📜 Status", interactive=False)
                apply_prefix_btn.click(
                    fn=apply_prefix_to_captions,
                    inputs=[files_df, prefix_input],  
                    outputs=[files_df, status_text]
                )
                delete_captions_btn.click(
                    fn=delete_caption_files,
                    inputs=[files_df, input_directory],  # On passe le DataFrame et le dossier source
                    outputs=[files_df, status_text]  # Mise à jour du tableau et affichage du statut
                )

                hidden_paths = gr.Textbox(visible=False)
                
                with gr.Accordion("Caption editor", open=True):
                    preview = gr.Image(label="Preview", interactive=False, type="pil", height=300)

                    next_btn = gr.Button("Next →")
                    prev_btn = gr.Button("← Previous") 
                    caption_selector = gr.Dropdown(label="Select an image file to edit", choices=[], interactive=True, allow_custom_value=True)
                    update_caption_btn = gr.Button("Update and save Caption", variant="primary")
                    caption_editor = gr.Textbox(label="Caption editor", lines=4, placeholder="Modify caption here...", interactive=True)
                    preview_caption = gr.Markdown(label="Preview Caption")
                
                generate_btn.click(
                    fn=batch_caption,
                    inputs=[input_directory, files_df, hidden_paths, model_selector, caption_type, prefix_input, max_tokens, num_beams, do_sample, temperature, top_k, top_p, repetition_penalty, min_new_tokens, early_stopping],
                    outputs=[files_df, hidden_paths, status_text]
                ).then(
                    fn=update_caption_selector,
                    inputs=[files_df, caption_selector],
                    outputs=caption_selector
                )
                    
                list_btn.click(
                    fn=ImageLoader.list_images,
                    inputs=[input_directory, load_captions, only_uncaptioned_checkbox],
                    outputs=[files_df, hidden_paths, status_text]
                ).then(
                    fn=reset_index,
                    inputs=[],
                    outputs=current_index_state
                ).then(
                    fn=update_caption_selector,
                    inputs=[files_df, caption_selector],
                    outputs=caption_selector
                )

                replace_btn.click(
                    fn=search_and_replace_in_captions,
                    inputs=[files_df, search_input, replace_input, match_case, whole_word],
                    outputs=[files_df, status_text]
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
                    inputs=[files_df, caption_selector],
                    outputs=caption_selector
                )
                rename_btn.click(
                    fn=ImageLoader.fix_filenames_and_captions,
                    inputs=[input_directory],
                    outputs=status_text
                )
                caption_selector.change(
                    fn=select_image,
                    inputs=[files_df, caption_selector, hidden_paths, current_index_state],
                    outputs=[current_index_state, caption_editor, preview]
                )
                                    
                caption_editor.change(
                    fn=update_preview,
                    inputs=caption_editor,
                    outputs=preview_caption
                )
                
                update_caption_btn.click(
                    fn=edit_caption, 
                    inputs=[files_df, caption_selector, caption_editor, hidden_paths],  
                    outputs=[files_df, status_text]
                ).then(
                    fn=update_caption_selector,
                    inputs=[files_df, caption_selector],  # On passe aussi la sélection courante
                    outputs=caption_selector
                )

                prev_btn.click(
                    fn=go_previous,
                    inputs=[files_df, current_index_state, hidden_paths],
                    outputs=[caption_selector, current_index_state]
                ).then(
                    fn=lambda df, sel, hp: (
                        load_caption_for_edit(df, sel), 
                        BatchCaptioningUI.update_gallery_selection(df, sel, hp)
                    ),
                    inputs=[files_df, caption_selector, hidden_paths],
                    outputs=[caption_editor, preview]
                )

                next_btn.click(
                    fn=go_next,
                    inputs=[files_df, current_index_state, hidden_paths],
                    outputs=[caption_selector, current_index_state]
                ).then(
                    fn=lambda df, sel, hp: (
                        load_caption_for_edit(df, sel), 
                        BatchCaptioningUI.update_gallery_selection(df, sel, hp)
                    ),
                    inputs=[files_df, caption_selector, hidden_paths],
                    outputs=[caption_editor, preview]
                )

           

                    
def list_lora_files(folder_path):

    if not os.path.isdir(folder_path):
        return None, "❌ Error: File not found"
    
    lora_files = [f for f in os.listdir(folder_path) if f.endswith(".safetensors")]
    if not lora_files:
        return None, "❌ No LoRA files found in this folder."
    
    return lora_files, None

def analyze_lora(folder_path, file_name):
  
    file_path = os.path.join(folder_path, file_name)
    if not file_name or not os.path.exists(file_path):
        return f"❌ Error : File `{file_name}` not available in `{folder_path}`."
    
    try:
        with safe_open(file_path, framework="pt") as f:
            keys = list(f.keys())
            
            # On va gérer la "metadata" en testant si c'est une méthode ou un attribut
            raw_metadata = getattr(f, "metadata", None)
            if callable(raw_metadata):
                # Si c'est une méthode, on l'appelle pour récupérer un dict
                metadata = raw_metadata()
            else:
                # Sinon, on le prend tel quel
                metadata = raw_metadata
            
        output = f"🔍 File analysis : {file_name}\n"
        output += f"Number of keys (tensors) : {len(keys)}\n"
        output += "Keys:\n" + "\n".join(keys)
        
        # Vérification : si c’est un dict non vide, on l’affiche
        if isinstance(metadata, dict) and metadata:
            output += "\n\n🗒️ Métadonnées :\n"
            for k, v in metadata.items():
                output += f"{k} : {v}\n"
        else:
            output += "\n\n⚠️ No metadata found."

        return output
    
    except Exception as e:
        return f"⚠️ Error when analysing LoRA : {str(e)}"

def load_folder(folder_path):

    files, error = list_lora_files(folder_path)
    if error:
        return error, gr.update(choices=[], value=None), "No analysis possible"
    else:
        return (
            "✅ File successfully loaded",
            gr.update(choices=files, value=files[0]),
            "Select a file"
        )



def build_lora_inspector_ui(container):
    with container:
        gr.Markdown("# 🛠️ **LoRA Inspector - Analysis LoRA files**")
        gr.Markdown("🔎 Enter the path to a folder containing LoRAs (`.safetensors`) and analyse their metadata.")
        folder_input = gr.Textbox(label="📂 Path to folder containing LoRA files", value="", interactive=True)
        load_button = gr.Button("📥 Loading LoRA files")
        status_output = gr.Textbox(label="📊 Statut", interactive=False)
        file_dropdown = gr.Dropdown(label="📂 Select a LoRA file", choices=[], interactive=True)
        analyze_button = gr.Button("🔍 Analysis")
        output_text = gr.Textbox(label="📑 Analysis result", interactive=False, lines=15)
        
        load_button.click(
            fn=load_folder,
            inputs=folder_input,
            outputs=[status_output, file_dropdown, output_text]
        )
        analyze_button.click(
            fn=analyze_lora,
            inputs=[folder_input, file_dropdown],
            outputs=output_text
        )
        pass


def create_integrated_interface():
    # Optionnel : définir un CSS pour harmoniser l’apparence
    css = """
    .generate-button { background: #2ecc71; }
    .save-button { background: #3498db; }
    .status-pending { color: #f39c12; }
    .status-success { color: #27ae60; }
    .status-error { color: #e74c3c; }
    .wrap .resizable-handle { display: none !important; }
    .wrap textarea { resize: none !important; }
    """
    with gr.Blocks(css=css) as demo:
        with gr.Tabs():
            with gr.Tab("Batch Captioning"):
                batch_container = gr.Column()
                build_batch_captioning_ui(batch_container)
            with gr.Tab("LoRA Inspector"):
                lora_container = gr.Column()
                build_lora_inspector_ui(lora_container)
    return demo


def main():
    demo = create_integrated_interface()
    demo.launch(inbrowser=True)

if __name__ == "__main__":
    main()
