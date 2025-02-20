# Florence-2 Batch Captioning Optimizer + Lora Inspector

This script is based on the work of [DenOfEquity](https://github.com/DenOfEquity) for his code in the [‘Spaces > Florence-2‘](https://github.com/lllyasviel/stable-diffusion-webui-forge/tree/main/extensions-builtin/forge_space_florence_2) part of [stable-diffusion-webui-forge](https://github.com/lllyasviel/stable-diffusion-webui-forge)

Florence-2 Batch Captioning Optimizer is a tool designed to generate, edit, and manage captions for an image dataset, primarily for training **LoRA models**.  

**My goal is to be able to have total control over your captions in a single application. Especially for the editing/supervision part**

![image](https://raw.githubusercontent.com/Fuhze73/Florence-2-Batch-Captioning-Optimizer/refs/heads/main/screen.png)

![image](https://raw.githubusercontent.com/Fuhze73/Florence-2-Batch-Captioning-Optimizer/refs/heads/main/screen2.png)

---

# 📌 Key Features

✔ Load and manage image datasets  
✔ Generate detailed captions  
✔ Edit and update captions in real-time  
✔ Edit each caption individually  
✔ Auto-save modifications  
✔ Search & replace tokens  
✔ Preview of captions with thumbnails  
✔ Optimized interface  
✔ Advanced options for refining captions's generation  
✔ Addition of a tab: ‘Lora Inspector’ (Useful if you need to watch captions from other LoRAs)

---

##  Installation

### 1️⃣ Clone the repository

Install Git, Python, Git Clone the forge repo `https://github.com/Fuhze73/Florence-2-Batch-Captioning-Optimizer.git` and then run run.bat.

## Install dependencies


pip install -r requirements.txt


## 🛠 Usage

### Load an image folder

Put the folder containing your images and click on "Load images" before generating or editing the captions.

### Modify or generate captions

- Select an image from the list.
- Edit or add a caption.
- Click "Update and Save Caption" to save changes.
- You can edit your past captions without having to generate them each time (just click "Load Images with captions (if already generated) ")



