Features

    📂 Automatically loads images from a selected folder.
    🤖 Batch caption generation using Florence-2.
    ✏️ Manual caption editing before saving.
    💾 Auto-saving captions in corresponding .txt files.
    🎯 Interactive file selection via a Gradio-based GUI.

🛠 Installation & Execution
1️⃣ Prerequisites

    Python 3.8+ installed on your system.
    Pip (Python package manager).

2️⃣ Install dependencies

Run the following command in your terminal (or CMD):

pip install -r requirements.txt

3️⃣ Run the application

If using the provided .bat file, simply double-click run.bat.
Otherwise, run manually in the terminal:

cd webui
python batch_caption.py

The interface will automatically open in your browser.

📷 How to Use

    Select the folder containing the images.
    Start the automatic caption generation.
    Edit captions if needed.
    Save the captions as .txt files.

📢 Notes

    This script runs locally and does not send any data to external servers.
    A good GPU setup is recommended for faster inference.
    Tested with Gradio 4.40.0 and Torch 1.12.1+.

📜 License

This project is released under the MIT License. Feel free to use, modify, and distribute it. 🚀
