import os
import warnings

import torch
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

# Suppress warnings
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
INPUT_FILE = "./book.pdf"
OUTPUT_DIR = "./out"
# ---------------------


def main():
    # 1. Force MPS (Metal Performance Shaders) for M4
    # The warning you saw about TableRecEncoderDecoderModel defaulting to CPU is NORMAL.
    # It just means table recognition runs on CPU, but text OCR will still use your GPU.
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🚀 Running on: {device.upper()}")

    # 2. Configure Models
    print("Loading AI models...")
    model_dict = create_model_dict()

    # 3. Configure Converter (THE FIX IS HERE)
    # We pass a simple dictionary, not a ConfigParser object.
    config = {
        "output_format": "markdown",
        "batch_multiplier": 2,  # Keep at 2 for stability
        "languages": "en",  # Force English for speed
    }

    # Initialize Converter with the dictionary directly
    converter = PdfConverter(
        artifact_dict=model_dict,
        config=config,
        processor_list=model_dict.get("processor_list"),  # Ensure processors are loaded
    )

    # 4. Run Conversion
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: {INPUT_FILE} not found.")
        return

    print(f"Processing '{INPUT_FILE}'... (This may take 1-2 mins)")
    rendered = converter(INPUT_FILE)

    # 5. Extract and Save
    full_text, _, images = text_from_rendered(rendered)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_filename = os.path.splitext(os.path.basename(INPUT_FILE))[0] + ".md"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"✅ Success! Saved to: {output_path}")


if __name__ == "__main__":
    main()
