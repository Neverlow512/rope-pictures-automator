# Save this as rope_automator2.py in your F:\Faceswap\Rope directory

import os
import cv2
import numpy as np
import json
import time
import torch # Ensure torch is available in the venv
import rope.Models as Models # Import Rope's Models class
import rope.VideoManager as VM # Import Rope's VideoManager class
from skimage import transform as trans # Needed for VideoManager/Models dependencies
from torchvision.transforms import v2 # Needed for VideoManager/Models dependencies
import traceback # For printing detailed errors

# --- Configuration ---
# Activate your Rope venv in the terminal before running this script!
# Example: cd F:\Faceswap\Rope && .\venv\Scripts\activate.bat && python rope_automator2.py

# --- Paths (Provided by you) ---
face_image_path = r"F:\Faceswap\Rope\Faces\chloe_face.jpg"
input_directory = r"F:\Faceswap\Rope\Pictures"
output_directory = r"F:\Faceswap\Rope\Output"
saved_parameters_path = r"F:\Faceswap\Rope\saved_parameters.json"

# --- Helper Functions ---
def print_separator():
    print("-" * 60)

def print_error(message):
    print(f"[ERROR] {message}")

def print_info(message):
    print(f"[INFO] {message}")

def print_success(message):
    print(f"[SUCCESS] {message}")

def print_warning(message):
    print(f"[WARNING] {message}")

# --- Main Automation Function ---
def automate_faceswap_headless(source_face_path, input_dir, output_dir, params_path):
    """
    Automates Rope faceswapping headlessly for images sequentially using its classes.
    MUST be run from an activated Rope virtual environment.
    Uses settings from the specified parameters JSON file.
    Only saves images where a swap was actually performed.
    """
    print_separator()
    print_info("Starting Rope Headless Faceswap Automation...")
    print_separator()

    # --- Path Validations ---
    paths_to_check = {
        "Source Face Image": source_face_path,
        "Input Directory": input_dir,
        "Parameters File": params_path,
    }
    valid_paths = True
    for name, path in paths_to_check.items():
        if not os.path.exists(path):
            print_error(f"{name} not found at: {path}")
            valid_paths = False
    if not valid_paths:
        print_error("One or more essential paths are invalid. Exiting.")
        return

    # --- Load Parameters ---
    try:
        with open(params_path, "r") as f:
            saved_params = json.load(f)
        print_info(f"Loaded parameters from: {params_path}")
    except Exception as e:
        print_error(f"Could not load or parse parameters file {params_path}: {e}")
        return

    # --- Ensure Output Directory Exists ---
    if not os.path.exists(output_dir):
        print_info(f"Output directory not found. Creating: {output_dir}")
        try:
            os.makedirs(output_dir)
        except OSError as e:
            print_error(f"Failed to create output directory {output_dir}: {e}")
            return

    # --- Initialize Rope Components ---
    models = None # Initialize to None for cleanup check
    vm = None     # Initialize to None for cleanup check
    try:
        print_info("Initializing Rope Models...")
        if 'CUDAExecutionProvider' in VM.onnxruntime.get_available_providers():
             print_info("CUDA available.")
        else:
             print_info("CUDA not available, using CPU (may be slow).")

        models = Models.Models()
        print_info("Initializing Rope VideoManager...")
        vm = VM.VideoManager(models)

        # Assign ALL loaded parameters to VideoManager
        vm.parameters = saved_params
        print_info("Assigned saved parameters to VideoManager.")

        # Set necessary Control flags for headless operation
        vm.control = {
            'SwapFacesButton': True,
            'MaskViewButton': False,
            'AudioButton': False,
        }
        print_info("Set control flags for headless operation.")
        print_info("Rope components initialized.")

    except Exception as e:
        print_error(f"Failed to initialize Rope components: {e}")
        print_error("Ensure you are running this script from an activated Rope virtual environment with all dependencies installed.")
        traceback.print_exc()
        return

    # --- Calculate Source Face Embedding ---
    try:
        print_info(f"Loading source face: {source_face_path}")
        source_img_bgr = cv2.imread(source_face_path)
        if source_img_bgr is None:
            raise ValueError(f"Failed to load source face image at {source_face_path}")

        source_img_rgb = cv2.cvtColor(source_img_bgr, cv2.COLOR_BGR2RGB)
        device = getattr(VM, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
        source_img_tensor = torch.from_numpy(source_img_rgb.astype(np.uint8)).to(device)
        source_img_tensor = source_img_tensor.permute(2, 0, 1) # CxHxW

        print_info("Detecting landmarks on source face...")
        detect_mode_param = vm.parameters.get('DetectTypeTextSel', 'Retinaface')
        detect_score_param = vm.parameters.get('DetectScoreSlider', 51) / 100.0

        kpss = models.run_detect(source_img_tensor, detect_mode=detect_mode_param, max_num=1, score=detect_score_param)

        # Check if detection failed
        if not isinstance(kpss, (list, np.ndarray)) or len(kpss) == 0:
            print_error("No face landmarks detected in the source image. Cannot proceed.")
            if models: models.delete_models()
            del models, vm
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            return

        print_info("Calculating embedding for source face...")
        source_embedding, _ = models.run_recognize(source_img_tensor, kpss[0])
        print_info("Source face embedding calculated.")

        # Prepare vm.found_faces structure
        vm.found_faces = [{
            "Embedding": source_embedding,
            "SourceFaceAssignments": [0],
            "AssignedEmbedding": source_embedding,
        }]
        print_info("Prepared VideoManager with source embedding.")

    except Exception as e:
        print_error(f"Failed during source face processing: {e}")
        traceback.print_exc()
        if models: models.delete_models()
        del models, vm
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return

    # --- Process Target Images ---
    print_separator()
    print_info(f"Processing images from: {input_dir}")
    print_info(f"Saving results to: {output_dir}")
    print_separator()

    processed_count = 0
    skipped_swap_count = 0 # Count images where no swap occurred
    error_count = 0
    skipped_file_count = 0 # Count non-image files

    try:
        files_in_input = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    except FileNotFoundError:
        print_error(f"Input directory not found: {input_dir}")
        files_in_input = []
    except Exception as e:
        print_error(f"Error listing files in input directory {input_dir}: {e}")
        files_in_input = []

    if not files_in_input:
        print_info("No files found to process in the input directory.")
    else:
        for filename in files_in_input:
            start_time = time.time()
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename) # Save with same name

            # Check for valid image extensions
            if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")):
                skipped_file_count += 1
                continue

            print_info(f"Processing: {filename}...")

            try:
                target_img_bgr = cv2.imread(input_path)
                if target_img_bgr is None:
                    raise ValueError(f"Could not read target image file: {input_path}")

                target_img_rgb_np = cv2.cvtColor(target_img_bgr, cv2.COLOR_BGR2RGB)

                # --- <<< PROCESSING HAPPENS HERE (SYNCHRONOUSLY) >>> ---
                # The script waits for this function to complete before continuing.
                result_img_rgb_np = vm.swap_video(target_img_rgb_np, 0, False)
                # --- <<< PROCESSING COMPLETE FOR THIS IMAGE >>> ---

                # --- Check if the image was actually modified ---
                if result_img_rgb_np is not None and result_img_rgb_np.size > 0:
                    # Compare input and output pixel data
                    if np.array_equal(target_img_rgb_np, result_img_rgb_np):
                        # Images are identical - no swap occurred (e.g., no face found, threshold not met)
                        print_warning(f"No swap performed on '{filename}' (check detection/threshold). Image not saved.")
                        skipped_swap_count += 1
                    else:
                        # Images are different - swap occurred, save the result
                        result_img_bgr_np = cv2.cvtColor(result_img_rgb_np, cv2.COLOR_RGB2BGR)
                        if not cv2.imwrite(output_path, result_img_bgr_np):
                             raise IOError(f"Failed to write output image to {output_path}")
                        end_time = time.time()
                        duration = end_time - start_time
                        print_success(f"Successfully processed and swapped '{filename}' in {duration:.2f} seconds. Output saved.")
                        processed_count += 1
                else:
                    # Handle cases where swap_video might return None or empty
                    raise ValueError("swap_video did not return a valid image.")

            # Catch specific and general exceptions during processing
            except (ValueError, IOError, cv2.error) as img_err:
                 end_time = time.time()
                 duration = end_time - start_time
                 print_error(f"Image processing error for '{filename}' after {duration:.2f}s: {img_err}")
                 error_count += 1
            except Exception as e:
                 end_time = time.time()
                 duration = end_time - start_time
                 print_error(f"Unexpected error processing '{filename}' after {duration:.2f} seconds: {e}")
                 traceback.print_exc()
                 error_count += 1

            # print_separator() # Optional: uncomment for separator between each file log

    # --- Final Summary ---
    print_separator()
    print_info("Batch processing complete.")
    print_success(f"Successfully processed and saved (swap performed): {processed_count} files")
    if skipped_swap_count > 0:
        print_warning(f"Skipped saving (no swap detected/performed): {skipped_swap_count} files")
    if error_count > 0:
        print_error(f"Errors occurred during processing: {error_count} files")
    if skipped_file_count > 0:
        print_info(f"Skipped non-image files: {skipped_file_count}")
    print_separator()

    # --- Cleanup ---
    print_info("Cleaning up Rope models and resources...")
    try:
        if models:
            models.delete_models()
        del models
        del vm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print_info("Cleanup complete.")
    except Exception as e:
        print_error(f"Error during cleanup: {e}")


# --- Run the automation ---
if __name__ == "__main__":
    automate_faceswap_headless(
        face_image_path,
        input_directory,
        output_directory,
        saved_parameters_path
    )