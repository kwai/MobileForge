"""
在项目的根目录运行 python -m utils.embedding_pipeline 即可启动服务，然后在SiglipMultimodalEmbeddingPipeline的构造函数的device填写"server"。
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from PIL import Image
import gradio as gr
from transformers import AutoProcessor, AutoModel
from typing import Any, List, Union, Dict, Tuple
import numpy as np
from gradio_client import Client
import io
import base64
from utils.utils import resize_pil_image
MAX_BATCH_SIZE = 32  # 最大批处理大小，根据显存大小调整，50张最大512像素的图片大约需要2.5GB
MAX_IMAGE_SIZE = 512  # 最大图片大小，512x512，超过这个大小会自动保留比例缩放到最长边为512


class SiglipMultimodalEmbeddingPipeline:
    IMAGE_PREFIX = "<is_image>"
    SEPARATOR = "<sep>"

    def __init__(
        self,
        model_id: str = "google/siglip-so400m-patch14-384",
        device: str = "cuda",  # "cuda", "mps", "cpu", or "server"
        server_endpoint: str = "http://127.0.0.1:8765",
    ):
        self.requested_device_str = device # Store original request
        self.model_id = model_id
        self.client = None
        self.model = None
        self.processor = None
        self.device = None # This will be a torch.device object or "server"
        self.dtype = torch.float32 # Default dtype
        self.autocast_device_type = None # 'cuda', 'cpu', 'mps' for torch.autocast

        if self.requested_device_str == "server":
            self.device = "server"
            try:
                self.client = Client(server_endpoint, verbose=False)
                #print(f"Initialized Gradio client for server: {server_endpoint}")
            except Exception as e:
                print(f"Failed to initialize Gradio client: {e}")
                raise
        else:
            # Determine actual device and dtype
            resolved_device_str = self.requested_device_str
            
            if resolved_device_str == "cuda":
                if torch.cuda.is_available():
                    self.device = torch.device("cuda")
                    self.autocast_device_type = "cuda"
                    if torch.cuda.is_bf16_supported():
                        self.dtype = torch.bfloat16
                        print(f"Using CUDA with bfloat16 for model {model_id}.")
                    else:
                        self.dtype = torch.float16 # Most CUDA GPUs support at least float16
                        print(f"Using CUDA with float16 for model {model_id} (bfloat16 not supported).")
                else:
                    print(f"Warning: CUDA device requested for model {model_id} but not available. Falling back to CPU.")
                    resolved_device_str = "cpu"

            if resolved_device_str == "mps":
                if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                    self.device = torch.device("mps")
                    self.autocast_device_type = "mps"
                    self.dtype = torch.float16 # MPS typically uses float16 well
                    print(f"Using MPS with float16 for model {model_id}.")
                else:
                    print(f"Warning: MPS device requested for model {model_id} but not available/built. Falling back to CPU.")
                    resolved_device_str = "cpu"
            
            if resolved_device_str == "cpu": # Covers explicit "cpu" or fallbacks
                self.device = torch.device("cpu")
                self.autocast_device_type = "cpu"
                # For CPU, bfloat16 can offer benefits if supported, float16 is generally not preferred for computation.
                # Check if torch.bfloat16 dtype exists (available in PyTorch 1.10+)
                # and if CPU has specific bfloat16 acceleration (more recent PyTorch for torch.backends.cpu.is_bf16_supported())
                if hasattr(torch, 'bfloat16') and \
                   (getattr(torch.backends.cpu, 'is_bf16_supported', lambda: False)()):
                    self.dtype = torch.bfloat16
                    print(f"Using CPU with bfloat16 for model {model_id} (native support).")
                # elif hasattr(torch, 'bfloat16'): # If bfloat16 exists but no specific CPU hardware support, it might be emulated.
                #     self.dtype = torch.bfloat16
                #     print(f"Using CPU with bfloat16 for model {model_id} (may be emulated).")
                else:
                    self.dtype = torch.float32 # Default, remains float32
                    print(f"Using CPU with float32 for model {model_id}.")

            try:
                self.processor = AutoProcessor.from_pretrained(model_id)
                # Load model and cast to the determined device and dtype
                self.model = AutoModel.from_pretrained(model_id) \
                                     .to(device=self.device, dtype=self.dtype) \
                                     .eval()
                print(f"Model {model_id} loaded successfully on {self.device} with dtype {self.dtype}.")
            except Exception as e:
                print(f"Error loading model {model_id} on {self.device} with {self.dtype}: {e}")
                # Potentially fall back further or raise
                if self.dtype != torch.float32:
                    print(f"Attempting to load model {model_id} with float32 as a fallback...")
                    self.dtype = torch.float32
                    try:
                        self.model = AutoModel.from_pretrained(model_id) \
                                             .to(device=self.device, dtype=self.dtype) \
                                             .eval()
                        print(f"Model {model_id} loaded successfully on {self.device} with fallback dtype {self.dtype}.")
                    except Exception as e2:
                        print(f"Fallback loading also failed: {e2}")
                        raise e2 from e
                else:
                    raise

    def __read_image_to_base64(self, image_path: str) -> str:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()

    def __pillow_image_to_base64(self, pillow_image: Image.Image) -> str:
        img_byte_arr = io.BytesIO()
        if pillow_image.size[0] > MAX_IMAGE_SIZE or pillow_image.size[1] > MAX_IMAGE_SIZE:
            pillow_image = resize_pil_image(pillow_image, MAX_IMAGE_SIZE)
        pillow_image.convert("RGB").save(img_byte_arr, format="WEBP", quality=95)
        return base64.b64encode(img_byte_arr.getvalue()).decode()

    def embedding_images_or_texts(
        self, images_or_texts_str: str
    ) -> Union[List[float], List[List[float]]]:
        processed_input: Union[Image.Image, List[Image.Image], str, List[str]]
        is_single_input = self.SEPARATOR not in images_or_texts_str

        if images_or_texts_str.startswith(self.IMAGE_PREFIX):
            content_str = images_or_texts_str.removeprefix(self.IMAGE_PREFIX)
            if is_single_input:
                processed_input = Image.open(
                    io.BytesIO(base64.b64decode(content_str))
                ).convert("RGB")
            else:
                processed_input = [
                    Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")
                    for img_b64 in content_str.split(self.SEPARATOR)
                    if img_b64
                ]
        else: # Text input
            if is_single_input:
                processed_input = images_or_texts_str
            else:
                processed_input = [
                    text for text in images_or_texts_str.split(self.SEPARATOR)
                    if text
                ]
        return self.__call__(processed_input)

    def __call__(
        self, images_or_texts: Union[Image.Image, List[Image.Image], str, List[str], List[np.ndarray]]
    ) -> Union[List[float], List[List[float]]]:
        try:
            if self.device == "server":
                if self.client is None:
                    raise ConnectionError("Client not initialized. Set device='server' and provide server_endpoint.")
                return self.__call__server(images_or_texts)
            else:
                if self.model is None or self.processor is None:
                    raise RuntimeError("Model or processor not initialized. Check __init__ errors.")
                return self.__call__local(images_or_texts)
        except Exception as e:
            print(f"Error during embedding: {e}")
            if isinstance(images_or_texts, list) and len(images_or_texts) > 0:
                print(f"Problematic input type: {type(images_or_texts[0])}, count: {len(images_or_texts)}")
            else:
                print(f"Problematic input type: {type(images_or_texts)}")
            print(f"Input snippet: {str(images_or_texts)[:200]}")
            raise e

    def __call__server(
        self, images_or_texts: Union[Image.Image, List[Image.Image], str, List[str], List[np.ndarray]]
    ) -> Union[List[float], List[List[float]]]:
        serialized_input: str

        if isinstance(images_or_texts, str):
            if os.path.exists(images_or_texts):
                serialized_input = self.IMAGE_PREFIX + self.__read_image_to_base64(images_or_texts)
            else:
                serialized_input = images_or_texts
        elif isinstance(images_or_texts, Image.Image):
            serialized_input = self.IMAGE_PREFIX + self.__pillow_image_to_base64(images_or_texts)
        elif isinstance(images_or_texts, list):
            if not images_or_texts:
                return []

            first_item = images_or_texts[0]
            if isinstance(first_item, str):
                # Assuming paths if os.path.exists on first_item, else treat all as text
                # This might be ambiguous if mixing paths and direct texts.
                # For robustness, client should ideally send pre-processed lists of one type.
                try:
                    is_path = os.path.exists(first_item)
                except TypeError: # os.path.exists can fail on non-string/bytes-like objects
                    is_path = False

                if is_path: # Assume list of image paths
                    b64_images = [self.__read_image_to_base64(p) for p in images_or_texts if isinstance(p, str) and os.path.exists(p)]
                    serialized_input = self.IMAGE_PREFIX + self.SEPARATOR.join(b64_images) + self.SEPARATOR
                else: # Assume list of texts
                    valid_texts = [t for t in images_or_texts if isinstance(t, str)]
                    serialized_input = self.SEPARATOR.join(valid_texts) + self.SEPARATOR
            elif isinstance(first_item, Image.Image):
                b64_images = [self.__pillow_image_to_base64(img) for img in images_or_texts if isinstance(img, Image.Image)]
                serialized_input = self.IMAGE_PREFIX + self.SEPARATOR.join(b64_images) + self.SEPARATOR
            elif isinstance(first_item, np.ndarray):
                pil_images = [Image.fromarray(img_arr).convert("RGB") for img_arr in images_or_texts if isinstance(img_arr, np.ndarray)]
                b64_images = [self.__pillow_image_to_base64(img) for img in pil_images]
                serialized_input = self.IMAGE_PREFIX + self.SEPARATOR.join(b64_images) + self.SEPARATOR
            else:
                raise ValueError(f"Unsupported list item type for server call: {type(first_item)}")
        else:
            raise ValueError(f"Unsupported input type for server call: {type(images_or_texts)}")

        rsp = self.client.predict(
            serialized_input,
            api_name="/embedding_images_or_texts",
        )
        return rsp

    @torch.no_grad()
    def __call__local(
        self, images_or_texts: Union[Image.Image, List[Image.Image], str, List[str], List[np.ndarray]]
    ) -> Union[List[float], List[List[float]]]:
        
        input_was_singular = False
        processed_input: Union[List[Image.Image], List[str]]

        if isinstance(images_or_texts, str):
            input_was_singular = True
            if os.path.exists(images_or_texts):
                img=Image.open(images_or_texts).convert("RGB")
                processed_input = [img]
            else:
                processed_input = [images_or_texts]
        elif isinstance(images_or_texts, Image.Image):
            input_was_singular = True
            processed_input = [images_or_texts.convert("RGB")]
        elif isinstance(images_or_texts, list):
            if not images_or_texts:
                return []
            
            first_item = images_or_texts[0]
            if isinstance(first_item, str):
                try:
                    is_path = os.path.exists(first_item)
                except TypeError:
                    is_path = False
                if is_path:
                    processed_input = [Image.open(p).convert("RGB") for p in images_or_texts if isinstance(p, str)]
                else:
                    processed_input = [t for t in images_or_texts if isinstance(t, str)]
            elif isinstance(first_item, Image.Image):
                processed_input = [img.convert("RGB") for img in images_or_texts if isinstance(img, Image.Image)]
            elif isinstance(first_item, np.ndarray):
                processed_input = [Image.fromarray(img_arr).convert("RGB") for img_arr in images_or_texts if isinstance(img_arr, np.ndarray)]
            else:
                raise ValueError(f"Unsupported list item type for local call: {type(first_item)}")
        else:
            raise ValueError(f"Unsupported input type for local call: {type(images_or_texts)}")

        if not processed_input: # If filtering removed all items
            return [] if not input_was_singular else [[]] # Match output shape expectation

        is_text_input = isinstance(processed_input[0], str)
        all_embeddings = []
        
        # Use autocast if not using float32
        autocast_enabled = self.dtype != torch.float32

        for i in range(0, len(processed_input), MAX_BATCH_SIZE):
            batch = processed_input[i : i + MAX_BATCH_SIZE]
            features_tensor = None # Initialize

            if is_text_input:
                # Processor returns tensors on CPU by default
                inputs = self.processor(
                    text=batch, padding="max_length", truncation=True, return_tensors="pt"
                )
                # Move inputs to the correct device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.autocast(device_type=self.autocast_device_type, dtype=self.dtype, enabled=autocast_enabled):
                    features_tensor = self.model.get_text_features(**inputs)
            else: # Image input
                # Ensure all images in batch are PIL Images and resized if necessary
                pil_batch = []
                for img_data in batch:
                    current_img = img_data # Already PIL Image from pre-processing
                    if current_img.size[0] > MAX_IMAGE_SIZE or current_img.size[1] > MAX_IMAGE_SIZE:
                       current_img = resize_pil_image(current_img, MAX_IMAGE_SIZE)
                    pil_batch.append(current_img)

                inputs = self.processor(images=pil_batch, return_tensors="pt")
                # Move inputs to the correct device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.autocast(device_type=self.autocast_device_type, dtype=self.dtype, enabled=autocast_enabled):
                    features_tensor = self.model.get_image_features(**inputs)
            
            if features_tensor is not None:
                # Move features to CPU and convert to list
                all_embeddings.extend(features_tensor.cpu().float().tolist()) # .float() before tolist for consistency if original was half

        return all_embeddings[0] if input_was_singular and all_embeddings else all_embeddings

if __name__ == "__main__":
    os.environ["no_proxy"] = "localhost,127.0.0.1,::1" # Removed /8 as it's usually not needed for local
    print("Text and Image Embedding Service")

    # Determine device for the server hosting the model
    server_device = os.getenv("SERVER_EMBEDDING_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {server_device}")

    print("Loading Multimodal Embedding Pipeline...")
    # This pipeline instance will run locally on the server
    multimodal_pipeline = SiglipMultimodalEmbeddingPipeline(
        model_id="google/siglip-so400m-patch14-384", # or "marco/mcdse-2b-v1"
        device=server_device,
    )
    print("Multimodal Embedding Pipeline Loaded.")

    # --- Gradio UI Helper Functions ---
    def embed_texts_gradio(text_input_str: str):
        if not text_input_str.strip():
            return [], {"error": "Input is empty"}
        texts = [t.strip() for t in text_input_str.splitlines() if t.strip()]
        if not texts:
            return [], {"error": "No valid text lines found"}
        
        # Prepare string for embedding_images_or_texts method
        # This simulates what __call__server would do if it were calling this server
        # If texts is a list: "text1<sep>text2<sep>"
        # If texts is single: "text1"
        if len(texts) == 1:
            api_input_str = texts[0]
        else:
            api_input_str = SiglipMultimodalEmbeddingPipeline.SEPARATOR.join(texts) + SiglipMultimodalEmbeddingPipeline.SEPARATOR
        
        embeddings = multimodal_pipeline.embedding_images_or_texts(api_input_str)
        shape = np.array(embeddings).shape
        return embeddings, {"shape": str(shape)}

    def embed_single_image_gradio(image_input: Image.Image): # Gradio gives PIL Image if type="pil"
        if image_input is None:
            return [], {"error": "No image uploaded"}
        
        # Prepare string for embedding_images_or_texts method
        # This simulates what __call__server would do
        b64_image = multimodal_pipeline._SiglipMultimodalEmbeddingPipeline__pillow_image_to_base64(image_input)
        api_input_str = SiglipMultimodalEmbeddingPipeline.IMAGE_PREFIX + b64_image
        
        embeddings = multimodal_pipeline.embedding_images_or_texts(api_input_str)
        shape = np.array(embeddings).shape
        return embeddings, {"shape": str(shape)}

    def embed_multiple_images_gradio(image_files: List[str]): # Gradio gives list of filepaths if type="filepath"
        if not image_files:
            return [], {"error": "No images uploaded"}
        
        # Prepare string for embedding_images_or_texts method
        b64_images = [multimodal_pipeline._SiglipMultimodalEmbeddingPipeline__read_image_to_base64(f.name) for f in image_files]
        api_input_str = (
            SiglipMultimodalEmbeddingPipeline.IMAGE_PREFIX + 
            SiglipMultimodalEmbeddingPipeline.SEPARATOR.join(b64_images) + 
            SiglipMultimodalEmbeddingPipeline.SEPARATOR
        )
        
        embeddings = multimodal_pipeline.embedding_images_or_texts(api_input_str)
        shape = np.array(embeddings).shape
        return embeddings, {"shape": str(shape)}

    print("Launching Gradio...")
    with gr.Blocks() as demo:
        gr.Markdown("### Text and Image Embedding Service")

        with gr.Tab("Text Embedding"):
            text_input = gr.Textbox(
                lines=7, label="Input Text(s)", info="Enter one text per line for multiple texts."
            )
            with gr.Row():
                text_button = gr.Button("Generate Text Embedding(s)")
            text_output_json = gr.JSON(label="Embedding Vector(s)")
            text_output_shape = gr.JSON(label="Output Shape")
            text_button.click(
                embed_texts_gradio,
                inputs=text_input,
                outputs=[text_output_json, text_output_shape],
                api_name="embed_texts" # For client.predict
            )
        
        with gr.Tab("Single Image Embedding"):
            # For single image, type="pil" is convenient to get PIL.Image directly
            single_image_input = gr.Image(type="pil", label="Upload Single Image")
            with gr.Row():
                single_image_button = gr.Button("Generate Image Embedding")
            single_image_output_json = gr.JSON(label="Embedding Vector")
            single_image_output_shape = gr.JSON(label="Output Shape")
            single_image_button.click(
                embed_single_image_gradio,
                inputs=single_image_input,
                outputs=[single_image_output_json, single_image_output_shape],
                api_name="embed_single_image"
            )

        with gr.Tab("Multiple Images Embedding"):
            # For multiple images, type="filepath" or type="bytes"
            # Using file_count="multiple" with gr.File
            multi_image_input = gr.File(
                label="Upload Multiple Images",
                file_count="multiple",
                file_types=["image"], # e.g., .png, .jpg, .jpeg, .webp
                type="filepath" # Gives list of tempfile._TemporaryFileWrapper objects, use .name for path
            )
            with gr.Row():
                multi_image_button = gr.Button("Generate Image Embeddings")
            multi_image_output_json = gr.JSON(label="Embedding Vectors")
            multi_image_output_shape = gr.JSON(label="Output Shape")
            multi_image_button.click(
                embed_multiple_images_gradio,
                inputs=multi_image_input,
                outputs=[multi_image_output_json, multi_image_output_shape],
                api_name="embed_multiple_images"
            )
        
        # This is the original endpoint that __call__server targets
        # It's kept for compatibility if you have clients using it directly
        # but the UI above uses more specific helper functions for clarity.
        gr.Textbox(
            label="Internal API input (for client calls)", 
            visible=False # Hide from UI unless debugging
        ).submit(
            multimodal_pipeline.embedding_images_or_texts,
            inputs=gr.Textbox(), # This needs to be a Textbox to accept the string
            outputs=gr.JSON(), # Output should be JSON for vectors
            api_name="embedding_images_or_texts" # This matches the client
        )


    demo.queue().launch(share=False, server_name="0.0.0.0", server_port=8765)