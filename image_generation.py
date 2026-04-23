import argparse
import base64
from google import genai
from google.genai import types
import pickle
from tqdm import tqdm
import random
import time

from rule_selection import *
from parse_predictions import parse_generation_prediction, generation_preds_to_csv, download_locally
from run_batch_job import monitor_batch_prediction_job, run_batch_prediction_job

MAX_WORKERS = 5  # adjust based on rate limits

SYSTEM_INSTRUCTIONS = """
You are a professional food product image generator.
Your primary goal is to create accurate, high-quality, e-commerce-ready food photographs based on provided product data and image instructions.
"""
# You must always adhere to the following core principles:
# 1. The structured product data provided (title, description, attributes) is factual ground truth and must be accurately represented in the image.
# 2. For every product, you must:
#     - Identify its protein category.
#     - Apply species-specific anatomical accuracy.
#     - Enforce correct bone, skin, trim, and portion structure corresponding to the product description and category.
# 3. Depict the product fully cooked and professionally plated as part of a complete main dish.
# 4. Include any side components on the same plate as the main dish.
# 5. The entire dish must be fully visible in frame, with no cropping at the edges.
# 6. Place the dish in a natural, realistic lifestyle setting. Avoid studio-only or abstract backgrounds.
# 7. Prioritize realistic and appetizing aesthetics in all generated images. Ensure the image is realistic and it does not look fake.
# 8. No humans or human hands visible. No labels, writing or letters.
# """

CONTENT_CONFIG_REF = {
    "temperature": 0.4,
    "topP": 0.8,
    "maxOutputTokens": 32768,
    "responseModalities": ["IMAGE"],
}

CONTENT_CONFIG_RAW = {
    "temperature": 0.4,
    "topP": 0.8,
    "maxOutputTokens": 32768,
    "responseModalities": ["IMAGE"],
}

IMAGE_SETTINGS = {
    "aspect_ratio": "5:4",
    "resolution": "2K"
}


def compose_prompts(product_dict, rule_selection=True):
    for product_id, product_info in tqdm(product_dict.items()):
        product_title = product_info['product_title']
        product_description = product_info['product_description']
        product_category = product_info['product_category']
        product_attributes = product_info['product_attributes']
        cos_sim = product_info['generation']['reference_similarity']

        relevant_rules = None
        if 'relevant_rules' in product_info['generation'].keys():
            if isinstance(product_info['generation']['relevant_rules'], str):
                relevant_rules = product_info['generation']['relevant_rules']

        if rule_selection and not relevant_rules:
            try:
                relevant_rules = select_relevant_rules(product_id, product_info)
            except Exception as e:
                print("Failed to select relevant rules for product {}".format(product_id))
                print(e)
                relevant_rules = PROTEIN_RULES[product_category]

        product_attributes_str = ""
        for attr in product_attributes:
            product_attributes_str += f"- {list(attr.keys())[0]}: {list(attr.values())[0]}\n"

        if isinstance(cos_sim, float) and cos_sim >= DATA['COS_SIM_THRESH']:
            prompt = f"""
                Task: Generate an accurate, high-quality, e-commerce-ready food photograph based on provided 
                product data, accuracy rules and image instructions.
                The food item should be in a cooked state and a styled setting.
                The product information is factual and should be treated as ground truth when generating the image.
                
                **PRODUCT INFORMATION**
                Product ID: {product_id}
                Product Title: "{product_title}".
                Product Description: "{product_description}".
                Product Attributes: {product_attributes_str}
                
                **ACCURACY RULES**
                The final image must depict the product exactly as described, respecting the following:
                {relevant_rules}

                **IMAGE INSTRUCTIONS**
                1. Using the provided image as a reference:
                - Use the provided image strictly as a reference to the protein's physical characteristics 
                (cut, shape, thickness, skin/bone presence, marbling, and muscle structure).  
                - If the reference image depicts a raw product, ensure that the generated image depicts the same product in a cooked state.
                - CRITICAL: Do not copy the background, props, lighting, or plating. Use a new lifestyle setting.
                
                2. Composition Requirements:
                - The product must be the primary visual focus, clearly visible and unobstructed.
                - The entire dish must be fully visible in frame, with no cropping at the edges.
                - The image must look like a real professional food photograph, with natural lighting, 
                realistic textures, and accurate colors. 
                - Avoid any artificial or computer-generated appearance.
                - No humans or human hands visible, nor labels, writing or letters.
            """

        else:
            prompt = f"""
                Task: Generate an accurate, high-quality, e-commerce-ready food photograph based on provided
                product data, accuracy rules and image instructions.
                The food item should be in a cooked state and a styled setting.
                The product information is factual and should be treated as ground truth when generating the image.

                **PRODUCT INFORMATION**
                Product ID: {product_id}##
                Product Title: "{product_title}".
                Product Description: "{product_description}".
                Product Attributes: {product_attributes_str}

                **ACCURACY RULES**
                The final image must depict the product exactly as described, respecting the following:
                {relevant_rules}

                **IMAGE INSTRUCTIONS**
                Composition Requirements:
                - The product must be the primary visual focus, clearly visible and unobstructed.
                - The entire dish must be fully visible in frame, with no cropping at the edges.
                - The image must look like a real professional food photograph, with natural lighting,
                realistic textures, and accurate colors.
                - Avoid any artificial or computer-generated appearance.
                - No humans or human hands visible, nor labels, writing or letters.
            """
        product_dict[product_id]['generation']['prompt'] = prompt
        product_dict[product_id]['generation']['accuracy_rules'] = relevant_rules
    return product_dict


def generate_batch_requests_for_generation(product_dict, n_images_per_product=5):
    lines = []
    for pid, prod_info in tqdm(product_dict.items()):

        cos_sim = prod_info['generation']['reference_similarity']
        prompt = prod_info['generation']['prompt']

        if isinstance(cos_sim, float) and cos_sim >= DATA['COS_SIM_THRESH']:
            image_file = prod_info['generation']['reference_image']

            if image_file.lower().endswith('png'):
                mime_type = 'image/png'
            else:
                mime_type = 'image/jpeg'

            if not image_file.startswith('gs://'):
                with open(image_file, "rb") as f:
                    image_file = base64.b64encode(f.read()).decode("utf-8")
            line = {"request":

                {"systemInstruction": {
                    "role": "system",
                    "parts": [
                        {"text": SYSTEM_INSTRUCTIONS}
                    ]
                },
                    "contents": [
                        {"role": "user",
                         "parts": [
                             {"text": prompt},
                             # {"inlineData": {"data": image_str, "mimeType": "image/jpeg"}}
                             {
                                 "file_data": {
                                     "file_uri": image_file,
                                     "mime_type": mime_type
                                 }
                             }

                         ]
                         }
                    ],
                    "generationConfig": CONTENT_CONFIG_REF}}

        else:
            line = {"request": {

                "systemInstruction": {
                    "role": "system",
                    "parts": [
                        {"text": SYSTEM_INSTRUCTIONS}
                    ]
                },
                "contents": [
                    {"role": "user",
                     "parts": [
                         {"text": prompt}
                     ]
                     }
                ],

                "generationConfig": CONTENT_CONFIG_RAW}}

        # Append line n times to generate multiple images for same product.
        for i in range(n_images_per_product):
            lines.append(line)
    return lines


def generate_contents(product_dict, images_per_product):
    contents = []
    for prod_id, prod_info in product_dict.items():
        prompt = prod_info['generation']['prompt']
        if "Using the provided image as a reference:" in prompt:
            config = CONTENT_CONFIG_REF
        else:
            config = CONTENT_CONFIG_RAW

        content_dict = {
                        'sys_instr': SYSTEM_INSTRUCTIONS,
                        'config': config,
                        'prompt': prompt,
                        'prod_id': prod_id,
                        'aspect_ratio': IMAGE_SETTINGS["aspect_ratio"],
                        'resolution': IMAGE_SETTINGS["resolution"],
                        }
        for i in range(images_per_product):
            contents.append(content_dict)

    return contents


def generate_image(content_dict, model):
    sys_instr = content_dict['sys_instr']
    config = content_dict['config']
    print(config)
    prompt = content_dict['prompt']
    aspect_ratio = content_dict['aspect_ratio']
    resolution = content_dict['resolution']

    client = genai.Client(
        vertexai=True,
        project=GCS["PROJECT_ID"],
        location="global"
    )

    start = time.time()
    min_delay = 1.0
    for attempt in range(5):  # retry logic
        try:

            response = client.models.generate_content(
                model=model,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    system_instruction=(
                        sys_instr
                    ),
                    response_modalities=["IMAGE"],
                    temperature=0.4,
                    topP=0.8,
                    max_output_tokens=32768,
                    image_config=types.ImageConfig(
                        aspect_ratio=aspect_ratio,
                        image_size=resolution
                    ),
                )
            )
            break

        except Exception as e:
            wait = (2 ** attempt) + random.uniform(0, 1)
            print(f"Retrying in {wait:.2f}s...")
            print(e)
            time.sleep(wait)

    else:
        raise Exception("Max retries exceeded")

    # Throttling: enforce minimum delay between calls
    elapsed = time.time() - start
    if elapsed < min_delay:
        time.sleep(min_delay - elapsed)
    return response


def main():
    # Set up argument parser similar to gdsn_extraction.py but with added batch options
    parser = argparse.ArgumentParser(
        description='Generate lifestyle images for selected products using reference images.')

    # Data source arguments - only required if not processing output only
    parser.add_argument('--batch_job', type=str, default='test',
                        help='Name of batch job')
    parser.add_argument('--job_type', type=str, default='batch', choices=["batch", "online"],
                        help='Job type: "batch" or "online"')
    parser.add_argument('--product_ids', type=str,
                        help='Product IDs separated by commas. E.g. "1234567,7654321,0987654')
    parser.add_argument('--product_dict', type=str, default=None,
                        help='Product data dict with reference images')
    parser.add_argument('--gcs_input_file', type=str, default=None,
                        help='Use existing GCS input file')
    parser.add_argument('--model', type=str, default="NANO_BANANA", choices=["NANO_BANANA", "NANO_BANANA_PRO", "NANO_BANANA_2"],
                        help='Model for image generation. Choose between "NANO_BANANA", "NANO_BANANA_PRO", and "NANO_BANANA_2"')
    parser.add_argument('--images_per_product', type=int, default=5,
                        help='Number of images to generate')
    parser.add_argument('--download_images', type=str, default='False',
                        help="Download the images")

    args = parser.parse_args()
    batch_job = args.batch_job
    job_type = args.job_type
    os.makedirs(batch_job, exist_ok=True)

    print("Loading product information...")
    if args.product_dict:
        with open(args.product_dict, "rb") as f:
            product_dict = pickle.load(f)
    else:
        if os.path.exists(f"{batch_job}/product_dict.pkl"):
            with open(f"{batch_job}/product_dict.pkl", "rb") as f:
                product_dict = pickle.load(f)
        elif os.path.exists("default/product_dict.pkl"):
            with open("default/product_dict.pkl", "rb") as f:
                product_dict = pickle.load(f)
        else:
            raise ValueError("No product information provided.")

    if args.product_ids:
        product_ids = args.product_ids.split(',')
        if all(item in product_dict.keys() for item in product_ids):
            product_dict = {k: product_dict[k] for k in product_dict if k in product_ids}
        else:
            missing = [item for item in product_ids if item not in product_dict.keys()]
            print(f"Invalid product IDs: {missing}. Images will not be generated for these products.")
    print(f"Collected {len(product_dict)} products.\n")

    # Create input file
    print("\nGenerating input data for generation...")
    product_dict = dict(list(product_dict.items()))
    product_dict = compose_prompts(product_dict)
    save_product_dict(product_dict, f"{batch_job}/product_dict_generation.pkl")

    # Generate Content
    if job_type == 'batch':
        if not args.gcs_input_file:
            input_data = generate_batch_requests_for_generation(product_dict, args.images_per_product)
            # Write batch input file
            input_path = f"./{batch_job}/input_file_{batch_job}_generation.jsonl"
            write_batch_input_file(input_data, input_path)
            print("\nUploading input file to GCS...")
            input_file_uri = upload_input_file_to_gcs(input_path, bucket_name=GCS['BUCKET'])
        else:
            print(f"Using existing GCS input file: {args.gcs_input_file}")
            input_path = args.gcs_input_file
            input_data = []
            with open(input_path, "r") as f:
                for line in f:
                    input_data.append(json.loads(line))
            print("\nUploading input file to GCS...")
            input_file_uri = upload_input_file_to_gcs(input_path, bucket_name=GCS['BUCKET'])

        if len(input_data) == 0:
            raise ValueError("Could not generate input data for generation.")
        else:
            print(f"Generated requests for {len(input_data)} products.")

        if 'pro' in args.model.lower() or '2' in args.model.lower():
            raise ValueError("Invalid model name. Selected model is not available for batch jobs.")
        else:
            try:
                job = run_batch_prediction_job(
                    input_file=input_file_uri,
                    output_uri=GCS["GCS_OUTPUT_URI"],
                    model_id=LLM_MODELS["NANO_BANANA"],
                    project_id=GCS["PROJECT_ID"],
                    location=GCS["LOCATION"],
                )

                # Monitor the batch prediction job
                img_gen_output_location = monitor_batch_prediction_job(job)
            except Exception as e:
                print(f"Error in batch prediction workflow: {e}")

            # Download and process output if requested
            if job.has_succeeded and img_gen_output_location:
                print(f"Downloading image generation response from {img_gen_output_location}...")
                output_dir = download_gcs_output(
                    img_gen_output_location,
                    local_dir=f"./{batch_job}",
                    suffix='generation',
                    project_id=GCS["PROJECT_ID"],
                )

    else:
            print(f'\nGenerating images using {args.model}')
            contents = generate_contents(product_dict, args.images_per_product)
            results = []
            model = LLM_MODELS[args.model]
            for content_dict in tqdm(contents):
                product_id = content_dict['prod_id']
                response = generate_image(content_dict, model)
                results.append({
                    "product_id": product_id,
                    "response": response
                })
                with open(f'{batch_job}/predictions_generation.pkl', "wb") as f:
                    pickle.dump(results, f)
            save_product_dict(product_dict, f"{batch_job}/product_dict_generation.pkl")

    batch_output = []
    batch_bool = None
    # Output for batch predictions
    if f"predictions_generation.jsonl" in os.listdir(f"./{batch_job}"):
        print(os.listdir(f"./{batch_job}"))
        output_dir = f"./{batch_job}/predictions_generation.jsonl"
        batch_bool = True
        with open(output_dir, "r") as f:
            for line in f:
                batch_output.append(json.loads(line))

    # Output for single request predictions
    elif f"predictions_generation.pkl" in os.listdir(f"./{batch_job}"):
        output_dir = f"./{batch_job}/predictions_generation.pkl"
        batch_bool = False
        with open(output_dir, "rb") as f:
            batch_output = pickle.load(f)
    else:
        raise ValueError("No predictions file found")
    print(f"Loading predictions from {output_dir}")
    print(f"Loaded {len(batch_output)} predictions.")

    print("Parsing predictions...")
    product_dict = parse_generation_prediction(batch_output, product_dict, batch=batch_bool)
    save_product_dict(product_dict, f"{batch_job}/product_dict_generation.pkl")
    df = generation_preds_to_csv(batch_job)

    # Download images locally
    if args.download_images.lower() == 'true':
        print("Downloading images...")
        download_locally(df, batch_job)

if __name__ == "__main__":
    main()
