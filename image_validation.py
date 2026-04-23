import argparse
import json
import os
import pickle
from PIL import Image
import imagehash
import base64
import pandas as pd
from PIL import Image
from io import BytesIO

from data_utils import *
from run_batch_job import monitor_batch_prediction_job, run_batch_prediction_job
from parse_predictions import parse_validation_prediction, validation_preds_to_csv
from image_generation import PROTEIN_RULES as PROTEIN_RULES_GENERATION
from tqdm import tqdm
from google import genai
from google.genai import types


SYSTEM_INSTRUCTIONS = """
You are a strict food product image compliance auditor.
Your task is to determine whether a generated image strictly follows the provided protein generation rules and structured product data.

You must:
- Treat the structured product data as absolute ground truth.
- Evaluate the image objectively.
- Never assume compliance.
- Explicitly verify each rule.
- Mark any mismatch as a violation.

"""

CONTENT_CONFIG_VAL = {
    "temperature": 0,
    "topP": 0.1,
    "maxOutputTokens": 32768,
    "responseMimeType": "application/json",
    "responseSchema": {
        "type": "object",
        "properties": {
            "rule_evaluations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "rule_name": {
                            "type": "string"
                        },
                        "rule_description": {
                            "type": "string"
                        },
                        "pass": {
                            "type": "boolean"
                        },
                        "reason": {
                            "type": "string"
                        },
                        "modifications": {
                            "type": "string"
                        }
                    },
                    "required": [
                        "rule_name",
                        "rule_description",
                        "pass",
                        "reason",
                        "modifications"
                    ]
                }
            },
            "aggregated_modifications": {
                "type": "string"
            }
        },
        "required": [
            "rule_evaluations",
            "aggregated_modifications"
        ]
    }
}

PROTEIN_RULES_VALIDATION = {

    'SEAFOOD': """
    1. Fish species: Verify that the depicted species (e.g., salmon, tuna, cod, sea bass) matches the product data exactly. 
      Confirm correct flesh color, muscle structure, fat distribution, flake size, and surface texture for that species.
    2. Cut type: Confirm whether the product is a fillet, steak, loin, whole fish, or portion cut, and verify anatomical consistency with the specified species.
    3. Form: Verify that the physical preparation (whole, filleted, sliced, cubed, minced) matches the product data. 
      Portion size and thickness must be realistic for the species
    4. Skin status:
      If skinless is specified, confirm that NO skin is visible — no skin patches, no silver membrane, no scale texture, and no skin-colored edges.
      If skin-on is specified, confirm that the skin texture, pattern, and color match the species and correctly cover the appropriate surface area. Make sure the skin is on the right side of the fish portion.
      Images that contradict the specified skin status must not be selected.
    5. Head status: Verify whether the fish is head-on or headless as specified.
    6. Fillet constraints: If described as a fillet, confirm that no head or tail is attached and that muscle structure and thickness are consistent with a true fillet of the species.
    7. Packaging: If the product is canned or packaged seafood (can, jar, tube, pouch, etc.), do not generate it inside its retail packaging. Always show the seafood out of the package, with the edible product as the primary focus.
    """,
    'PORK': """
    1. Pork cut: Verify that the depicted cut (e.g., loin, shoulder/Boston butt, belly, tenderloin, ham, ribs) matches the product data exactly and reflects correct fat distribution and muscle structure.
    Special note about Country-style vs spare ribs distinction (CRITICAL when applicable): Country-style ribs must be boneless and originate from the shoulder. Spare ribs must be bone-in and originate from the lower ribcage.
    2. Shape: Confirm that the overall geometric shape matches the expected form of the specified cut.
    3. Form: Verify that physical preparation (sliced, cubed, ground, chopped) matches the product data.
    4. Bone status: Confirm bone-in or boneless status exactly as described. 
      If not explicitly stated, infer only when the cut is unambiguously defined as bone-in or boneless by standard definition.
    """,
    'BEEF': """
    1. Beef cut: Verify that the depicted cut (e.g., ribeye, sirloin, brisket) matches the product data and reflects correct marbling characteristics.
    Cube steak clarification (CRITICAL when applicable): A cube steak must be flat and thin with visible mechanical tenderization indentations. It must NOT be depicted as beef cut into cubes.
    3. Shape: Confirm that the overall geometric shape matches the expected form of the specified cut.
    4. Form: Verify that preparation (sliced, ground, chopped) matches the product data.
    5. Bone status: Confirm bone-in or boneless status as described.
      If not explicitly stated, infer only when the cut is unambiguously defined as bone-in or boneless by standard definition.
    """,
    'POULTRY': """
    1. Poultry type: Verify that the depicted bird (chicken, turkey, duck) matches the product data, including typical color, fat distribution, and proportions.
    2. Cut type: Confirm that the specific cut (breast, thigh, drumstick, leg quarter, wing, whole bird, spatchcocked, etc.) matches the product data.
    3. Meat color: The flesh must match the natural color of cooked poultry. For example: cooked breast meat is white to off-white (white meat); whereas cooked thigh or leg meat is darker beige to light brown (dark meat). 
      If multiple muscle types are specified in the product information (e.g., 60% thigh / 40% breast), verify that the visual proportion matches this ratio realistically.
    4. Bone status:
      If bone-in, confirm visible bones that anatomically correspond to the specified cut.
      If deboned, confirm no visible or implied bones.
      If partially deboned, confirm the depiction matches the precise description.
    5. Skin status:
      If skin-on is specified, confirm continuous skin coverage with no exposed muscle.
      If skinless is specified, confirm no skin is visible.
    6. Form:
      Verify that preparation (whole, split, spatchcocked, butterflied, diced, sliced, ground, pounded thin) matches the product data.
      If spatchcocked or split, confirm the bird is flattened appropriately with backbone removed.
    """
}

# **Food Product Image Evaluation Rules**
# 1. Product accuracy: Verify that the dish depicted in the image matches the structured product data (title, description, attributes) and accurately represents the product.
# 2. Cooking state: Verify that the product is fully cooked and presented as part of a complete main dish.
# 3. Dish composition: Verify that the full dish is visible in the frame without cropping and that any side components described in the product data appear on the same plate as the main dish.
# 4. Scene realism: Verify that the dish is presented in a natural, realistic lifestyle setting and that the image appears visually realistic and appetizing (not artificial, CGI-like, or obviously fake).
# 5. No people or text: Verify that no humans or human hands are visible and that there are no labels, writing, or letters present in the image.

def compose_prompts(product_dict):
    for product_id, product_info in product_dict.items():
        product_title = product_info['product_title']
        product_description = product_info['product_description']
        product_category = product_info['product_category']
        product_attributes = product_info['product_attributes']
        prompt = product_info['generation']['prompt']

        # Fetching Accuracy Rules
        if 'accuracy_rules' in product_info['generation'].keys():
            accuracy_rules = product_info['generation']['accuracy_rules']
        else:
            try:
                accuracy_rules = prompt.split("ACCURACY RULES")[1].strip("\n").strip("")
            except:
                print("No accuracy rules found in prompt. Retrieving all protein-specific rules.")
                accuracy_rules = PROTEIN_RULES_GENERATION[product_category]

        # Parsing attributes
        product_attributes_str = ""
        for attr in product_attributes:
            product_attributes_str += f"- {list(attr.keys())[0]}: {list(attr.values())[0]}\n"

        for image_id in product_info['generation']['generated_images'].keys():
            prompt = f"""
                Consider the following food product (ID: {product_id}): 
    
                PRODUCT CATEGORY: {product_category}
    
                STRUCTURED PRODUCT DATA (GROUND TRUTH):
                Product Title: '{product_title}'.
                Product Description: '{product_description}'.
                Product Attributes: {product_attributes_str}
                IMPORTANT: Evaluate only attributes visible in a cooked product.
                
                Perform evaluation in 3 steps considering the image provided (IMAGE ID: {image_id}): 
    
                STEP 1 — Extract Visual Observations
                Objectively describe:
                - Species or protein type
                - Cut type
                - Form
                - Bone status
                - Skin status
                - Packaging presence
                - Shape and muscle/fat characteristics
    
                Do not evaluate yet. Only describe what is visible.
    
                STEP 2 — Rule-by-Rule Compliance Check
                Consider the set of rules below:
                
                **ACCURACY RULES**
                {accuracy_rules}
                
                For each rule:
                - State the rule
                - Compare it to the product information
                - Compare it to the observed image
                - Mark as: PASS / FAIL
                - Explain precisely why it passed or failed.
                - If FAIL, specify what modifications should be made to the image so that it can pass the rule. 
                Modifications should only refer to the image, not the product information. 
    
                STEP 3 — Verdict and Output 
                3.1. For each rule above, create one dictionary with the keys and values below.
                a. 'rule_name': include rule title provided
                b. 'rule_description': include rule description provided
                c. 'pass': True if it passes the rule, False otherwise.
                d. 'reason': reasons for passing or failing the rule
                e. 'modifications': if FAIL, pass the modifications needed so that the image reflects accurately the 
                product information. If PASS, return 'None'. 
                Use the response schema provided, adding each dictionary rule to the "rule_evaluations" array.
                
                3.2. Using the 'modifications' from all failed rules, write one concise paragraph describing the 
                required image corrections. 
                Merge redundant instructions and include all necessary constraints. Do not reference rule names.
                Use the response schema provided, assigning this paragraph as a value to the 
                "aggregated_modifications" key.
                """

            product_dict[product_id]['validation']['validation_prompt'][image_id] = prompt
            product_dict[product_id]['validation']['accuracy_rules'] = accuracy_rules
    return product_dict


def generate_batch_requests_for_validation(product_dict):

    lines = []
    for pid, prod_info in product_dict.items():
        generated_images = prod_info['generation']['generated_images']
        for image_id, image_dict in generated_images.items():
            if product_dict[pid]['validation'][image_id]['reference_sim_bool'] is True:
                continue
            prompt = prod_info['validation']['validation_prompt'][image_id]
            generated_image = image_dict['image']
            if isinstance(generated_image, bytes):
                generated_image = base64.b64encode(generated_image).decode("utf-8")
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
                             {"inlineData": {"data": generated_image, "mimeType": "image/jpeg"}}
                         ]
                         }
                    ],
                    "generationConfig": CONTENT_CONFIG_VAL}}

            lines.append(line)
    return lines


def reference_vs_generated_similarity(product_dict):

    for product_id, product_info in tqdm(product_dict.items()):
        reference_image = product_info['generation']['reference_image']
        if not isinstance(reference_image, str):
            continue
        generated_images = product_info['generation']['generated_images']
        for image_id, image_dict in generated_images.items():
            generated_image = image_dict['image']

            # Fetching Reference Image
            path = reference_image[5:] # remove gs://
            bucket_name, blob_name = path.split("/", 1)
            client = storage.Client()
            blob = client.bucket(bucket_name).blob(blob_name)
            reference_image = blob.download_as_bytes()

            try:
                if isinstance(generated_image, str):
                    generated_image = base64.b64decode(generated_image)
                generated_image = Image.open(BytesIO(generated_image))
                reference_image = Image.open(BytesIO(reference_image))

                reference_hash = imagehash.phash(reference_image)
                generated_hash = imagehash.phash(generated_image)

                distance = reference_hash - generated_hash
                if distance < 5:
                    reference_sim_bool = True
                elif 5 < distance < 10:
                    reference_sim_bool = False
                else:
                    reference_sim_bool = False
                product_dict[product_id]['validation'][image_id] = {'reference_sim_bool': reference_sim_bool}
            except Exception as e:
                print(f"Not able to compare reference image for product ID {product_id} (Image ID: {image_id}): {e}")

    return product_dict


def generate_contents(product_dict):
    contents = []
    # ['generated_images'][image_id] = {'image_str': image_str}
    for prod_id, prod_info in product_dict.items():
        for image_id in prod_info['generation']['generated_images'].keys():
            prompt = prod_info['validation']['validation_prompt'][image_id]
            content_dict = {
                            'prompt': prompt,
                            'prod_id': prod_id,
                            'image_id': image_id,
                            }
            contents.append(content_dict)
    return contents


def run_llm_validator(content_dict):

    prod_id = content_dict['prod_id']
    image_id = content_dict['image_id']
    prompt = content_dict['prompt']

    client = genai.Client(
        vertexai=True,
        project=GCS['PROJECT_ID'],
        location=GCS['LOCATION'],
    )
    model = LLM_MODELS["FLASH_2.5"]
    try:
        response = client.models.generate_content(
            model=model,  # "gemini-3-pro-preview",  # ensure this model exists in your project
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=prompt),
                    ],
                )
            ],
            config=types.GenerateContentConfig(
                system_instruction=(
                    SYSTEM_INSTRUCTIONS
                ),
                response_modalities=["TEXT"],
                temperature=CONTENT_CONFIG_VAL["temperature"],
                # top_p=CONTENT_CONFIG_VAL["topP"],
                max_output_tokens=CONTENT_CONFIG_VAL["maxOutputTokens"],
                response_mime_type=CONTENT_CONFIG_VAL["responseMimeType"],
                response_schema=CONTENT_CONFIG_VAL["responseSchema"],
            ),
        )
        return response
    except Exception as e:
        print(e)
        print(f"No response for product ID {prod_id} (Image ID: {image_id}).")
        return None

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

    args = parser.parse_args()
    batch_job = args.batch_job
    job_type = args.job_type
    os.makedirs(batch_job, exist_ok=True)

    # Load product dict
    try:
        product_dict_file = f"./{batch_job}/product_dict_generation.pkl"
        with open(product_dict_file, "rb") as f:
            product_dict = pickle.load(f)
        print(f"Loaded product dict from {product_dict_file}")
    except Exception as e:
        raise ValueError("Could not load product dict from file.")

    # Filter by product ids
    if args.product_ids:
        product_ids = args.product_ids.split(',')
        if all(item in product_dict.keys() for item in product_ids):
            product_dict = {k: product_dict[k] for k in product_dict if k in product_ids}
        else:
            missing = [item for item in product_ids if item not in product_dict.keys()]
            print(f"Invalid product IDs: {missing}. Images will not be validated for these products.")
    print(f"Collected {len(product_dict)} products.\n")

    # # Image similarity: reference vs. generated image
    print("Image similarity check...")
    product_dict = reference_vs_generated_similarity(product_dict)

    print("\nGenerating input data for validation...")
    product_dict = compose_prompts(product_dict)
    save_product_dict(product_dict, f"{batch_job}/product_dict_validation.pkl")

    if job_type == "batch":
        # Create input file
        input_data = generate_batch_requests_for_validation(product_dict)
        print(f"Generated requests for {len(input_data)} products.")

        # Write batch input file
        input_path = f"./{batch_job}/input_file_{batch_job}_validation.jsonl"
        write_batch_input_file(input_data, input_path)

        # Upload to GCS
        print("\nUploading input file to GCS...")
        input_file_uri = upload_input_file_to_gcs(input_path, bucket_name=GCS["BUCKET"])

        # # Image Validation
        try:
            print(f"Running batch prediction job with model: {LLM_MODELS["FLASH_2.5"]}")
            job = run_batch_prediction_job(
                input_file=input_file_uri,
                output_uri=GCS["GCS_OUTPUT_URI"],
                model_id=LLM_MODELS["FLASH_2.5"],
                project_id=GCS["PROJECT_ID"],
                location=GCS["LOCATION"],
            )

            # Monitor the batch prediction job
            print("Monitoring batch prediction job...")
            img_val_output_location = monitor_batch_prediction_job(job)

        except Exception as e:
            print(f"Error in batch prediction workflow: {e}")

        # Download and process output if requested
        if job.has_succeeded and img_val_output_location:
            print(f"Downloading image generation response from {img_val_output_location}...")
            output_dir = download_gcs_output(
                img_val_output_location,
                local_dir=f"./{batch_job}",
                suffix='validation',
                project_id=GCS["PROJECT_ID"],
            )
    else:
        contents = generate_contents(product_dict)
        results = []
        for content_dict in tqdm(contents):
            product_id = content_dict['prod_id']
            image_id = content_dict['image_id']
            response = run_llm_validator(content_dict)
            results.append({
                "product_id": product_id,
                "image_id": image_id,
                "response": response
            })
            with open(f'{batch_job}/predictions_validation.pkl', "wb") as f:
                pickle.dump(results, f)
        save_product_dict(product_dict, f"{batch_job}/product_dict_validation.pkl")

    batch_output = []
    # Output for batch predictions
    if f"predictions_validation.jsonl" in os.listdir(f"./{batch_job}"):
        print(os.listdir(f"./{batch_job}"))
        output_dir = f"./{batch_job}/predictions_validation.jsonl"
        batch_bool = True
        with open(output_dir, "r") as f:
            for line in f:
                batch_output.append(json.loads(line))

    # Output for single request predictions
    elif f"predictions_validation.pkl" in os.listdir(f"./{batch_job}"):
        output_dir = f"./{batch_job}/predictions_validation.pkl"
        batch_bool = False
        with open(output_dir, "rb") as f:
            batch_output = pickle.load(f)
    else:
        raise ValueError("No predictions file found")
    print(f"Loading predictions from {output_dir}")
    print(f"Loaded {len(batch_output)} predictions.")
    product_dict = parse_validation_prediction(batch_output, product_dict, batch=batch_bool)
    save_product_dict(product_dict, f"{batch_job}/product_dict_validation.pkl")
    validation_preds_to_csv(batch_job)

if __name__ == "__main__":
    main()
