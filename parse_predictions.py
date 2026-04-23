import base64
import pandas as pd
from PIL import Image
from io import BytesIO
from ast import literal_eval
import argparse
from tqdm import tqdm
import uuid

from data_utils import *


def parse_generation_prediction(output, product_info, batch=True):

    if batch is True:
        for line in output:
            prod_id = str(line).split("ID: ")[1].split("Product Title:")[0]
            prod_id = prod_id[:7] # double check
            if 'generated_images' not in product_info[prod_id]['generation'].keys():
                product_info[prod_id]['generation']['generated_images'] = {}
            try:
                image_id = str(uuid.uuid4())
                image_str = line['response']['candidates'][0]['content']['parts'][0]['inlineData']['data']
                product_info[prod_id]['generation']['generated_images'][image_id] = {'image': image_str}
                print(f"Successfully fetched image for product {prod_id}")
            except Exception as e:
                print("Failed to fetch image")
                print(e)
    else:
        for line in output:
            prod_id = line['product_id']
            response = line['response']
            if 'generated_images' not in product_info[prod_id]['generation'].keys():
                product_info[prod_id]['generation']['generated_images'] = {}
            try:
                image_id = str(uuid.uuid4())
                image_str = response.candidates[0].content.parts[0].inline_data.data
                product_info[prod_id]['generation']['generated_images'][image_id] = {'image': image_str}
            except Exception as e:
                print("Failed to fetch image")
                print(e)
    return product_info


def generation_preds_to_csv(batch_job):

    with open(f"{batch_job}/product_dict_generation.pkl", "rb") as f:
        product_dict = pickle.load(f)

    all_rows = []
    for prod_id, prod_info in tqdm(product_dict.items()):
        row = {
            'product_id': prod_id,
            'product_title': prod_info.get('product_title', ""),
            'product_description': prod_info.get('product_description', ""),
            'product_attributes': prod_info.get('product_attributes', ""),
            'reference_title': prod_info.get('reference_description', ""),
            'reference_similarity': prod_info.get('reference_similarity', 0),
            'reference_bool': False
        }

        # cosine similarity
        if isinstance(row['reference_similarity'], (int, float)) and row['reference_similarity'] >= DATA['COS_SIM_THRESH']:
            row['reference_bool'] = True

        reference_image = prod_info['generation']['reference_image']
        if not reference_image.startswith("gs://"):
            print("Invalid GCS URI for reference image.")
            row['reference_image'] = Image.new("RGB", (96, 96), color=(0, 0, 0))
        else:
            try:
                client = storage.Client()
                bucket = client.bucket(GCS['BUCKET'])
                blob_name = reference_image.replace(f"gs://{GCS['BUCKET']}/", "")
                blob = bucket.blob(blob_name)
                reference_bytes = blob.download_as_bytes()
                reference_image = Image.open(BytesIO(reference_bytes))
                row['reference_image'] = reference_image
            except Exception as e:
                print("Failed to fetch reference image from GCS")
                print(e)
                row['reference_image'] = Image.new("RGB", (96, 96), color=(0, 0, 0))

        # Process images
        generated_images = prod_info['generation']['generated_images']
        for i, image_id in enumerate(generated_images.keys()):
            try:
                if isinstance(generated_images[image_id]['image'], str):
                    generated_image_str = generated_images[image_id]['image']
                    generated_image_bytes = base64.b64decode(generated_image_str)
                else:
                    generated_image_bytes = generated_images[image_id]['image']
                generated_image = Image.open(BytesIO(generated_image_bytes))
            except Exception as e:
                print("Failed to fetch image")
                print(e)
                generated_image = Image.new("RGB", (96, 96), color=(0, 0, 0))

            row[f'generated_image_{i}'] = generated_image
        all_rows.append(row)

    df = pd.DataFrame(all_rows)
    df_file = f"{batch_job}/generation_preds"
    df.to_pickle(f"{df_file}.pkl")
    df.to_csv(f"{df_file}.csv", index=False)
    print(f"Predictions saved at {df_file}.csv and {df_file}.pkl")
    return df


def parse_validation_prediction(batch_output, product_dict, acceptable_threshold=1, batch=True):

    parsed_output = nested_dict()
    if batch is True:
        for line in batch_output:
            prod_id = str(line).split("(ID: ")[1].split("):")[0]
            image_id = str(line).split("IMAGE ID: ")[1].split("):")[0]

            # response
            try:
                resp_str = line['response']['candidates'][0]['content']['parts'][0]['text']
                resp_str = resp_str.replace("\n", " ").replace("   ", " ").replace("   ", " ")
                resp_str = resp_str.replace("\"pass\": true", "\"pass\": \"PASS\"").replace("\"pass\": false", "\"pass\": \"FAIL\"")
                resp_json = literal_eval(resp_str)
                rules = resp_json['rule_evaluations']
                aggregated_modifications = resp_json['aggregated_modifications']
                parsed_output[prod_id][image_id]['rules'] = rules
                parsed_output[prod_id][image_id]['aggregated_modifications'] = aggregated_modifications
            except Exception as e:
                print()
                print(f"Failed to fetch LLM ruling and verdict for product {prod_id} (image {image_id}).")
                print(str(line)[:111])
                print(e)
                parsed_output[prod_id][image_id] = None
    else:
        for line in batch_output:
            prod_id = line['product_id']
            image_id = line['image_id']
            response = line['response']
            try:
                # resp_json = literal_eval(response)
                resp_str = response.candidates[0].content.parts[0].text
                resp_str = resp_str.replace("\"pass\": true", "\"pass\": \"PASS\"").replace("\"pass\": false",
                                                                                            "\"pass\": \"FAIL\"")
                resp_json = literal_eval(resp_str)
                rules = resp_json['rule_evaluations']
                aggregated_modifications = resp_json['aggregated_modifications']
                parsed_output[prod_id][image_id]['rules'] = rules
                parsed_output[prod_id][image_id]['aggregated_modifications'] = aggregated_modifications
            except Exception as e:
                print()
                print(f"Failed to fetch LLM ruling and verdict for product {prod_id} (image {image_id}).")
                print(str(line)[:111])
                print(e)
                parsed_output[prod_id][image_id] = None

    modifications = []
    verdicts = []
    for prod_id in parsed_output.keys():
        for image_id in parsed_output[prod_id].keys():
            print(image_id)
            rules = parsed_output[prod_id][image_id]['rules']
            aggregated_modifications = parsed_output[prod_id][image_id]['aggregated_modifications']
            for rule in rules:
                modifications.append({'rule_name': rule['rule_name'], 'modifications': rule.get('modifications', None)})
                verdicts.append(rule.get('pass', "False"))

            if all(item == "PASS" for item in verdicts):
                verdict = 'PASS'
            elif verdicts.count("FAIL") == acceptable_threshold:
                verdict = 'ACCEPTABLE'
            else:
                verdict = 'FAIL'

            validation_dict = {
                                'rule_evaluations': rules,
                                'modifications': modifications,
                                'aggregated_modifications': aggregated_modifications,
                                'final_verdict': verdict
            }

            product_dict[prod_id]['validation']['results'][image_id] = validation_dict

    return product_dict


def validation_preds_to_csv(batch_job):

    # Load product dict
    with open(f"{batch_job}/product_dict_validation.pkl", "rb") as f:
        product_dict = pickle.load(f)

    # df columns
    product_ids = []
    product_titles = []
    product_descriptions = []
    product_attrs = []
    final_verdicts = []
    modifications = []
    aggreagated_modifications = []
    image_ids = []
    generated_images = []

    for prod_id, prod_info in product_dict.items():
        if 'validation' in prod_info.keys():
            for image_id in prod_info['validation']['results'].keys():
                # Product data
                image_ids.append(image_id)
                product_ids.append(prod_id)
                product_titles.append(prod_info['product_title'])
                product_descriptions.append(prod_info['product_description'])
                product_attrs.append(prod_info['product_attributes'])

                # Validation data
                val_dict = prod_info['validation']['results'][image_id]
                final_verdicts.append(val_dict['final_verdict'])
                aggreagated_modifications.append(val_dict['aggregated_modifications'])

                modif_list = val_dict['modifications']
                modif_filtered = []
                for modif in modif_list:
                    if modif['modifications'] != "None":
                        modif_filtered.append({'rule_name': modif['rule_name'], 'modifications': modif.get('modifications', None)})
                modifications.append(modif_filtered)

                # Read image as PIL object
                image_str = prod_info['generation']['generated_images'][image_id]['image']
                if isinstance(image_str, str):
                    image_bytes = base64.b64decode(image_str)
                    generated_img = Image.open(BytesIO(image_bytes))
                else:
                    generated_img = Image.new("RGB", (96, 96), color=(0, 0, 0))
                generated_images.append(generated_img)


    df = pd.DataFrame(data={'product_id': product_ids, 'product_title': product_titles, 'product_description': product_descriptions,
                            'attributes': product_attrs, 'final_verdict': final_verdicts, 'modifications': modifications,
                            'aggregated_modifications': aggreagated_modifications, 'image_id': image_ids, 'generated_image': generated_images})

    df_file = f"{batch_job}/validation_preds"
    df.to_csv(f"{df_file}.csv", index=False)
    df.to_pickle(f"{df_file}.pkl")
    print("Predictions saved at {}".format(df_file))
    return df


def download_locally(df, batch_job):

    folder = f"{batch_job}/generated_images"
    os.makedirs(folder, exist_ok=True)

    for i,row in tqdm(df.iterrows()):

        product_id = row['product_id']
        product_description = row['product_description']

        for col in df.columns:
            if 'image' in col:
                try:
                    image = row[f'{col}']
                    image_file = f"{folder}/{product_description.replace("/", "-")}_{product_id}_{col}"
                    image.save(f"{image_file}.png")
                except Exception as e:
                    print("Failed to fetch image")
                    print(e)
    print(f"Images saved at {folder}")


def main():
    # Set up argument parser similar to gdsn_extraction.py but with added batch options
    parser = argparse.ArgumentParser(
        description='Generate lifestyle images for selected products using reference images.')

    # Data source arguments - only required if not processing output only
    parser.add_argument('--batch_job', type=str, default='test',
                        help='Name of batch job')
    parser.add_argument('--step', type=str, default='generation',
                        help="Choose between 'generation' or 'validation'")
    parser.add_argument('--download_images', type=str, default='False',
                        help="Download the images")

    args = parser.parse_args()
    batch_job = args.batch_job
    step = args.step
    os.makedirs(batch_job, exist_ok=True)

    # Load product dict
    prod_dict_path = f"{batch_job}/product_dict_{step}.pkl"
    print("Loading prod dict from {}".format(prod_dict_path))
    with open(prod_dict_path, "rb") as f:
        product_dict = pickle.load(f)

    # Load predictions
    batch_output = []
    # output of batch predictions
    if f"predictions_{step}.jsonl" in os.listdir(f"./{batch_job}"):
        batch_bool = True
        output_dir = f"./{batch_job}/predictions_{step}.jsonl"
        with open(output_dir, "r") as f:
            for line in f:
                batch_output.append(json.loads(line))

    # output of single request predictions
    elif f"predictions_{step}.pkl" in os.listdir(f"./{batch_job}"):
        batch_bool = False
        output_dir = f"./{batch_job}/predictions_{step}.pkl"
        with open(output_dir, "rb") as f:
            batch_output = pickle.load(f)
    else:
        raise ValueError("No predictions file found")
    print(f"Loading predictions from {output_dir}")
    print(f"Loaded {len(batch_output)} predictions.")

    # Parse Predictions
    if step == 'generation':
        product_dict = parse_generation_prediction(batch_output, product_dict, batch=batch_bool)
        save_product_dict(product_dict, f"{batch_job}/product_dict_generation_parsed.pkl")
        print("Loading predictions into a dataframe...")
        df = generation_preds_to_csv(batch_job)
    elif step == 'validation':
        product_dict = parse_validation_prediction(batch_output, product_dict)
        save_product_dict(product_dict, f"{batch_job}/product_dict_validation.pkl")
        df = validation_preds_to_csv(batch_job)
    else:
        raise ValueError("Step must be 'generation', 'validation' or 'regeneration'")

    # Download images locally
    if args.download_images.lower() == 'true':
        print("Downloading images...")
        download_locally(df, batch_job)

if __name__ == "__main__":
    main()
