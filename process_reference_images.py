import argparse
import re
import xml.etree.ElementTree as ET

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from data_utils import *

# Load once (do NOT reload inside the function in real code)
model = SentenceTransformer("all-MiniLM-L6-v2")


def fetch_reference_images(target_label):

    asset_data = []
    client = storage.Client()
    bucket_name = GCS["BUCKET"]
    prefix = "reference_image_bank"
    blobs = client.list_blobs(bucket_name, prefix=prefix)
    for blob in blobs:
        if blob.name == 'reference_image_bank/':
            continue

        label = blob.name.split("reference_image_bank/")[-1].split("/")[1].split("/")[0]
        if label == target_label:
            business_center = blob.name.split("reference_image_bank/")[-1].split("/")[0]

            filename = blob.name.split("reference_image_bank/")[-1].split("/")[2]
            supc = filename.split("_")[0]
            description = filename.split("_")[1].split(".")[0]
            path = f"gs://{bucket_name}/{blob.name}"

            ref_img_dict = {'supc': supc,
                            'description': description,
                            'image_path': path,
                            'business_center': business_center,
                            'label': label,
                            'sysco_brand': True,
                            'published': False,}
            asset_data.append(ref_img_dict)

    print(f"Collected {len(asset_data)} reference images.\n")
    return asset_data


def semantic_similarity(text1: str, text2: str) -> float:
    """
    Returns semantic similarity between two strings in [0, 1].
    Higher = more similar.
    """
    embeddings = model.encode([text1, text2])
    similarity = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]]
    )[0][0]
    return float(similarity)


def select_reference_images(products, asset_data):
    print("Selecting reference images by computing semantic similarity...")
    sim_pairs = []
    for pid, prod_info in tqdm(products.items()):
        title = prod_info['product_title']
        max_sim = 0
        sim_desc = ""
        sim_img = None
        for asset_dict in asset_data:
            if prod_info['product_category'] != asset_dict['business_center']:
                continue
            if prod_info['target_label'] != asset_dict['label']:
                continue
            material_description = asset_dict['description']
            sem_sim = semantic_similarity(title, material_description)

            # Ignore contradictory terms
            skin_off = ['SKINLESS', 'SKLS', 'SK\\OFF', 'SK/OFF', 'SKOFF', 'SKIN\\OFF', 'SKIN/OFF', 'SKIN-OFF', 'SKIN OFF', 'B\\S', 'B/S']
            skin_on = ['SKON', 'SK\\ON', 'SK/ON', 'SKIN ON','SKIN-ON', 'SKIN/ON', 'SKIN\\ON']
            boneless = ['BONELESS', 'BNLS', 'B\\S', 'B/S']
            bone_in = ['B\\I', 'B/I']
            if any(s in title.upper() for s in skin_off) and any(s in material_description.upper() for s in skin_on):
                sem_sim = 0
            if any(s in material_description.upper() for s in skin_off) and any(s in title.upper() for s in skin_on):
                sem_sim = 0
            if any(s in title.upper() for s in boneless) and any(s in material_description.upper() for s in bone_in):
                sem_sim = 0
            if any(s in material_description.upper() for s in boneless) and any(s in title.upper() for s in bone_in):
                sem_sim = 0

            if sem_sim > max_sim:
                max_sim = sem_sim
                sim_desc = material_description
                sim_img = asset_dict['image_path']

        sim_pairs.append((title, sim_desc, max_sim))
        products[pid]['generation']['reference_similarity'] = max_sim
        products[pid]['generation']['reference_description'] = sim_desc
        products[pid]['generation']['reference_image'] = sim_img

    return products


def main():
    parser = argparse.ArgumentParser(
        description='Generate lifestyle images for selected products using reference images.')
    parser.add_argument('--batch_job', type=str, default='default',
                        help='Name of batch job')
    parser.add_argument('--target_label', type=str, default='STYLED',
                        help='Target label for generated images')
    parser.add_argument('--input_product_file', type=str, default=DATA["INPUT_PRODUCT_FILE"],
                        help='JSONL file with product data')
    parser.add_argument('--product_ids', type=str,
                        help='Product IDs separated by commas. E.g. "1234567,7654321,0987654')

    args = parser.parse_args()
    batch_job = args.batch_job
    os.makedirs(batch_job, exist_ok=True)

    # Read input file and process product information
    print("Loading product information ...")
    input_product_file = args.input_product_file
    target_label = args.target_label
    product_dict = process_product_information(input_product_file, target_label)

    # Filtering by product ids
    if args.product_ids:
        product_ids = args.product_ids.split(',')
        if all(item in product_dict.keys() for item in product_ids):
            product_dict = {k: product_dict[k] for k in product_dict if k in product_ids}
        else:
            missing = [item for item in product_ids if item not in product_dict.keys()]
            print(f"Invalid product IDs: {missing}. Images will not be generated for these products.")
    print(f"Collected {len(product_dict)} products from product dict.\n")

    # # Fetch and select reference images
    print(f'Reading reference images from GCS: {GCS["BUCKET"]}/reference_image_bank...')
    asset_data = fetch_reference_images(target_label)
    product_dict = select_reference_images(product_dict, asset_data)

    # Saving product dict (for parsing predictions)
    save_product_dict(product_dict, f"{batch_job}/{DATA["PRODUCT_DICT_FILE"]}")

if __name__ == "__main__":
    main()
