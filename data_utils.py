import json
import os
import pickle
from collections import defaultdict
from google.cloud import storage

# Constants
LLM_MODELS = {
                "NANO_BANANA": "gemini-2.5-flash-image",  # Update with your preferred model
                "NANO_BANANA_PRO": "gemini-3-pro-image-preview",
                "NANO_BANANA_2": "gemini-3.1-flash-image-preview",
                "FLASH_2.5": "gemini-2.5-flash"
             }

GCS = {
        "PROJECT_ID": "syy-cx-shop-np-c5e4",  # Update with your GCP project ID
        "LOCATION": "us-central1",
        "BUCKET": "sysco-smarter-catalog-ce-image-generation-poc",
        "GCS_INPUT_URI": "gs://sysco-smarter-catalog-ce-image-generation-poc/batch_inputs",
        "GCS_OUTPUT_URI": "gs://sysco-smarter-catalog-ce-image-generation-poc/batch_outputs",
        "GCS_REFERENCE_URI": "gs://sysco-smarter-catalog-ce-image-generation-poc/reference_images"
      }

DATA = {
        "REFERENCE_IMG_FOLDER": "/Users/carolinacaprile/Documents/sysco/image_generation/reference_image_bank",
        "PRODUCT_DICT_FILE": "product_dict.pkl",
        "INPUT_PRODUCT_FILE": "/Users/carolinacaprile/Documents/sysco/image_generation/input_protein.jsonl",
        "COS_SIM_THRESH": 0.85
       }

def add_product_information(input_file, product_dict, target_label):
    # Create input file
    print("Adding product information ...")
    print(f"Processing input file: {input_file}")
    data = []
    with open(input_file, "r") as f:
        for line in f:
            data.append(json.loads(line))

    for d in data:

        pid = (d["product_id"])
        if pid in product_dict.keys():

            title = (d["name"][0]["value"])
            desc = (d["description"][0]["value"])
            cat = d["taxonomy"]["business_center"]
            attr_list = (d["taxonomy"]["attributes"])

            attr = []
            exclude_attr_key = ['aged', 'feed', 'origin', 'fresh', 'frozen', 'packag', 'harvest', 'imported', 'domestic',
                                'bioengineer', 'organic', 'aqua', 'natural', 'child', 'grade', 'injected', 'shelf', 'sugar',
                                'chemical', 'water', 'allergen', 'nutrient', 'region', 'claim', 'kosher', 'casing',
                                'certified', 'flavor', 'diet type']
            exclude_attr_value = ['not ', 'no ', 'free', 'packag', 'fresh', 'frozen', 'imported', 'domestic', 'farm',
                                  'bioengineer', 'injected', 'organic', 'aqua', 'natural', 'may contain', 'sugar',
                                  'non breed', 'standard', 'flavor', 'claim', 'kosher', 'cooking', 'casing', 'certified',
                                  'artificial', 'raw', 'contains']
            for a in attr_list:
                if not any(sub.lower() in a['name'].lower() for sub in exclude_attr_key):
                    if not any(sub.lower() in a['value'].lower() for sub in exclude_attr_value):
                        attr.append({a['name']: a['value']})

            # product_dict[pid] = nested_dict()
            product_dict[pid]["product_title"] = title
            product_dict[pid]["product_description"] = desc
            product_dict[pid]["product_category"] = cat
            product_dict[pid]["product_attributes"] = attr
            product_dict[pid]["target_label"] = target_label

    print(f"Collected {len(product_dict)} products.\n")
    return product_dict

def nested_dict():
    return defaultdict(nested_dict)


def process_product_information(input_file, target_label):
    print(f"Processing input file: {input_file}")
    data = []
    with open(input_file, "r") as f:
        for line in f:
            data.append(json.loads(line))

    product_dict = nested_dict()
    for d in data:
        pid = (d["product_id"])
        title = (d["name"][0]["value"])
        desc = (d["description"][0]["value"])
        cat = d["taxonomy"]["business_center"]
        attr_list = (d["taxonomy"]["attributes"])

        attr = []
        exclude_attr_key = ['aged', 'feed', 'origin', 'fresh', 'frozen', 'packag', 'harvest', 'imported', 'domestic',
                            'bioengineer', 'organic', 'aqua', 'natural', 'child', 'grade', 'injected', 'shelf', 'sugar',
                            'chemical', 'water', 'allergen', 'nutrient', 'region', 'claim', 'kosher', 'casing',
                            'certified', 'flavor', 'diet type']
        exclude_attr_value = ['not ', 'no ', 'free', 'packag', 'fresh', 'frozen', 'imported', 'domestic', 'farm',
                              'bioengineer', 'injected', 'organic', 'aqua', 'natural', 'may contain', 'sugar',
                              'non breed', 'standard', 'flavor', 'claim', 'kosher', 'cooking', 'casing', 'certified',
                              'artificial', 'raw', 'contains']
        for a in attr_list:
            if not any(sub.lower() in a['name'].lower() for sub in exclude_attr_key):
                if not any(sub.lower() in a['value'].lower() for sub in exclude_attr_value):
                    attr.append({a['name']: a['value']})

        product_dict[pid]["product_title"] = title
        product_dict[pid]["product_description"] = desc
        product_dict[pid]["product_category"] = cat
        product_dict[pid]["product_attributes"] = attr
        product_dict[pid]["target_label"] = target_label

    print(f"Collected {len(product_dict)} products from input file.\n")
    return product_dict


# Saving product dict so that it can be used when parsing predictions
def save_product_dict(products, product_dict_filename):
    with open(product_dict_filename, "wb") as f:
        pickle.dump(products, f)
    print(f"Product dict saved at {product_dict_filename}")


def write_batch_input_file(input_data, input_filename):
    with open(input_filename, "w") as f:
        for line in input_data:
            try:
                f.write(json.dumps(line) + "\n")
            except Exception as e:
                print(e)
                print(line)
    print(f"Saved input file: {input_filename}")


def download_gcs_output(output_uri=GCS["GCS_OUTPUT_URI"], project_id=GCS["PROJECT_ID"], local_dir="batch_output", filter_pattern=None, suffix=None):
    """
    Download and process batch prediction output files from GCS.

    Args:
        output_location (str): GCS URI to the output location
        local_dir (str): Local directory to download files to
        filter_pattern (str, optional): Pattern to filter output files
        project_id (str, optional): GCP project ID

    Returns:
        str: Path to the local directory containing downloaded files
    """
    # Create local directory if it doesn't exist
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    # Parse the GCS URI
    if not output_uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI: {output_uri}")

    gcs_path = output_uri.replace("gs://", "")
    bucket_name, prefix = gcs_path.split("/", 1)

    # Get the storage client
    storage_client = storage.Client(project=project_id)

    bucket = storage_client.get_bucket(bucket_name)

    # List all blobs in the specified location
    blobs = list(bucket.list_blobs(prefix=prefix))

    # Filter blobs if pattern is specified
    if filter_pattern:
        blobs = [blob for blob in blobs if filter_pattern in blob.name]

    # Download each blob
    downloaded_files = []
    for blob in blobs:
        # Skip directory markers
        if blob.name.endswith('/'):
            continue

        # Create local file path
        rel_path = blob.name[len(prefix):].lstrip('/')
        if suffix:
            rel_path = rel_path.replace(f".{rel_path.split(".")[1]}", f"_{suffix}.{rel_path.split(".")[1]}")
        local_file_path = os.path.join(local_dir, rel_path)

        # Create directory structure if needed
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

        # Download the blob
        blob.download_to_filename(local_file_path)
        downloaded_files.append(local_file_path)
        print(f"Downloaded: {local_file_path}")

    return local_dir


def upload_input_file_to_gcs(input_file_path, bucket_name=GCS["BUCKET"]):

    input_filename = input_file_path.split("/")[-1]
    storage_client = storage.Client(project=GCS["PROJECT_ID"])
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(f"batch_inputs/{input_filename}")
    # blob.upload_from_filename(input_file_path)
    blob.chunk_size = 15 * 1024 * 1024  # 15MB
    blob.upload_from_filename(
        input_file_path,
        timeout=600
    )

    upload_uri = f"gs://{bucket_name}/batch_inputs/{input_filename}"
    print(f"Uploaded input to {upload_uri}")
    return upload_uri
