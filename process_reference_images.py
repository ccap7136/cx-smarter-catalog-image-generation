import argparse
import re
import xml.etree.ElementTree as ET

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from data_utils import *

# Load once (do NOT reload inside the function in real code)
model = SentenceTransformer("all-MiniLM-L6-v2")


def extract_entities(xml_text):
    """
    Extract entity declarations:
    <!ENTITY asset0000001 SYSTEM "12PX_Scale_A-S.JPG" NDATA image_jpeg>

    Returns dict:
    {
        "asset0000001": "12PX_Scale_A-S.JPG",
        "asset0000002": "7056839-50794297871553_C1CH-S.JPG",
        ...
    }
    """
    entity_pattern = r'<!ENTITY\s+(asset\d+)\s+SYSTEM\s+"([^"]+)"'
    return dict(re.findall(entity_pattern, xml_text))


def extract_metadata(metadata, key):
    """
    Extract SUPC from METADATA.
    """

    # Case 1: <SUPC>1234</SUPC>
    tag = metadata.find(key)
    if tag is not None and tag.text:
        return tag.text.strip()

    # Case 2: SUPC attribute <METADATA SUPC="1234">
    if key in metadata.attrib:
        return metadata.attrib[key]

    # Case 3: SUPC attribute inside nested tags
    for child in metadata.iter():
        if key in child.attrib:
            return child.attrib[key]

    return None


def parse_assets(xml_path, asset_data=[], gcp_prefix=None):
    """
    Main function: links asset_id → SYSTEM filename → SUPC.
    """
    # Read raw XML text to parse entity declarations
    with open(xml_path, 'r', encoding='utf-8') as f:
        xml_text = f.read()

    # Step 1: Extract SYSTEM filenames from entity declarations
    entity_map = extract_entities(xml_text)
    tree = ET.ElementTree(ET.fromstring(xml_text))
    root = tree.getroot()

    # Step 2: Extract supc
    for asset in root.findall(".//ASSET"):
        metadata = asset.find("METADATA")
        asset_id = asset.find("CONTENT").find("MASTER").find("OBJECT").get("FILE")

        # SUPC
        supc = extract_metadata(metadata, 'SUPC')
        desc = extract_metadata(metadata, 'MATERIALDESCRIPTION')
        bc = extract_metadata(metadata, 'LEVEL_2')
        published = extract_metadata(metadata, 'IMAGE_STATUS_CODE')
        brand = extract_metadata(metadata, 'BRANDCODE')

        if published.lower() == 'published':
            published = True
        else:
            published = False

        if brand.lower() == 'y':
            sysco_brand = True
        else:
            sysco_brand = False

        if gcp_prefix is not None:
            img_path = f"gs://{GCS['BUCKET']}/{gcp_prefix}{entity_map[asset_id]}"
        else:
            img_path = xml_path.replace("assetProperties.xml", entity_map[asset_id])
        asset_data.append({'supc': supc, 'description': desc, 'image_path': img_path, 'business_center': bc,
                           'published': published, 'sysco_brand': sysco_brand})
    return asset_data


def fetch_reference_images(local_folder=None, bucket=None):

    asset_data = []

    if local_folder:
        print(f"Fetching reference images from {local_folder}")
        for folder in os.listdir(local_folder):
            if folder.startswith("."):
                continue
            xml_file = f"{local_folder}/{folder}/assetProperties.xml"
            asset_data = parse_assets(xml_file, asset_data)

    if bucket:
        client = storage.Client()
        bucket_name = GCS["BUCKET"]
        prefix = "reference_images"
        blobs = client.list_blobs(bucket_name, prefix=prefix)
        for blob in blobs:
            if blob.name.endswith(".xml"):
                temp_asset_file = "assets_temp.xml"
                blob.download_to_filename(temp_asset_file)
                blob_prefix = blob.name.replace("assetProperties.xml", "")
                asset_data = parse_assets(temp_asset_file, asset_data, gcp_prefix=blob_prefix)

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
    parser.add_argument('--input_product_file', type=str, default=DATA["INPUT_PRODUCT_FILE"],
                        help='JSONL file with product data')
    parser.add_argument('--reference_img_folder', type=str, default=GCS["GCS_REFERENCE_URI"],
                        help='Path to local reference images folder or GCS URI')
    parser.add_argument('--product_ids', type=str,
                        help='Product IDs separated by commas. E.g. "1234567,7654321,0987654')

    args = parser.parse_args()
    batch_job = args.batch_job
    os.makedirs(batch_job, exist_ok=True)

    # Read input file and process product information
    print("Loading product information ...")
    product_dict = process_product_information(DATA["INPUT_PRODUCT_FILE"])

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
    if args.reference_img_folder.startswith('gs://'):
        print(f'Reading reference images from GCS: {GCS["BUCKET"]}/reference_images...')
        asset_data = fetch_reference_images(bucket=GCS["BUCKET"])
    elif args.reference_img_folder is not None:
        print(f'Reading reference images from local directory: {DATA["REFERENCE_IMG_FOLDER"]}...')
        asset_data = fetch_reference_images(local_folder=DATA["REFERENCE_IMG_FOLDER"])
    else:
        print("No reference image bucket selected")
    product_dict = select_reference_images(product_dict, asset_data)

    # Saving product dict (for parsing predictions)
    save_product_dict(product_dict, f"{batch_job}/{DATA["PRODUCT_DICT_FILE"]}")

if __name__ == "__main__":
    main()