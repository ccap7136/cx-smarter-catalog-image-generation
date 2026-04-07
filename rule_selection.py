import argparse
import base64
from google import genai
from google.genai import types
import pickle
from tqdm import tqdm
import random
import time

from data_utils import *
from parse_predictions import parse_generation_prediction, generation_preds_to_csv, download_locally
from run_batch_job import monitor_batch_prediction_job, run_batch_prediction_job

MAX_WORKERS = 5  # adjust based on rate limits

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

PROTEIN_RULES = {
    'SEAFOOD': """
    - Fish species (CRITICAL):
      Species visual cues must be respected:
        • Salmon / trout: orange to pink flesh with visible white fat lines running through the muscle.
        • Tuna: deep red to dark pink flesh with dense, smooth muscle and minimal visible fat lines.
        • Cod / haddock / pollock: pale white flesh with large, delicate flakes that separate easily.
        • Sea bass: white to off-white flesh with firm muscle and small to medium flakes.
        • Snapper: white to light pink flesh with firm muscle, fine flakes, and a smooth surface texture.
        • Catfish: pale off-white to light pink flesh with smooth, slightly translucent muscle and very fine flake structure; typically thicker and more uniform in fillets.
        • Mackerel: darker flesh with visible oiliness and pronounced muscle grain.
      Do not depict a generic fish texture that contradicts the species.

    - Cut type:
      Respect the anatomical geometry of the cut:
      • Fillet: a long, tapered piece of fish muscle removed from the side of the fish, typically thicker at one end and thinner at the tail end.
      • Loin: a thick, rectangular or block-like center cut from the upper back of the fish, usually uniform in thickness and without tail tapering.
      • Steak (cross-section cut): a vertical slice through the whole fish body, often showing a central bone and symmetrical muscle sections.
      • Whole fish: intact fish body with natural proportions and anatomical features.
      • Bias cut (salmon): thin slices cut diagonally across the fillet at an angle rather than straight across. The slices should appear elongated and slightly oval with smooth edges, showing the salmon’s orange flesh and white fat lines running diagonally across the slice.

    - Preparation:
      Accurately depict the physical preparation (e.g., whole, filleted, sliced, cubed, minced, flakes).

    - Muscle structure (CRITICAL):
      Respect the natural muscle flake pattern of the species. Do not depict unrealistic muscle patterns.
      • White fish (cod, haddock): large, distinct flakes that separate easily.
      • Salmonids: medium flakes with visible fat marbling.
      • Tuna: dense, smooth muscle with minimal flaking.

    - Skin status (CRITICAL):
      If skinless is specified:
       • The fish must have NO skin whatsoever: no skin patches, no silver membrane, no scale texture, and no skin-colored edges. 
       • Only exposed flesh should be visible.
       • Do not add skin by default.

      If skin-on is specified:
       • Ensure the skin texture, pattern, and color match the species
       • Ensure that the skin sits on the correct outer surface of the fish portion.

    - Head status:
      If headless, ensure the head is completely absent.

    - Fillet constraints:
      If the product is described as a fillet:
       • Ensure it has no head and no tail attached
       • Ensure that the muscle thickness and tapering match a true fillet of the specified species.

    - Packaging:
      If the product is canned or packaged seafood (can, jar, tube, pouch, etc.):
       • Always show the seafood outside the package, with the edible product as the primary focus.
    """,

    'PORK': """
    - Pork cut (CRITICAL):
      Identify the exact pork cut (e.g., pork loin, pork shoulder/Boston butt, pork belly, tenderloin, ham, ribs) and match its characteristic muscle structure, fat distribution, and proportions.

      Visual cues by cut:
      • Pork loin: lean meat with a relatively uniform cylindrical or rectangular shape and a thin outer fat cap.
      • Pork tenderloin: small, long, narrow cylindrical muscle with very little visible fat.
      • Pork shoulder / Boston butt: thicker, irregular cut with heavier marbling and visible connective tissue.
      • Pork belly: flat slab with clearly layered bands of fat and meat running horizontally.
      • Ham: large hind-leg muscle with firm structure and moderate fat coverage.

    - Rib type (CRITICAL):
      When the product is ribs, strictly match the correct rib category and anatomy:
      • Baby back ribs: curved rack from the upper ribcage near the spine. Bones are shorter, curved, and closely spaced with more meat on top of the bones and less between them.
      • Spare ribs: flatter rack from the lower ribcage. Bones are longer and straighter with wider spacing, and the meat layer is thinner with more visible fat.
      • St. Louis–style ribs: trimmed version of spare ribs with a more rectangular rack shape, straight bone line, and removed rib tips.
      • Country-style ribs (boneless): thick, meat-heavy pieces cut from the pork shoulder/loin area. They must clearly appear as a boneless cut of pork with no bones present. They appear as chunky, irregular strips or blocks of meat with visible marbling and connective tissue.

      Do not confuse these rib types. For example:
      • Do not depict curved baby back racks when spare ribs are specified.
      • Do not show bones when country-style ribs are specified.

    - Shape:
      Match the overall geometry of the cut.
      • Rack of ribs: long connected bones forming a rack.
      • Pork belly: flat rectangular slab with layered fat and meat.
      • Tenderloin: narrow cylindrical shape tapering slightly at the ends.
      • Loin roast: thicker rectangular or cylindrical roast.
      • Country-style ribs: thick individual strips or chunks rather than a rib rack.

    - Fat distribution:
      Ensure the fat pattern matches the cut.
      • Pork belly: heavy fat layers alternating with meat.
      • Pork shoulder: strong marbling and connective tissue.
      • Pork loin / tenderloin: lean meat with minimal internal fat.
      • Ribs: fat concentrated around and between the rib bones.

    - Preparation:
      Depict the physical preparation accurately (e.g., whole, sliced, cubed, diced, ground, chopped).
      • If sliced, show natural muscle grain and realistic thickness.
      • If cubed or diced, ensure consistent cube size.
      • If ground or minced, depict loose strands of ground meat.
      • If product is a pork butt, depict it as pulled pork.

    - Bone status:
      If bone-in: bones must be visible and anatomically consistent with the cut.
      If boneless: ensure no bones or bone fragments are visible.
    """,

    'BEEF': """
    - Beef cut (CRITICAL):
      Identify the exact cut (e.g., ribeye, sirloin, brisket, striploin, cube steak) and strictly match its typical muscle structure, fat distribution, and marbling.

      Visual cues by cut:
      • Ribeye: round, thick, and well-marbled central eye of meat with surrounding cap (spinalis) muscle; slightly oval cross-section; can be bone-in (rib bone visible) or boneless.
      • Sirloin: leaner rectangular cut with moderate marbling; typically boneless, uniform thickness, firm muscle grain.
      • Brisket: long, flat, and thick cut from the chest; heavily marbled with connective tissue; tapering towards the edges; often used whole or trimmed.
      • Striploin / New York strip: long rectangular cut with a moderate fat cap on top; uniform thickness; lean central muscle with consistent marbling.
      • Cube steak: top round or top sirloin mechanically tenderized, leaving small square indentations on the surface; flat and thin; not actually cut into cubes.
      • Chuck: thicker, irregularly shaped, heavily marbled with connective tissue; often used for roasting or slow cooking.
      • Steak Strip: long, rectangular slice of beef, usually cut from a larger steak. It has a uniform thickness, visible muscle grain, and light marbling, with clean edges and a defined, elongated shape.

    - Shape:
      Account for the geometric shape and size of the cut.
      • Round ribeye: oval to circular cross-section.
      • Striploin: rectangular prism with rounded edges.
      • Brisket: long, flat, slightly tapered slab.
      • Cube steak: thin, flat rectangle with visible cubing pattern.

    - Marbling (CRITICAL):
      Match the expected intramuscular fat pattern for the cut.
      • Ribeye: heavy marbling in the central eye, fat cap on edge.
      • Sirloin: moderate marbling, mostly uniform.
      • Brisket: thick streaks of fat running along the length of the muscle.
      • Striploin: uniform marbling along the main muscle.
      • Chuck: irregular marbling with connective tissue strands.

    - Preparation:
      Depict the physical preparation accurately (e.g., whole, sliced, diced, ground, chopped, pounded).
      • Sliced: show realistic slice thickness and natural muscle grain.
      - If a slice includes a “tail” (e.g., “1 Inch Tail”), depict the tail tapering from the main body with the correct thickness (e.g., 1 inch thick). 
      Maintain the cut’s geometry while showing the thinner taper at the end.
      • Ground or chopped: depict loose strands or small chunks of meat.
      • Cube steak: include tenderized square indentations; maintain flat, thin profile.

    - Bone status:
      If Bone-in ribeye: show rib bone aligned with central eye of meat.
      If Boneless cuts: no bone visible.

    - Fat cap / trim:
      If the cut specifies a fat cap (e.g., striploin, ribeye), ensure it is visible along the top edge.
      If the product specifies a fat cap: do not add fat where the cut is described as lean or trimmed.
    """,

    'POULTRY': """
    - Poultry type (e.g., chicken, turkey, duck).
      Identify the bird and strictly match its typical color, fat distribution, muscle texture, and proportions.

    - Cut type:
      Identify and depict the specified poultry cut (e.g., breast, breast half, double breast, thigh, drumstick, leg quarter, wing, tenderloin, whole bird, split bird, spatchcocked bird).
      When depicting a specific cut, respect its characteristic anatomy and shape:
      • Breast: a thick boneless muscle with a smooth rounded top and tapered end.
      • Double breast: two connected breast lobes forming a natural heart-like or butterfly shape. One lobe is slightly larger, and both are joined along a central seam.
      • Breast half: a single breast lobe, thicker at the top and tapering toward the tip.
      • Thigh: a rounded, compact cut with darker meat and visible muscle grain.
      • Drumstick: elongated with a thicker rounded end and a narrower bone handle.
      • Leg quarter: thigh and drumstick connected together as one piece.
      • Wing: small segmented cut composed of drumette, flat, and tip.
      • Tenderloin: a thin elongated strip of white meat attached beneath the breast.

    - Shape:
      • For poultry breast depictions, enforce a realistic, lean appearance. Do not generate overly plump, thick, or inflated shapes; the breast should be visibly thinner and naturally proportioned.
      • When depicting sausages, ensure their size matches the product’s specified dimensions or weight

    - Color:
      • Ensure the meat color matches the cut.
      • Chicken breast meat must appear light white or pale pink.
      • Chicken thigh meat must appear darker brown or reddish-brown.
      • Do not depict breast meat as dark or thigh meat as white.
      • When a product includes multiple meat types (e.g., thigh + breast mix), respect the specified ratio when depicting color (e.g., 60% dark meat / 40% white meat).

    - Bone status:
      • If bone-in, ensure bones are visible and anatomically correct for the cut (e.g., breastbone for bone-in breast, drum bone for drumstick).
      • If deboned, ensure no bones are visible or implied.
      • When the product description includes the term ‘boned’, treat the product as deboned. Do not depict any bones. If the description includes ‘drum on’, retain the bones in the drum portion.”
      • If partially deboned (e.g., first joint on, second joint removed),reflect this precisely.


    - Skin status:
      If skin-off is specified:   
       • Only exposed flesh should be visible.
       • Do not add skin by default.
      If skin-on is specified, the skin must fully and continuously cover the portion, with no exposed muscle tissue.

    - Preparation:
      • Account for the physical preparation (e.g., whole, split, spatchcocked, butterflied, diced, sliced, ground, pounded thin).
      • If spatchcocked or split, depict a whole chicken cut entirely through the backbone, opened and laid flat, with neck and giblets removed. Show each half clearly detached from each other.
      • If diced, ensure realistic cube size and accurate meat-type proportions.
    """
}


def select_relevant_rules(product_id, product_info,
                          model_id=LLM_MODELS["FLASH_2.5"],
                          project_id=GCS["PROJECT_ID"],
                          location=GCS["LOCATION"], ):
    product_title = product_info['product_title']
    product_description = product_info['product_description']
    product_category = product_info['product_category']
    product_attributes = product_info['product_attributes']

    sys_instr = """
    You are an expert AI prompt engineer for image generation models. 
    You need to verify that the constraints provided to the prompt are relevant to the product in question.
    """

    prompt = f"""
                Consider the following food product (ID: {product_id}): 

                Product Title: "{product_title}".
                Product Description: "{product_description}".
                Product Attributes: {product_attributes}

                Task: Using the product information provided and analyzing the rules below, select the rules 
                that would be relevant when generating an image for this product.

                STEP 1 — Rule-by-Rule Compliance Check
                Consider the set of rules below:

                **{product_category}-Specific Accuracy Rules**
                {PROTEIN_RULES[product_category]}

                For each rule:
                - Analyze each of the rule's conditions.
                - Compare them to the product information.
                - Evaluate if the condition is relevant to the product above.

                STEP 2 — Output 
                Output: Return the same set of rules but only maintaining the conditions that are relevant for this product.
                Adjust provided examples to the product in question.
                Do not include any header statement before returning the rules.
    """

    client = genai.Client(
        vertexai=True,
        project=project_id,
        location=location,
    )
    rule_selection = None
    try:
        response = client.models.generate_content(
            model=model_id,  # "gemini-3-pro-preview",  # ensure this model exists in your project
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
                    sys_instr
                ),
                response_modalities=["TEXT"],
                temperature=0,
                # topP=0.1,
                max_output_tokens=32768,
            ),
        )

        # parse response
        rule_selection = response.candidates[0].content.parts[0].text  # ['content']['parts']#[0]['text']
    except Exception as e:
        print(e)

    return rule_selection

