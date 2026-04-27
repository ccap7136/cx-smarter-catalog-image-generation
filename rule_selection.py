from google import genai
from google.genai import types

from accuracy_rules import *
from data_utils import *

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


def select_relevant_rules(product_id, product_info,
                          model_id=LLM_MODELS["FLASH_2.5"],
                          project_id=GCS["PROJECT_ID"],
                          location=GCS["LOCATION"], ):
    product_title = product_info['product_title']
    product_description = product_info['product_description']
    product_category = product_info['product_category']
    product_attributes = product_info['product_attributes']
    target_label = product_info['target_label']

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
                {ACCURACY_RULES[target_label][product_category]}

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
