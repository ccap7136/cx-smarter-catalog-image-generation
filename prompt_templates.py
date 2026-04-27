PROMPT_TEMPLATES = {
    'STYLED': {
        'PROTEINS': {
            "TASK":
                """
                    Task: Generate an accurate, high-quality, e-commerce-ready food photograph based on provided 
                    product data, accuracy rules and image instructions.
                    The food item should be in a cooked state and a styled setting.
                    The product information is factual and should be treated as ground truth when generating the image.
                """,
            "REFERENCE_IMG_INSTRUCTIONS":
                """
                    - Use the provided image strictly as a reference to the protein's physical characteristics 
                    (cut, shape, thickness, skin/bone presence, marbling, and muscle structure).  
                    - If the reference image depicts a raw product, ensure that the generated image depicts the same product in a cooked state.
                    - CRITICAL: Do not copy the background, props, lighting, or plating. Use a new lifestyle setting.
                """,
            "COMPOSITION_REQUIREMENTS":
                """
                    - The product must be the primary visual focus, clearly visible and unobstructed.
                    - The entire dish must be fully visible in frame, with no cropping at the edges.
                    - The image must look like a real professional food photograph, with natural lighting, 
                    realistic textures, and accurate colors. 
                    - Avoid any artificial or computer-generated appearance.
                    - No humans or human hands visible, nor labels, writing or letters.
                """,

        },
    },
    'RAW': {
        'PROTEINS': {
            "TASK": "",
            "REFERENCE_IMG_INSTRUCTIONS": "",
            "COMPOSITION_REQUIREMENTS": "",
            }
    },
}