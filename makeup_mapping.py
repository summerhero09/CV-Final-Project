SINGLE_FEATURE_RULES = {
    "Narrow_Eyes": [
        "Use lighter eyeshadow shades on the inner corners and center of the lid to visually open the eyes.",
        "Focus eyeliner and mascara toward the outer third of the eye to create width."
    ],

    "High_Cheekbones": [
        "Apply blush slightly lower on the cheeks rather than directly on the cheekbones for balance.",
        "Use subtle contour beneath the cheekbones to enhance structure without over-definition."
    ],

    "Rosy_Cheeks": [
        "Use a green-toned color corrector before foundation to neutralize redness.",
        "Choose neutral or peach-toned blushes instead of pink to avoid emphasizing redness."
    ],

    "Pointy_Nose": [
        "Apply soft matte contour along the sides of the nose and blend thoroughly to soften sharp angles.",
        "Avoid strong highlight on the nose tip to reduce emphasis."
    ],

    "Big_Nose": [
        "Use matte contour along the sides of the nose bridge to create balance.",
        "Shift visual focus toward the eyes or lips to draw attention away from the center of the face."
    ],

    "Big_Lips": [
        "Use soft matte or satin lipstick finishes instead of high-gloss to balance lip volume.",
        "Follow the natural lip line and avoid overlining."
    ],

    "Pale_Skin": [
        "Use soft peach or rose blush shades to add warmth to the complexion.",
        "Avoid overly dark contour shades; opt for light, neutral tones instead."
    ],

    "Oval_Face": [
        "Minimal contouring is needed; focus on blush placement to enhance natural balance.",
        "Experiment freely with eye and lip makeup, as most styles complement an oval face."
    ],

    "Arched_Eyebrows": [
        "Follow the natural brow arch and avoid flattening it with heavy filling.",
        "Use lighter brow products and softer strokes to keep the look balanced."
    ]
}
COMBO_RULES = {
    frozenset(["Narrow_Eyes", "Arched_Eyebrows"]): [
        "Keep brow makeup soft while using light-reflecting eyeshadows to open the eye area without over-sharpening features."
    ],

    frozenset(["Narrow_Eyes", "Big_Lips"]): [
        "Use brightening eye makeup to open the eyes while choosing neutral lip tones to maintain facial balance."
    ],

    frozenset(["High_Cheekbones", "Oval_Face"]): [
        "Use blush sparingly and focus on subtle highlighting to enhance structure without overpowering natural balance."
    ],

    frozenset(["Rosy_Cheeks", "Pale_Skin"]): [
        "Prioritize color correction and lightweight foundation to even skin tone while maintaining a natural finish."
    ],

    frozenset(["Pointy_Nose", "Big_Nose"]): [
        "Use soft, blended contouring techniques and avoid harsh highlights to create a more balanced nose appearance."
    ],

    frozenset(["Arched_Eyebrows", "Pointy_Nose"]): [
        "Balance strong facial angles with soft eye makeup and diffused contouring across the center of the face."
    ],

    frozenset(["Big_Lips", "Pale_Skin"]): [
        "Choose muted or cool-toned lip colors and balance with warm blush to avoid high contrast."
    ],

    frozenset(["High_Cheekbones", "Rosy_Cheeks"]): [
        "Apply blush lower on the cheeks and blend outward to reduce emphasis on redness while maintaining structure."
    ]
}