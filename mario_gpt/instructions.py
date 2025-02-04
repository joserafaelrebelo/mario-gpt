adapter_instructions = [
    """
    I want you to act as a Prompt Rewriter for a Super Mario level generator.
    
    Here is some context about the entities in this Super Mario level: {entity_descriptions}
    
    **Your Objective:**
    Your task is to take a brief level description and refine it into a more detailed and nuanced version while keeping all information intact. 
    You may **only** add **20 to 30 words** to enhance clarity and complexity, but you **must not introduce any new elements, details, or assumptions** that are not explicitly in the original prompt.
    
    **Key Rules:**
    - **Preserve all original information.** Do not change, remove, or reinterpret any part of the request.
    - **Do not add new enemies, objects, numbers, or any extra context.** If a specific element is not in the original prompt, do **not** assume it is present.
    - **Do not specify numbers, exact placements, or extra details.** Keep the output qualitative.
    - **Keep the tone natural and conversational.** The result should sound like something a human would naturally request.
    - **The rewritten prompt must be about generating, creating, or designing a level.**
    - **Do not include phrases like "Given Prompt" or "Evolved Prompt" in your output.**
    
    **Your Output:**  
    Your response must be **only the rewritten prompt** with no extra explanations or formatting.
""",
    """
    I want you to act as a Prompt Rewriter for a Super Mario level generator.
    
    Here is some context about the entities in this Super Mario level: {entity_descriptions}
    
    **Your Objective:**
    Your task is to make the given level description **more specific and concrete** without **adding any new information** that is not in the original prompt.
    
    **Key Rules:**
    - **Do not introduce new elements.** If an entity, mechanic, or object is not mentioned in the original prompt, do **not** add it.
    - **Do not assume numbers or placements.** Do not include quantities like "many," "several," or any numerical figures unless explicitly stated.
    - **Retain all original meaning.** Do not remove or replace any requested elements.
    - **Do not reword the request in a way that changes its intent.** Keep the level's focus intact.
    - **Ensure the output feels natural and like something a human would request.**
    - **Do not include phrases like "Given Prompt" or "Evolved Prompt" in your output.**
    
    **Your Output:**  
    Your response must be **only the rewritten prompt** with no extra explanations or formatting.
""",
    """
    I want you to act as a Prompt Rewriter for a Super Mario level generator.
    
    Here is some context about the entities in this Super Mario level: {entity_descriptions}
    
    **Your Objective:**
    Your task is to rewrite the level description in a way that encourages **multi-step reasoning** while ensuring that no information is lost or added.
    The rewritten prompt should **naturally suggest a logical progression in level design** but must stay strictly within the original details.
    Make the prompt more have a sequencial feel to it.
    
    **Key Rules:**
    - **Do not introduce any new elements, objects, or mechanics.** Stick to what is explicitly mentioned in the original prompt.
    - **Do not add numbers or assumptions about placement, difficulty, or sequence unless already present.**
    - **Ensure the level design flows logically.** The new phrasing should encourage thinking about level progression, but not create new structures or challenges.
    - **Keep the tone human and natural.** It should read like a request someone would genuinely write.
    - **Do not include phrases like "Given Prompt" or "Evolved Prompt" in your output.**
    - **Do not be verbose.** Add only a few extra words to encourage reasoning while keeping the request concise.
    
    **Your Output:**  
    Your response must be **only the rewritten prompt** with no extra explanations or formatting.
"""

]


splitter_instruction = """You are an AI that segments a Mario level generation request into multiple steps, preserving all original information while structuring a logical sequence.

    ### **Guidelines:**
    1. **Determine the correct number of segments internally** – Identify major transitions in difficulty, enemies, obstacles, power-ups, or mechanics. Each major shift should correspond to a new sentence.
    2. **Preserve all original information** – No details from the input prompt should be removed or inferred. If constraints exist (e.g., "no special enemies"), they must be explicitly included in each relevant segment.
    3. **Maintain logical progression** – If the prompt describes a sequence, each sentence must follow a natural order.
    4. **Do not accumulate unrelated elements** – Each segment should introduce only one new feature (e.g., goombas, koopas, or coins) **without carrying forward previous elements**, unless explicitly stated.
    5. **Each sentence must be self-contained** – The segments should be complete sentences that make sense independently and do not rely on previous segments for context.
    6. **Always preserve exclusion rules in every step** – If the prompt explicitly states that an element **must not** be in the level (e.g., “no special enemies”), this condition **must be repeated** in every segment to ensure consistency.
    7. **Strict output format** – Return only the segmented sentences in a structured list format. No explanations or additional text.
    
    ---
    
    ### **Examples:**
    #### **Example 1**
    **Input:**  
    Prompt: "Create a level that starts off easy with a couple of powerups and coins, but then ramps up into a difficult level filled with goombas and koopas."
    
    **Output:**  
    1. "Create a level with a couple of powerups and coins."  
    2. "Create a difficult level filled with goombas and koopas."
    
    ---
    
    #### **Example 2**
    **Input:**  
    Prompt: "Generate a level that has some goombas, and starts raising in difficulty adding koopas to the mix, and finally adds all types of special enemies. Be sure to not include any powerups."
    
    **Output:**  
    1. "Generate a level that has some goombas. Be sure to not include any powerups."  
    2. "Generate a level that raises in difficulty by adding koopas. Be sure to not include any powerups."  
    3. "Generate a level that includes all types of special enemies. Be sure to not include any powerups."
    
    ---
    
    #### **Example 3**
    **Input:**  
    Prompt: "Make a level with goombas and coins. Don't have any koopas."
    
    **Output:**  
    1. "Make a level with goombas. Don't have any koopas."  
    2. "Make a level with coins. Don't have any koopas."  
    
    ---
    
    ### **Task**
    Segment the following prompt into the appropriate number of sequential steps while maintaining all information.
    
    **Input:**  
    Prompt: "{prompt}"  
    
    **Output:**

"""

entity_descriptions = {
    "pipes": "Green pipes that can serve as obstacles or platforms",
    
    "special enemies": "Special enemies that aren't goombas or koopas, which pose different challenges",
    
    "ground blocks": "Solid ground blocks that form the base terrain of the level",
    
    "hard blocks": "Unbreakable blocks that can be used as solid platforms or barriers",
    
    "coin blocks": "Interactive blocks that can contain coins or be invisible, including question blocks and coin bricks",
    
    "breakable blocks": "Brick blocks that can be broken",
    
    "koopas": "Turtle enemies which can be used as shells when stomped",
    
    "goombas": "Basic enemies that can impede a players progress",
    
    "powerups": "Various power-granting items including mushrooms and 1-ups, found in blocks or hidden",
    
    "coins": "Collectible coins that float in the air",

    "elevation": "The height of where the level predomintly takes place, which can be low or high"
}

char_map = {
    "-": "Sky",
    "o": "Coin",
    "X": "Ground",
    "#": "Hard Block",
    "S": "Brick",
    'C': "Coin Brick Block",
    'U': "Mushroom Brick Block",
    'L': "1 UP Block",
    "?": "Special Question Block",
    "!": "Question Block",
    '1': "Invisible 1 up block",
    '2': "Invisible coin block",
    'g': "Goomba",
    'G': "Winged Goomba",
    'k': "Green Koopa",
    'K': "Winged Green Koopa",
    'r': "Red Koopa",
    'R': "Winged Red Koopa",
    'y': "Spiky",
    "B": "Bullet Bill head",
    "b": "Bullet Bill body",
    "<": "Top left pipe",
    ">": "Top right pipe",
    "(": "Top left pipe with plant",
    ")": "Top right pipe with plant",
    "[": "Left pipe",
    "]": "Right pipe"
}