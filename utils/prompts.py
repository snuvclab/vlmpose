INSTRUCTIONS_POSE_EST = """# Role
You are a pose estimator that predicts which object in an image should be moved and specifies how it should be moved, using a provided text description.

# Instructions
1. List all distinct objects in the image.
2. Analyze the text description to determine:
- The specific object to move.
- The manner or goal of its movement.
3. Only proceed if exactly one object is identified as the movement target; otherwise, return an error JSON (see Output format).
4. Example: For "Pour the tea into a teacup using a teapot.", identify the teapot and teacup. Then, select the teapot as the moving object, deduce its movement (tilt to pour), determine pose change (position and orientation), and estimate the target pose.
5. Predict and output the pose change for the selected object:
- 3D translation as [Tx, Ty, Tz] (float array, minimum 3 decimal places).
- Dominant rotation axis: "x", "y", or "z" (lowercase string).
- Rotation angle in degrees (float, at least 2 decimal places).
- Rotation is applied counterclockwise around the axis (right-hand rule). 
6. The length of each axis arrow shown in the image is 1. Scale all translation values accordingly.
7. Explain logically and in detail the reasons for your decisions. When inferring translation, you must explain how many times each axis' length must be moved to achieve the goal. When inferring rotation, you must explain why an axis is dominant and explain the specific reasoning behind how many degrees of rotation should be made about that axis. See 'Example Output'.

# Axis Definitions
- Coordinate system is shown explicitly in the image (around the target object):
  - x-axis: Red arrow in the input image
  - y-axis: Green arrow in the input image
  - z-axis: Blue arrow in the input image
- Positive rotation about an axis is the direction your fingers curl when your right thumb points along the +axis. Equivalently, if you look from the tip of the +axis toward the origin, positive rotation appears counterclockwise. (right-hand rule).
- Each axis overlaid around the target object has length 1. When inferring translation, use the length of this axis to infer.

# The Meaning of [Behind, Front, Left, Right] in the Text Prompt
The text prompt may specify relations such as behind, front, left, and right. However, these are relative notions. Depending on the camera viewpoint, the meaning of behind/front/left/right in world coordinates can vary, which may confuse your reasoning. Therefore, I will define clear standards for these four words as follows:
1) behind corresponds to the −z-axis direction.
For example, if the prompt says “the apple is placed behind the bottle,” then the apple should be located more in the −z direction than the bottle.
2) front corresponds to the +z-axis direction.
For example, if the prompt says “the apple is placed in front of the bottle,” then the apple should be located more in the +z direction than the bottle.
3) left corresponds to the −x-axis direction.
For example, if the prompt says “the cap of the water bottle is facing left,” then the cap should face the −x direction.
4) right corresponds to the +x-axis direction.
For example, if the prompt says “the cap of the water bottle is facing right,” then the cap should face the +x direction.
5) top corresponds to the +y-axis direction, bottom corresponds to the −y-axis direction.

# Rotation Sign Convention
Use the sign of the rotation angle according to the right-hand rule, but determine it with the following concrete convention:
- A positive rotation about +x means the object rotates from +y toward +z.
- A positive rotation about +y means the object rotates from +z toward +x.
- A positive rotation about +z means the object rotates from +x toward +y.
Equivalently:
- +Rx: y -> z
- +Ry: z -> x
- +Rz: x -> y
Negative rotation means the opposite direction:
- -Rx: z -> y
- -Ry: x -> z
- -Rz: y -> x

# Output
Output a JSON object with these fields, in order:
- **Target object** (string or null): Name of the object to move, or null if undetermined.
- **Reasoning**: Detailed explanation of your analysis and decision (multi-line allowed).
- **Translation**: [Tx, Ty, Tz] (float array), or null.
- **Dominant rotation axis**: "x", "y", or "z", or null.
- **Angle**: Rotation in degrees (float), or null.

## Error Handling
- If a single, clear target object cannot be determined, set all fields except "Reasoning" to null. Explain the issue in "Reasoning".
- For numeric values, use floats with the specified precision; use null if value is indeterminate.

## Example Output
```json
{
"Target object": "teapot",
"Reasoning": "The description specifies pouring with a teapot into a teacup, so the teapot should move to pour into the cup. In the figure, the teacup lies behind the teapot, i.e., along the teapot’s negative z direction, so the teapot needs to translate backward by approximately 4.7× the teapot’s z-axis arrow length to reach the vicinity of the teacup and align the spout above the cup opening. Additionally, the teapot should translate upward by about 1.5× the teapot’s y-axis arrow length to place the spout at an appropriate height above the cup. After these translations, the teapot must be rotated to create a pouring attitude: specifically, it should be tilted by 38° about the z-axis (right-hand rule) so that the spout points downward into the cup, matching the expected pose change for pouring.",
"Translation": [0.000, 1.500, -4.700],
"Dominant rotation axis": "z",
"Angle": 38.00
}
```

## Error Example
```json
{
"Target object": null,
"Reasoning": "No single object could be identified for movement based on the description.",
"Translation": null,
"Dominant rotation axis": null,
"Angle": null
}
```"""

INSTRUCTIONS_TARGET_SELECT1 = """# Role
Given {N} multi-view images of a scene, identify all objects present and select the image that most clearly displays all of them.

# Instructions
- Review all provided images (numbered from 1 to {N} in the top left corner of the image).
- List all objects identified in the scene.
- Choose the image number that best and most comprehensively displays every object.

# Output Format
Return your answer as a JSON object:
- **Image number**: Integer (1 to {N}), indicating the chosen image.
- **Reasoning**: Short explanation of the selection.

## Example Output
```json
{{
  "Image number": {N},
  "Reasoning": "There are three objects in the scene: a table, a teacup, and a teapot. Image {N} is the view where all three objects are visible and most clearly distinguished."
}}
```"""

INSTRUCTIONS_TARGET_SELECT2 = """# Role
Select the target object (the object to be moved) and related objects in the image based on the provided text description.

# Input
- First image: original scene.
- Second image: object masks with:
  - Numbered white labels on black squares
  - Thin bounding boxes for each object
  - Labels centered in bounding boxes
- Text description: A text prompt describing the interaction between objects.

# Instructions
1. Identify what each label corresponds to using the masks and labels.
2. From the text description, determine:
   - The single object that should be moved or manipulated (target object)
   - Any other object(s) that are directly involved in or relevant to the intended interaction with the target object (related objects)
   - Exactly one target object must be selected
   - Multiple related objects may be selected when necessary
   - Example: For "Pour the tea into a teacup using a teapot.", the target object is the teapot, and the related object is the teacup
   
# Output Format
Return a JSON object with these fields in order:
- **Target label**: Number matching the target object's label
- **Target object**: Name of the one object to be moved or manipulated
- **Related labels**: List of label number(s) for the object(s) directly involved in or relevant to the target object's intended interaction
- **Related objects**: List of names of the related object(s)
- **Reasoning**: Short explanation of why the target object and related objects were selected

## Example Output
```json
{
  "Target label": 2,
  "Target object": "teapot",
  "Related labels": [3],
  "Related objects": ["teacup"],
  "Reasoning": "The description specifies pouring with a teapot into a teacup, so the teapot is the target object to be moved, and the teacup is a related object involved in the interaction."
}
```"""

INSTRUCTIONS_EVAL_FAITHFULNESS = """# Role
Evaluate multi-view scene renders for faithfulness to a text description and select the best supporting view.

# Instructions
Given {N} separate images, each a different camera view of the same rendered scene (labeled 1 to {N} at the top-left), assess whether the scene matches the provided text description. For the view that most clearly shows this, specify the image number. Use these criteria:
1. Do all objects named in the text appear in the scene?
2. Are the spatial relationships between objects described in the text description clear and plausible in the image?
3. Are there any physically invalid collisions between objects?
4. If information about rotation is specified in the text description, analyze in detail whether the rotation of the target object shown in the image matches the specified information.
5. Use all view images jointly to decide Faithfulness. You shouldn't judge faithfulness based on just one image. Keep in mind that the input images are multi-view images. Then pick one view image number that best supports your final judgment (faithful or unfaithful).

# Axis Definitions
- Coordinate system is shown explicitly in the image (around the target object):
  - x-axis: Red arrow in the input image
  - y-axis: Green arrow in the input image
  - z-axis: Blue arrow in the input image

# The Meaning of [Behind, Front, Left, Right] in the Text Prompt
The text prompt may specify relations such as behind, front, left, and right. However, these are relative notions. In multi-view images, the meaning of behind/front/left/right in world coordinates can vary depending on the camera viewpoint, which may confuse your reasoning. Therefore, I will define clear standards for these four words as follows:
1) behind corresponds to the −z-axis direction.
For example, if the prompt says “the apple is placed behind the bottle,” then the apple should be located more in the −z direction than the bottle.
2) front corresponds to the +z-axis direction.
For example, if the prompt says “the apple is placed in front of the bottle,” then the apple should be located more in the +z direction than the bottle.
3) left corresponds to the −x-axis direction.
For example, if the prompt says “the cap of the water bottle is facing left,” then the cap should face the −x direction.
4) right corresponds to the +x-axis direction.
For example, if the prompt says “the cap of the water bottle is facing right,” then the cap should face the +x direction.
5) top corresponds to the +y-axis direction, bottom corresponds to the −y-axis direction.

## Output Format
Return a JSON object with these fields:
- **Faithfulness**: "Yes" or "No" (capitalize exactly as shown), indicating if the multi-view images match the text description.
- **Image number**: Integer (1-{N}) of the view image best showing faithfulness. If multiple, select the lowest.
- **Reasoning**: Concise explanation for your choices, briefly summarizing your analysis. You should also specify which view image best supports your judgment. That is, if you judged it to be faithful, specify the image number that best shows faithfulness, and if you judged it to be unfaithful, specify the image number that best shows unfaithfulness.

### Example Output
```json
{{
  "Faithfulness": "Yes",
  "Image number": {N},
  "Reasoning": "The multi-view images overall accurately reflect the input text description, 'Pour the tea into a teacup using a teapot.' The teapot's spout is well positioned on the teacup, and it is tilted toward gravity, allowing the tea to flow naturally into the teacup. Image {N} is the view that most clearly demonstrates this."
}}
```

```json
{{
  "Faithfulness": "No",
  "Image number": 1,
  "Reasoning": "Analysis of the multi-view images reveals that the input text description 'Pour the tea into a teacup using a teapot' does not accurately reflect this. Although the teapot is positioned close to the teacup, its spout is not angled toward the teacup, preventing the tea from being poured into it in image 1."
}}
```"""
