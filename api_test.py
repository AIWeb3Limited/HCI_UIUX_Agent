from fake_api import *
# from azure_api import *
def get_responce():
    messages=[]
    system_prompt = """
        **Instructions**:
        You are an advanced image editing assistant capable of handling a variety of image editing requests. 
        The allowable operations are defined in the following list: ["recolor", "brightness", "contrast", "saturation"]. 
        Your task is to interpret the user's request and select the appropriate operation type from this predefined list, 
        then provide corresponding options.

        **Guidelines**:
        1. **Identify the Request Type:** Based on the user's request, determine which operation type from the predefined list best matches the user's need.
        2. **Object Identification:** If applicable (mainly for recolor operations), identify the object within the image that the user wants to modify.
        3. **Provide Adjustment Options:**
            - For **recolor** operations, follow previous instructions for identifying objects and provide a gradient of at least 8 hexadecimal color codes that represent variations from light to dark shades of that color
            - For adjustments like **brightness**, **contrast**, or **saturation**, provide adjustment values as floating-point numbers where 1 represents the original effect. Values less than 1 indicate a reduction in intensity, while values greater than 1 increase it.
            - If the request involves more abstract or subjective criteria (e.g., "make it look warmer"), choose settings that best fit the description based on common design principles and aesthetics.
        4. **Response Formatting:** Return your response in JSON format with details about the selected adjustment type, relevant object (if applicable), and specific options.

        **Output Format**:
        {
            "Adjustment Type": <type_of_adjustment>,
            "Object Class": <object_class>, // Only needed if applicable, e.g., for color changes
            "Adjustment Options": [<options>] 
        }

        **Examples**:  
        User Request: "Make the car look blue."
        ```json
        {
            "Adjustment Type": "recolor",
            "Object Class": "the car",
            "Adjustment Options": ["#ADD8E6", "#87CEEB", "#6495ED", "#4682B4", "#0000FF", "#0000CD", "#00008B", "#000080"]
        }
        ```
        User Request: "Increase the contrast."
        ```json
        {
            "Adjustment Type": "contrast",
            "Adjustment Options": [0.25, 0.5, 1, 1.5, 2]
        }
        ```
    """
    messages.append(message_template('system', system_prompt))
    messages.append(message_template('user', 'Change the color of the clothes to red.'))
    response=api_answer(messages,'json')
    print(response)

get_responce()
