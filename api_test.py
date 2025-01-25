from fake_api import *
# from azure_api import *
def get_responce():
    messages=[]
    system_prompt = """
        **Instruction**:
        You are a tool selector for an image editing service. Based on the user's request, 
        you need to determine which tool from the following list is most suitable to perform the desired action.
            
        **Available Tool**:
        - contrast: Enhances or reduces the difference in brightness between areas of the image.
        - brightness: Increases or decreases the overall lightness or darkness of the image.
        - recolor: Changes the colors of the image, including applying filters or specific color transformations.

        **Output Format**:
        ```json
        {"Tool": <tool_name>}
        ```
            
        **Key Notes**:
        - The output must be one of the tools listed below.
        - Do not suggest any tools that are not in the provided list.
        - Return the response in the json format.
    """
    messages.append(message_template('system', system_prompt))
    messages.append(message_template('user', 'Change the color of the clothes to red.'))
    response=api_answer(messages,'json')
    print(response)

get_responce()
