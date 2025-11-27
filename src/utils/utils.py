import re
import ast
import json

# from json_repair import repair_json

from .mylogger import logger

def read_file(path: str) -> list | dict:
    """Read data from txt or json or jsonl file."""
    if not isinstance(path, str):
        path = str(path)
    if path.endswith('.txt'):
        with open(path, 'r', encoding='utf-8') as f:
            return [i.rstrip() for i in f.readlines()] # list
    elif path.endswith('.json'):
        with open(path, 'r', encoding='utf8') as f:
            return json.load(f) # dict
    elif path.endswith('.jsonl'):
        with open(path, 'r', encoding='utf8') as f:
            return [json.loads(i) for i in f.readlines()] # list
    else:
        raise ValueError(f"File extension [{path.split('.')[-1]}] not valid.")
        
def write_file(path: str, data: list | dict) -> None:
    # 检查文件路径是否以 '.txt' 结尾
    if not isinstance(path, str):
        path = str(path)
    if path.endswith('.txt'):
        with open(path, 'w', encoding='utf-8') as f:
            for i in data:
                f.write(i + '\n')
    elif path.endswith('.json'):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    elif path.endswith('.jsonl'):
        with open(path, 'w', encoding='utf-8') as f:
            for i in data:
                f.write(json.dumps(i, ensure_ascii=False) + '\n')
    else:
        print(f'File extension [{path}] not valid.')
        raise ValueError(f"File extension [{path.split('.')[-1]}] not valid.")

def re_search(text, type):
    '''
    搜索text中符合type类型的内容,
    如果text中存在多个符合type类型的内容，则返回最后一个
    如果搜索不到，则抛出异常
    '''
    pattern = ''
    flags = re.DOTALL  # 添加 DOTALL 标志，让 . 也匹配换行符
    
    if type == 'json':
        pattern = r'```json\s*(.*?)\s*```'
    elif type == 'box':
        pattern = r'boxed{(.*?)}'
    elif type == 'xml':
        pattern = r'```xml\s*(.*?)\s*```'

    matches = re.findall(pattern, text, flags)
    if not matches and type == 'json':
        # 使用贪婪匹配来匹配完整的字典结构
        pattern = r'\{.*\}'
        matches = re.findall(pattern, text, flags)

    if len(matches) > 1:
        logger.warning(f"the number of matches is greater than 1:")
        for i, match in enumerate(matches):
            logger.debug(f"match{i}:\n{match}\n")
    elif not matches:
        logger.error(f"no matches found for {type} in text:\n{text}\n")
        raise ValueError()
        return text

    structured_text = matches[-1]
    # if type == 'json':
    #     structured_text = repair_json(structured_text)
    return structured_text

def extract_json(response: str) -> dict:
    # First, try to find JSON within ```json ... ``` blocks
    # The (.*?) captures the content within the ```json ... ``` block.
    # response = response.replace('', '')
    markdown_matches = re.findall(r'```json\s*(.*?)\s*```', response, re.DOTALL)
    
    candidate_strings_for_eval = []
    if markdown_matches:
        candidate_strings_for_eval = markdown_matches 
    else:
        # Fallback to the original regex if no markdown blocks are found
        # This regex finds strings starting with { and ending with }
        candidate_strings_for_eval = re.findall(r'\{.*?\}', response, re.DOTALL)


    if len(candidate_strings_for_eval) != 1:
        logger.debug(f'response:\n{response}\n')
        raise ValueError(f'JSON匹配数量不一致！！！')
    
    parsed_objects = []

    for match_str in candidate_strings_for_eval:
        try:
            # Clean up the match string (though ast.literal_eval is often robust to this)
            clean_match_str = match_str.strip()
            if not clean_match_str: # Skip empty strings that might result from regex or strip
                continue

            res = ast.literal_eval(clean_match_str)
            
            # Check if 'Negative' is a value in the dictionary
            if isinstance(res, dict):
                # Check all values for 'Negative'
                # Using res.values() is fine if you expect 'Negative' to be a direct value.
                # If 'Negative' could be nested or part of a key, logic would need adjustment.
                if 'Negative' in res.values():
                    return 'Negative'
            
            parsed_objects.append(res)
        except SyntaxError:
            # This occurs if match_str is not a valid Python literal
            logger.warning(f"SyntaxError parsing segment: >>>{clean_match_str}<<< (from original: >>>{match_str}<<<)")
            pass # Continue to the next match
        except ValueError:
            # This can occur if it's a malformed literal (e.g. dict with too many commas)
            logger.warning(f"ValueError parsing segment: >>>{clean_match_str}<<< (from original: >>>{match_str}<<<)")
            pass # Continue to the next match
        except Exception as e:
            # Catch any other unexpected errors during parsing of a segment
            logger.warning(f"Unexpected error parsing segment: {e}")
            logger.warning(f"Problematic segment: >>>{clean_match_str}<<< (from original: >>>{match_str}<<<)")
            # Optionally, print the full response for context, but can be verbose
            # print(f"Original response context: \n{response}\n")

    if parsed_objects:
        # If 'Negative' was not found in any of the parsed dicts,
        # return the last successfully parsed object.
        return parsed_objects[-1]
    
    # If nothing was successfully parsed from any candidate string
    # print(f"Could not find or parse any valid JSON-like structure in the response.")
    # print(f"Original response was:\n{response}\n")
    logger.debug(f'debug: response:\n{response}\n')
    raise ValueError(f'Could not find or parse any valid JSON-like structure in the response.')