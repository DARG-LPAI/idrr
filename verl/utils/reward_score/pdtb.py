import re
from typing import Tuple, Optional


def extract_solution(solution_str: str) -> Tuple[Optional[str], Optional[str]]:
    content = None
    reasoning_content = None
    if m := re.match(r"<think>\n(.+)</think>\n\n", solution_str, flags=re.DOTALL):
        content = solution_str[len(m.group(0)):].strip()
        pattern = r'boxed{(.*?)}'
        matches = re.findall(pattern, content)
        if matches:
            content = matches[-1].strip()
        if thinking_content := m.group(1).strip():
            reasoning_content = thinking_content
    # if (content is None) or (reasoning_content is None):
    #     print("[Error] 思维链与答案解析出错")
    return content, reasoning_content

def compute_score(solution_str: str, ground_truth: str, algorithm: str = 'grpo'):
    """计算总得分"""
    # print("\n" + "="*80)
    # print(" 开始新的采样 ".center(80, '='))
    # 从模型输出中分离答案和思考过程
    answer_text, processed_str = extract_solution(solution_str)
    # 成功解析
    if answer_text and processed_str:
        if algorithm == 'grpo': # 长尾奖励
            alpha = 0.6 # 控制对长尾分布的关注程度
            label2weight = {
                'A': 1.32,
                'B': 1.13,
                'C': 1.0,
                'D': 2.0,
            }
            if answer_text == ground_truth:
                total_score = (1 - alpha) * 1.0 + alpha * label2weight.get(ground_truth, 1.0)
            else:
                total_score = -0.1
        elif answer_text == ground_truth: # 标准二值奖励
            total_score = 1
        else:
            total_score = -1 # 错误答案得分
    else:
        total_score = -1 # 解析失败得分
    # print(f" 最终得分{total_score} ".center(80, '-'))

    if algorithm == 'dapo':
        acc = 1 if total_score > 0 else 0
        return {
            "score": total_score,
            "acc": acc,
            "pred": answer_text,
        }
    else:
        return total_score
    
if __name__ == "__main__":
    # 测试代码
    answer_text = r'\boxed{A}'
    # answer = remove_boxed(answer_text)
    # print(f"去除boxed后的答案为: {answer}")