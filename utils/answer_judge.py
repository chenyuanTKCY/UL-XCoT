########################################################################
#
# @author : Chenyuan Zhang
# @when : Winter Semester 2025/2026
# @where : Harbin Institute of Technology
# @title : Answer judge by a LLM
# @component: utils
# @file : answer_judge.py
#
#########################################################################

import openai
import re
from openai import OpenAI

class answerJudge:
    def __init__(self, model_name:str = "deepseek-reasoner",  api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key

    
    def get_answer(self, pred: str, answer: str) -> int:
        client = OpenAI(
            api_key=self.api_key)
        instruction = f"""As a mathematics expert, compare these two mathematical answers:

        Answer A: {pred}
        Answer B: {answer}

        Determine if Answer A is mathematically equivalent to Answer B. Consider:
        - Numerical equality for exact values
        - Algebraic equivalence for expressions  
        - Simplified forms and different representations
        - Common mathematical identities

        If they are equivalent, respond with "1". If not equivalent, respond with "0".
        Provide only the number as your response."""
        
        messages = [{"role": "user", "content": instruction}]
        
        try:
            completion = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1,  
                top_p=0.9,
            )
            
            content = completion.choices[0].message.content.strip()

            if content in ['1', '0']:
                return int(content)
            else:
                numbers = re.findall(r'\b[01]\b', content)
                if numbers:
                    return int(numbers[-1])
                else:
                    print(f"Unexpected response: {content}")
                    return 0
                    
        except Exception as e:
            print(f"API Error: {e}")
            return 0
        