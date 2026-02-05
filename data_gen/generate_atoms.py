"""
Generate Atomic Skill Data for Arithmetic Reasoning

Atomic skills are simple, foundational operations that underlie more complex tasks.
These serve as pre-training data before the sibling tasks.

Skills:
- single_op: Basic arithmetic operations (7 + 5 = ?)
- parity: Even/odd classification (Is 7 even or odd?)
- sign_check: Positive/negative classification (Is -5 positive?)
- comparison: Numerical comparison (Is 7 > 12?)
- order_of_ops: Order of operations (2 + 3 * 4 = ?)
"""

import pandas as pd
import random
import os
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "data" / "atoms"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def save_parquet(data: list, filename: str):
    """Save data as parquet file."""
    df = pd.DataFrame(data)
    filepath = OUTPUT_DIR / f"{filename}.parquet"
    df.to_parquet(filepath)
    print(f"Saved {len(data)} rows to {filepath}")


def format_sft(question: str, reasoning: str, answer: str) -> dict:
    """Format as SFT training example with thinking tokens."""
    response = f"<think>\n{reasoning}\n</think>\n\n#### {answer}"
    return {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": response}
        ]
    }


def gen_single_op(n: int = 2000) -> None:
    """Generate single arithmetic operation examples."""
    data = []
    operators = ['+', '-', '*', '/']
    
    for _ in range(n):
        op = random.choice(operators)
        
        if op == '/':
            # Ensure clean division
            b = random.randint(1, 20)
            result = random.randint(1, 20)
            a = b * result
        else:
            a = random.randint(-50, 50)
            b = random.randint(-50, 50)
            if op == '+':
                result = a + b
            elif op == '-':
                result = a - b
            elif op == '*':
                result = a * b
        
        question = f"Calculate: {a} {op} {b} = ?"
        reasoning = f"Computing {a} {op} {b}.\n{a} {op} {b} = {result}"
        answer = str(result)
        
        data.append(format_sft(question, reasoning, answer))
    
    save_parquet(data, "atom_single_op")


def gen_parity(n: int = 1500) -> None:
    """Generate even/odd classification examples."""
    data = []
    
    for _ in range(n):
        num = random.randint(-100, 100)
        is_even = num % 2 == 0
        
        question = f"Is {num} even or odd?"
        
        if is_even:
            reasoning = f"Checking {num}. {num} ÷ 2 = {num // 2} with no remainder."
            answer = "even"
        else:
            reasoning = f"Checking {num}. {num} ÷ 2 = {num // 2} with remainder 1."
            answer = "odd"
        
        data.append(format_sft(question, reasoning, answer))
    
    save_parquet(data, "atom_parity")


def gen_sign_check(n: int = 1500) -> None:
    """Generate positive/negative/zero classification examples."""
    data = []
    
    for _ in range(n):
        num = random.randint(-100, 100)
        
        question = f"Is {num} positive, negative, or zero?"
        
        if num > 0:
            reasoning = f"{num} is greater than 0."
            answer = "positive"
        elif num < 0:
            reasoning = f"{num} is less than 0."
            answer = "negative"
        else:
            reasoning = f"{num} equals 0."
            answer = "zero"
        
        data.append(format_sft(question, reasoning, answer))
    
    save_parquet(data, "atom_sign_check")


def gen_comparison(n: int = 2000) -> None:
    """Generate numerical comparison examples."""
    data = []
    comparisons = ['>', '<', '>=', '<=', '==', '!=']
    
    for _ in range(n):
        a = random.randint(-100, 100)
        b = random.randint(-100, 100)
        comp = random.choice(comparisons)
        
        # Evaluate comparison
        if comp == '>':
            result = a > b
        elif comp == '<':
            result = a < b
        elif comp == '>=':
            result = a >= b
        elif comp == '<=':
            result = a <= b
        elif comp == '==':
            result = a == b
        else:  # !=
            result = a != b
        
        question = f"Is {a} {comp} {b}? Answer Yes or No."
        reasoning = f"Comparing {a} and {b}.\n{a} {comp} {b} is {'True' if result else 'False'}."
        answer = "Yes" if result else "No"
        
        data.append(format_sft(question, reasoning, answer))
    
    save_parquet(data, "atom_comparison")


def gen_order_of_ops(n: int = 2000) -> None:
    """Generate order of operations examples."""
    data = []
    
    for _ in range(n):
        # Generate 3-term expressions with mixed operators
        a = random.randint(1, 20)
        b = random.randint(1, 20)
        c = random.randint(1, 20)
        
        # Choose pattern
        pattern = random.choice([
            ('+', '*'),  # a + b * c
            ('-', '*'),  # a - b * c
            ('*', '+'),  # a * b + c
            ('*', '-'),  # a * b - c
            ('+', '/'),  # a + b / c (ensure clean division)
            ('-', '/'),  # a - b / c
        ])
        
        op1, op2 = pattern
        
        # Ensure clean division
        if op2 == '/':
            c = random.randint(1, 10)
            b = c * random.randint(1, 10)
        
        expr = f"{a} {op1} {b} {op2} {c}"
        result = eval(expr)
        
        # Only keep integer results
        if result != int(result):
            continue
        result = int(result)
        
        question = f"Calculate: {expr} = ?"
        
        if op2 in ['*', '/']:
            reasoning = f"Following order of operations (PEMDAS):\nFirst: {b} {op2} {c} = {eval(f'{b} {op2} {c}')}\nThen: {a} {op1} {int(eval(f'{b} {op2} {c}'))} = {result}"
        else:
            first_result = eval(f"{a} {op1} {b}")
            reasoning = f"Following order of operations (PEMDAS):\nFirst: {a} {op1} {b} = {first_result}\nThen: {first_result} {op2} {c} = {result}"
        
        answer = str(result)
        
        data.append(format_sft(question, reasoning, answer))
    
    # Trim to exact count
    data = data[:n]
    save_parquet(data, "atom_order_of_ops")


if __name__ == "__main__":
    print("Generating Atomic Skill Data for Arithmetic...")
    print("=" * 50)
    
    random.seed(42)
    
    gen_single_op(2000)
    gen_parity(1500)
    gen_sign_check(1500)
    gen_comparison(2000)
    gen_order_of_ops(2000)
    
    print("=" * 50)
    print(f"All atomic data saved to {OUTPUT_DIR}")
