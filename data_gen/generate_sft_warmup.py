"""
Generate SFT Warmup Data for Base Models (Task-Agnostic)

Creates a small JSONL dataset (~500 examples) with <think>...</think> and
<answer>...</answer> structured outputs to teach a pretrained-only model
(e.g. Qwen3-4B-Base) the expected output format before GRPO training.

IMPORTANT: Content is deliberately task-agnostic — no reasoning-gym task
content is used. This prevents the warmup from giving the model a head
start on any specific benchmark task.

Categories:
- General knowledge trivia
- Unit conversions
- Simple word/letter puzzles
- Date/time reasoning
- Vocabulary/definitions

Usage:
    python data_gen/generate_sft_warmup.py [--output data_gen/data/sft_warmup.jsonl] [--num_samples 500]
"""

import json
import argparse
import random
from pathlib import Path


SYSTEM_PROMPT = (
    "You are a helpful assistant. Think step by step inside "
    "<think>...</think> tags, then provide your final answer "
    "inside <answer>...</answer> tags."
)


# ---------------------------------------------------------------
# Task-agnostic Q&A generators
# ---------------------------------------------------------------

def generate_trivia_examples(num_samples: int, rng: random.Random) -> list:
    """General knowledge trivia — no overlap with reasoning-gym tasks."""
    trivia = [
        ("What is the capital of France?", "Paris is the capital of France.", "Paris"),
        ("What is the largest ocean on Earth?", "The Pacific Ocean is the largest, covering about 63 million square miles.", "Pacific Ocean"),
        ("Who wrote Romeo and Juliet?", "Romeo and Juliet was written by William Shakespeare.", "William Shakespeare"),
        ("What planet is known as the Red Planet?", "Mars is known as the Red Planet due to iron oxide on its surface.", "Mars"),
        ("What is the chemical symbol for gold?", "Gold's chemical symbol is Au, from the Latin 'aurum'.", "Au"),
        ("How many continents are there?", "There are 7 continents: Africa, Antarctica, Asia, Australia, Europe, North America, and South America.", "7"),
        ("What is the smallest country in the world by area?", "Vatican City is the smallest country, at about 0.44 km².", "Vatican City"),
        ("What gas do plants absorb from the atmosphere?", "Plants absorb carbon dioxide (CO2) during photosynthesis.", "Carbon dioxide"),
        ("Who painted the Mona Lisa?", "The Mona Lisa was painted by Leonardo da Vinci.", "Leonardo da Vinci"),
        ("What is the boiling point of water in Celsius?", "Water boils at 100 degrees Celsius at standard atmospheric pressure.", "100"),
        ("What is the largest mammal?", "The blue whale is the largest mammal, reaching up to 30 meters in length.", "Blue whale"),
        ("What year did World War II end?", "World War II ended in 1945.", "1945"),
        ("What is the speed of light in km/s (approximately)?", "The speed of light is approximately 300,000 km/s.", "300,000 km/s"),
        ("What is the hardest natural substance?", "Diamond is the hardest natural substance.", "Diamond"),
        ("How many bones are in the adult human body?", "An adult human body has 206 bones.", "206"),
        ("What is the currency of Japan?", "The currency of Japan is the Yen.", "Yen"),
        ("What is the tallest mountain in the world?", "Mount Everest is the tallest mountain at 8,849 meters.", "Mount Everest"),
        ("What element does 'O' represent on the periodic table?", "O represents Oxygen on the periodic table.", "Oxygen"),
        ("Who discovered penicillin?", "Penicillin was discovered by Alexander Fleming in 1928.", "Alexander Fleming"),
        ("What is the longest river in the world?", "The Nile is generally considered the longest river at about 6,650 km.", "Nile"),
        ("What language has the most native speakers?", "Mandarin Chinese has the most native speakers.", "Mandarin Chinese"),
        ("What is the freezing point of water in Fahrenheit?", "Water freezes at 32 degrees Fahrenheit.", "32"),
        ("What organ pumps blood through the body?", "The heart pumps blood through the circulatory system.", "Heart"),
        ("How many sides does a hexagon have?", "A hexagon has 6 sides.", "6"),
        ("What is the main ingredient in guacamole?", "The main ingredient in guacamole is avocado.", "Avocado"),
    ]

    examples = []
    for _ in range(num_samples):
        q, reasoning, a = rng.choice(trivia)
        think_content = f"Let me recall this fact.\n{reasoning}"
        assistant_response = f"<think>\n{think_content}\n</think>\n<answer>{a}</answer>"
        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q},
                {"role": "assistant", "content": assistant_response},
            ]
        })
    return examples


def generate_unit_conversion_examples(num_samples: int, rng: random.Random) -> list:
    """Unit conversion — teaches reasoning format with simple math, no reasoning-gym overlap."""
    conversions = [
        ("km", "miles", 0.621371, "multiply by 0.621371"),
        ("miles", "km", 1.60934, "multiply by 1.60934"),
        ("kg", "pounds", 2.20462, "multiply by 2.20462"),
        ("pounds", "kg", 0.453592, "multiply by 0.453592"),
        ("meters", "feet", 3.28084, "multiply by 3.28084"),
        ("feet", "meters", 0.3048, "multiply by 0.3048"),
        ("liters", "gallons", 0.264172, "multiply by 0.264172"),
        ("gallons", "liters", 3.78541, "multiply by 3.78541"),
        ("Celsius", "Fahrenheit", None, "multiply by 9/5 then add 32"),
        ("inches", "centimeters", 2.54, "multiply by 2.54"),
    ]

    examples = []
    for _ in range(num_samples):
        from_unit, to_unit, factor, method = rng.choice(conversions)

        if from_unit == "Celsius":
            value = rng.randint(0, 100)
            result = round(value * 9 / 5 + 32, 1)
        else:
            value = rng.randint(1, 500)
            result = round(value * factor, 2)

        question = f"Convert {value} {from_unit} to {to_unit}."
        think_content = (
            f"To convert {from_unit} to {to_unit}, I need to {method}.\n"
            f"{value} × conversion factor = {result} {to_unit}"
        )
        assistant_response = f"<think>\n{think_content}\n</think>\n<answer>{result} {to_unit}</answer>"

        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
                {"role": "assistant", "content": assistant_response},
            ]
        })
    return examples


def generate_letter_puzzle_examples(num_samples: int, rng: random.Random) -> list:
    """Simple letter/word puzzles — teaches structured reasoning."""
    examples = []
    for _ in range(num_samples):
        puzzle_type = rng.choice(["count_vowels", "reverse", "letter_position", "uppercase_count"])

        if puzzle_type == "count_vowels":
            words = ["elephant", "university", "beautiful", "orange", "algorithm",
                     "concatenate", "hypothesis", "education", "automobile", "sequential"]
            word = rng.choice(words)
            vowels = sum(1 for c in word.lower() if c in "aeiou")
            question = f"How many vowels are in the word '{word}'?"
            think_content = (
                f"Let me count the vowels (a, e, i, o, u) in '{word}'.\n"
                f"Going letter by letter: {', '.join(c + ('(vowel)' if c in 'aeiou' else '') for c in word.lower())}\n"
                f"Total vowels: {vowels}"
            )
            answer = str(vowels)

        elif puzzle_type == "reverse":
            words = ["hello", "python", "computer", "science", "language", "training", "network"]
            word = rng.choice(words)
            reversed_word = word[::-1]
            question = f"What is the word '{word}' spelled backwards?"
            think_content = f"I need to reverse the letters in '{word}'.\nReading from right to left: {reversed_word}"
            answer = reversed_word

        elif puzzle_type == "letter_position":
            word = rng.choice(["mathematics", "programming", "philosophy", "understanding"])
            pos = rng.randint(1, len(word))
            letter = word[pos - 1]
            question = f"What is the {pos}{'st' if pos == 1 else 'nd' if pos == 2 else 'rd' if pos == 3 else 'th'} letter of '{word}'?"
            think_content = f"I need to find position {pos} in '{word}'.\nCounting: {', '.join(f'{i+1}={c}' for i, c in enumerate(word[:pos]))}\nThe letter at position {pos} is '{letter}'."
            answer = letter

        else:  # uppercase_count
            sentence = rng.choice([
                "The Quick Brown Fox Jumps Over The Lazy Dog",
                "Hello World From Python Programming",
                "AI Is Transforming The World Today",
                "New York City Is In The United States",
            ])
            count = sum(1 for c in sentence if c.isupper())
            question = f"How many uppercase letters are in: '{sentence}'?"
            think_content = f"Let me count each uppercase letter in the sentence.\n"
            think_content += f"Uppercase letters: {', '.join(c for c in sentence if c.isupper())}\n"
            think_content += f"Total: {count}"
            answer = str(count)

        assistant_response = f"<think>\n{think_content}\n</think>\n<answer>{answer}</answer>"
        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
                {"role": "assistant", "content": assistant_response},
            ]
        })
    return examples


def generate_date_reasoning_examples(num_samples: int, rng: random.Random) -> list:
    """Simple date/time reasoning."""
    examples = []
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    months_days = [
        ("January", 31), ("February", 28), ("March", 31), ("April", 30),
        ("May", 31), ("June", 30), ("July", 31), ("August", 31),
        ("September", 30), ("October", 31), ("November", 30), ("December", 31),
    ]

    for _ in range(num_samples):
        puzzle_type = rng.choice(["days_later", "month_days", "time_zones"])

        if puzzle_type == "days_later":
            start_idx = rng.randint(0, 6)
            offset = rng.randint(1, 14)
            result_idx = (start_idx + offset) % 7
            question = f"If today is {days[start_idx]}, what day will it be in {offset} days?"
            think_content = (
                f"Starting from {days[start_idx]} (day {start_idx} of the week).\n"
                f"Adding {offset} days: ({start_idx} + {offset}) mod 7 = {result_idx}.\n"
                f"That corresponds to {days[result_idx]}."
            )
            answer = days[result_idx]

        elif puzzle_type == "month_days":
            month, num_days = rng.choice(months_days)
            question = f"How many days are in {month}?"
            think_content = f"Recalling the number of days in each month.\n{month} has {num_days} days."
            answer = str(num_days)

        else:  # time_zones
            hour = rng.randint(1, 12)
            offset = rng.choice([3, 5, 8, -5, -8])
            period = rng.choice(["AM", "PM"])
            result_hour_24 = (hour + (0 if period == "AM" else 12) + offset) % 24
            result_period = "AM" if result_hour_24 < 12 else "PM"
            result_hour = result_hour_24 % 12
            if result_hour == 0:
                result_hour = 12
            direction = "ahead" if offset > 0 else "behind"
            question = f"If it is {hour}:00 {period} in City A, and City B is {abs(offset)} hours {direction}, what time is it in City B?"
            think_content = (
                f"City A time: {hour}:00 {period}.\n"
                f"City B is {abs(offset)} hours {direction}, so I {'add' if offset > 0 else 'subtract'} {abs(offset)} hours.\n"
                f"Result: {result_hour}:00 {result_period}"
            )
            answer = f"{result_hour}:00 {result_period}"

        assistant_response = f"<think>\n{think_content}\n</think>\n<answer>{answer}</answer>"
        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
                {"role": "assistant", "content": assistant_response},
            ]
        })
    return examples


def generate_definition_examples(num_samples: int, rng: random.Random) -> list:
    """Vocabulary / definition questions."""
    definitions = [
        ("What does 'ephemeral' mean?", "Something that lasts for a very short time.", "Lasting for a very short time"),
        ("What does 'ubiquitous' mean?", "Present, appearing, or found everywhere.", "Present everywhere"),
        ("What does 'pragmatic' mean?", "Dealing with things in a practical rather than theoretical way.", "Practical, dealing with things realistically"),
        ("What does 'ambiguous' mean?", "Open to more than one interpretation; not clear.", "Open to multiple interpretations"),
        ("What does 'resilient' mean?", "Able to recover quickly from difficulties.", "Able to recover quickly from difficulties"),
        ("What does 'verbose' mean?", "Using more words than needed; wordy.", "Using more words than needed"),
        ("What does 'benevolent' mean?", "Well-meaning and kindly; charitable.", "Kind and well-meaning"),
        ("What does 'concise' mean?", "Giving a lot of information clearly in few words.", "Brief but comprehensive"),
        ("What does 'diligent' mean?", "Having or showing care and effort in one's work.", "Hardworking and careful"),
        ("What does 'eloquent' mean?", "Fluent or persuasive in speaking or writing.", "Fluent and persuasive in expression"),
        ("What does 'frugal' mean?", "Sparing or economical with regard to money or resources.", "Economical with money or resources"),
        ("What does 'meticulous' mean?", "Showing great attention to detail; very careful.", "Very careful and precise"),
    ]

    examples = []
    for _ in range(num_samples):
        q, reasoning, a = rng.choice(definitions)
        think_content = f"Let me recall the definition of this word.\n{reasoning}"
        assistant_response = f"<think>\n{think_content}\n</think>\n<answer>{a}</answer>"
        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q},
                {"role": "assistant", "content": assistant_response},
            ]
        })
    return examples


def main():
    parser = argparse.ArgumentParser(description="Generate task-agnostic SFT warmup data")
    parser.add_argument(
        "--output",
        type=str,
        default="data_gen/data/sft_warmup.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Total number of examples to generate",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Split evenly across 5 categories
    per_category = args.num_samples // 5
    remainder = args.num_samples - per_category * 5

    print(f"Generating {args.num_samples} task-agnostic warmup examples...")
    print(f"  Trivia:           {per_category}")
    print(f"  Unit conversions: {per_category}")
    print(f"  Letter puzzles:   {per_category}")
    print(f"  Date reasoning:   {per_category}")
    print(f"  Definitions:      {per_category + remainder}")

    all_examples = []
    all_examples.extend(generate_trivia_examples(per_category, rng))
    all_examples.extend(generate_unit_conversion_examples(per_category, rng))
    all_examples.extend(generate_letter_puzzle_examples(per_category, rng))
    all_examples.extend(generate_date_reasoning_examples(per_category, rng))
    all_examples.extend(generate_definition_examples(per_category + remainder, rng))

    # Shuffle
    rng.shuffle(all_examples)

    # Write
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\n✓ Wrote {len(all_examples)} examples to {output_path}")
    print("  (No reasoning-gym task content — safe for unbiased benchmarking)")


if __name__ == "__main__":
    main()
