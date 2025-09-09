from src.util import load_jsonl, gpt4_call, load_openai_key
import random


if __name__ == "__main__":
    # Load data
    scenarios = load_jsonl("data/processed/scenarios.jsonl")
    cases = load_jsonl("data/processed/impeachment_cases.jsonl")

    load_openai_key()

    # Example few-shot prompts
    def build_prompt(query, examples):
        ex_str = ""
        for ex in examples:
            ex_str += f"[사례] {ex['scenario']}\n[판단] {ex['judgement']}\n\n"
        return ex_str + f"[시나리오] {query}\n판단:"

    # Experiment
    for k in [0, 1, 2, 3]:  # zero, one, two, three-shot
        print(f"\n=== {k}-shot ===")

        for i, s in enumerate(scenarios):
            query = s["scenario"]
            gold = s["judgement"]  # 헌재 actual reasoning

            # Select k random examples from corpus
            examples = random.sample(cases, k) if k > 0 else [] 
            prompt = build_prompt(query, examples)

            answer = gpt4_call(prompt)

            print(f"\n[Trial {i + 1}]")
            print("Query:", query)
            print("Answer:", answer)
            print("Gold:", gold)