"""Minimal reproduction of FActScore atomic fact decomposition.

Scope:
- Reuses the repository's original `AtomicFactGenerator` implementation.
- Only performs: input text -> sentence splitting -> few-shot prompt construction
  -> InstructGPT decomposition -> post-processing -> atomic facts.
- Does NOT run retrieval, NPM, factual verification, or FActScore aggregation.

This script is intentionally kept close to the original FActScore code path so
that the atomic-fact decomposition process can be studied in isolation.
"""

import argparse
import os

from factscore.atomic_facts import AtomicFactGenerator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run only the atomic-fact decomposition stage of FActScore."
    )
    parser.add_argument(
        "--text",
        required=True,
        help="Text to decompose into atomic facts.",
    )
    parser.add_argument(
        "--api-key-file",
        default="api.key",
        help="Path to the OpenAI API key file used by the original FActScore implementation.",
    )
    parser.add_argument(
        "--demon-dir",
        default=".cache/factscore/demos",
        help="Directory containing the original FActScore demonstration file demons.json.",
    )
    parser.add_argument(
        "--cache-file",
        default=".cache/factscore/InstructGPT-minimal.pkl",
        help="Cache file for InstructGPT decomposition outputs.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    demon_file = os.path.join(args.demon_dir, "demons.json")
    if not os.path.isfile(demon_file):
        raise FileNotFoundError(
            f"Missing FActScore demonstrations: {demon_file}\n"
            "Download the original FActScore data first so that demons.json is available."
        )

    if not os.path.isfile(args.api_key_file):
        raise FileNotFoundError(
            f"Missing OpenAI API key file: {args.api_key_file}\n"
            "The original AtomicFactGenerator uses InstructGPT through OpenAIModel."
        )

    cache_dir = os.path.dirname(args.cache_file)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    generator = AtomicFactGenerator(
        key_path=args.api_key_file,
        demon_dir=args.demon_dir,
        gpt3_cache_file=args.cache_file,
    )

    atomic_fact_pairs, paragraph_breaks = generator.run(args.text)

    print("\n=== INPUT ===")
    print(args.text)

    print("\n=== ATOMIC FACT DECOMPOSITION ===")
    fact_id = 1
    for sentence_id, (sentence, facts) in enumerate(atomic_fact_pairs, start=1):
        print(f"\nSentence {sentence_id}: {sentence}")
        if not facts:
            print("  (no atomic facts)")
            continue

        for fact in facts:
            print(f"  Fact {fact_id}: {fact}")
            fact_id += 1

    print("\n=== PARAGRAPH BREAKS ===")
    print(paragraph_breaks)

    generator.save_cache()


if __name__ == "__main__":
    main()
