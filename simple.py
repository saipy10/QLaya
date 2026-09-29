"""QLaya - Simple Quickstart & Test Script

Demonstrates the core capabilities of the QLaya decision engine:
1. Version & available quantized model variants
2. Script and language detection
3. Intelligent model routing (English vs Multilingual checkpoints)
4. Built-in decision presets (routing, moderation, triage)
5. Email text sanitization
6. Probability confidence metrics
"""

import sys
import numpy as np
import qlaya

# Ensure standard output can display multilingual characters on Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def main():
    print("=" * 60)
    print(f"  QLaya Decision Engine v{qlaya.__version__} - Quickstart Demo")
    print("=" * 60)

    # 1. Package & Available Model Variants
    print("\n[1] Available Model Variants:")
    for variant in qlaya.QLAYA_MODEL_IDS:
        print(f"  - {variant}")

    # 2. Script & Language Detection
    print("\n[2] Script & Language Detection:")
    samples = [
        "Refund my duplicate charge immediately.",
        "お客様は二重に請求されたため返金を希望しています。",
        "ग्राहक से दो बार शुल्क लिया गया और वह धनवापसी चाहता है।",
        "Bonjour, je souhaite annuler ma commande s'il vous plaît.",
    ]
    for text in samples:
        info = qlaya.detect_language(text)
        preview = text[:45] + ("..." if len(text) > 45 else "")
        print(f"  Text:    \"{preview}\"")
        print(f"  Script:  {info['script']} (English: {info['is_english']})")
        print()

    # 3. Router Demonstration (Pure, instant routing without downloading weights)
    print("[3] Intelligent Checkpoint Routing:")
    router = qlaya.Router()
    queries = [
        "Can you route this support ticket to billing?",
        "请帮我处理这个退款申请",
        "Guten Tag, können Sie mir bitte weiterhelfen?",
    ]
    for query in queries:
        decision = router.route(query)
        print(f"  Query:  \"{query}\"")
        print(f"  Route:  Model='{decision.model}' | Reason='{decision.reason}'")
        print()

    # 4. Preset Questions (Routing, Moderation, Guardrails)
    print("[4] Built-in Workflow Presets:")
    presets = {
        "Router": qlaya.presets.router_questions(),
        "Guardrails": qlaya.presets.guard_questions(),
        "Moderation": qlaya.presets.moderation_questions(),
        "Triage": qlaya.presets.triage_questions(),
    }
    for name, questions in presets.items():
        print(f"  - {name} preset: {len(questions)} question template(s)")

    # 5. Email Text Cleaning
    print("\n[5] Email Sanitization:")
    raw_email = (
        "Hi Support,\n\n"
        "I need an update on ticket #4810.\n\n"
        "Thanks,\nAlice\n\n"
        "On Mon, Jan 15, 2026 at 10:00 AM, Support <support@example.com> wrote:\n"
        "> Thank you for contacting us. We will look into this."
    )
    cleaned = qlaya.clean_email_body(raw_email)
    print(f"  Raw length:     {len(raw_email)} chars")
    print(f"  Cleaned length: {len(cleaned)} chars")
    print(f"  Cleaned text:\n    {cleaned.strip()}")

    # 6. Confidence Scoring
    print("\n[6] Confidence Scoring Metrics:")
    probs = np.array([0.88, 0.08, 0.04])
    k = len(probs)
    ans_conf = qlaya.answer_confidence(probs, k)
    entropy_conf = qlaya.confidence_from_probs(probs, k)
    print(f"  Probabilities:         {probs}")
    print(f"  Top-answer confidence: {ans_conf:.4f} (88% likelihood)")
    print(f"  Entropy confidence:    {entropy_conf:.4f} (distribution sharpness)")

    print("\n" + "=" * 60)
    print("  All QLaya checks completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
