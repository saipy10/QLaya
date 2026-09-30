import sys
import time
import qlaya

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def main():
    print("=" * 60)
    print("QLaya 0.4.2 Test")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Check version
    # ---------------------------------------------------------
    print("\n[1] Package")
    print("QLaya version:", getattr(qlaya, "__version__", "unknown"))

    # ---------------------------------------------------------
    # 2. Check available models
    # ---------------------------------------------------------
    print("\n[2] Available model variants")

    for model in qlaya.QLAYA_MODEL_IDS:
        print(" -", model)

    # ---------------------------------------------------------
    # 3. Test automatic routing
    # ---------------------------------------------------------
    print("\n[3] Testing Router")

    router = qlaya.Router(default="QLaya-HighSpeedProduction")


    test_texts = [
        "Refund my duplicate order please",
        "ग्राहक से दो बार शुल्क लिया गया और वह धनवापसी चाहता है।",
        "お客様は二重に請求されたため返金を希望しています。",
    ]

    for text in test_texts:
        start = time.perf_counter()

        decision = router.route(text)

        elapsed = (time.perf_counter() - start) * 1000

        print("\nText:")
        print(text)

        print("Decision:")
        print(decision)

        print(f"Routing time: {elapsed:.3f} ms")

    # ---------------------------------------------------------
    # 4. Test confidence utilities
    # ---------------------------------------------------------
    print("\n[4] Testing confidence calculation")

    import numpy as np

    probabilities = np.array([
        0.88,
        0.08,
        0.04
    ])

    confidence = qlaya.answer_confidence(
        probabilities,
        k=len(probabilities)
    )

    entropy_confidence = qlaya.confidence_from_probs(
        probabilities,
        k=len(probabilities)
    )

    print("Probabilities:", probabilities)
    print("Answer confidence:", confidence)
    print("Entropy confidence:", entropy_confidence)

    # ---------------------------------------------------------
    # 5. Test email cleaning
    # ---------------------------------------------------------
    print("\n[5] Testing email cleaning")

    email = """Hi Support,

I need help with my account.

On Mon, Jan 15, 2026 at 10:00 AM,
Support <support@example.com> wrote:

> Thank you for contacting us.
"""

    cleaned = qlaya.clean_email_body(email)

    print("\nOriginal:")
    print(email)

    print("Cleaned:")
    print(cleaned)

    print("\n" + "=" * 60)
    print("QLaya test completed")
    print("=" * 60)


if __name__ == "__main__":
    main()