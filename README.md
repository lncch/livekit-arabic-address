# Arabic address form for LiveKit Agents (demo)

## The gap
Arabic verbs and adjectives change with the speaker's gender. A voice agent needs to know which form to use.
LiveKit issue #4640 asks for gender detection. In livekit-agents 1.8.13 there is no built-in support.
The only related piece is `stt_context`, an empty slot that an STT plugin may fill. Nothing in core fills it.

## Design choice
This demo does not build a voice-based gender classifier. Guessing gender from a voice is error-prone and sensitive.
Public classifiers already exist, so a generic one is not new. This demo reads the words only.

1. Default: gender-neutral Arabic phrasing.
2. The caller's own words come first: "أنا جاهزة" vs "أنا جاهز", "أنا امرأة", or "خاطبني بصيغة المؤنث".
3. One self-reference is not enough. Two matching ones switch the form. A conflict returns to neutral.
4. An explicit statement switches at once and locks. Later cues cannot override it.
5. The caller can correct at any time ("لا، خاطبني بالمذكر").
6. Nothing is stored. State lives in memory for one session.

The hook is `ArabicAddressAgent.on_user_turn_completed`. It calls `update_instructions`. Both are real `Agent` methods.

## Run
    uv venv && uv pip install livekit-agents
    .venv/bin/python demo.py
    .venv/bin/python test_arabic_address.py

## Limits
- Small word list (about 40 stems) and rules. New words and unusual phrasing are missed.
- Known misses in the tests: "مستعداً" with tanween spelling, and "أنا ما زلت مستعدة".
- Dialects are only partly covered. STT can also drop or add the final ة.
- The agent's replies still depend on the LLM following the instruction.
- Quoted speech ("هي قالت أنا جاهزة") is not handled.
- The 27 test cases are a small scripted set. They are not a benchmark, and no real speech was tested.
