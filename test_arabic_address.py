import asyncio
from arabic_address import AddressTracker, ArabicAddressAgent, detect, NEUTRAL, MALE, FEMALE
from livekit.agents import llm

# (text, expected detect() result: None, or "cue:male", "cue:female", "explicit:male", ...)
CASES = [
    ("أنا جاهزة", "cue:female"),
    ("أنا جاهز", "cue:male"),
    ("أنا تعبانة اليوم", "cue:female"),
    ("أنا فعلاً مشغول الآن", "cue:male"),
    ("كنت متأكدة من ذلك", "cue:female"),
    ("أنا لست مستعداً", "cue:male"),  # tanween spelling
    ("أنا لست مستعدة", "cue:female"),
    ("أنا مهندسة", "cue:female"),
    ("أنا طالب في الجامعة", "cue:male"),
    ("أنا امرأة", "explicit:female"),
    ("أنا رجل", "explicit:male"),
    ("خاطبني بصيغة المؤنث", "explicit:female"),
    ("كلمني كأنثى من فضلك", "explicit:female"),
    ("خاطبني بالمذكر", "explicit:male"),
    ("أريد صيغة محايدة", "explicit:neutral"),
    ("أنا سعيدة جداً بالتحدث", "cue:female"),
    ("أنا ما زلت مستعدة", "cue:female"),
    ("أحب أختي وهي جاهزة", None),
    ("قال لي أبي إنه متعب", None),
    # traps: third person or no self-reference
    ("أختي جاهزة", None),
    ("أخي جاهز وأنا أنتظر", None),
    ("أنا وأختي جاهزة", None),
    ("زوجتي تقول إنها متعبة", None),
    ("هل أنتِ جاهزة؟", None),
    ("أنا أعتقد أن أمي مشغولة", None),
    ("مرحبا كيف حالك", None),
    ("أنا أريد حجز موعد", None),
]


def label(r):
    return None if r is None else f"{r[0]}:{r[1]}"


def run():
    ok, bad = 0, []
    for text, want in CASES:
        got = label(detect(text))
        if got == want:
            ok += 1
        else:
            bad.append((text, want, got))
    return ok, bad


def test_policy_and_agent():
    t = AddressTracker()
    assert t.update("أنا جاهزة") == NEUTRAL          # one cue is not enough
    assert t.update("أنا تعبانة") == FEMALE          # two matching cues
    assert t.update("أنا جاهز") == NEUTRAL           # conflict drops back to neutral
    assert t.update("خاطبني بالمذكر") == MALE        # explicit wins and locks
    assert t.update("أنا تعبانة") == MALE            # cues cannot override a lock
    assert t.update("لا، أنا امرأة") == FEMALE       # caller corrects
    agent = ArabicAddressAgent(instructions="أنت مساعد صوتي.")
    msg = lambda s: llm.ChatMessage(role="user", content=[s])
    ctx = llm.ChatContext()
    asyncio.run(agent.on_user_turn_completed(ctx, msg("أنا امرأة")))
    assert "المؤنث" in agent.instructions
    asyncio.run(agent.on_user_turn_completed(ctx, msg("أنا رجل")))
    assert "المذكر" in agent.instructions and "المؤنث" not in agent.instructions


if __name__ == "__main__":
    ok, bad = run()
    print(f"detect accuracy: {ok}/{len(CASES)} = {ok/len(CASES):.0%}")
    for b in bad:
        print("  FAIL", b)
    test_policy_and_agent()
    print("policy + agent hook tests: pass")
