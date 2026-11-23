"""Offline demo. No keys, no audio. Shows the address form before and after each scripted turn."""
import asyncio
from arabic_address import ArabicAddressAgent
from livekit.agents import llm

TURNS = [
    ("start", ""),
    ("one cue only (stays neutral)", "أنا جاهزة للبدء"),
    ("second matching cue", "أنا تعبانة قليلا"),
    ("third person trap", "أخي جاهز أيضا"),
    ("caller corrects, explicit", "لا، خاطبني بصيغة المذكر"),
    ("cue after lock, ignored", "أنا مشغولة"),
    ("caller asks neutral", "الأفضل صيغة محايدة"),
    ("ambiguous, no cue", "أريد تغيير موعدي"),
    ("cue after neutral lock, ignored", "أنا مهندس"),
]


async def main():
    agent = ArabicAddressAgent(instructions="أنت مساعد صوتي لحجز المواعيد.")
    ctx = llm.ChatContext()
    for name, text in TURNS:
        before = agent.tracker.form
        if text:
            await agent.on_user_turn_completed(ctx, llm.ChatMessage(role="user", content=[text]))
        print(f"{name:30} | {text or '-':28} | {before:8} -> {agent.tracker.form}")


asyncio.run(main())
