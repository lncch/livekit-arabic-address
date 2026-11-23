"""Arabic address-form tracker for LiveKit Agents. Text only. No voice analysis. Nothing stored."""
import re
from dataclasses import dataclass, field

from livekit.agents import Agent, llm

NEUTRAL, MALE, FEMALE = "neutral", "male", "female"

# Adjective/participle stems: male form = stem, female = stem + ة (or ه in dialect spelling).
STEMS = """جاهز مستعد متعب تعبان سعيد حزين غاضب زعلان مشغول متأكد موجود متحمس مهتم ممتن
مرتبك قلقان خايف خائف محتاج مضطر عاجز مبسوط مستغرب متفاجئ متأخر مريض محبط
طالب معلم مهندس طبيب موظف مدير محاسب ممرض مسافر متزوج أعزب""".split()
FILLERS = {"لست", "مش", "مو", "ما", "جدا", "فعلا", "الان", "اليوم", "حاليا", "كثير", "كتير",
           "بالفعل", "دايما", "دائما", "حقا", "والله", "تماما"}
SELF = {"انا", "كنت", "صرت", "اصبحت", "اصبح", "اظل", "ظللت"}
# Words after a self-reference that carry no cue: stop scanning so a third person cannot leak in.
NOUN_F = {"امراة", "بنت", "انثى", "ام", "سيدة", "فتاة"}
NOUN_M = {"رجل", "ولد", "ذكر", "اب", "شاب"}

_DIAC = re.compile(r"[ً-ْـ]")


def norm(word: str) -> str:
    w = _DIAC.sub("", word)
    return re.sub("[أإآ]", "ا", w).replace("ى", "ي").replace("ئ", "ي").strip("،.؟?!,:;\"'")


_ST = {norm(s) for s in STEMS}


def _adj_gender(w: str):
    if w in _ST:
        return MALE
    if w[-1:] in "ةه" and w[:-1] in _ST:
        return FEMALE
    return None


def detect(text: str):
    """Return (kind, gender) or None. kind: 'explicit' (stated outright) or 'cue' (self-reference)."""
    t = " ".join(norm(w) for w in text.split())
    # explicit preference about how to be addressed
    m = re.search(r"(?:خاطب|كلم|نادي)\w*(?:\s+\w+){0,2}?\s+(?:ب|ك)?(المؤنث|المذكر|مؤنث|مذكر|انثي|انثى|ذكر)\b", t)
    if m:
        return ("explicit", MALE if m.group(1) in ("المذكر", "مذكر", "ذكر") else FEMALE)
    if re.search(r"(بدون|من غير|لا)\s+(تحديد|تفترض|تخمن)|صيغة محايدة|محايد", t):
        return ("explicit", NEUTRAL)
    ws = t.split()
    for i, w in enumerate(ws):
        if w not in SELF:
            continue
        j = i + 1
        while j < len(ws) and ws[j] in FILLERS and j - i <= 3:
            j += 1
        if j >= len(ws):
            continue
        nxt = ws[j]
        if w == "انا" and nxt in NOUN_F | NOUN_M:
            return ("explicit", FEMALE if nxt in NOUN_F else MALE)
        g = _adj_gender(nxt)
        if g:
            return ("cue", g)
    return None


@dataclass
class AddressTracker:
    """Session memory only. Needs 2 matching cues; an explicit statement wins at once and locks."""
    form: str = NEUTRAL
    locked: bool = False
    _votes: list = field(default_factory=list)

    def update(self, text: str) -> str:
        hit = detect(text)
        if not hit:
            return self.form
        kind, g = hit
        if kind == "explicit":
            self.form, self.locked, self._votes = g, True, []
        elif not self.locked:
            self._votes = (self._votes + [g])[-2:]
            self.form = g if len(self._votes) == 2 and len(set(self._votes)) == 1 else NEUTRAL
        return self.form


ADDENDUM = {
    NEUTRAL: "خاطب المستخدم بصيغة محايدة: استخدم الجمع أو المصدر أو أسلوب غير مباشر، وتجنب صيغة المذكر والمؤنث.",
    MALE: "خاطب المستخدم بصيغة المذكر. إن صحح لك ذلك فاتبع تصحيحه فورا.",
    FEMALE: "خاطب المستخدمة بصيغة المؤنث. إن صححت لك ذلك فاتبع تصحيحها فورا.",
}


class ArabicAddressAgent(Agent):
    """Agent that re-writes its own instructions when the caller's self-reference changes the form."""

    def __init__(self, *, instructions: str, **kw):
        self._base = instructions
        self.tracker = AddressTracker()
        super().__init__(instructions=f"{instructions}\n{ADDENDUM[NEUTRAL]}", **kw)

    async def on_user_turn_completed(self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage) -> None:
        before = self.tracker.form
        self.tracker.update(new_message.text_content or "")
        if self.tracker.form != before:
            await self.update_instructions(f"{self._base}\n{ADDENDUM[self.tracker.form]}")
