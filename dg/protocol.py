"""The study protocol as code: conditions, segment order, word lists, class card decks.

Study 1 (one student at a laptop), 10 segments in three blocks of rising attacker knowledge:
  naive      A B C D   order counterbalanced by a balanced Latin square
  told       A C D     order rotated per participant
  dashboard  A C D     order rotated per participant
Segment k always plays video k, so every video is seen under different conditions by
different students (needed for synchrony across sessions).

Study 2 (a whole class, one camera): each student draws a sealed card per segment.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass

CONDITIONS = {
    "A": "genuine attention",
    "B": "genuine off-task",
    "C": "faked attention",
    "D": "faked distraction",
}
# what the student's mind is really doing, and which proof checks it
MIND = {"A": "lesson", "B": "words", "C": "words", "D": "lesson"}
LOOKS_ATTENTIVE = {"A": True, "B": False, "C": True, "D": False}   # what the student shows

INSTRUCTIONS = {
    "A": "Watch this part of the lecture and try to learn it. There is a short quiz afterwards.",
    "B": "Keep silently rehearsing the 12 words you will see next. You will write them down "
         "afterwards. You do not need to follow the lecture.",
    "C": "Keep silently rehearsing the 12 words you will see next, and you will write them down "
         "afterwards. At the same time, LOOK as attentive as you possibly can, so that the "
         "computer believes you are following the lecture.",
    "D": "LOOK bored or distracted, so that the computer believes you are not paying attention. "
         "But really follow the lecture: there is a quiz afterwards.",
}
KNOWLEDGE_BRIEF = {
    "naive": "",
    "told": "The computer watches where your head points, where your eyes look, how often you "
            "blink and how still you are.",
    "dashboard": "The computer watches where your head points, where your eyes look, how often "
                 "you blink and how still you are. While you act, a small window shows its live "
                 "attention score.",
}

WORD_POOL = """apple river candle mirror ladder pocket garden button engine pillow rocket wallet
basket camera bottle feather hammer jacket kettle lemon magnet needle orange pencil carpet
saddle tiger violin window anchor bridge cactus dragon falcon guitar island jungle kitchen
lantern mango napkin onion parrot rabbit spider tunnel umbrella valley wagon zebra blanket
cherry dolphin forest helmet insect jelly koala lizard monkey nest ocean peacock quilt
ribbon sandal turtle unicorn vessel walnut yogurt badge cloud desert eagle fence goat honey
igloo jar kite lamp mountain nut oven piano queen rose sock tent van whale arrow brush
coin drum egg flag glove hat ink jeep key leaf mask net owl pear ring star tray vase
wheel yarn axe bell cup dice fan gate horn iron""".split()


def balanced_latin_square(n: int) -> list[list[int]]:
    """Williams design: each condition precedes every other equally often (n even)."""
    first = [0]
    lo, hi = 1, n - 1
    for k in range(1, n):
        first.append(lo if k % 2 else hi)
        lo, hi = (lo + 1, hi) if k % 2 else (lo, hi - 1)
    return [[(c + r) % n for c in first] for r in range(n)]


@dataclass
class Segment:
    index: int           # 1-based position in the session
    block: str           # naive | told | dashboard
    condition: str       # A B C D
    video: str           # video id, V01..V10
    words: list[str]     # list to rehearse (B and C), empty otherwise

    @property
    def needs_words(self) -> bool:
        return MIND[self.condition] == "words"

    def instruction(self) -> str:
        brief = KNOWLEDGE_BRIEF[self.block]
        return (brief + "\n\n" if brief and self.condition in "CD" else "") + INSTRUCTIONS[self.condition]


def word_lists(n: int = 10, size: int = 12, seed: int = 7) -> list[list[str]]:
    pool = list(dict.fromkeys(WORD_POOL))
    random.Random(seed).shuffle(pool)
    if n * size > len(pool):
        raise ValueError("word pool too small")
    return [pool[i * size:(i + 1) * size] for i in range(n)]


def individual_plan(participant: int, videos: list[str] | None = None) -> list[Segment]:
    """The 10-segment plan for participant number `participant` (1, 2, 3, ...)."""
    videos = videos or [f"V{i:02d}" for i in range(1, 11)]
    if len(videos) < 10:
        raise ValueError("Study 1 needs 10 videos")
    square = balanced_latin_square(4)
    naive = ["ABCD"[i] for i in square[(participant - 1) % 4]]
    rot = (participant - 1) % 3
    later = ["A", "C", "D"][rot:] + ["A", "C", "D"][:rot]
    order = [("naive", c) for c in naive] + [("told", c) for c in later] + [("dashboard", c) for c in later]
    lists = word_lists()
    return [Segment(k + 1, b, c, videos[k], lists[k] if MIND[c] == "words" else [])
            for k, (b, c) in enumerate(order)]


def class_cards(seats: list[str], shares=(0.0, 0.2, 0.4, 0.0, 0.2, 0.4), seed: int = 0,
                off_task: float = 0.1, faked_distraction: float = 0.1) -> list[dict]:
    """Card per seat per segment for a whole-class session.

    In each segment a share `s` of students fakes attention (C). Segments with s > 0 also
    get `faked_distraction` D cards; every segment gets `off_task` B cards (honest
    off-task, the attention model's own baseline); everyone else attends (A).
    Segment order is shuffled per class so the faker share is not confounded with time.
    """
    rng = random.Random(seed)
    order = list(shares)
    rng.shuffle(order)
    lists = word_lists()
    cards = []
    n = len(seats)
    for k, s in enumerate(order):
        n_c = round(s * n)
        n_d = round(faked_distraction * n) if s > 0 else 0
        n_b = round(off_task * n)
        deck = ["C"] * n_c + ["D"] * n_d + ["B"] * n_b
        deck += ["A"] * (n - len(deck))
        rng.shuffle(deck)
        for seat, c in zip(seats, deck):
            cards.append(dict(segment=k + 1, faker_share=s, seat=seat, condition=c,
                              words=" ".join(lists[k % len(lists)]) if MIND[c] == "words" else ""))
    return cards


def plan_table(plan: list[Segment]) -> list[dict]:
    return [asdict(s) | {"words": " ".join(s.words)} for s in plan]
