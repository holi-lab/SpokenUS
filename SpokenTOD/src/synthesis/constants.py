NEUTRAL = 0
FEARFUL = 1
DISSATISFIED = 2
APOLOGETIC = 3
ABUSIVE = 4
EXCITED = 5
SATISFIED = 6

# 대분류 이름 매핑
EMOTION_CATEGORIES = {
    NEUTRAL: "neutral",
    FEARFUL: "fearful",
    DISSATISFIED: "dissatisfied",
    APOLOGETIC: "apologetic",
    ABUSIVE: "abusive",
    EXCITED: "excited",
    SATISFIED: "satisfied",
}

# 하위 감정/특성 매핑
EMOTION_SUBCATEGORIES = {
    NEUTRAL: ["calm", "indifferent", "patient", "relaxed"],
    FEARFUL: [
        "fearful",
        "shocked",
        "surprised",
    ],
    DISSATISFIED: [
        "angry",
        "contempt",
        "disgusted",
        "defiant",
    ],
    APOLOGETIC: ["compassionate", "selfless", "humble"],
    ABUSIVE: [
        "commanding",
        "authoritative",
        "merciless",
        "loud",
        "vengeful",
    ],
    EXCITED: [
        "adventurous",
        "energetic",
        "passionate",
        "curious",
        "creative",
        "joyful",
    ],
    SATISFIED: [
        "proud",
        "hopeful",
        "happy",
        "cheerful",
    ],
}

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
LANGUAGE = "English"
MAX_NEW_TOKENS = 4608
REFERENCE_TRANSCRIPT = "Please call Stella.  Ask her to bring these things with her from the store:  Six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob.  We also need a small plastic snake and a big toy frog for the kids.  She can scoop these things into three red bags, and we will go meet her Wednesday at the train station."
