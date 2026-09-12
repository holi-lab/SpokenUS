"""Emotion tagging prompts with EmoWOZ few-shot examples."""

import re

from augmentation.constants import EMOTION_LABELS

# Few-shot examples without context: (utterance, label)
FEWSHOT_EXAMPLES_SIMPLE = [
    # Neutral (0)
    ("I'd like a reservation for 7 people Monday at 15:30 please.", 0),
    ("Could you recommend one of the expensive ones?", 0),
    # Fearful/Sad (1)
    ("That's disappointing. Can you try international food instead?", 1),
    ("Hmm, okay, how about another restaurant in the same area and price range?", 1),
    ("Thats disappointing,thank you andgoodbye.", 1),
    ("I guess I don't have a choice. Book me in one of them a room for 3 people for 5 nights starting from this thursday.", 1),
    ("Oh, bummer. How about a different hotel in the same price range?", 1),
    ("Darn, I was really hoping to find a cheap Austrian restaurant. Would you mind checking one more time?", 1),
    ("Hmm, well that isn't good. Oh well, how about trying if there is a guesthouse thats cheap then. I suppose it will have to do.", 1),
    ("That's too bad. Would you be able to find me a different hotel in the same price range?", 1),
    ("Are you sure there's nothing available? I really need something in the same price range as Limehouse.", 1),
    ("Oh no.  Can you get us in at 13:15 then?", 1),
    ("That's too bad.  Please book any other restaurant that is in the west and expensive.", 1),
    ("That's a pity. Ok, but I want it to be expensive alright?", 1),
    ("Oh, that's a bummer. Ok, how about trying in the north?", 1),
    # Dissatisfied (2)
    ("No.What other Hotels are nearby ?", 2),
    ("Unlisted!  What a pain.  Okay, I suppose give me the phone number, thanks.", 2),
    ("I'll take any type of cuisine, then. I just need a reservation for 6 at 13:15 on Saturday.", 2),
    ("Yes, please try another one of the 4 restaurants you found.", 2),
    ("How about 15:15 then?", 2),
    ("Do you have a moderately priced restaurant serving lebanese food?", 2),
    ("No, in that case I think I would prefer to try a place that serves international food.", 2),
    ("No, could you double check that? I want a restaurant in the same area and price range as the Eraina. Type of food doesn't matter.", 2),
    ("OK, can you please pay attention to my questions? I need a room, please.", 2),
    ("Are you sure the Alexander is in the centre? You said earlier that it's in the east.", 2),
    ("Please double check and make sure the booking is good.", 2),
    ("I am not looking for British, I am looking for scottish food.", 2),
    ("Shucks. No Polish?", 2),
    ("I really have no desire for Cote. Are there other restaurants available?", 2),
    ("Not the phone number, the postcode, please.", 2),
    ("I will try and restate so you can look again, I need free wifi and parking, cheap place, type of room does not matter.", 2),
    ("That's too late. I need something that arrives by 8:00, please.", 2),
    ("Hmmm, could you try again. That is what I really need.", 2),
    ("Are you sure there are no moderately priced italian restaurants in the centre? I could have sworn someone told me about one...", 2),
    # ("hmmm. Can you check again? it should be in the centre.", 2),
    # ("I don't understand, did you book the reservation or not?", 2),
    # Apologetic (3)
    ("Oh yes, actually I need the postcode too.", 3),
    ("Actually, come to think of it I might need free wifi. Do either of those offer that?", 3),
    # Abusive (4)
    ("No I JUST told you I only want the travel time!", 4),
    ("Um what? I need to arrive by 23:00. Can you help or not?", 4),
    ("I already told you how many people. are you paying attention?", 4),
    ("You didn't answer my question. I need the postcode for The Cambridge Belfry and I need to know if they have free parking or not.", 4),
    ("okay well then do so, I don't have all day", 4),
    ("You should have booked it last time you just didn't want to so skipped it.", 4),
    ("ok thanks go away now", 4),
    ("well what is it? This is ridiculous! Your very rude", 4),
    ("I am in a hurry, this is odd. Hello? Can anyone help me? Can I speak with your supervisor?", 4),
    # ("Why do you need to know my preference? You already booked me a table at michaelhouse cafe...", 4),
    # ("I literally JUST told you the time! And right before that I told you to book it for ONE. Please pay attention! My ferret hates unprofessional people!", 4),
    # Excited (5)
    ("I would love to have some Indian food please.", 5),
    ("Yes, I would like it to be a guesthouse and I need free wifi!", 5),
    ("Aloha! Can you help me find a hotel, please?", 5),
    ("I'm coming to town for an overdue visit with my relatives. Can you help me find a place to stay?", 5),
    ("That is great, I'm so excited.  Thanks for you help.  Bye.", 5),
    ("Okay, what about an attraction that is in the type of entertainment?", 5),
    ("Surprise me, I just want something pricey to impress my friends. It'll be a party of seven.", 5),
    ("Lets try for Asian please", 5),
    ("I'm not sure.  Just something expensive.  We are celebrating.", 5),
    ("It is a special occasion so I am hoping for an expensive place.", 5),
    ("Great!  Did you find any Indian restaurants in the east part of town?", 5),
    ("Sounds good, let's book it! 8 people for 4 nights, starting Friday.", 5),
    ("Really anything will do. I have to kill some time in between appointments. Can you recommend something and send me the address please?", 5),
    ("I'd like to visit a college in the center of town. Could you help me find something interesting?", 5),
    ("Good afternoon. It's such a beautiful day out and I am visiting here. Can you tell me if there is a park nearby to where I am?", 5),
    ("I am trying to locate a really nice guesthouse to take my wife to. Can you suggest any?", 5),
    ("Do you have any information on the University Arms Hotel? I've heard it's a nice place.", 5),
    ("Greetings! Can you help me in locating a train to get me to Cambridge?", 5),
    ("Hi there. I'll be coming into the centre of town to visit relatives. Can you help me find a place to stay?", 5),
    ("Sure! Which one do you recommend?", 5),
    ("Hi! Can you give me some information on the Royal Spice restaurant?", 5),
    ("That sounds great! Can you tell me more about it?", 5),
    # Satisfied (6)
    ("Yes, please. Thank you very much.", 6),
    ("You do the same!", 6),
    ("Thanks! I really don't care about the area. I need stay in a guesthouse that has free wifi and parking.", 6),
    ("Thanks so much.  That is all I need.  Bye.", 6),
    ("Thank you. I think that's all I need today, goodbye.", 6),
    ("Thank you.  No, that's all I need.  Goodbye!", 6),
]

# Few-shot examples with context: (context, utterance, label)
FEWSHOT_EXAMPLES_CONTEXT = [
    # Neutral (0)
    ([], "I want a train that arrives in broxbourne by 09:45.", 0),
    ([("user", "I'm looking for a restaurant in the centre."),
      ("system", "I found several restaurants. What cuisine?")],
     "The area doesn't matter, but I would like it to be in the moderate price range.", 0),
    # Fearful/Sad (1)
    ([("user", "I'd like to book a table for Thursday at 7pm."),
      ("system", "I'm sorry, there are no tables available at that time on Thursday.")],
     "Oh, that's a shame. Okay, can you try the same time on Friday instead, please?", 1),
    ([("user", "I need a hotel for 3 nights starting Monday."),
      ("system", "Unfortunately, there are no rooms available for 3 nights.")],
     "Ok, well lets try it for just one night then.", 1),
    # Dissatisfied (2)
    ([("user", "I need a train arriving around 16:00."),
      ("system", "I found a train arriving at 09:15. Shall I book it?")],
     "No, I'm sorry, that's much too early. Could you find one closer to 16:00 or 16:20?", 2),
    ([("user", "I need to go to London Liverpool Street."),
      ("system", "I've booked you a train to Leicester.")],
     "Let's back up a bit here. I need to get to London Liverpool Street, not Leicester.", 2),
    ([("user", "Please book for 5 people for 2 nights."),
      ("system", "I booked a table for 3 people for 1 night.")],
     "The booking should be for 5 people and 2 nights. Can you please try again?", 2),
    # Apologetic (3)
    ([("user", "I need a reservation for Friday for 2 people."),
      ("system", "What time would you like the reservation?")],
     "Oh, around 17:15. But, I meant to say on Monday, not Friday and for 7 people not 2. Thanks!", 3),
    ([("user", "I'm looking for an expensive restaurant."),
      ("system", "I've found an expensive Italian restaurant for you.")],
     "Actually, I was looking for a cheap restaurant. Are there any cheap Indian restaurants? If so, let's cancel that reservation.", 3),
    # Abusive (4)
    ([("user", "That sounds great. Please make me a reservation for 6 at 13:15 on Friday"),
      ("system", "How many people?")],
     "I already told you how many people. are you paying attention?", 4),
    ([("user", "Well, can you tell me what they are so I can decide?"),
      ("system", "All Expensive and all in the center. Do you need anything else?")],
     "i'll ask you again: what are my choices?", 4),
    ([("user", "Yes their postcode and whether they have free parking or not."),
      ("system", "The postcode for the nandos is cb17dy")],
     "You didn't answer my question. I need the postcode for The Cambridge Belfry and I need to know if they have free parking or not.", 4),
    ([("user", "Perfect. Can you book a table for 1 at 18:30 on Sunday?"),
      ("system", "I can go ahead and book that now.")],
     "You should have booked it last time you just didn't want to so skipped it.", 4),
    ([("user", "Just one, thank you."),
      ("system", "I will book that for you now.")],
     "okay well then do so, I don't have all day", 4),
    # Excited (5)
    ([("user", "What attractions are there in the area?"),
      ("system", "There's a nice boat tour available on the river.")],
     "The boat sounds like it will be fun! Do you have the phone number handy?", 5),
    ([("user", "I'm looking for something to do in town."),
      ("system", "What type of attraction are you interested in?")],
     "What's fun to do on the south side?", 5),
    ([("user", "Can you recommend a restaurant?"),
      ("system", "I found several options for you.")],
     "I'm open to anything. Surprise me with your favorite please!", 5),
    # Satisfied (6)
    ([("user", "Please book that restaurant for me."),
      ("system", "I've booked the restaurant for you. Your reference number is ABC123.")],
     "Great, thanks! That's all I need.", 6),
    ([("user", "Can you give me the hotel details?"),
      ("system", "The hotel address is 123 Main Street and the phone number is 555-1234.")],
     "Thank you. That was all I needed.", 6),
]


def build_emotion_prompt(
    utterance: str,
    context: list[dict] | None = None,
) -> str:
    """Build few-shot emotion classification prompt.

    Args:
        utterance: The user utterance to classify
        context: Optional list of previous turns [{"role": "user"|"assistant", "text": "...", "emotion": int|None}]
        fewshot_context: Whether to include context in few-shot examples

    Returns:
        Complete prompt for LLM
    """
    # Build examples section
    # if fewshot_context:
    #     examples = []
    #     for ex_ctx, ex_text, label in FEWSHOT_EXAMPLES_CONTEXT:
    #         label_name = EMOTION_LABELS[label]
    #         ctx_lines = []
    #         for role, text in ex_ctx:
    #             role_label = "User" if role == "user" else "System"
    #             ctx_lines.append(f"[{role_label}]: {text}")
    #         ctx_str = "\n".join(ctx_lines)
    #         if ctx_str:
    #             ctx_str += "\n"
    #         examples.append(f"{ctx_str}[User]: {ex_text}\n→ {label} ({label_name})")
    #     examples_text = "\n\n".join(examples)
    examples = []
    for ex_text, label in FEWSHOT_EXAMPLES_SIMPLE:
            label_name = EMOTION_LABELS[label]
            examples.append(f'"{ex_text}" → {label} ({label_name})')
    examples_text = "\n".join(examples)

    # Build context section if provided
    context_section = ""
    if context:
        context_lines = []
        for turn in context:
            role = "User" if turn["role"] == "user" else "System"
            if role == "User" and "emotion" in turn and turn["emotion"] is not None:
                label = turn["emotion"]
                label_name = EMOTION_LABELS[label]
                context_lines.append(f"[{role}]: {turn['text']} → {label} ({label_name})")
            else:
                context_lines.append(f"[{role}]: {turn['text']}")
        context_section = "\n".join(context_lines) + "\n"

    prompt = f"""You are an emotion classifier for task-oriented dialogues (EmoWOZ annotation scheme).
Classify the emotion of the LAST user utterance based on the conversation context.

Labels:
- 0: neutral - No emotion expressed. Plain requests or factual statements without enthusiasm, frustration, or apology.
- 1: fearful/sad - Disappointment about external circumstances outside system's control; resigned or saddened tone.
- 2: dissatisfied - Frustration with the system's mistakes or misalignment; user corrects, insists, or asks to retry.
- 3: apologetic - User apologizes for THEIR OWN mistake or change of mind.
- 4: abusive - Rude, dismissive, or hostile expression toward the system.
- 5: excited - Interest/enthusiasm about exploring options or getting recommendations; positive curiosity.
- 6: satisfied - Gratitude or closure about the system's help (even if followed by another request).

Examples:
{examples_text}

Now classify:
{context_section}
---

Based on the conversation above, the LAST user utterance is: "{utterance}"

Predict the emotion label (0-6) for this utterance.
- 0: neutral (plain factual question)
- 1: fearful/sad (disappointment about external circumstances)
- 2: dissatisfied (challenging/correcting the system)
- 3: apologetic (user's own mistake)
- 4: abusive (rude/hostile)
- 5: excited (enthusiasm/curiosity)
- 6: satisfied (gratitude/closure)

Respond with only the number (0-6)."""

    return prompt
def parse_emotion_response(response: str) -> int:
    """Parse emotion label from LLM response.

    Args:
        response: LLM response text

    Returns:
        Emotion label (0-6), defaults to 0 (neutral) on error
    """
    try:
        # Extract first digit from response
        for char in response.strip():
            if char.isdigit():
                label = int(char)
                if 0 <= label <= 6:
                    return label
        return 0
    except (ValueError, IndexError):
        return 0

def parse_emotion_think_response(response: str) -> int:
                """
                모델의 응답에서 0-6 사이의 숫자를 추출합니다.
                <think> 태그가 있다면 태그 이후의 내용에서 찾습니다.
                """
                clean_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

                if not clean_response: # 만약 think 태그 밖에 내용이 없다면 전체에서 찾음
                    clean_response = response

                numbers = re.findall(r'[0-6]', clean_response)

                if numbers:
                    return int(numbers[-1])

                # 찾지 못했을 경우 기본값 0 (또는 에러 상황을 알리기 위해 -1 반환 후 처리)
                return -1
