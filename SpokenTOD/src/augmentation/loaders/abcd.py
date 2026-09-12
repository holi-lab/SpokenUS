"""ABCD dataset loader with user goal from flow/subflow."""

import json
import random
from pathlib import Path
from typing import Iterator

import re

from augmentation.schema import Dialogue, Goal, SlotSpan, StructuredGoal
from augmentation.constants import FILLERS, FP, EDIT, DM, SEGMENTABLE_SLOTS
from .base import BaseLoader


class ABCDLoader(BaseLoader):
    """Load ABCD dialogues with goals from scenario flow/subflow."""

    # ABCD uses "dev" instead of "valid"
    SPLIT_MAP = {"valid": "dev", "validation": "dev"}

    def __init__(self, data_dir: Path, split: str = "train"):
        super().__init__(data_dir, split)
        self._data_cache: dict | None = None

    @property
    def name(self) -> str:
        return "abcd"

    def _get_split_key(self) -> str:
        """Map split name to ABCD's internal naming."""
        return self.SPLIT_MAP.get(self.split, self.split)

    def _load_data(self) -> dict:
        """Load and cache ABCD data."""
        if self._data_cache is None:
            data_path = self.data_dir / "data" / "abcd_v1.1.json"
            with open(data_path, "r") as f:
                self._data_cache = json.load(f)
        return self._data_cache

    def __len__(self) -> int:
        data = self._load_data()
        return len(data.get(self._get_split_key(), []))

    def load(self) -> Iterator[Dialogue]:
        """Yield ABCD dialogues."""
        data = self._load_data()
        dialogues = data.get(self._get_split_key(), [])

        for dlg in dialogues:
            dlg_id = dlg.get("convo_id", "")
            scenario = dlg.get("scenario", {})

            # Extract goal from scenario
            goal = self._extract_goal(scenario)

            # Build initial state from scenario
            cumulative_state = self._build_initial_state(scenario)

            # Extract turns from original conversation with slot spans
            turns = []
            delexed_turns = dlg.get("delexed", [])
            original_turns = dlg.get("original", [])
            merged_turns = self._merge_consecutive_turns(
                delexed_turns, original_turns
            )

            for turn_data in merged_turns:
                speaker = turn_data.get("speaker", "").lower()
                role = "user" if speaker == "customer" else "assistant"
                delexed_text = turn_data.get("delexed_text", "")
                original_text = turn_data.get("original_text", delexed_text)

                # Extract slots per original segment to avoid cross-merge span bleed.
                slots = []
                segments = turn_data.get("segments") or [{
                    "delexed_text": delexed_text,
                    "original_text": original_text,
                    "original_offset": 0,
                }]
                for segment in segments:
                    segment_slots = self._extract_slots(
                        segment.get("delexed_text", ""),
                        segment.get("original_text", ""),
                        scenario,
                    )
                    offset = segment.get("original_offset", 0)
                    if offset:
                        segment_slots = [
                            SlotSpan(
                                slot=slot.slot,
                                value=slot.value,
                                start=slot.start + offset,
                                end=slot.end + offset,
                            )
                            for slot in segment_slots
                        ]
                    slots.extend(segment_slots)
                
                # Update cumulative state only on user turns
                if role == "user":
                    cumulative_state = self._update_cumulative_state(
                        cumulative_state, slots
                    )
                
                turn_dict = {
                    "role": role,
                    "text": original_text,
                    "slots": slots,
                    "state": {"customer_service": dict(cumulative_state.get("customer_service", {}))},
                }
                
                # Pass disfluency from merge (for user turns with fillers)
                merge_disfluency = turn_data.get("disfluency", [])
                if role == "user" and merge_disfluency:
                    turn_dict["disfluency"] = merge_disfluency
                
                turns.append(turn_dict)

            # Final state
            state = cumulative_state

            yield Dialogue(
                id=dlg_id,
                source=self.name,
                turns=turns,
                goal=goal,
                state=state,
                metadata={"scenario": scenario},
            )

    def _extract_goal(self, scenario: dict) -> Goal:
        """Extract user goal from scenario."""
        flow = scenario.get("flow", "")
        subflow = scenario.get("subflow", "")
        personal = scenario.get("personal", {})
        order = scenario.get("order", {})
        product = scenario.get("product", {})

        # Build text goal
        text = self._build_goal_text(flow, subflow, personal, order, product)

        # Build structured goal
        structured = self._build_structured_goal(flow, subflow, personal, order, product)

        return Goal(text=text, structured=structured)

    def _merge_consecutive_turns(
        self,
        delexed_turns: list[dict],
        original_turns: list[list],
    ) -> list[dict]:
        """Merge consecutive turns from the same speaker into a single turn.
        
        When merging user turns, inserts fillers and tracks them as disfluency
        annotations so inject_disfluency_dialogue can augment rather than duplicate.
        """
        merged = []
        current = None
        fillers = [FP, EDIT, DM]
        # Map tag strings to type codes
        tag_to_type = {FP: "FP", EDIT: "EDIT", DM: "DM"}

        for i, turn_data in enumerate(delexed_turns):
            speaker = turn_data.get("speaker", "")
            delexed_text = turn_data.get("text", "")
            original_speaker = ""

            if i < len(original_turns):
                original_speaker = original_turns[i][0] if original_turns[i] else ""
                original_text = (
                    original_turns[i][1] if len(original_turns[i]) > 1 else delexed_text
                )
            else:
                original_text = delexed_text

            if speaker == "action" or original_speaker == "action":
                continue

            if current and current["speaker"] == speaker:
                # Only add fillers between consecutive USER turns (not agent)
                if speaker.lower() == "customer":
                    filler = random.choice(fillers)
                    filler_word = random.choice(FILLERS[filler])
                    # Keep `text`/`original_text` tag-free; tags are rendered later into `tagged`.
                    join_str = f" {filler_word}, "
                    
                    # Track disfluency annotation for this filler
                    # Position is right after current text + space (where tag starts)
                    filler_position = len(current["original_text"]) + 1
                    current.setdefault("disfluency", []).append({
                        "type": tag_to_type[filler],
                        "position": filler_position,
                        "tag": filler,
                        "text": f"{filler_word}, ",
                    })
                else:
                    join_str = " "
                    
                segment_offset = 0
                if current.get("original_text"):
                    segment_offset = len(current["original_text"]) + len(join_str)

                if current["delexed_text"]:
                    current["delexed_text"] += join_str + delexed_text
                else:
                    current["delexed_text"] = delexed_text
                if current["original_text"]:
                    current["original_text"] += join_str + original_text
                else:
                    current["original_text"] = original_text
                current.setdefault("segments", []).append({
                    "delexed_text": delexed_text,
                    "original_text": original_text,
                    "original_offset": segment_offset,
                })
            else:
                if current:
                    merged.append(current)
                current = {
                    "speaker": speaker,
                    "delexed_text": delexed_text,
                    "original_text": original_text,
                    "segments": [{
                        "delexed_text": delexed_text,
                        "original_text": original_text,
                        "original_offset": 0,
                    }],
                    "disfluency": [],
                }

        if current:
            merged.append(current)

        return merged

    def _build_goal_text(
        self,
        flow: str,
        subflow: str,
        personal: dict,
        order: dict,
        product: dict,
    ) -> str:
        """Build comprehensive natural language goal from scenario.

        Includes ALL information needed to reconstruct the conversation:
        - Task description (flow/subflow)
        - User profile (name, email, phone, username, account_id, member_level)
        - Order details (order_id, address, products, payment, dates)
        """
        sentences = []

        # 1. Task description
        task_desc = self._get_task_description(flow, subflow)
        
        # Strip "Generic Task: " prefix if present to make it more natural
        if task_desc.startswith("Generic Task: "):
            task_desc = task_desc[len("Generic Task: "):]
            
        sentences.append(task_desc)

        # 2. User profile
        profile_parts = self._format_personal_profile(personal)

        if profile_parts:
            # Join with commas and "and" for the last item
            if len(profile_parts) > 1:
                profile_str = ", ".join(profile_parts[:-1]) + ", and " + profile_parts[-1]
            else:
                profile_str = profile_parts[0]
            sentences.append(profile_str + ".")

        # 3. Order details
        order_parts = []
        if order.get("order_id"):
            order_parts.append(f"the order ID is {order['order_id']}")
        if order.get("street_address"):
            order_parts.append(f"address is {order['street_address']}")
        if order.get("city"):
            city_state_zip = order['city']
            if order.get("state"):
                city_state_zip += f", {order['state']}"
            if order.get("zip_code"):
                city_state_zip += f" {order['zip_code']}"
            order_parts.append(f"in {city_state_zip}")
        if order.get("payment_method"):
            order_parts.append(f"paid via {order['payment_method']}")
        if order.get("purchase_date"):
            order_parts.append(f"purchased on {order['purchase_date']}")
        if order.get("shipping_status"):
            order_parts.append(f"shipping status is {order['shipping_status']}")
        if order.get("packaging"):
            order_parts.append(f"packaging is {order['packaging']}")
        if order.get("num_products"):
            order_parts.append(f"number of products is {order['num_products']}")

        if order_parts:
            # Join with commas and "and" for the last item
            if len(order_parts) > 1:
                order_str = ", ".join(order_parts[:-1]) + ", and " + order_parts[-1]
            else:
                order_str = order_parts[0]
            sentences.append("For this request, " + order_str + ".")

        # 4. Product details
        product_names = product.get("names", [])
        product_amounts = product.get("amounts", [])

        if product_names:
            product_parts = []
            for i, name in enumerate(product_names):
                item_str = name.replace("_", " ")
                if i < len(product_amounts):
                    item_str += f" (${product_amounts[i]})"
                product_parts.append(item_str)

            if len(product_parts) > 1:
                products_str = ", ".join(product_parts[:-1]) + ", and " + product_parts[-1]
            else:
                products_str = product_parts[0]
            
            total = sum(product_amounts) if product_amounts else 0
            
            info = f"The products involved are {products_str}"
            if total > 0:
                info += f", with a total of ${total}"
            sentences.append(info + ".")

        return " ".join(sentences)

    def _format_personal_profile(self, personal: dict) -> list[str]:
        """Render every scalar personal field into the goal text."""
        profile_parts = []
        field_templates = {
            "customer_name": "Your name is {value}",
            "username": "your username is {value}",
            "email": "your email is {value}",
            "phone": "your phone number is {value}",
            "account_id": "your account ID is {value}",
            "member_level": "you are a {value} member",
            "pin_number": "your PIN is {value}",
            "security_answer": "your security answer is {value}",
            "password": "your password is {value}",
            "order_id": "your personal order ID is {value}",
        }

        handled = set()
        for field, template in field_templates.items():
            value = personal.get(field)
            if value in (None, "", [], {}):
                continue
            profile_parts.append(template.format(value=value))
            handled.add(field)

        for field, value in personal.items():
            if field in handled or value in (None, "", [], {}):
                continue
            label = field.replace("_", " ")
            profile_parts.append(f"your {label} is {value}")

        return profile_parts

    def _get_task_description(self, flow: str, subflow: str) -> str:
        """Get detailed task description with conversation flow for flow/subflow."""
        flow_instructions = {
            "account_access": {
                "recover_password": "Your goal is to recover your forgotten password. Start by stating that you have forgotten your password. When prompted, provide your username or email address. To wrap up, follow the agent's instructions to reset it.",
                "recover_username": "Your goal is to recover your forgotten username. Begin by explaining that you have forgotten your username. Provide your email address or account details when requested. Once you have the username, confirm you've received it.",
                "reset_2fa": "Your goal is to reset your two-factor authentication. Tell the agent you need to reset your two-factor authentication to start with. Verify your identity with the requested information. Lastly, set up your new 2FA method.",
            },
            "manage_account": {
                "manage_change_address": "Your goal is to update your account address. Initially, inform the agent you want to update your address. Provide the new address details when asked. To finish, confirm the address has been updated correctly.",
                "manage_change_name": "Your goal is to change your account name. State that you need to change your name to begin. Provide the new name and any necessary verification. Follow through until the change is reflected.",
                "manage_change_phone": "Your goal is to update your phone number. To start, tell the agent you want to change your phone number. Give them the new number when asked. After that, verify the update is complete.",
                "manage_payment_method": "Your goal is to update your payment method. Indicate you want to update or add a payment method. Provide the card details securely when requested. Once done, confirm the new method is active.",
                "status_credit_missing": "Your goal is to inquire about missing store credit. Complain that your store credit is missing to initiate the request. Provide order or account context if asked. Make sure the credit is restored to your account before finishing.",
                "status_service_added": "Your goal is to ask about a service added to your account. To begin, ask why a certain service was added to your account. Discuss whether you want to keep or remove it. Conclude by confirming the resolution.",
                "status_service_removed": "Your goal is to ask about a service removed from your account. Start by asking why a service was removed from your account. If it was a mistake, ask to have it reinstated. Finally, confirm the service status.",
                "status_shipping_question": "Your goal is to ask about shipping preferences. Ask about or request to change your shipping preferences first. Discuss options like express or standard shipping. To wrap up, confirm your new preference.",
            },
            "order_issue": {
                "manage_cancel": "Your goal is to cancel an order. Start by stating that you want to cancel your order. Provide the order ID when requested. To finish, confirm the cancellation and check on the refund status.",
                "manage_create": "Your goal is to create a new order. Tell the agent you want to place a new order to begin. Specify the items and quantities. After that, provide shipping and payment details to complete the purchase.",
                "manage_downgrade": "Your goal is to downgrade an order or service. Explain you want to downgrade your current order/subscription first. Choose the lower tier or option. Conclude by confirming the change and the price adjustment.",
                "manage_upgrade": "Your goal is to upgrade an order. Express interest in upgrading your order or shipping speed. Select the upgrade option. Then, confirm the additional cost and update the order.",
                "status_delivery_date": "Your goal is to check your delivery date. Ask when your order is expected to arrive. Provide your order ID. Make sure to note the estimated delivery date.",
                "status_delivery_time": "Your goal is to check your delivery time. Start by asking for the specific time window of your delivery. Confirm if you need to be present. Lastly, acknowledge the time slot.",
                "status_mystery_fee": "Your goal is to inquire about an unexpected fee. Point out an unexpected fee on your bill or order. Ask for an explanation. Conclude by requesting a refund if the fee is unjustified.",
                "status_payment_method": "Your goal is to ask about the payment method used. Ask which payment method was used for a specific order. If needed, ask to change it. Confirm the payment details to finish.",
                "status_quantity": "Your goal is to ask about order quantity. Ask or complain about the quantity of items in your order. Verify against your receipt. To wrap up, resolve the quantity discrepancy.",
            },
            "product_defect": {
                "refund_initiate": "Your goal is to initiate a refund for a defective product. Report that you received a defective product. Describe the defect. Once done, initiate the refund process.",
                "refund_status": "Your goal is to check the status of a refund. Ask about the status of a previously requested refund. Provide the reference number. Lastly, get the estimated timeline for the funds.",
                "refund_update": "Your goal is to update refund information. Say you need to update information for a refund. Provide the new details. Conclude by confirming the update.",
                "return_color": "Your goal is to return a product because of the wrong color. State you received an item in the wrong color. Specify what color you ordered vs. what you got. After that, arrange for a return or exchange.",
                "return_size": "Your goal is to return a product because of the wrong size. Explain that an item doesn't fit or is the wrong size. Mention the correct size needed if exchanging. To finish, start the return process.",
                "return_stain": "Your goal is to return a product because it has a stain. Complain that the item you received has a stain. Describe the damage. Conclude by demanding a replacement or return.",
            },
            "purchase_dispute": {
                "bad_price_competitor": "Your goal is to request a price match with a competitor. Point out that a competitor has a lower price for the same item. Provide details of the competitor's offer. Lastly, ask to match the price.",
                "bad_price_yesterday": "Your goal is to request a price adjustment due to a price drop. Mention the price dropped right after you bought the item. Ask for a price adjustment. Conclude with confirming the credit.",
                "mistimed_billing_already_returned": "Your goal is to dispute a charge for a returned item. State you were charged for an item you already returned. Provide return proof if asked. To finish, ensure the charge is reversed.",
                "mistimed_billing_never_bought": "Your goal is to dispute a charge for an item you never bought. Claim you are being charged for something you never bought. Ask to investigate the charge. Lastly, have the charge removed.",
                "out_of_stock_general": "Your goal is to inquire about an out-of-stock item. Ask if an item is back in stock or when it will be. If unavailable, ask for alternatives. Finally, decide whether to wait or choose another item.",
                "out_of_stock_one_item": "Your goal is to inquire about one item being out of stock. Ask why one item from your order was not shipped. Decide if you want to cancel that item or wait. Conclude with your choice.",
                "promo_code_invalid": "Your goal is to report an invalid promo code. Mention your promo code isn't working. Provide the code. Lastly, ask the agent to apply the discount manually.",
                "promo_code_out_of_date": "Your goal is to ask about an expired promo code. Ask if an expired promo code can still be used or if there is a new specific one. Conclude by applying the discount if granted.",
            },
            "shipping_issue": {
                "cost": "Your goal is to ask about shipping costs. Ask why the shipping cost is so high or what the rates are. Ask for free shipping if applicable. Confirm the cost to finish.",
                "manage": "Your goal is to manage your shipping options. Ask to change the shipping method for an active order. Select the new method. Then, confirm the change.",
                "missing": "Your goal is to report a missing shipment. Report that your package hasn't arrived despite the tracking saying otherwise. Ask for an investigation. Lastly, request a deeper search or refund.",
                "status": "Your goal is to check your shipment status. Ask where your package is. Provide the tracking number. Conclude by getting the current location and ETA.",
            },
            "subscription_inquiry": {
                "manage_dispute_bill": "Your goal is to dispute a subscription charge. Dispute a specific charge on your subscription bill. Explain why it's wrong. To finish, request a correction.",
                "manage_extension": "Your goal is to extend your subscription. Say you want to extend your subscription. Choose the duration. Conclude by confirming the extension.",
                "manage_pay_bill": "Your goal is to pay your subscription bill. State you want to pay your subscription bill. Confirm the amount. Lastly, complete the payment.",
                "status_due_amount": "Your goal is to check the amount due for your subscription. To begin, ask how much you owe for your subscription. Conclude by acknowledging the amount.",
                "status_due_date": "Your goal is to check your subscription due date. Ask when your subscription payment is due. Finally, note the date.",
                "status_questions": "Your goal is to ask general questions about your subscription. Start by asking general questions about what the subscription covers. To wrap up, confirm your understanding.",
            },
            "troubleshoot_site": {
                "credit_card": "Your goal is to resolve a credit card processing issue. Report that your credit card isn't satisfying the system. Describe the error message. Finally, try a different card or solution.",
                "search_results": "Your goal is to report that search results are not working. Mention that the search function isn't finding items correctly. Give an example. Lastly, ask for help finding the item.",
                "shopping_cart": "Your goal is to report a shopping cart issue. Say you can't add items to the cart or checkout. Detail the problem. Conclude by trying to complete the purchase.",
                "slow_speed": "Your goal is to report that the website is running slowly. Complain that the website is loading very slowly. Ask if they are experiencing issues. Conclude by trying to complete your task.",
            },
        }

        if flow in flow_instructions and subflow in flow_instructions[flow]:
            return flow_instructions[flow][subflow]

        # Handle single_item_query
        if flow == "single_item_query":
            item = subflow.split("_")[0] if "_" in subflow else "item"
            query_type = "care/usage" if "how" in subflow else "general"
            return f"Your goal is to inquire about a product ({item}, {query_type}). Please focus on the product '{item}'. First, say you have a question about this item. Ask specifically about {query_type}. Finally, thank the agent for the information."

        # Handle storewide_query
        if flow == "storewide_query":
            if "membership" in subflow:
                return "Your goal is to ask about the membership program. Please focus on membership benefits. First, ask about the benefits or tiers of the membership program. Finally, decide if you want to join or upgrade."
            elif "policy" in subflow:
                return "Your goal is to ask about store policies. Please focus on understanding the rules. First, ask about a specific store policy (e.g., returns, privacy). Finally, confirm you understand the rules."
            elif "pricing" in subflow:
                return "Your goal is to ask about pricing. Please focus on pricing details. First, ask about the pricing of services or general price matching policies. Finally, clarify any conditions."
            elif "timing" in subflow:
                return "Your goal is to ask about store hours. Please focus on opening and closing times. First, ask when the store opens or closes. Finally, note the hours."

        return f"Your goal is to resolve an issue regarding {subflow} ({flow}). Follow a natural conversation flow to address this."

    def _build_structured_goal(
        self,
        flow: str,
        subflow: str,
        personal: dict,
        order: dict,
        product: dict,
    ) -> StructuredGoal:
        """Build structured goal."""
        # Combine all slots
        slots = {}
        slots.update(personal)
        slots.update({f"order_{k}": v for k, v in order.items()})
        slots.update({f"product_{k}": v for k, v in product.items()})

        # Filter out complex nested values
        slots = {k: v for k, v in slots.items() if isinstance(v, (str, int, float))}

        return StructuredGoal(
            domains=["customer_service"],
            intents=[{
                "domain": "customer_service",
                "intent": f"{flow}.{subflow}",
                "slots": slots,
                "requests": [],
            }],
            metadata={
                "personal": personal,
                "order": order,
                "product": product,
            },
        )

    def _build_initial_state(self, scenario: dict) -> dict:
        """Build initial state from scenario data.
        
        Args:
            scenario: ABCD scenario with personal/order info
            
        Returns:
            Initial state dict {customer_service: {...}}
        """
        state = {"customer_service": {}}
        personal = scenario.get("personal", {})
        order = scenario.get("order", {})

        state["customer_service"].update(personal)
        state["customer_service"].update(order)

        return state

    def _clean_slot_value(self, value: str) -> str:
        """Remove disfluency tags from slot value."""
        # Remove [FP], [DM], [EDIT] tags
        clean = re.sub(r"\[(FP|DM|EDIT)\]", "", value)
        # Remove extra spaces
        return " ".join(clean.split())

    def _update_cumulative_state(self, state: dict, slots: list[SlotSpan]) -> dict:
        """Update cumulative state from extracted slots.
        
        Args:
            state: Current cumulative state {customer_service: {slot: value}}
            slots: List of SlotSpan from the current turn
            
        Returns:
            Updated state dict
        """
        new_state = {"customer_service": dict(state.get("customer_service", {}))}
        
        for slot in slots:
            new_state["customer_service"][slot.slot] = self._clean_slot_value(slot.value)
        
        return new_state

    def _extract_slots(
        self, delexed_text: str, original_text: str, scenario: dict | None = None
    ) -> list[SlotSpan]:
        """Extract slot spans by comparing delexed and original text.

        ABCD uses <slot_name> placeholders in delexed text:
        delexed:  "username: <username>"
        original: "Username: cminh730"

        We find <slot> patterns and map them to the original text.
        
        Additionally, ABCD does NOT delexicalize customer_name, so we search
        for it directly using scenario.personal.customer_name.
        """
        slots = []

        # Find all <slot_name> patterns in delexed text
        pattern = re.compile(r"<(\w+)>")
        matches = list(pattern.finditer(delexed_text))

        if not matches:
            return slots

        # For each match, find corresponding value in original text
        for match in matches:
            slot_name = match.group(1)
            delex_start = match.start()
            delex_end = match.end()

            # Get text before the slot in delexed version
            prefix = delexed_text[:delex_start].lower()

            # Find where prefix ends in original text (case-insensitive)
            original_lower = original_text.lower()
            prefix_pos = original_lower.find(prefix) if prefix else 0

            if prefix_pos == -1:
                prefix_pos = 0

            # Calculate start position in original
            start = prefix_pos + len(prefix)

            # Find text after the slot in delexed version
            suffix = delexed_text[delex_end:].lower()

            # Find where suffix starts in original text
            if suffix:
                suffix_pos = original_lower.find(suffix, start)
                if suffix_pos != -1:
                    end = suffix_pos
                else:
                    # No suffix found, take rest of string
                    end = len(original_text)
            else:
                # No suffix, take rest of string
                end = len(original_text)

            # Extract the value
            value = original_text[start:end].strip()

            if value:
                # Adjust start/end to match stripped value position
                actual_start = original_text.find(value, start - 1)
                if actual_start != -1:
                    slots.append(SlotSpan(
                        slot=slot_name,
                        value=value,
                        start=actual_start,
                        end=actual_start + len(value),
                    ))

        # Find customer_name directly from scenario (not delexicalized in ABCD)
        if scenario:
            slots = self._find_scenario_slots(original_text, scenario, slots)

        return slots

    def _find_scenario_slots(
        self,
        text: str,
        scenario: dict,
        existing_slots: list[SlotSpan],
    ) -> list[SlotSpan]:
        """Find slots that are not delexicalized by comparing with scenario values.
        
        ABCD does not use placeholders for customer_name, so we need to find it
        by matching the text against scenario.personal.customer_name.
        """
        slots = list(existing_slots)
        existing_slot_names = {s.slot for s in slots}
        
        # Get personal info from scenario
        personal = scenario.get("personal", {})
        order = scenario.get("order", {})
        
        segmentable = SEGMENTABLE_SLOTS.get("abcd", {})
        direct_slots = {}
        for slot_name in segmentable:
            if slot_name in personal:
                direct_slots[slot_name] = personal.get(slot_name, "")
            elif slot_name in order:
                direct_slots[slot_name] = order.get(slot_name, "")
        
        text_lower = text.lower()
        
        for slot_name, slot_value in direct_slots.items():
            if not slot_value or slot_name in existing_slot_names:
                continue

            slot_value_str = str(slot_value)
            value_lower = slot_value_str.lower()
            start_pos = text_lower.find(value_lower)
            
            if start_pos != -1:
                # Get the actual casing from original text
                actual_value = text[start_pos:start_pos + len(slot_value_str)]
                slots.append(SlotSpan(
                    slot=slot_name,
                    value=actual_value,
                    start=start_pos,
                    end=start_pos + len(actual_value),
                ))
        
        return slots
