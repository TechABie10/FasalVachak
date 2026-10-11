import re

from llm.prompts import (
    CROP_DETECTION_PROMPT,
    DISAMBIGUATION_PROMPT,
    FOLLOWUP_INTENT_PROMPT,
    FOLLOWUP_QA_PROMPT,
    MANAGEMENT_PROMPT,
)
from llm.client import get_fast_llm_completion, get_reasoning_completion
from llm.translation import translate_to_english
from rag.retriever import get_candidates, get_disease_record

MAX_DISAMBIGUATION_QUESTIONS = 2
MAX_FOLLOWUPS = 2

# All fixed replies are in Devanagari so they match the LLM-generated Hindi
# and are read consistently by the Hindi (hi-IN) TTS voice.
GREETING = (
    "नमस्ते, कृषिसेवा में आपका स्वागत है। "
    "आपकी धान या कपास की फसल में क्या लक्षण दिख रहे हैं, कृपया बताइए?"
)
GENERIC_ERROR_REPLY = "माफ़ कीजिए, थोड़ी दिक्कत आ गई। कृपया दोबारा बोलिए।"
CALL_ALREADY_ENDED_REPLY = "यह कॉल समाप्त हो चुकी है। नए सवाल के लिए कृपया नई कॉल करें।"
CROP_UNCLEAR_REPLY = "माफ़ कीजिए, मैं ठीक से समझ नहीं पाया। यह धान की समस्या है या कपास की?"
NO_DIAGNOSIS_REPLY = (
    "माफ़ कीजिए, लक्षणों के आधार पर मैं सही बीमारी नहीं पहचान पा रहा हूँ। "
    "कृपया कृषि विशेषज्ञ से संपर्क करें।"
)
REPEAT_SYMPTOMS_REPLY = "माफ़ कीजिए, मैं ठीक से समझ नहीं पाया। क्या आप लक्षण दोबारा बता सकते हैं?"
ASK_FOLLOWUP_TOPIC_REPLY = "जी ज़रूर, बताइए, आप क्या जानना चाहते हैं?"
MORE_QUESTIONS_REPLY = "क्या आपको इसके बारे में और कुछ जानना है?"
GOODBYE_REPLY = "ठीक है। कृषिसेवा में कॉल करने के लिए धन्यवाद। नमस्ते!"
FOLLOWUP_LIMIT_REPLY = (
    "माफ़ कीजिए, इस कॉल का समय समाप्त हो गया है। "
    "और जानकारी के लिए कृपया नई कॉल करें। धन्यवाद!"
)


class DialogueManager:
    def __init__(self):
        self.phase = "crop_detection"
        self.crop = None
        self.symptoms_text = ""
        # accumulates farmer text while crop is still unknown
        self.pending_symptom_turns = []
        self.candidates_text = ""
        self.candidate_names = []
        self.qa_history = []
        self.questions_asked = 0
        self.identified_disease = None
        self.followups_asked = 0
        # True after the farmer said "yes" without a question and we asked
        # what they want to know — their next turn is the question itself.
        self.awaiting_followup_question = False

        # (hindi_input, english_translation) pairs, for later glossary-building / debugging
        self.translation_log = []

    async def process_turn(self, farmer_text_hindi: str) -> str:
        # No point translating once the call is over.
        if self.phase == "completed":
            return CALL_ALREADY_ENDED_REPLY

        try:
            translated_text = await translate_to_english(farmer_text_hindi)
            self.translation_log.append((farmer_text_hindi, translated_text))
        except Exception as e:
            print(f"[DialogueManager] Translation failed: {e}")
            return GENERIC_ERROR_REPLY

        try:
            if self.phase == "crop_detection":
                return await self._handle_crop_detection(translated_text)
            elif self.phase == "symptom_diagnosis":
                return await self._handle_disambiguation(translated_text)
            elif self.phase == "followup":
                return await self._handle_followup(translated_text)
            else:
                return GENERIC_ERROR_REPLY
        except Exception as e:
            print(
                f"[DialogueManager] Unhandled error in phase={self.phase}: {e}")
            return GENERIC_ERROR_REPLY

    async def _handle_crop_detection(self, text_en: str) -> str:
        """Phase 0: Figure out the crop from what the farmer says (may take multiple turns)."""

        # Always bank this turn's text — it might contain symptom info even if
        # it doesn't (yet) reveal the crop.
        self.pending_symptom_turns.append(text_en)

        prompt = CROP_DETECTION_PROMPT.format(farmer_text=text_en)
        llm_decision = await get_fast_llm_completion(prompt, temperature=0.0)
        decision = self._first_word(llm_decision)

        if decision == "RICE":
            self.crop = "rice"
        elif decision == "COTTON":
            self.crop = "cotton"

        if not self.crop:
            return CROP_UNCLEAR_REPLY

        # Crop is now known — combine EVERYTHING said so far (across however many
        # turns it took) into the symptom text used for retrieval, not just this
        # last confirmation message.
        self.symptoms_text = " ".join(self.pending_symptom_turns)
        self.phase = "symptom_diagnosis"

        self.candidates_text = get_candidates(self.crop, self.symptoms_text)
        self.candidate_names = self._parse_candidate_names(
            self.candidates_text)

        return await self._trigger_llm_turn()

    async def _handle_disambiguation(self, text_en: str) -> str:
        """Phase 1: Handles the farmer's answers to our clarifying questions (in English)."""
        self.qa_history.append(f"Farmer: {text_en}")
        self.questions_asked += 1
        return await self._trigger_llm_turn()

    async def _trigger_llm_turn(self) -> str:
        """The Reasoning Engine for Phase 1. Decides whether to ASK or ANSWER."""

        if not self.candidate_names:
            self.phase = "completed"
            return NO_DIAGNOSIS_REPLY

        limit_reached = self.questions_asked >= MAX_DISAMBIGUATION_QUESTIONS

        formatted_history = "\n".join(self.qa_history)
        prompt = DISAMBIGUATION_PROMPT.format(
            symptom_text=self.symptoms_text,
            candidates=self.candidates_text,
            qa_history=formatted_history,
            questions_asked_so_far=self.questions_asked,
            max_questions=MAX_DISAMBIGUATION_QUESTIONS
        )
        if limit_reached:
            # Restated last, where it is hardest to ignore: without reasoning the
            # model otherwise tends to keep asking past the limit.
            prompt += (
                "\nThe maximum number of questions has been reached. "
                "You MUST respond with ACTION: ANSWER now.\n"
            )

        # The ASK content the LLM produces here is in Hindi (it's spoken to the
        # farmer directly) — DISAMBIGUATION_PROMPT instructs this explicitly.
        llm_output = await get_reasoning_completion(prompt)
        action, content = self._parse_action_content(llm_output)

        if action == "ANSWER":
            disease = self._match_candidate(content)
            if disease is None:
                # The LLM named something outside the candidate list; fall back
                # to the best retrieval match instead of ending the call.
                print(
                    f"[DialogueManager] ANSWER {content!r} not in candidates "
                    f"{self.candidate_names}; using top candidate.")
                disease = self.candidate_names[0]
            self.identified_disease = disease
            return await self._generate_management_plan()

        if limit_reached:
            # The LLM was told it must answer now; if it still didn't (or its
            # output was unparseable), use the best retrieval match.
            print(
                f"[DialogueManager] No ANSWER at question limit "
                f"(output: {llm_output!r}); using top candidate.")
            self.identified_disease = self.candidate_names[0]
            return await self._generate_management_plan()

        if action == "ASK" and content:
            # Stored as-is (Hindi) in qa_history; DISAMBIGUATION_PROMPT reasons fine over
            # a mixed-language history since the farmer side is consistently English.
            self.qa_history.append(f"Agent: {content}")
            return content

        print(
            f"[DialogueManager] Could not parse reasoning output: {llm_output!r}")
        return REPEAT_SYMPTOMS_REPLY

    async def _generate_management_plan(self) -> str:
        """Phase 2: Fetch DB record, generate spoken treatment (in Hindi), transition to Phase 3."""

        kb_record = get_disease_record(self.identified_disease)

        if not kb_record:
            self.phase = "completed"
            return (
                f"लगता है यह {self.identified_disease} है, पर मेरे पास इसकी पूरी दवाई की "
                f"जानकारी अभी नहीं है। कृपया किसी कृषि विशेषज्ञ से संपर्क करें।"
            )

        self.identified_disease = kb_record.get(
            "disease_name", self.identified_disease)

        prompt = MANAGEMENT_PROMPT.format(
            disease=self.identified_disease,
            kb_context=self._format_kb_record(kb_record)
        )

        treatment_audio_text = await get_fast_llm_completion(prompt, temperature=0.1)

        self.phase = "followup"
        closing_question = f"क्या आपको {self.identified_disease} के बारे में और जानकारी चाहिए?"

        return f"{treatment_audio_text}\n\n{closing_question}"

    async def _handle_followup(self, text_en: str) -> str:
        """Phase 3: The Follow-Up Loop (farmer input already translated to English)."""

        intent_prompt = FOLLOWUP_INTENT_PROMPT.format(farmer_text=text_en)
        intent_response = await get_fast_llm_completion(intent_prompt, temperature=0.0)
        intent = self._first_word(intent_response)

        if intent == "NO":
            self.phase = "completed"
            return GOODBYE_REPLY

        if intent == "YES" and not self.awaiting_followup_question:
            # A bare "yes" is not a question — ask what they want to know
            # instead of answering "yes" as if it were one.
            self.awaiting_followup_question = True
            return ASK_FOLLOWUP_TOPIC_REPLY

        # QUESTION, an unrecognised label, or a second vague reply after we
        # already asked for the topic — answer it as a question.
        self.awaiting_followup_question = False

        self.followups_asked += 1
        if self.followups_asked > MAX_FOLLOWUPS:
            self.phase = "completed"
            return FOLLOWUP_LIMIT_REPLY

        kb_record = get_disease_record(self.identified_disease)
        qa_prompt = FOLLOWUP_QA_PROMPT.format(
            disease=self.identified_disease,
            kb_context=self._format_kb_record(kb_record),
            farmer_question=text_en
        )

        answer_text = await get_fast_llm_completion(qa_prompt, temperature=0.1)
        return f"{answer_text}\n\n{MORE_QUESTIONS_REPLY}"

    # ---------- helpers ----------

    @staticmethod
    def _first_word(llm_output: str) -> str:
        """Upper-cased first word of an LLM label output, punctuation stripped."""
        match = re.search(r"[A-Za-z]+", llm_output or "")
        return match.group(0).upper() if match else ""

    def _match_candidate(self, name: str | None) -> str | None:
        """Maps the LLM's ANSWER onto one of the retrieved candidate names."""
        if not name:
            return None
        cleaned = name.strip().strip("\"'.*").strip().lower()
        for candidate in self.candidate_names:
            if candidate.lower() == cleaned:
                return candidate
        # Tolerate small additions/omissions, e.g. "Blast disease" vs "Blast".
        # Longest candidate first so "Alternaria Leaf Blight" beats "Blight".
        for candidate in sorted(self.candidate_names, key=len, reverse=True):
            if candidate.lower() in cleaned or cleaned in candidate.lower():
                return candidate
        return None

    @staticmethod
    def _parse_action_content(llm_output: str):
        match = re.search(
            r"ACTION:\s*(\w+)\s*\n+CONTENT:\s*(.*)",
            (llm_output or "").strip(),
            re.IGNORECASE | re.DOTALL
        )
        if not match:
            return None, None
        action = match.group(1).strip().upper()
        content = match.group(2).strip()
        return action, content

    @staticmethod
    def _parse_candidate_names(candidates_text: str):
        names = []
        for line in candidates_text.splitlines():
            m = re.match(r"\s*\d+\.\s*([^:]+):", line)
            if m:
                names.append(m.group(1).strip())
        return names

    @staticmethod
    def _format_kb_record(kb_record: dict) -> str:
        if not kb_record:
            return "No knowledge base record available."

        bullets = kb_record.get("symptoms_bullets", [])
        symptoms = " ".join(bullets) if bullets else "Not available."

        lines = [
            f"Disease: {kb_record.get('disease_name', 'Unknown')}",
            f"Causal organism: {kb_record.get('causal_organism') or 'Not available.'}",
            f"Symptoms: {symptoms}",
            f"Favourable conditions: {kb_record.get('favourable_conditions') or 'Not available.'}",
            f"Survival and spread: {kb_record.get('survival_and_spread') or 'Not available.'}",
            f"Management: {kb_record.get('management') or 'Not available.'}",
        ]
        return "\n".join(lines)
