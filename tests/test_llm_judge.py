import json
import sys
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

import llm_judge

class LlmJudgeTests(unittest.TestCase):
    def test_null_payload_fields_do_not_stop_judging(self):
        evidence = {
            "Sub_OAJ1": {
                "kode": None,
                "nama": None,
                "objective": None,
                "assessment_question": None,
                "evidence_to_find": None,
                "juknis": None,
                "regulasi": [
                    {"chunk_id": "reg-1", "text": None},
                    {"chunk_id": None, "text": "ignored"},
                ],
            }
        }
        assessments = {
            "Sub_OAJ1": {
                "kode": None,
                "status": None,
                "assessment_level": None,
                "alasan": None,
                "evidence_juknis": None,
                "evidence_regulasi": None,
            }
        }

        raw_prompts = []

        def fake_ollama(prompt):
            raw_prompts.append(prompt)
            return '''```json
            {"decision_correctness": null, "reasoning_faithfulness": 1,
             "citation_groundedness": 1, "completeness": null,
             "verdict": "undeterminable", "supported_chunk_ids": ["reg-1", "fake"],
             "rationale": "Evidence tidak cukup."}
            ```'''

        result = llm_judge.judge_all_subcategories(
            evidence, assessments, call_model=fake_ollama
        )

        self.assertEqual(len(raw_prompts), 1)
        payload = json.loads(raw_prompts[0].split("PAKET EVALUASI:", 1)[1])
        self.assertEqual(payload["sub_kriteria"]["objective"], "")
        self.assertEqual(payload["sub_kriteria"]["assessment_question"], "")
        self.assertEqual(payload["evidence_regulasi"][0]["text"], "")
        self.assertIsNone(payload["output_sistem_qwen"]["assessment_level"])
        self.assertEqual(result["Sub_OAJ1"]["supported_chunk_ids"], ["reg-1"])
        self.assertIsNone(result["Sub_OAJ1"]["decision_correctness"])
        self.assertEqual(result["Sub_OAJ1"]["reasoning_faithfulness"], 1)

    def test_invalid_response_becomes_a_saved_fallback_judgment(self):
        result = llm_judge.judge_all_subcategories(
            {"Sub_OAJ1": {"juknis": [], "regulasi": []}},
            {"Sub_OAJ1": {"status": "Sesuai"}},
            call_model=lambda _: "bukan json",
        )

        judgment = result["Sub_OAJ1"]
        self.assertEqual(judgment["verdict"], "undeterminable")
        self.assertIn("Judge gagal dijalankan", judgment["rationale"])


class OllamaResponseTests(unittest.TestCase):
    def test_response_text_reads_message_content(self):
        self.assertEqual(
            llm_judge._response_text({"message": {"content": '{"ok": true}'}}),
            '{"ok": true}',
        )


if __name__ == "__main__":
    unittest.main()
