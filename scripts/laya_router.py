"""
laya_router.py - System 1 Decision & Reflex Engine for 2ndBrain

Powered by Convai Innovations' non-autoregressive Laya architecture (ModernBERT-large, 421M).
Evaluates typed schema questions (Choice, Score, Noul) in a single parallel forward pass,
with zero-latency calibrated heuristic fallbacks and passive Strategy 3 telemetry logging.
"""

import os
import re
import json
import datetime
import warnings
from typing import Dict, List, Any, Optional

try:
    import laya
    LAYA_AVAILABLE = True
except ImportError:
    LAYA_AVAILABLE = False


class LayaDecisionEngine:
    """
    System 1 Decision Engine powered by Convai Innovations' Laya model.
    Evaluates typed schema questions (Choice, Score, Noul) in a single forward pass.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        subfolder: Optional[str] = "typed-decisions",
        device: str = "cpu",
        enable_neural: bool = True,
        telemetry_path: Optional[str] = None
    ):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        local_model_path = os.path.join(base_dir, "models", "laya-2ndbrain")
        project_fallback = r"C:\Users\Fabian\Desktop\Y2T1\RSE2802 Concept Defintion\Second Brain\models\laya-2ndbrain"
        if not (os.path.exists(local_model_path) and os.path.exists(os.path.join(local_model_path, "model.safetensors"))):
            if os.path.exists(project_fallback) and os.path.exists(os.path.join(project_fallback, "model.safetensors")):
                local_model_path = project_fallback

        # Auto-detect local fine-tuned and calibrated checkpoint
        if model_id is None:
            if os.path.exists(local_model_path) and os.path.exists(os.path.join(local_model_path, "model.safetensors")):
                self.model_id = local_model_path
                self.subfolder = None
                self.is_fine_tuned = True
            else:
                self.model_id = "convaiinnovations/laya"
                self.subfolder = subfolder
                self.is_fine_tuned = False
        else:
            self.model_id = model_id
            self.subfolder = subfolder
            self.is_fine_tuned = "laya-2ndbrain" in model_id

        self.device = device
        self.enable_neural = enable_neural
        self._agent = None
        self._load_attempted = False

        if telemetry_path:
            self.telemetry_path = telemetry_path
        else:
            self.telemetry_path = os.path.join(base_dir, "vault_template", "04-Agent-Memory", "telemetry.jsonl")

        self.isotonic_noul = None
        iso_path = os.path.join(self.model_id, "noul_isotonic.json")
        if os.path.exists(iso_path):
            try:
                with open(iso_path, "r", encoding="utf-8") as f:
                    self.isotonic_noul = json.load(f)
            except Exception:
                pass

    def calibrate_noul(self, prob: float) -> float:
        """Applies exact non-parametric Isotonic Regression (PAVA) calibration to Noul probabilities."""
        if self.isotonic_noul and "x" in self.isotonic_noul and "y" in self.isotonic_noul:
            try:
                import numpy as np
                cal = float(np.interp(prob, self.isotonic_noul["x"], self.isotonic_noul["y"], left=0.0, right=1.0))
                return round(cal, 3)
            except Exception:
                pass
        return round(prob, 3)

    @property
    def agent(self):
        """Lazy-loads the genuine Laya ModernBERT agent on first invocation."""
        if not self.enable_neural or not LAYA_AVAILABLE:
            return None
        if not self._load_attempted:
            self._load_attempted = True
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self._agent = laya.load(self.model_id, subfolder=self.subfolder, device=self.device)
            except Exception as e:
                print(f"[Laya] Warning: Could not initialize neural agent: {e}. Using calibrated fallback.")
                self._agent = None
        return self._agent

    def log_telemetry(
        self,
        task: str,
        state: str,
        decision: Any,
        confidence: float,
        engine_type: str,
        is_feedback: bool = False,
        feedback_notes: str = ""
    ):
        """Strategy 3: Passive log-as-you-go telemetry recording for continuous distillation."""
        try:
            os.makedirs(os.path.dirname(self.telemetry_path), exist_ok=True)
            entry = {
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "task": task,
                "state": state[:2000],
                "decision": decision,
                "confidence": round(confidence, 4),
                "engine_type": engine_type,
                "model_id": self.model_id,
                "is_feedback": is_feedback,
                "feedback_notes": feedback_notes
            }
            with open(self.telemetry_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass

    def record_feedback(self, query: str, correct_decision: str, reason: str = ""):
        """Strategy 3 Feedback: Explicitly records a user correction as a gold training label for future retraining."""
        self.log_telemetry(
            task="user_correction",
            state=query,
            decision=correct_decision,
            confidence=1.0,
            engine_type="gold_user_correction",
            is_feedback=True,
            feedback_notes=reason
        )

    def route_query(self, query: str) -> Dict[str, Any]:
        """
        Typed Question: Choice
        Which engine should resolve this question?
        Options: CRG_AST_CALL_GRAPH, GRAPHIFY_SEMANTIC_GRAPHRAG, OBSIDIAN_VAULT_SEARCH
        """
        if self.agent:
            try:
                questions = {
                    "target_engine": {
                        "type": "choice",
                        "instructions": "Which engine should answer this question?",
                        "criteria": {
                            "CRG_AST_CALL_GRAPH": "Questions regarding code function call trees, callers, callees, imports, class inheritance, or code blast radius",
                            "GRAPHIFY_SEMANTIC_GRAPHRAG": "Questions about system architecture, high-level design, or module concepts",
                            "OBSIDIAN_VAULT_SEARCH": "Questions asking about past meeting notes, decisions, user preferences, remembered rules, or session history"
                        }
                    }
                }
                res = self.agent.system_one(query, questions)
                ans = res["answers"]["target_engine"]
                decision = ans["choice"]
                conf = ans.get("answer_confidence", ans.get("confidence", 0.75))
                result = {
                    "decision": decision,
                    "confidence": round(conf, 3),
                    "reasoning": f"Neural Laya ({'Fine-Tuned 2ndBrain' if self.is_fine_tuned else 'ModernBERT-large'}) routed via {decision} ({conf*100:.1f}% confidence)",
                    "engine_type": "neural"
                }
                self.log_telemetry("query_triage", query, decision, conf, "neural")
                return result
            except Exception:
                pass

        # --- Calibrated Heuristic Fallback ---
        q_lower = query.lower()

        ast_triggers = ["call", "caller", "callee", "who calls", "blast radius", "break", "signature", 
                        "function", "class", "symbol", "inherit", "dependency", "ast", "imported"]
        concept_triggers = ["why", "architecture", "overview", "community", "high-level", "design", 
                            "how does the system", "concept", "cluster", "bridge", "module relationship"]
        memory_triggers = ["what did we decide", "rule", "preference", "session", "yesterday", 
                           "past discussion", "consultation", "transcript", "note", "remember"]

        ast_score = sum(1 for t in ast_triggers if t in q_lower)
        concept_score = sum(1 for t in concept_triggers if t in q_lower)
        memory_score = sum(1 for t in memory_triggers if t in q_lower)

        if ast_score > concept_score and ast_score > memory_score:
            decision = "CRG_AST_CALL_GRAPH"
            confidence = min(0.95, 0.6 + ast_score * 0.1)
        elif concept_score > ast_score and concept_score > memory_score:
            decision = "GRAPHIFY_SEMANTIC_GRAPHRAG"
            confidence = min(0.95, 0.6 + concept_score * 0.1)
        elif memory_score > 0:
            decision = "OBSIDIAN_VAULT_SEARCH"
            confidence = min(0.95, 0.6 + memory_score * 0.1)
        else:
            decision = "GRAPHIFY_SEMANTIC_GRAPHRAG"
            confidence = 0.65

        result = {
            "decision": decision,
            "confidence": round(confidence, 3),
            "reasoning": f"Heuristic fallback routed via {decision} with confidence {confidence:.2f}",
            "engine_type": "heuristic"
        }
        self.log_telemetry("query_triage", query, decision, confidence, "heuristic")
        return result

    def score_ingestion_novelty(self, text: str) -> Dict[str, Any]:
        """
        Typed Question: Score (1 to 5)
        Rate the architectural significance and conceptual novelty of this document.
        """
        if self.agent:
            try:
                questions = {
                    "novelty_score": {
                        "type": "score",
                        "instructions": "Rate the architectural importance and conceptual novelty of this document on a scale of 1 to 5.",
                        "criteria": [
                            "Trivial changes, changelogs, formatting, license files",
                            "Minor updates, routine bug fixes, test snapshots",
                            "Standard module design, implementation details",
                            "Significant system features, architectural designs, APIs",
                            "Core architecture, foundational ADRs, critical protocol designs"
                        ]
                    }
                }
                res = self.agent.system_one(text[:2500], questions)
                score_ans = res["answers"]["novelty_score"]
                lvl = score_ans.get("level", 3)
                final_score = max(1, min(5, lvl + 1))
                conf = score_ans.get("confidence", 0.8)
                result = {
                    "score": final_score,
                    "should_extract_graph": final_score >= 3,
                    "reasoning": f"Neural Laya novelty scored at {final_score}/5",
                    "engine_type": "neural"
                }
                self.log_telemetry("ingestion_novelty", text[:200], final_score, conf, "neural")
                return result
            except Exception:
                pass

        # --- Calibrated Heuristic Fallback ---
        text_sample = text[:3000].lower()
        score = 2

        novelty_indicators = ["architecture", "adr", "decision", "trade-off", "system design", 
                              "database schema", "protocol", "security", "pipeline", "consensus"]
        routine_indicators = ["typo", "formatting", "license", "changelog", "test snapshot", "bump version"]

        for ind in novelty_indicators:
            if ind in text_sample:
                score += 0.5
        for ind in routine_indicators:
            if ind in text_sample:
                score -= 0.5

        final_score = max(1, min(5, round(score)))
        result = {
            "score": final_score,
            "should_extract_graph": final_score >= 3,
            "reasoning": f"Heuristic architectural novelty scored at {final_score}/5",
            "engine_type": "heuristic"
        }
        self.log_telemetry("ingestion_novelty", text[:200], final_score, 0.7, "heuristic")
        return result

    def score_blast_risk(self, diff_summary: str) -> Dict[str, Any]:
        """
        Typed Question: Noul (Probability 0.0 - 1.0)
        Does this change introduce breaking API changes, alter core boundaries, or impact critical schemas?
        """
        if self.agent:
            try:
                questions = {
                    "is_high_risk": {
                        "type": "noul",
                        "instructions": "Does this code diff introduce breaking API changes, drop database tables/columns, remove public functions, or break security?",
                        "criteria": {
                            "true": "Contains breaking changes, signature modifications, deletions, or database alterations",
                            "false": "Safe, non-breaking, documentation, formatting, or internal additive changes"
                        }
                    }
                }
                res = self.agent.system_one(diff_summary[:2500], questions)
                noul_ans = res["answers"]["is_high_risk"]
                raw_prob = noul_ans.get("noul", noul_ans.get("probability", 0.2))
                prob = self.calibrate_noul(raw_prob)
                result = {
                    "probability": round(prob, 2),
                    "is_high_risk": prob > 0.7,
                    "recommendation": "ESCALATE_TO_SYSTEM2" if prob > 0.7 else "AUTO_LOG",
                    "engine_type": "neural"
                }
                self.log_telemetry("blast_risk", diff_summary[:200], prob > 0.7, prob, "neural")
                return result
            except Exception:
                pass

        # --- Calibrated Heuristic Fallback ---
        diff_lower = diff_summary.lower()
        risk_prob = 0.2

        critical_markers = [
            "breaking", "delete", "drop table", "remove function", "signature changed", 
            "public api", "auth", "security", "database migration", "critical",
            "-def ", "-class ", "drop column", "alter table", "deprecated"
        ]
        for marker in critical_markers:
            if marker in diff_lower:
                risk_prob += 0.25

        risk_prob = min(0.99, round(risk_prob, 2))
        result = {
            "probability": risk_prob,
            "is_high_risk": risk_prob > 0.7,
            "recommendation": "ESCALATE_TO_SYSTEM2" if risk_prob > 0.7 else "AUTO_LOG",
            "engine_type": "heuristic"
        }
        self.log_telemetry("blast_risk", diff_summary[:200], risk_prob > 0.7, risk_prob, "heuristic")
        return result

    def evaluate_conversational_turn(self, user_msg: str, assistant_msg: str) -> Dict[str, Any]:
        """
        Typed Question: Noul + Choice
        Evaluates whether a conversation turn contains a durable rule, preference, or architectural decision.
        """
        if self.agent:
            try:
                state = f"User: {user_msg}\nAssistant: {assistant_msg}"
                questions = {
                    "is_durable_memory": {
                        "type": "noul",
                        "instructions": "Does this turn establish a persistent user preference, explicit rule, bug fix pattern, or architectural constraint?",
                        "criteria": {
                            "true": "Establishes a rule, constraint, or preference that should be remembered across future sessions",
                            "false": "Chit-chat, temporary question, transient task execution, or one-off conversation"
                        }
                    },
                    "memory_type": {
                        "type": "choice",
                        "instructions": "What category of persistent memory is this?",
                        "criteria": {
                            "user_preference": "User instructions on style, tool preferences, coding habits",
                            "architectural_constraint": "Project architecture decisions, ADRs, design requirements",
                            "bug_pattern": "Specific bug fixes, pitfalls to avoid",
                            "none": "Not a durable memory"
                        }
                    }
                }
                res = self.agent.system_one(state[:2500], questions)
                raw_prob = res["answers"]["is_durable_memory"].get("noul", 0.1)
                prob = self.calibrate_noul(raw_prob)
                m_type = res["answers"]["memory_type"].get("choice", "none")
                result = {
                    "is_durable_memory": prob >= 0.7 and m_type != "none",
                    "probability": round(prob, 2),
                    "memory_type": m_type,
                    "rule_text": user_msg.strip() if prob >= 0.7 else None,
                    "engine_type": "neural"
                }
                self.log_telemetry("memory_distillation", user_msg[:200], prob >= 0.7, prob, "neural")
                return result
            except Exception:
                pass

        # --- Calibrated Heuristic Fallback ---
        user_lower = user_msg.lower()
        is_memory = False
        memory_type = "none"
        extracted_rule = None
        prob = 0.1

        preference_patterns = [
            (r"(always|never|do not|don't|prefer)\s+(use|import|modify|touch|run|apply|create)\s+(.*)", "user_preference"),
            (r"(we decided|decision:|let's go with|our architecture requires)\s+(.*)", "architectural_constraint"),
            (r"(remember that|make sure to always|make sure that|note that)\s+(.*)", "user_preference"),
            (r"(fix|bug):?\s+(.*)", "bug_pattern")
        ]

        for pattern, m_type in preference_patterns:
            match = re.search(pattern, user_lower)
            if match:
                is_memory = True
                memory_type = m_type
                prob = 0.88
                extracted_rule = user_msg.strip()
                break

        if not is_memory and ("we decided" in assistant_msg.lower() or "architectural decision" in assistant_msg.lower()):
            is_memory = True
            memory_type = "architectural_constraint"
            prob = 0.78
            extracted_rule = "Synthesized architectural decision from conversation."

        result = {
            "is_durable_memory": is_memory and prob >= 0.7,
            "probability": prob,
            "memory_type": memory_type,
            "rule_text": extracted_rule,
            "engine_type": "heuristic"
        }
        self.log_telemetry("memory_distillation", user_msg[:200], is_memory and prob >= 0.7, prob, "heuristic")
        return result

    def classify_vault_category(self, title: str, content: str) -> str:
        """
        Typed Question: Choice
        Which vault category does this note belong to?
        """
        combined = (title + " " + content[:500]).lower()
        if "decision" in combined or "adr" in combined:
            return "01-Concepts/Decisions"
        elif "transcript" in combined or "consultation" in combined or "meeting" in combined:
            return "03-Sources/Transcripts"
        elif "rule" in combined or "preference" in combined or "correction" in combined:
            return "04-Agent-Memory/Corrections"
        elif "module" in combined or "ast" in combined or "class" in combined or "function" in combined:
            return "02-Codebase/Modules"
        else:
            return "01-Concepts/Architecture"


# Global singleton instance
router = LayaDecisionEngine()
