"""
cognitive_module.nemo_scorer
────────────────────────────
NVIDIA NeMo Prompt Task and Complexity Classifier Integration.
Maps DeBERTa-v3 output to CORA CognitiveProfile objects.
"""
from __future__ import annotations
import logging
import os
from typing import Optional

from .models import CognitiveProfile, TaskType
from .scorer import BaseScorer
from .config import CognitiveConfig, DEFAULT_CONFIG

logger = logging.getLogger("cora.nemo_scorer")

_ML_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import numpy as np
    from huggingface_hub import PyTorchModelHubMixin
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    _ML_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    np = None
    PyTorchModelHubMixin = object  # dummy

# ── NVIDIA's Architecture from HF Model Card ─────────────────────────────────

if _ML_AVAILABLE:
    class MeanPooling(nn.Module):
        def __init__(self):
            super(MeanPooling, self).__init__()
            
        def forward(self, last_hidden_state, attention_mask):
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            )
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask
            return mean_embeddings

    class MulticlassHead(nn.Module):
        def __init__(self, input_size, num_classes):
            super(MulticlassHead, self).__init__()
            self.fc = nn.Linear(input_size, num_classes)
            
        def forward(self, x):
            x = self.fc(x)
            return x

    class CustomModel(nn.Module, PyTorchModelHubMixin):
        def __init__(self, target_sizes, task_type_map, weights_map, divisor_map):
            super(CustomModel, self).__init__()
            self.backbone = AutoModel.from_pretrained("microsoft/DeBERTa-v3-base")
            self.target_sizes = target_sizes.values()
            self.task_type_map = task_type_map
            self.weights_map = weights_map
            self.divisor_map = divisor_map
            
            self.heads = [
                MulticlassHead(self.backbone.config.hidden_size, sz) 
                for sz in self.target_sizes
            ]
            for i, head in enumerate(self.heads):
                self.add_module(f"head_{i}", head)
                
            self.pool = MeanPooling()

        def compute_results(self, preds, target, decimal=4):
            if target == "task_type":
                top2_indices = torch.topk(preds, k=2, dim=1).indices
                softmax_probs = torch.softmax(preds, dim=1)
                top2_probs = softmax_probs.gather(1, top2_indices)
                top2 = top2_indices.detach().cpu().tolist()
                top2_prob = top2_probs.detach().cpu().tolist()
                
                top2_strings = [
                    [self.task_type_map[str(idx)] for idx in sample] for sample in top2
                ]
                top2_prob_rounded = [
                    [round(value, 3) for value in sublist] for sublist in top2_prob
                ]
                
                counter = 0
                for sublist in top2_prob_rounded:
                    if sublist[1] < 0.1:
                        top2_strings[counter][1] = "NA"
                    counter += 1
                    
                task_type_1 = [sublist[0] for sublist in top2_strings]
                task_type_2 = [sublist[1] for sublist in top2_strings]
                task_type_prob = [sublist[0] for sublist in top2_prob_rounded]
                return (task_type_1, task_type_2, task_type_prob)
            else:
                preds = torch.softmax(preds, dim=1)
                weights = np.array(self.weights_map[target])
                weighted_sum = np.sum(np.array(preds.detach().cpu()) * weights, axis=1)
                scores = weighted_sum / self.divisor_map[target]
                scores = [round(value, decimal) for value in scores]
                if target == "number_of_few_shots":
                    scores = [x if x >= 0.05 else 0 for x in scores]
                return scores

        def process_logits(self, logits):
            result = {}
            target_names = [
                "task_type", "creativity_scope", "reasoning", "contextual_knowledge",
                "number_of_few_shots", "domain_knowledge", "no_label_reason", "constraint_ct"
            ]
            for i, target in enumerate(target_names):
                res = self.compute_results(logits[i], target=target)
                if target == "task_type":
                    result["task_type_1"] = res[0]
                    result["task_type_2"] = res[1]
                    result["task_type_prob"] = res[2]
                else:
                    result[target] = res

            result["prompt_complexity_score"] = [
                round(
                    0.35 * creativity + 0.25 * reasoning + 0.15 * constraint + 
                    0.15 * domain_knowledge + 0.05 * contextual_knowledge + 0.05 * few_shots, 
                    5,
                )
                for creativity, reasoning, constraint, domain_knowledge, contextual_knowledge, few_shots in zip(
                    result["creativity_scope"], result["reasoning"], result["constraint_ct"], 
                    result["domain_knowledge"], result["contextual_knowledge"], result["number_of_few_shots"]
                )
            ]
            return result

        def forward(self, input_ids, attention_mask):
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state
            mean_pooled_representation = self.pool(last_hidden_state, attention_mask)
            logits = [
                self.heads[k](mean_pooled_representation) for k in range(len(self.target_sizes))
            ]
            return self.process_logits(logits)


# ── CORA Mapping Logic ───────────────────────────────────────────────────────

NVIDIA_TO_CORA_TASK_MAP = {
    "Open QA": TaskType.FACTUAL,
    "Closed QA": TaskType.FACTUAL,
    "Summarization": TaskType.ANALYTICAL,
    "Extraction": TaskType.ANALYTICAL,
    "Classification": TaskType.ANALYTICAL,
    "Text Generation": TaskType.CREATIVE,
    "Rewrite": TaskType.CREATIVE,
    "Brainstorming": TaskType.CREATIVE,
    "Code Generation": TaskType.CODE,
    "Chatbot": TaskType.CONVERSATIONAL,
    "Other": TaskType.MULTI_STEP
}

class NeMoScorer:
    """NVIDIA NeMo Classifier disguised as a BaseScorer."""
    
    def __init__(self, config: CognitiveConfig | None = None):
        self.config = config or DEFAULT_CONFIG
        self._model = None
        self._tokenizer = None
        self._ready = False
        
        from .rule_scorer import RuleBasedScorer
        self._fallback = RuleBasedScorer(config=self.config)
        
        self._try_load_model()
        
    def _try_load_model(self):
        if not _ML_AVAILABLE:
            logger.warning("torch/transformers missing. Falling back to rules.")
            return
            
        try:
            model_id = "nvidia/prompt-task-and-complexity-classifier"
            logger.info(f"Loading {model_id}...")
            
            # Load the custom config as raw JSON (AutoConfig fails because
            # NVIDIA's config.json has no 'model_type' key, which transformers
            # 5.x requires).
            import json
            from huggingface_hub import hf_hub_download
            config_path = hf_hub_download(model_id, "config.json")
            with open(config_path, "r") as f:
                hf_config_dict = json.load(f)
            
            # Tokenizer comes from the DeBERTa backbone, not the custom head
            self._tokenizer = AutoTokenizer.from_pretrained("microsoft/DeBERTa-v3-base")
            
            self._model = CustomModel(
                target_sizes=hf_config_dict["target_sizes"],
                task_type_map=hf_config_dict["task_type_map"],
                weights_map=hf_config_dict["weights_map"],
                divisor_map=hf_config_dict["divisor_map"],
            ).from_pretrained(model_id)
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model.to(device)
            self._model.eval()
            
            self._device = device
            self._ready = True
            logger.info(f"NeMo model ready on {device}.")
        except Exception as e:
            logger.error(f"Failed to load NeMo model: {e}")
            self._ready = False

    def score(self, prompt: str) -> CognitiveProfile:
        if not self._ready:
            return self._fallback.score(prompt)
            
        try:
            inputs = self._tokenizer(
                [prompt], 
                return_tensors="pt", 
                add_special_tokens=True, 
                max_length=512, 
                padding=True, 
                truncation=True
            )
            input_ids = inputs["input_ids"].to(self._device)
            attention_mask = inputs["attention_mask"].to(self._device)
            
            with torch.no_grad():
                res = self._model(input_ids, attention_mask)
            
            # Extract scores (convert to 0-100)
            cora_task = NVIDIA_TO_CORA_TASK_MAP.get(res["task_type_1"][0], TaskType.CONVERSATIONAL)
            conf = res["task_type_prob"][0]
            
            creativity = int(round(res["creativity_scope"][0] * 100))
            reasoning = int(round(res["reasoning"][0] * 100))
            domain_know = int(round(res["domain_knowledge"][0] * 100))
            precision = int(round(res["constraint_ct"][0] * 100))
            struct = int(round(res["contextual_knowledge"][0] * 100))
            
            # Provide at least some code complexity if task is CODE or DEBUGGING
            code_complexity = reasoning if cora_task in (TaskType.CODE, TaskType.DEBUGGING) else 0

            # ── Sanity check & Blending: cross-reference with rule scorer on normalized text ──
            rule_profile = self._fallback.score(prompt)
            
            # 1. Technical Task Type Upgrading / Boosting
            # If the rule-based keyword/regex scorer detects a highly specific technical/coding/math task
            # but NeMo's general classifier missed it or gave it a lower-priority classification, upgrade it.
            TASK_PRIORITY = {
                TaskType.MATHEMATICAL: 4,
                TaskType.CODE: 3,
                TaskType.DEBUGGING: 3,
                TaskType.ANALYTICAL: 2,
                TaskType.MULTI_STEP: 1,
                TaskType.FACTUAL: 0,
                TaskType.CREATIVE: -1,
                TaskType.CONVERSATIONAL: -2,
            }
            rule_pri = TASK_PRIORITY.get(rule_profile.task_type, -2)
            nemo_pri = TASK_PRIORITY.get(cora_task, -2)
            if rule_pri > nemo_pri:
                cora_task = rule_profile.task_type
                conf = max(conf, rule_profile.confidence)

            # 2. Score Blending & Upgrading for Technical tasks
            # Ensure NeMo doesn't under-score technical tasks that are clearly identified by keyword rules
            if rule_profile.task_type in (TaskType.CODE, TaskType.DEBUGGING, TaskType.MATHEMATICAL, TaskType.ANALYTICAL):
                reasoning = max(reasoning, rule_profile.reasoning_depth)
                domain_know = max(domain_know, rule_profile.domain_specificity)
                creativity = max(creativity, rule_profile.creative_demand)
                precision = max(precision, rule_profile.precision_required)
                struct = max(struct, rule_profile.structural_complexity)
                if cora_task in (TaskType.CODE, TaskType.DEBUGGING):
                    code_complexity = max(reasoning, rule_profile.code_complexity)

            # 3. Downward Sanity Check (fooling NeMo with fancy vocabulary)
            # If NeMo scores are very high but the rule scorer sees almost nothing, NeMo is likely over-scoring.
            nemo_dims = [reasoning, domain_know, creativity, precision, struct]
            rule_dims = [
                rule_profile.reasoning_depth,
                rule_profile.domain_specificity,
                rule_profile.creative_demand,
                rule_profile.precision_required,
                rule_profile.structural_complexity,
            ]
            nemo_avg = sum(nemo_dims) / max(len(nemo_dims), 1)
            rule_avg = sum(rule_dims) / max(len(rule_dims), 1)

            # If rule scorer sees <30% of what NeMo sees, blend scores downward
            if nemo_avg > 10 and rule_avg < nemo_avg * 0.30:
                blend = 0.5
                reasoning = int(reasoning * blend + rule_profile.reasoning_depth * (1 - blend))
                domain_know = int(domain_know * blend + rule_profile.domain_specificity * (1 - blend))
                creativity = int(creativity * blend + rule_profile.creative_demand * (1 - blend))
                precision = int(precision * blend + rule_profile.precision_required * (1 - blend))
                struct = int(struct * blend + rule_profile.structural_complexity * (1 - blend))
                if cora_task in (TaskType.CODE, TaskType.DEBUGGING):
                    code_complexity = int(code_complexity * blend + rule_profile.code_complexity * (1 - blend))
                
                if rule_profile.task_type == TaskType.CONVERSATIONAL:
                    cora_task = TaskType.CONVERSATIONAL
                    conf = max(conf * 0.6, rule_profile.confidence)
                logger.info(
                    f"NeMo downward sanity check triggered: nemo_avg={nemo_avg:.1f}, "
                    f"rule_avg={rule_avg:.1f} — blended scores downward."
                )

            # Merge rule-based signals into NeMo to prevent unwarranted verbosity penalties
            # and enrich frontend explainability cards
            nemo_signals = [f"nemo_base_score:{res['prompt_complexity_score'][0]:.3f}"]
            if rule_profile.signals:
                nemo_signals.extend(rule_profile.signals)

            return CognitiveProfile(
                reasoning_depth=min(100, reasoning),
                domain_specificity=min(100, domain_know),
                code_complexity=min(100, code_complexity),
                creative_demand=min(100, creativity),
                precision_required=min(100, precision),
                structural_complexity=min(100, struct),
                task_type=cora_task,
                confidence=float(conf),
                signals=nemo_signals,
                scorer_used="nemo"
            )
            
        except Exception as e:
            logger.error(f"NeMo inference failed: {e}")
            return self._fallback.score(prompt)
