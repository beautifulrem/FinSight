# Query Intelligence Large-Scale Evaluation

## Eval Set
- Total queries: 10000
- Buckets: {"ood_must_pass": 21, "finance_must_pass_open_evaluation": 2, "finance_must_pass_unknown_entity": 2, "finance_public": 3983, "finance_retrieval": 1397, "ood_public": 2743, "ood_dialogue_public": 1000, "finance_generated_why": 713, "finance_generated_fact": 108, "finance_generated_macro": 13, "finance_generated_compare": 16, "finance_generated_clarification": 2}
- Domains: {"ood": 3764, "finance": 6236}
- Languages: {"en": 4135, "zh": 5865}

## Metrics
- finance_domain_recall: 0.975
- ood_rejection_accuracy: 0.9729
- must_pass_ood_accuracy: 1.0
- product_type_accuracy: 1.0
- question_style_accuracy: 1.0
- intent_micro_precision: 0.9046
- intent_micro_recall: 0.9798
- intent_micro_f1: 0.9407
- topic_micro_precision: 0.8549
- topic_micro_recall: 0.9977
- topic_micro_f1: 0.9208
- source_plan_hit_at_5: 1.0
- source_plan_recall_at_5: 0.9957
- clarification_recall: 0.75
- retrieval_eval_queries: 1397
- ood_retrieval_queries: 500
- recall_at_10: 0.9664
- mrr_at_10: 0.889
- ndcg_at_10: 0.8872
- source_plan_support: 0.8382
- ood_retrieval_abstention: 0.986

## Threshold Check
- finance_domain_recall: value=0.975, threshold=0.97, pass=True
- ood_rejection_accuracy: value=0.9729, threshold=0.95, pass=True
- must_pass_ood_accuracy: value=1.0, threshold=1.0, pass=True
- product_type_accuracy: value=1.0, threshold=0.9, pass=True
- question_style_accuracy: value=1.0, threshold=0.84, pass=True
- intent_micro_f1: value=0.9407, threshold=0.78, pass=True
- topic_micro_f1: value=0.9208, threshold=0.82, pass=True
- source_plan_hit_at_5: value=1.0, threshold=0.85, pass=True
- source_plan_recall_at_5: value=0.9957, threshold=0.7, pass=True
- clarification_recall: value=0.75, threshold=0.9, pass=False
- recall_at_10: value=0.9664, threshold=0.75, pass=True
- mrr_at_10: value=0.889, threshold=0.45, pass=True
- ndcg_at_10: value=0.8872, threshold=0.5, pass=True
- source_plan_support: value=0.8382, threshold=0.8, pass=True
- ood_retrieval_abstention: value=0.986, threshold=0.95, pass=True

## Benchmark Proxies
- BEIR / FiQA-2018: NDCG@10 >= 0.3 (https://huggingface.co/datasets/BeIR/fiqa)
- T2Ranking: MRR@10 >= 0.45 (https://huggingface.co/datasets/THUIR/T2Ranking)

## Failed Metrics
- clarification_recall

## Failure Examples
- None
