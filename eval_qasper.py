import argparse
import json
import os
import re
import string
from collections import Counter
from statistics import mean

from htsir import RetrievalAugmentation, RetrievalAugmentationConfig


def get_answers_and_evidence(paper_entry):
    answers_and_evidence = {}
    for qa_info in paper_entry["qas"]:
        question_id = qa_info["question_id"]
        references = []
        for annotation_info in qa_info["answers"]:
            answer_info = annotation_info["answer"]
            if answer_info["unanswerable"]:
                references.append(
                    {"answer": "Unanswerable", "evidence": [], "type": "none"}
                )
            else:
                if answer_info["extractive_spans"]:
                    answer = ", ".join(answer_info["extractive_spans"])
                    answer_type = "extractive"
                elif answer_info["free_form_answer"]:
                    answer = answer_info["free_form_answer"]
                    answer_type = "abstractive"
                elif answer_info["yes_no"]:
                    answer = "Yes"
                    answer_type = "boolean"
                elif answer_info["yes_no"] is not None:
                    answer = "No"
                    answer_type = "boolean"
                else:
                    raise RuntimeError(
                        f"Annotation {answer_info['annotation_id']} does not contain an answer"
                    )
                evidence = [
                    text
                    for text in answer_info["evidence"]
                    if "FLOAT SELECTED" not in text
                ]
                references.append(
                    {"answer": answer, "evidence": evidence, "type": answer_type}
                )
        answers_and_evidence[question_id] = references
    return answers_and_evidence


def normalize_answer(text):
    """Lower text and remove punctuation, articles, and extra whitespace (SQuAD v1.1)."""

    def remove_articles(s):
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def white_space_fix(s):
        return " ".join(s.split())

    def remove_punc(s):
        exclude = set(string.punctuation)
        return "".join(ch for ch in s if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(str(text).lower())))


def token_metrics(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0, 0.0, 0.0
    prec = 1.0 * num_same / len(prediction_tokens)
    rec = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * prec * rec) / (prec + rec)
    return f1, prec, rec


def evaluate(gold, predicted):
    max_answer_f1s = []
    max_answer_precisions = []
    max_answer_recalls = []
    max_answer_f1s_by_type = {
        "extractive": [],
        "abstractive": [],
        "boolean": [],
        "none": [],
    }
    evidences_f1_scores = {}
    num_missing_predictions = 0

    for question_id, references in gold.items():
        if question_id not in predicted:
            num_missing_predictions += 1
            max_answer_f1s.append(0.0)
            max_answer_precisions.append(0.0)
            max_answer_recalls.append(0.0)
            continue

        pred_answer = predicted[question_id]["answer"]
        scored = [
            (*token_metrics(pred_answer, ref["answer"]), ref["type"])
            for ref in references
        ]
        max_f1, _, _, best_type = max(scored, key=lambda x: x[0])
        max_prec = max(x[1] for x in scored)
        max_rec = max(x[2] for x in scored)

        max_answer_f1s.append(max_f1)
        max_answer_f1s_by_type[best_type].append(max_f1)
        max_answer_precisions.append(max_prec)
        max_answer_recalls.append(max_rec)

        best_ev_f1 = 0.0
        for ref in references:
            for evidence in ref["evidence"]:
                evidence_str = (
                    " ".join(evidence) if isinstance(evidence, list) else evidence
                )
                ev_f1, _, _ = token_metrics(pred_answer, evidence_str)
                if ev_f1 > best_ev_f1:
                    best_ev_f1 = ev_f1
        evidences_f1_scores[question_id] = best_ev_f1

    safe_mean = lambda vals: sum(vals) / len(vals) if vals else 0.0
    return {
        "Answer F1": safe_mean(max_answer_f1s),
        "Answer Precision": safe_mean(max_answer_precisions),
        "Answer Recall": safe_mean(max_answer_recalls),
        "Answer F1 by type": {
            k: safe_mean(v) for k, v in max_answer_f1s_by_type.items()
        },
        "Evidence F1": safe_mean(list(evidences_f1_scores.values())),
        "Missing predictions": num_missing_predictions,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate HTSIR on the QASPER dataset."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to QASPER JSON file (e.g., data/qasper-dev-v0.3.json).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs/qasper_results.json",
        help="Path to save evaluation results JSON.",
    )
    parser.add_argument("--start_idx", type=int, default=0, help="Start paper index (0-based).")
    parser.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="End paper index (exclusive). Defaults to all papers.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=100,
        help="Leaf chunk token length l (default: 100).",
    )
    parser.add_argument(
        "--chunk_group_size",
        type=int,
        default=2,
        help="Number of chunks r grouped for intermediate summaries (default: 2).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of candidate nodes retrieved before reranking (default: 10).",
    )
    parser.add_argument(
        "--reranker_top_k",
        type=int,
        default=4,
        help="Number of top nodes kept after reranking (default: 4).",
    )
    parser.add_argument(
        "--no_reranker",
        action="store_true",
        help="Disable the reranking stage.",
    )
    args = parser.parse_args()

    with open(args.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    config = RetrievalAugmentationConfig(
        tb_max_tokens=args.chunk_size,
        use_reranker=not args.no_reranker,
        reranker_top_k=args.reranker_top_k,
    )
    ra = RetrievalAugmentation(config=config)

    items = list(data.items())
    end_idx = args.end_idx if args.end_idx is not None else len(items)
    selected_items = items[args.start_idx : end_idx]

    predicted_all = {}
    answer_f1s, answer_precisions, answer_recalls = [], [], []

    for offset, (paper_id, paper_entry) in enumerate(selected_items, start=args.start_idx):
        try:
            ra.add_documents(
                paper_entry,
                max_tokens=args.chunk_size,
                chunk_group_size=args.chunk_group_size,
            )
            predicted_answers = {}
            for qa_info in paper_entry["qas"]:
                question_id = qa_info["question_id"]
                question = qa_info["question"]
                context, answer = ra.answer_question(
                    question=question,
                    top_k=args.top_k,
                    reranker_top_k=args.reranker_top_k,
                )
                predicted_answers[question_id] = {
                    "answer": answer,
                    "evidence": context,
                }

            gold_answers = get_answers_and_evidence(paper_entry)
            evaluation_output = evaluate(gold_answers, predicted_answers)

            answer_f1s.append(evaluation_output["Answer F1"])
            answer_precisions.append(evaluation_output["Answer Precision"])
            answer_recalls.append(evaluation_output["Answer Recall"])

            predicted_all[paper_id] = {
                "results": evaluation_output,
                "predicted_answers_and_evidence": predicted_answers,
            }
            print(
                f"[{offset}] Paper {paper_id} | "
                f"Avg F1: {mean(answer_f1s):.4f}, "
                f"Avg Precision: {mean(answer_precisions):.4f}, "
                f"Avg Recall: {mean(answer_recalls):.4f}"
            )
        except Exception as e:
            print(f"Error processing paper {paper_id} (index {offset}): {e}")
            continue

    summary = {
        "Answer F1": mean(answer_f1s) if answer_f1s else 0.0,
        "Answer Precision": mean(answer_precisions) if answer_precisions else 0.0,
        "Answer Recall": mean(answer_recalls) if answer_recalls else 0.0,
        "Num Papers": len(answer_f1s),
    }
    print("Final results:", summary)

    out_dir = os.path.dirname(args.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(
            {"summary": summary, "papers": predicted_all},
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Saved results to {args.output_path}")


if __name__ == "__main__":
    main()
