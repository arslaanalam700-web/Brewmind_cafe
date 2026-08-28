"""Download and curate 25 landmark papers on RLHF & LLM Alignment from arXiv."""
import os
import sys
import time
import requests
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PAPERS_DIR

# Curated 25 Landmark RLHF & Alignment Papers
CORPUS_PAPERS = [
    {
        "id": "instructgpt",
        "title": "Training language models to follow instructions with human feedback",
        "authors": "Ouyang et al.",
        "year": "2022",
        "arxiv_id": "2203.02155",
        "category": "PPO / Core RLHF"
    },
    {
        "id": "dpo",
        "title": "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
        "authors": "Rafailov et al.",
        "year": "2023",
        "arxiv_id": "2305.18290",
        "category": "Direct Preference Optimization"
    },
    {
        "id": "constitutional_ai",
        "title": "Constitutional AI: Harmlessness from AI Feedback",
        "authors": "Bai et al. (Anthropic)",
        "year": "2022",
        "arxiv_id": "2212.08073",
        "category": "RLAIF / Self-Critique"
    },
    {
        "id": "kto",
        "title": "KTO: Model Alignment as Prospect Theoretic Optimization",
        "authors": "Ethayarajh et al.",
        "year": "2024",
        "arxiv_id": "2402.01306",
        "category": "Unpaired Alignment"
    },
    {
        "id": "ipo",
        "title": "A General Theoretical Paradigm to Understand Learning from Human Preferences",
        "authors": "Azar et al. (DeepMind)",
        "year": "2023",
        "arxiv_id": "2310.12036",
        "category": "Identity Preference Optimization"
    },
    {
        "id": "orpo",
        "title": "ORPO: Monolithic Preference Optimization without Reference Model",
        "authors": "Hong et al.",
        "year": "2024",
        "arxiv_id": "2403.07691",
        "category": "Reference-Free Alignment"
    },
    {
        "id": "simpo",
        "title": "SimPO: Simple Preference Optimization with a Reference-Free Reward",
        "authors": "Meng et al.",
        "year": "2024",
        "arxiv_id": "2405.14734",
        "category": "Reference-Free Alignment"
    },
    {
        "id": "rlhf_overoptimization",
        "title": "Scaling Laws for Reward Model Overoptimization in Direct Policy Optimization",
        "authors": "Gao et al. (OpenAI)",
        "year": "2022",
        "arxiv_id": "2210.10760",
        "category": "Reward Hacking"
    },
    {
        "id": "ppo_deep_rl",
        "title": "Proximal Policy Optimization Algorithms",
        "authors": "Schulman et al. (OpenAI)",
        "year": "2017",
        "arxiv_id": "1707.06347",
        "category": "RL Fundamentals"
    },
    {
        "id": "secrets_of_rlhf",
        "title": "The Secrets of RLHF in Large Language Models Part I: PPO",
        "authors": "Zheng et al.",
        "year": "2023",
        "arxiv_id": "2307.04964",
        "category": "PPO Implementation"
    },
    {
        "id": "rlhf_survey",
        "title": "A Comprehensive Survey on Pretrained Foundation Models: A History from BERT to ChatGPT",
        "authors": "Zhao et al.",
        "year": "2023",
        "arxiv_id": "2303.18223",
        "category": "Survey"
    },
    {
        "id": "iterative_dpo",
        "title": "Self-Rewarding Language Models",
        "authors": "Yuan et al. (Meta)",
        "year": "2024",
        "arxiv_id": "2401.10020",
        "category": "Iterative Alignment"
    },
    {
        "id": "length_bias_dpo",
        "title": "Disentangling Length from Quality in Direct Preference Optimization",
        "authors": "Park et al.",
        "year": "2024",
        "arxiv_id": "2403.19159",
        "category": "Alignment Biases"
    },
    {
        "id": "safe_rlhf",
        "title": "Safe RLHF: Constrained Reinforcement Learning from Human Feedback",
        "authors": "Dai et al. (PKU-Alignment)",
        "year": "2023",
        "arxiv_id": "2310.12773",
        "category": "Safety Alignment"
    },
    {
        "id": "reward_bench",
        "title": "RewardBench: Evaluating Reward Models for Language Modeling",
        "authors": "Lambert et al. (HuggingFace)",
        "year": "2024",
        "arxiv_id": "2403.13787",
        "category": "Evaluation"
    },
    {
        "id": "dpo_vs_ppo_empirical",
        "title": "Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study",
        "authors": "Xu et al.",
        "year": "2024",
        "arxiv_id": "2404.10719",
        "category": "Empirical Comparison"
    },
    {
        "id": "spin",
        "title": "Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models",
        "authors": "Chen et al. (UCLA)",
        "year": "2024",
        "arxiv_id": "2401.01335",
        "category": "Self-Play Alignment"
    },
    {
        "id": "cpo",
        "title": "Contrastive Preference Optimization: Pushing the Boundaries of LLM Translation",
        "authors": "Xu et al.",
        "year": "2024",
        "arxiv_id": "2401.08417",
        "category": "Preference Optimization"
    },
    {
        "id": "star_reasoning",
        "title": "STaR: Bootstrapping Reasoning With Reasoning",
        "authors": "Zelikman et al. (Stanford)",
        "year": "2022",
        "arxiv_id": "2203.14465",
        "category": "Reasoning & Alignment"
    },
    {
        "id": "rlhf_human_values",
        "title": "Fine-Tuning Language Models from Human Preferences",
        "authors": "Ziegler et al. (OpenAI)",
        "year": "2019",
        "arxiv_id": "1909.08593",
        "category": "Foundations"
    },
    {
        "id": "deep_rl_human_pref",
        "title": "Deep Reinforcement Learning from Human Preferences",
        "authors": "Christiano et al. (OpenAI/DeepMind)",
        "year": "2017",
        "arxiv_id": "1706.03741",
        "category": "Foundations"
    },
    {
        "id": "red_teaming_lm",
        "title": "Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned",
        "authors": "Ganguli et al. (Anthropic)",
        "year": "2022",
        "arxiv_id": "2209.07858",
        "category": "Safety & Robustness"
    },
    {
        "id": "alpaca_eval",
        "title": "AlpacaEval: An Automatic Evaluator of Instruction-following Models",
        "authors": "Li et al. (Stanford)",
        "year": "2023",
        "arxiv_id": "2305.14387",
        "category": "Evaluation"
    },
    {
        "id": "rlhf_vs_sft",
        "title": "LIMA: Less Is More for Alignment",
        "authors": "Zhou et al. (Meta)",
        "year": "2023",
        "arxiv_id": "2305.11206",
        "category": "SFT vs RLHF"
    },
    {
        "id": "online_dpo",
        "title": "Online Direct Preference Optimization",
        "authors": "Guo et al.",
        "year": "2024",
        "arxiv_id": "2402.04792",
        "category": "Online Preference Optimization"
    }
]

def download_paper(paper: dict, target_dir: Path) -> bool:
    """Download a single paper from arXiv PDF endpoint."""
    pdf_path = target_dir / f"{paper['id']}.pdf"
    if pdf_path.exists() and pdf_path.stat().st_size > 10000:
        print(f"  [OK] {paper['id']} already exists ({pdf_path.stat().st_size // 1024} KB).")
        return True

    arxiv_id = paper["arxiv_id"]
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    print(f"  [>] Downloading {paper['id']} ({paper['title'][:40]}...) from {pdf_url}")
    
    headers = {
        "User-Agent": "LiteratureReviewAssistantBot/1.0 (academic research tool)"
    }
    
    try:
        response = requests.get(pdf_url, headers=headers, timeout=30)
        if response.status_code == 200 and len(response.content) > 10000:
            with open(pdf_path, "wb") as f:
                f.write(response.content)
            print(f"      Downloaded successfully ({len(response.content) // 1024} KB).")
            return True
        else:
            print(f"      Warning: Failed with status {response.status_code}, length {len(response.content)}")
            return False
    except Exception as e:
        print(f"      Error downloading {paper['id']}: {e}")
        return False

def download_all_papers():
    """Download all curated papers with polite rate limiting."""
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Starting download of {len(CORPUS_PAPERS)} landmark RLHF/alignment papers into {PAPERS_DIR}...\n")
    
    success_count = 0
    for idx, paper in enumerate(CORPUS_PAPERS, 1):
        print(f"[{idx}/{len(CORPUS_PAPERS)}] {paper['id']} - {paper['title']}")
        if download_paper(paper, PAPERS_DIR):
            success_count += 1
        time.sleep(1.0)  # Polite sleep to respect arXiv API rate limits
        
    print(f"\nCompleted: {success_count}/{len(CORPUS_PAPERS)} papers downloaded and ready for ingestion.")

if __name__ == "__main__":
    download_all_papers()
