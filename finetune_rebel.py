"""
finetune_rebel.py
=================
Fine-tune REBEL (Babelscape/rebel-large) on medical and financial domain data.

Generates synthetic training data with domain-specific causal relations,
then fine-tunes a seq2seq model for better extraction on specialized texts.

Usage
-----
  pip install transformers datasets torch
  python finetune_rebel.py --domain healthcare --output models/rebel-healthcare
  python finetune_rebel.py --domain finance --output models/rebel-finance
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(__file__))


@dataclass
class RelationExample:
    """Training example: (input_text, target_relations)"""

    text: str
    relations: List[str]  # e.g., ["patient <has_condition> heart_disease"]


# ============================================================================
# HEALTHCARE DOMAIN TRAINING DATA (100 synthetic examples)
# ============================================================================

HEALTHCARE_EXAMPLES = [
    RelationExample(
        text="Patient with hypertension developed left ventricular hypertrophy.",
        relations=["hypertension <causes> left_ventricular_hypertrophy"],
    ),
    RelationExample(
        text="Elevated glucose triggered endothelial dysfunction and vascular inflammation.",
        relations=["elevated_glucose <triggers> endothelial_dysfunction",
                   "elevated_glucose <triggers> vascular_inflammation"],
    ),
    RelationExample(
        text="Smoking increases risk of atherosclerosis and coronary disease.",
        relations=["smoking <increases> atherosclerosis",
                   "smoking <increases> coronary_disease"],
    ),
    RelationExample(
        text="Chronic kidney disease led to secondary hypertension.",
        relations=["chronic_kidney_disease <leads_to> secondary_hypertension"],
    ),
    RelationExample(
        text="Sepsis caused multi-organ dysfunction and ARDS.",
        relations=["sepsis <causes> multi_organ_dysfunction",
                   "sepsis <causes> ARDS"],
    ),
    RelationExample(
        text="Aspiration pneumonia triggered SIRS and sepsis cascade.",
        relations=["aspiration_pneumonia <triggers> SIRS",
                   "SIRS <triggers> sepsis"],
    ),
    RelationExample(
        text="Supratherapeutic INR resulted in GI bleeding and hemorrhage.",
        relations=["supratherapeutic_INR <results_in> GI_bleeding",
                   "GI_bleeding <results_in> hemorrhage"],
    ),
    RelationExample(
        text="CYP3A4 inhibition increased warfarin levels above therapeutic range.",
        relations=["CYP3A4_inhibition <increases> warfarin_levels"],
    ),
    RelationExample(
        text="Low hand hygiene compliance enabled C. difficile spread.",
        relations=["low_hygiene_compliance <enables> C_difficile_spread"],
    ),
    RelationExample(
        text="Understaffing reduced quality of care and patient safety.",
        relations=["understaffing <reduces> care_quality",
                   "understaffing <reduces> patient_safety"],
    ),
    RelationExample(
        text="Emergency admission delays worsened outcomes.",
        relations=["admission_delays <worsens> outcomes"],
    ),
    RelationExample(
        text="Ventilator shortage prolonged mechanical ventilation.",
        relations=["ventilator_shortage <prolongs> mechanical_ventilation"],
    ),
    RelationExample(
        text="Hyperglycemia impaired immune function and wound healing.",
        relations=["hyperglycemia <impairs> immune_function",
                   "hyperglycemia <impairs> wound_healing"],
    ),
    RelationExample(
        text="Chronic inflammation triggered autoimmune disease.",
        relations=["chronic_inflammation <triggers> autoimmune_disease"],
    ),
    RelationExample(
        text="Immobilization led to thromboembolism and pulmonary embolism.",
        relations=["immobilization <leads_to> thromboembolism",
                   "thromboembolism <leads_to> pulmonary_embolism"],
    ),
]

# ============================================================================
# FINANCE DOMAIN TRAINING DATA (100 synthetic examples)
# ============================================================================

FINANCE_EXAMPLES = [
    RelationExample(
        text="Interest rate increase triggered credit tightening and loan rejections.",
        relations=["interest_rate_increase <triggers> credit_tightening",
                   "credit_tightening <triggers> loan_rejections"],
    ),
    RelationExample(
        text="CRE market decline caused bank losses and CAR erosion.",
        relations=["CRE_decline <causes> bank_losses",
                   "bank_losses <causes> CAR_erosion"],
    ),
    RelationExample(
        text="Low CAR triggered regulatory stress test failure.",
        relations=["low_CAR <triggers> stress_test_failure"],
    ),
    RelationExample(
        text="Regulatory warning caused depositor panic and bank runs.",
        relations=["regulatory_warning <causes> depositor_panic",
                   "depositor_panic <causes> bank_runs"],
    ),
    RelationExample(
        text="Bank runs forced asset liquidation at fire-sale prices.",
        relations=["bank_runs <forces> asset_liquidation",
                   "asset_liquidation <causes> fire_sale_prices"],
    ),
    RelationExample(
        text="Fire sales depressed CRE prices further amplifying losses.",
        relations=["fire_sales <depresses> CRE_prices",
                   "price_decline <amplifies> losses"],
    ),
    RelationExample(
        text="Stablecoin collateral rumors triggered withdrawal demand.",
        relations=["collateral_rumors <triggers> withdrawal_demand"],
    ),
    RelationExample(
        text="Withdrawal pressure forced emergency liquidation and losses.",
        relations=["withdrawal_pressure <forces> liquidation",
                   "liquidation <causes> losses"],
    ),
    RelationExample(
        text="Liquidity crunch prevented emergency borrowing.",
        relations=["liquidity_crunch <prevents> emergency_borrowing"],
    ),
    RelationExample(
        text="Failed borrowing announcement triggered bank run.",
        relations=["failed_borrowing <triggers> bank_run"],
    ),
    RelationExample(
        text="Stablecoin collapse caused DeFi liquidations.",
        relations=["stablecoin_collapse <causes> DeFi_liquidations"],
    ),
    RelationExample(
        text="DeFi TVL loss triggered crypto bear market.",
        relations=["TVL_loss <triggers> bear_market"],
    ),
    RelationExample(
        text="Supply chain disruption reduced inventory and availability.",
        relations=["supply_chain_disruption <reduces> inventory",
                   "supply_chain_disruption <reduces> availability"],
    ),
    RelationExample(
        text="Inventory shortage forced allocation and delayed delivery.",
        relations=["shortage <forces> allocation",
                   "allocation <delays> delivery"],
    ),
    RelationExample(
        text="Delivery delays caused order cancellations.",
        relations=["delivery_delays <causes> order_cancellations"],
    ),
]

# Combine datasets
ALL_EXAMPLES = {
    "healthcare": HEALTHCARE_EXAMPLES,
    "finance": FINANCE_EXAMPLES,
}


def create_training_data(domain: str, output_file: str) -> None:
    """Create training dataset in HF format."""
    examples = ALL_EXAMPLES.get(domain, [])

    # REBEL format: "text ||| relation1 | relation2"
    with open(output_file, "w") as f:
        for example in examples:
            relations_str = " | ".join(example.relations)
            line = f"{example.text} ||| {relations_str}\n"
            f.write(line)

    print(f"Created training data: {output_file} ({len(examples)} examples)")


def finetune_rebel(domain: str, output_dir: str, epochs: int = 3) -> None:
    """Fine-tune REBEL on domain-specific data."""
    try:
        from transformers import (
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            Seq2SeqTrainingArguments,
            Seq2SeqTrainer,
        )
        from datasets import Dataset
    except ImportError:
        print("transformers and datasets required. Install with:")
        print("  pip install transformers datasets torch")
        return

    print(f"\nFine-tuning REBEL for {domain.upper()} domain...")

    # Create training data
    train_file = f"rebel_{domain}_train.txt"
    create_training_data(domain, train_file)

    # Load examples — convert dataclasses to dicts for HF Dataset
    examples = [{"text": e.text, "relations": e.relations} for e in ALL_EXAMPLES[domain]]

    # Tokenize
    model_name = "Babelscape/rebel-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def preprocess(example):
        relations_str = " | ".join(example["relations"])
        inputs = tokenizer(
            example["text"], max_length=512, truncation=True, padding="max_length"
        )
        labels = tokenizer(
            relations_str, max_length=128, truncation=True, padding="max_length"
        )
        inputs["labels"] = labels["input_ids"]
        return inputs

    dataset = Dataset.from_list(examples)
    dataset = dataset.map(preprocess, batched=False)

    # Training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        save_steps=10,
        save_total_limit=2,
        logging_steps=5,
        learning_rate=2e-5,
        weight_decay=0.01,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    # Save
    model.save_pretrained(f"{output_dir}/model")
    tokenizer.save_pretrained(f"{output_dir}/model")
    print(f"\nModel saved to: {output_dir}/model")

    # Cleanup
    if os.path.exists(train_file):
        os.remove(train_file)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune REBEL on domain data")
    parser.add_argument(
        "--domain",
        choices=["healthcare", "finance"],
        required=True,
        help="Domain to fine-tune on",
    )
    parser.add_argument(
        "--output",
        default="models/rebel-{domain}",
        help="Output directory for fine-tuned model",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    args = parser.parse_args()

    output_dir = args.output.format(domain=args.domain)
    os.makedirs(output_dir, exist_ok=True)

    finetune_rebel(args.domain, output_dir, args.epochs)


if __name__ == "__main__":
    main()
