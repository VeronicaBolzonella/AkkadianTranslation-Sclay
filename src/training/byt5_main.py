from training.byt5_training import ByT5Trainer
import argparse
import torch
import os


def main():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(
                f"GPU {i} memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB"
            )
    else:
        print("WARNING: No GPU available, running on CPU!")

    parser = argparse.ArgumentParser(
        description="Train a ByT5 model for Akkadian to English translation."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the trained model and logs.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to a pretrained model, use only if not putting byt5 as base",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Define configurations
    config = {
        # Model paths
        "base_model_name": "google/byt5-small",
        "pretrained_model": args.model_path,
        # Task
        "task": "translation",  # tasks = "translation", "reconstruction", "span_corruption"   # CHANGE
        # For logging
        "model_name": "byt5-akkadian-sclay-base",  # CHANGE
        "run_name": "all-data-training",  # CHANGE
        # Data paths
        "internal_train_data_path": "dataset/training_input/internal/maas_aligned_v3.csv",
        "external_train_data_path": "dataset/training_input/external/akkademia_cleaned-v2.csv",
        "third_data_path": "dataset/training_input/internal/merged_human_translations.csv",
        "dictionary_path": "dataset/training_input/dictionary/dictionary_train.csv",
        "output_dir": args.output_dir,
        "morphemes_csv_path": "dataset/dictionaries/akkadian_morphemes.csv",
        "lexicon_csv_path": "dataset/dictionaries/OA_Lexicon_eBL.csv",
        "onomasticon_path": "dataset/training_input/dictionary/onomasticon-cleaned.csv",
        # Text processing Arguments
        "separate_compounds": False,  # should be false
        "with_hyphens": False,  # should be false
        "named_determinatives": True,
        "normalize_chars": True,  # FALSE: with diacritics
        "diacritic_mode": False,
        # Parameters
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "train_batch_size": 4,
        "eval_batch_size": 2,
        "num_epochs": 20,  # CHANGE
        "eval_every_n_epochs": 5,  # CHANGE
        "learning_rate": 1e-5,
        "early_stopping_patience": 5,
        "generation_max_length": 512,
        "max_length": 512,
        "fp16": True,
        "gradient_accumulation_steps": 1,
        "gradient_checkpointing": False,
        "dropout": 0,
        # Data Augmentation and Alignment
        "name_swapping": True,  # CHANGE
        "data_alignment": False,  # CHANGE
        "use_external_data": True,
        "use_dictionary": True,
        "use_onomasticon": True,
        "amount_of_external_data": 15000,
        "amount_of_internal_new_data": 10000,
        "pretraining_dictionary": False,
        "pretraining_external": False,
        # Weights for data
        "internal_weight": 10,
        "external_weight": 5,
        "dictionary_weight": 1,
        # Span Corruption Parameters
        "noise_density": 0.15,
        "mean_noise_span_length": 3,
        # LoRa Configuration
        "use_lora": False,
        "lora_r": 128,
        "lora_alpha": 128,
        "lora_dropout": 0.05,
        "lora_target_modules": ["q", "v", "k", "o", "wi", "wo"],
        "lora_bias": "none",
    }

    trainer = ByT5Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
