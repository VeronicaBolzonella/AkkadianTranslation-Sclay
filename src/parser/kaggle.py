import argparse
from src.config import TrainingMode


class KaggleTrainingParser:
    """
    This class contains the parsing logic to get the user inputs from the kaggle notebook

    parser = KaggleTrainingParser()
    config = parser.parse()
    print(config["learning_rate"])
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self._add_arguments()

    def _add_arguments(self):
        # Run parameters
        self.parser.add_argument(
            "--learning_rate",
            type=float,
            default=1e-4,
            help="Learning rate for the optimizer.",
        )
        self.parser.add_argument(
            "--max_epochs",
            type=int,
            default=200,
            help="Maximum number of training epochs.",
        )
        self.parser.add_argument(
            "--batch_size", type=int, default=4, help="Batch size per GPU."
        )
        self.parser.add_argument(
            "--precision",
            type=str,
            default="32",
            help="Training precision. Options: 32, 16-mixed, bf16-mixed.",
        )
        self.parser.add_argument(
            "--early_stopping_patience",
            type=int,
            default=30,
            help="Epochs to wait without val_loss improvement before stopping.",
        )
        self.parser.add_argument(
            "--arabic_init",
            action="store_true",
            help="Start the akkadian token with the same embeddings as arabic.",
        )
        self.parser.add_argument(
            "--max_length",
            type=int,
            default=512,
            help="Max token length of training and validation data",
        )
        # Model saving

        self.parser.add_argument(
            "--save_model_every",
            type=int,
            required=True,
            help="Save a checkpoint every N epochs.",
        )
        self.parser.add_argument(
            "--save_model_path",
            type=str,
            required=True,
            help="Directory to save model checkpoints.",
        )
        self.parser.add_argument(
            "--checkpoint_path",
            type=str,
            default=None,
            help="Path to a checkpoint to resume training from.",
        )
        self.parser.add_argument(
            "--base_model_path",
            type=str,
            default=None,
            help="Path to the base_model start training from.",
        )
        # LoRA parameters
        self.parser.add_argument(
            "--lora_r",
            type=int,
            default=32,
            help="LoRA rank. Higher = more trainable params. Recommended: 32-64.",
        )
        self.parser.add_argument(
            "--lora_alpha",
            type=int,
            default=16,
            help="LoRA scaling factor. Keep alpha/r ratio constant when changing r.",
        )
        # GPU settings
        self.parser.add_argument(
            "--gpu_num",
            type=int,
            required=True,
            help="Number of GPUs to use for training.",
        )
        # WandB
        self.parser.add_argument(
            "--project_name",
            type=str,
            default="mbart-iterative-improvements",
            help="WandB project name for experiment tracking.",
        )
        self.parser.add_argument(
            "--model_name",
            type=str,
            required=True,
            help="The run name.",
        )
        # Overfitting / debugging
        self.parser.add_argument(
            "--max_samples",
            type=int,
            default=None,
            help="Limit dataset to N samples. Useful for overfitting a single batch.",
        )
        self.parser.add_argument(
            "--dropout", type=float, default=0.1, help="mBART dropout, defaults to 0.1"
        )
        self.parser.add_argument(
            "--attention_dropout",
            type=float,
            default=0,
            help="mBART attention dropout, defaults to 0",
        )
        # Preprocessing
        self.parser.add_argument(
            "--separate_compounds",
            action="store_true",
            help="Split compound tokens during preprocessing.",
        )
        self.parser.add_argument(
            "--normalize_chars",
            action="store_true",
            help="Normalize the characters.",
        )
        self.parser.add_argument(
            "--diacritic_mode",
            action="store_true",
            help="You know what this does.",
        )
        self.parser.add_argument(
            "--with_hyphens",
            action="store_true",
            help="Preserve hyphens in transliteration tokens.",
        )
        self.parser.add_argument(
            "--named_determinatives",
            action="store_true",
            help="Keep named determinatives (e.g. m, f, d) as explicit tokens.",
        )
        self.parser.add_argument(
            "--name_swapping",
            action="store_true",
            help="Apply name swapping to the dataset. e.g (Marcelo eats cats -> John eats cats, Luca eats cats,...)",
        )
        # Datasets
        self.parser.add_argument(
            "--dataset_configs",
            type=str,
            nargs="+",
            help=(
                "Dataset configs in format: path:type:mode:weight. "
                "E.g. --dataset_configs "
                "dataset/internal:internal:supervised:2.0 "
                "dataset/internal:internal:self_supervised:1.0 "
                "dataset/external:external:supervised:1.0"
            ),
        )
        self.parser.add_argument(
            "--train_data_ratio",
            type=float,
            default=0.9,
            help="Train/validation split ratio. Default 0.9 = 90%% train, 10%% val.",
        )
        self.parser.add_argument(
            "--eval_every",
            type=int,
            default=10,
            help="How often to compute geo score.",
        )
        self.parser.add_argument(
            "--align",
            action="store_true",
            help="triggers alignemnt of the internal dataset",
        )
        self.parser.add_argument(
            "--model_type",
            type=str,
            required=True,
            help="Pick either `mbart`, `mbart_lora`, `mt5` or `akk_m300`",
        )
        self.parser.add_argument(
            "--checkpoint_monitor",
            required=True,
            help="Track either the `geo` or `val_loss` or `train_loss`",
        )

    def parse(self, args_list=None):
        # e.g., parser.parse(args_list=["--batch_size", "8"])
        args = self.parser.parse_args(args_list)

        # 2. Automatically turn ALL arguments into a dictionary
        config = vars(args)

        # 3. Feedback block (using the new config dict)
        header = " Kaggle Training Configuration "
        print(f"\n{header:=^50}")
        for key, value in config.items():
            print(f"{key: <25} -> {value}")
        print("=" * 50 + "\n")

        return config
