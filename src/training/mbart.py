from src.config import DatasetType, TrainingMode
from src.training.trainer import KaggleTrainer
from src.parser.kaggle import KaggleTrainingParser

"""
This is the file you are suppose to execute in your kaggle notebook to train your model.
"""


def main():
    config = KaggleTrainingParser().parse()
    kt = KaggleTrainer(config=config, checkpoint_dir=config["save_model_path"])

    for dataset_config in config["dataset_configs"]:
        path, dataset_type, mode, weight = dataset_config.split(":")
        kt.load_and_prepare_dataset(
            max_samples=config["max_samples"],
            model_type=config["model_type"],
            data_folder_path=path,
            dataset_type=DatasetType(dataset_type),
            training_mode=TrainingMode(mode),
            weight=float(weight),
            name_swapping=config["name_swapping"],
            train_data_ratio=config["train_data_ratio"],
            # preprocessing params:
            normalize_chars=config["normalize_chars"],
            separate_compounds=config["separate_compounds"],
            with_hyphens=config["with_hyphens"],
            named_determinatives=config["named_determinatives"],
            diacritic_mode=config["diacritic_mode"],
        )

    kt.create_dataloaders(batch_size=config["batch_size"])
    kt.fit()


if __name__ == "__main__":
    main()
