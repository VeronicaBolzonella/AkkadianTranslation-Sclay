from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.types import (
    OptimizerLRSchedulerConfig,
)
from src.config import TrainingMode
import torch
from sacrebleu.metrics.chrf import CHRF
import math
import lightning as L
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Adafactor,
    get_linear_schedule_with_warmup,
)
from torchmetrics.text import SacreBLEUScore, CHRFScore


L.seed_everything(42, workers=True)


class mBartFineTuner(L.LightningModule):
    def __init__(
        self,
        lr=1e-2,
        dropout=0.8,
        attention_dropout=0.8,
        model_type="mbart",
        src_lang="ak_XX",
        tgt_lang="en_XX",
        base_model_path=None,
        num_beams=5,
        length_penalty=1.2,
        no_repeat_ngram_size=3,
        use_mbr=False,  # add these
        mbr_num_sample_cands=2,
        mbr_num_beam_cands=4,
        early_stopping=True,
        repetition_penalty=1,
        eval_every=20,
    ):
        """
        It uses mBART as a baseline along with LoRA to finetune for a new given src_lang.

        :param lr : learning rate
        :param src_lang: language that is being translated
        :param base_model_path: If None then download the hugging face model. Else, use the local version
        """

        super().__init__()
        self.save_hyperparameters()

        self.val_preds = []
        self.val_refs = []
        self.validation_step_outputs = []
        # Evaluation
        self.val_bleu = SacreBLEUScore()
        self.val_chrf = CHRFScore(n_word_order=2)
        self._chrf = CHRF(word_order=2)
        self.base_model_path = base_model_path
        self.model_type = model_type
        self.eval_every = eval_every

        if base_model_path is not None:
            path = base_model_path
        else:
            if self.model_type == "mbart_lora":
                path = "MarkSpaghetti/mbart-lora-r32-alpha-16"
            elif self.model_type == "mbart-expanded":
                path = "MarkSpaghetti/mbart-tokens-lora-r32-alpha-16"
            elif self.model_type == "mbart":
                path = "veronicabolzonella/full_mbart"

            elif self.model_type == "mt5":
                path = "google/mt5-base"
            elif self.model_type == "akk_300m":
                path = "Thalesian/AKK_300m"
            else:
                raise ValueError("Choose either `mbart` or `mt5` or akk_300m ")

        # Load tokenizer + base model

        self.tokenizer = AutoTokenizer.from_pretrained(path)
        if self.model_type == "mbart-expanded":
            self.eng_tokenizer = AutoTokenizer.from_pretrained(
                path,
                subfolder="english_tokenizer",
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                path,
                subfolder="akkadian_tokenizer",
            )
        else:
            self.eng_tokenizer = None

        config = AutoConfig.from_pretrained(path)

        if "mt5" in path.lower() or "akk_300m" in path.lower():
            config.dropout_rate = dropout
        else:
            # mBART
            config.dropout = dropout
            config.attention_dropout = attention_dropout

        self.model = AutoModelForSeq2SeqLM.from_pretrained(path, config=config)

        if "mbart" in model_type.lower():
            for layer in self.model.model.encoder.layers:
                layer.dropout = dropout

            for layer in self.model.model.decoder.layers:
                layer.dropout = dropout
        # Set forced BOS for the target language
        # Set the source and target language
        # mBART-specific: language tokens
        self.tokenizer.src_lang = src_lang

        self.supervised_tgt_lang = "en_XX"
        self.self_supervised_tgt_lang = "ak_XX"

    def _bleu_loss_fn(self, logits, labels):
        """
        Differentiable BLEU proxy: token-level expected log-prob.

        Args:
            logits: (B, T, V)
            labels: (B, T)  — -100 for padding
        """
        safe_labels = labels.clone()
        safe_labels[safe_labels == -100] = self.tokenizer.pad_token_id

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # (B, T, V)

        losses = []
        for i in range(logits.shape[0]):
            lp = log_probs[i]  # (T, V)
            ref = safe_labels[i]  # (T,)

            non_pad = ref != self.tokenizer.pad_token_id
            lp = lp[non_pad]  # (T', V)
            ref = ref[non_pad]  # (T',)
            T = lp.shape[0]

            if T < 4:
                continue

            # log prob of each token
            token_lp = lp[torch.arange(T, device=lp.device), ref]  # (T',)

            # pos weights with geom decay (bleu favors early precision?)
            weights = torch.pow(
                torch.tensor(0.95, device=logits.device),
                torch.arange(T, device=logits.device, dtype=torch.float),
            )
            weights = weights / weights.sum()

            # weighted average
            bleu_proxy = (token_lp * weights).sum()
            losses.append(-bleu_proxy)  # neg for descent

        if not losses:
            return logits.sum() * 0.0

        return torch.stack(losses).mean()

    def _get_forced_bos_token_id(self, training_mode: str):
        """Returns the correct forced_bos_token_id based on the training mode."""
        if training_mode == TrainingMode.SUPERVISED.value:
            tgt_lang = self.supervised_tgt_lang
        else:
            tgt_lang = self.self_supervised_tgt_lang
        return self.tokenizer.convert_tokens_to_ids(tgt_lang)

    def _generate(self, encoded, training_mode: str = TrainingMode.SUPERVISED.value):
        if self.hparams["use_mbr"]:
            return self._generate_mbr(encoded, training_mode=training_mode)
        forced_bos_token_id = self._get_forced_bos_token_id(training_mode)

        generate_kwargs = dict(
            **encoded,
            max_length=512,
            num_beams=self.hparams["num_beams"],
            length_penalty=self.hparams["length_penalty"],
            no_repeat_ngram_size=self.hparams["no_repeat_ngram_size"],
            early_stopping=self.hparams["early_stopping"],
            use_cache=True,
        )
        if self.model_type in ("mbart", "mbart_lora", "mbart-expanded"):
            generate_kwargs["forced_bos_token_id"] = forced_bos_token_id

        generated_tokens = self.model.generate(**generate_kwargs)
        tokenizer = (
            self.eng_tokenizer
            if self.model_type == "mbart-expanded"
            else self.tokenizer
        )
        assert tokenizer is not None
        return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    def _generate_mbr(
        self, encoded, training_mode: str = TrainingMode.SUPERVISED.value
    ):
        B = encoded["input_ids"].shape[0]

        tgt_lang_id = (
            self._get_forced_bos_token_id(training_mode)
            if self.model_type in ("mbart", "mbart_lora", "mbart-expanded")
            else None
        )
        n_beams = self.hparams["mbr_num_beam_cands"]
        n_samples = self.hparams["mbr_num_sample_cands"]

        # beam candidates
        beam_kwargs = dict(
            **encoded,
            max_length=512,
            num_beams=max(self.hparams["num_beams"], n_beams),
            num_return_sequences=n_beams,
            length_penalty=self.hparams["length_penalty"],
            repetition_penalty=self.hparams["repetition_penalty"],
            early_stopping=self.hparams["early_stopping"],
            use_cache=True,
        )
        if tgt_lang_id is not None:
            beam_kwargs["forced_bos_token_id"] = tgt_lang_id

        if self.model_type == "mbart-expanded":
            tokenizer = self.eng_tokenizer
        else:
            tokenizer = self.tokenizer

        assert tokenizer is not None

        beam_out = self.model.generate(**beam_kwargs)
        beam_texts = tokenizer.batch_decode(beam_out, skip_special_tokens=True)

        pools = []
        for i in range(B):
            pools.append(beam_texts[i * n_beams : (i + 1) * n_beams])

        if n_samples > 0:
            sample_kwargs = dict(
                **encoded,
                max_length=512,
                do_sample=True,
                num_beams=1,
                top_p=0.9,
                temperature=0.8,
                repetition_penalty=self.hparams["repetition_penalty"],
                num_return_sequences=n_samples,
                use_cache=True,
            )
            if tgt_lang_id is not None:
                sample_kwargs["forced_bos_token_id"] = tgt_lang_id

            sample_out = self.model.generate(**sample_kwargs)
            assert tokenizer is not None
            sample_texts = tokenizer.batch_decode(sample_out, skip_special_tokens=True)

            for i in range(B):
                pools[i].extend(sample_texts[i * n_samples : (i + 1) * n_samples])

        return [self._mbr_pick(candidates) for candidates in pools]

    def _mbr_pick(self, candidates):
        candidates = list(dict.fromkeys(c.strip() for c in candidates if c.strip()))

        if len(candidates) == 0:
            return ""
        if len(candidates) == 1:
            return candidates[0]

        n = len(candidates)
        scores = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                a = candidates[i]
                b = candidates[j]
                if not a or not b:
                    s = 0.0
                else:
                    s = self._chrf.sentence_score(a, [b]).score
                scores[i][j] = s
                scores[j][i] = s

        best, best_score = candidates[0], -1
        for i in range(n):
            avg = sum(scores[i]) / (n - 1)
            if avg > best_score:
                best_score = avg
                best = candidates[i]

        return best

    def forward(self, batch):
        """
        It is the forward function, `batch` is expected to be tokenized before hand
        Args:
            batch (input): It is expected to be tokenized before hand with `tokenize()`.
        """
        encoded_input = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch.get("attention_mask"),
        }
        translation = self._generate(encoded_input)
        return translation

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        optimizer = Adafactor(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.hparams["lr"],  # 1e-4 default
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            clip_threshold=1.0,
            weight_decay=1e-2,
        )

        total_steps = self.trainer.estimated_stepping_batches
        if total_steps == float("inf") or total_steps <= 0:
            total_steps = 1000
        total_steps = int(total_steps)

        warmup_steps = int(total_steps * 0.10)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def training_step(self, batch, batch_idx):
        training_mode = batch.pop("training_mode")[0]  # same mode for whole batch
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        training_mode = batch.pop("training_mode")[0]
        outputs = self.model(**batch)
        loss = outputs.loss

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)

        if batch_idx < 15 and self.global_rank == 0 and loss.item() < 5.0:
            total_epochs = self.trainer.max_epochs
            assert isinstance(total_epochs, int)

            checkpoints = set(
                range(
                    self.hparams["eval_every"],
                    total_epochs + 1,
                    self.hparams["eval_every"],
                )
            )
            if total_epochs not in checkpoints:
                checkpoints.add(total_epochs)

            if self.current_epoch + 1 in sorted(checkpoints):
                # translations are not meaningful to log
                if training_mode == TrainingMode.SUPERVISED.value:
                    encoded_input = {
                        "input_ids": batch["input_ids"],
                        "attention_mask": batch.get("attention_mask"),
                    }
                    preds = self._generate(encoded_input, training_mode=training_mode)

                    tokenizer = (
                        self.eng_tokenizer
                        if self.model_type == "mbart-expanded"
                        else self.tokenizer
                    )
                    inputs = self.tokenizer.batch_decode(
                        batch["input_ids"], skip_special_tokens=True
                    )

                    labels = batch["labels"].clone()
                    assert tokenizer is not None
                    labels[labels == -100] = tokenizer.pad_token_id
                    targets = tokenizer.batch_decode(labels, skip_special_tokens=True)

                    targets_wrapped = [[t] for t in targets]
                    self.val_bleu.update(preds, targets_wrapped)
                    self.val_chrf.update(preds, targets_wrapped)

                    for i, r, p in zip(inputs, targets, preds):
                        self.validation_step_outputs.append([i, r, p])

        return loss

    def on_validation_epoch_end(self):
        """This funciton logs the metrics BLEU, chrF++ and the
        scoring metric used by Kaggle to WANDB. It also saves
        a table showing the translations performed by the model.
        It sends its translations in form of a table to WANDB

        """

        bleu_score = self.val_bleu.compute().detach().item()
        chrf_res = self.val_chrf.compute()

        # If chrf_res is a tuple (Score, Total), we take the first element (the score)
        if isinstance(chrf_res, tuple):
            chrf_score = chrf_res[0].detach().item()
        else:
            chrf_score = chrf_res.detach().item()

        # Torchmetrics usually returns 0-1.
        # The official metric expects 0-100.
        if bleu_score <= 1.0:
            bleu_score *= 100.0
        if chrf_score <= 1.0:
            chrf_score *= 100.0

        geo_mean_score = math.sqrt(bleu_score * chrf_score + 1e-9)
        self.log("val_bleu", bleu_score, sync_dist=False)
        self.log("val_chrf", chrf_score, sync_dist=False)
        self.log("val_geo_score", geo_mean_score, prog_bar=True, sync_dist=False)

        if (
            isinstance(self.logger, WandbLogger)
            and self.validation_step_outputs
            and self.global_rank == 0
        ):
            # Log the table
            self.logger.log_table(
                key="validation_translations",
                columns=[
                    "transliteration",
                    "translation",
                    "predicted_translation",
                ],
                data=self.validation_step_outputs,
            )

        # Clear the list for the next epoch
        self.validation_step_outputs.clear()
        self.val_bleu.reset()
        self.val_chrf.reset()
