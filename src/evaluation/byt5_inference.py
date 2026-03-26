"""
# Pipeline for inference with ByT5


# This file should only be used in Kaggle, as imports are very much specific to
# Kaggle input directory where datasets.py and processing.py is uploaded
# manually. If one wishes to use this for local/cluster inference they have to
# adjust the imports. 

"""

from data_processing.datasets import AkkadianTranslationDatasetT5
from data_processing.processing import TextProcessor
import torch
import tqdm
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq
import pandas as pd
from torch.utils.data import DataLoader, Sampler
from sacrebleu.metrics import CHRF


class byT5Inference:
    """
    Inference class for byT5 Akkadian translation
    All configurations defined directly in Kaggle notebook
    """
    def __init__(self, config):
        """
        Args:
            config (dict): Configuration dictionary defined in Kaggle notebook
        """
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_path"],
                                                           local_files_only=True)
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(config["model_path"], 
                                                           attn_implementation="sdpa",
                                                            local_files_only=True).to(config["device"])
        except ValueError:
            print("SDPA not supported, falling back to default")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(config["model_path"], 
                                                               local_files_only=True).to(config["device"])
            
        self.processor = TextProcessor()
        if self.config["mbr"]:
            self.mbr_metric = CHRF(word_order=2)
            
    def prepare_dataloader(self):
        """
        Loads and preprocesses the test CSV, returns ready dataset
        """
        data = pd.read_csv(self.config["test_data_path"])
        
        data["transliteration"] = data["transliteration"].apply(
            lambda x: self.processor.preprocess_transliteration_text(
                   transliteration_text=x,
                   separate_compounds=self.config["separate_compounds"],
                   with_hyphens=self.config["with_hyphens"], 
                   named_determinatives=self.config["named_determinatives"]))
        
        test_dataset = AkkadianTranslationDatasetT5(data, self.tokenizer, max_length=self.config["max_length"], mode="inference")
        
        return test_dataset


    def mbr_select(self, candidates):
        """
        Picks best candidate via MBR. Builds full similarity matrix and returns
        candidate with highest average CHRF similarity to all other candidates.

        Args:
            candidates (list[str]): Candidate strings from generation

        Returns:
            str: Best candidate or empty string if there are no valid candidates 
        """
        # Strip whitespaces, drop empty candidates, remove duplicates
        candidates = list(dict.fromkeys(c.strip() for c in candidates if c.strip()))
        
        n = len(candidates)

        if n == 0:
            return ""
        if n==1:
            return candidates[0]
            
        similarity = [[0.0]*n for _ in range (n)]
        
        # Only compute upper triangle, then mirror
        for i in range(n):
            for j in range(i+1,n):
                a = candidates[i]
                b = candidates[j]

                if not a or not b:
                    s = 0.0
                else:
                    # Changed from corpus score to sentence score
                    s = self.mbr_metric.sentence_score(a, [b]).score

                similarity[i][j] = s
                similarity[j][i] = s
        
        best_ids = max(range(n), key=lambda i: sum(similarity[i]))

        return candidates[best_ids]

    def prune_candidates(self, candidates, scores, keep_k=6):
        """
        Keeps top-k candidates before expensive MBR. 

        Args:
            candidates (list[str]): Candidate translations
            scores (list[float]): Huggingface generation scores
            keep_k (int): Number of candidates to keep

        Returns:
            list[str]: Pruned candidate list
        """
        if len(candidates) <= keep_k:
            return candidates
        
        pairs = list(zip(candidates, scores))
        pairs.sort(key=lambda x: float(x[1]), reverse = True)
        pruned = [c for c,_ in pairs[:keep_k]]
        return pruned


    def _base_generate_kwargs(self, attention_mask):
        """
        Generates kwargs for regular (beam-based) inference
        """
        return dict(
            attention_mask=attention_mask,
            max_new_tokens=self.config["max_new_tokens"],
            use_cache=True,
            no_repeat_ngram_size=self.config["no_repeat_ngram_size"],
            early_stopping=self.config["early_stopping"],
            length_penalty=self.config["length_penalty"]
        )


    def _run_generate(self, input_ids, generate_kwargs):
        """
        Wrapper for running model.generate() with or without mixed precision
        """
        if self.config["mixed_precision"]:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = self.model.generate(input_ids, **generate_kwargs)
        else:
            outputs = self.model.generate(input_ids,**generate_kwargs)
        return outputs

    def _decode_generate_output(self, outputs):
        """
        Decodes output, handling both raw tensors and HF generation objects 
        in the case of pruning
        """
        if hasattr(outputs, "sequences"):
            sequences = outputs.sequences

        else:
            sequences = outputs

        return self.tokenizer.batch_decode(
            sequences,
            skip_special_tokens=True
        )
        
    
    def _get_sequence_scores(self, outputs, fallback_len):
        """
        Extract Huggingface scores if available
        """
        if hasattr(outputs, "sequences_scores") and outputs.sequences_scores is not None:
            return outputs.sequences_scores.detach().cpu().tolist()
                            
        return [0.0]*fallback_len # Fallback: assign all candidates same dummy score



    
    def _generate_sample_candidates(self, input_ids, attention_mask):
        """
        Generates  candidates via sampling for MBR

        Returns:
            tuple: (decoded, scores, n_samples)
        """
        n_samples = self.config.get("mbr_samples", 10)

        kwargs = self._base_generate_kwargs(attention_mask)
        kwargs.update(dict(
            do_sample=True,
            num_return_sequences=n_samples,
            temperature=self.config.get("mbr_temperature", 0.7),
            num_beams=1,  
            return_dict_in_generate=True, 
            output_scores=True, # For pruning
        ))

        outputs = self._run_generate(input_ids, kwargs)
        decoded = self._decode_generate_output(outputs)
        scores = self._get_sequence_scores(outputs, len(decoded)) 

        return decoded, scores, n_samples


    def _generate_beam_candidates(self, input_ids, attention_mask):
        """
        Generates beam candidates for MBR

        Returns:
            tuple: (decoded, scores, n_samples)
        """
        n_beams = self.config.get("mbr_beams", self.config.get("num_beams", 4))
        n_beams = max(1, n_beams)
        
        kwargs = self._base_generate_kwargs(attention_mask)
        kwargs.update(dict(
            do_sample=False,
            num_beams=n_beams,
            num_return_sequences=n_beams,
            return_dict_in_generate=True, 
            output_scores=True, # For pruning
        ))
        
        outputs = self._run_generate(input_ids, kwargs)
        decoded = self._decode_generate_output(outputs)
        scores = self._get_sequence_scores(outputs, len(decoded))     

        return decoded, scores, n_beams

    def _build_mbr_candidate_groups(self, input_ids, attention_mask):
        """
        Returns MBR candidate pools for three modes:
            - "sample": stochastic sampling only
            - "beam": beam search only
            - "hybrid": beam and sample candidates combined

        Returns:
            tuple: (candidates_list, scores_list), one per sample in the batch
        """
        mbr_mode = self.config.get("mbr_mode", "sample")
        batch_size = input_ids.size(0)
        

        if mbr_mode == "sample":
            decoded_all, all_scores, group_size = self._generate_sample_candidates(
                input_ids, attention_mask
            )
            candidates_list = []
            scores_list = []

        elif mbr_mode == "beam":  
            decoded_all, all_scores, group_size = self._generate_beam_candidates(
                input_ids, attention_mask
            )
            candidates_list = []
            scores_list = []
        
        elif mbr_mode == "hybrid":
            sample_decoded, sample_scores, n_samples = self._generate_sample_candidates(
                input_ids, attention_mask
            )

            beam_decoded, beam_scores, n_beams = self._generate_beam_candidates(
                input_ids, attention_mask
            )

            candidates_list = []
            scores_list = []

            for i in range(batch_size):
                sample_start = i * n_samples
                sample_end = (i+1) * n_samples

                beam_start = i * n_beams
                beam_end = (i+1) * n_beams

                candidates = beam_decoded[beam_start:beam_end] + sample_decoded[sample_start:sample_end]
                scores = beam_scores[beam_start:beam_end] + sample_scores[sample_start:sample_end]

                candidates_list.append(candidates)
                scores_list.append(scores)
                
                        
            return candidates_list, scores_list

        
        else:
            raise ValueError(f"Unsupported mbr_mode: {mbr_mode}")

        # for sample and beam modes
        candidates_list = [decoded_all[i * group_size:(i + 1) * group_size] for i in range(batch_size)]
        scores_list = [all_scores[i * group_size:(i + 1) * group_size] for i in range(batch_size)]
        
        return candidates_list, scores_list

    def translate(self, test_dataset):
        """
        Runs inference on test set and returns predictions.
        Supports regular beam search, MBR (sample, beam, hybrid), 
        MBR pruning, and bucket batching.

        Args:
            test_dataset (AkkadianTranslationDatasetT5): Preprocessed dataset from prepare_dataloader()

        Returns:
            list[str]: Predicted translations, in matching order with the test dataset
            
        """

        collator = InferenceCollator(self.tokenizer, model=self.model)

        if self.config.get("use_bucket_batching", False):
            batch_sampler = BucketBatchSampler(
                lengths=test_dataset.input_lengths,
                batch_size=self.config["batch_size"],
                num_buckets=self.config["num_buckets"],
                drop_last=False
            )
            
            test_loader = DataLoader(
                test_dataset, 
                batch_sampler=batch_sampler,
                num_workers=self.config["num_workers"], 
                collate_fn=collator,
                pin_memory=True  
            )
            
        else:
            test_loader = DataLoader(
                test_dataset, 
                batch_size=self.config["batch_size"], 
                num_workers=self.config["num_workers"], 
                shuffle=False,
                collate_fn=collator,
                pin_memory=True
            )


        # Initialising None-filled list to preserve origanl order when adding predictions
        predictions = [None] * len(test_dataset)
        
        with torch.no_grad():
            for batch in tqdm(test_loader):
                batch_indices = batch["idx"].tolist()
                input_ids = batch["input_ids"].to(self.config["device"])
                attention_mask = batch["attention_mask"].to(self.config["device"])
   
                # Standard beam search
                if not self.config["mbr"]:
                    generate_kwargs = self._base_generate_kwargs(attention_mask)
                    generate_kwargs.update(dict(
                        num_beams=self.config["num_beams"]
                    ))

                    outputs = self._run_generate(input_ids, generate_kwargs)
                    decoded = self._decode_generate_output(outputs)

                # MBR Path
                else:
                    keep_k = self.config.get("mbr_keep_k", 6)
                    grouped_candidates, grouped_scores = self._build_mbr_candidate_groups(
                        input_ids, attention_mask
                    )

                    decoded = []
                    for candidates, candidate_scores in zip(grouped_candidates, grouped_scores):
                        if self.config.get("mbr_pruning", False): # Set to False if pruning not specified
                            candidates = self.prune_candidates(
                                candidates,
                                candidate_scores,
                                keep_k=keep_k
                            )

                        # Run MBR metric 
                        best = self.mbr_select(candidates)
                        decoded.append(best)

                post_processed = [
                    self.processor.postprocess_translation_output(text) 
                    for text in decoded
                ]

                for idx, pred in zip(batch_indices, post_processed):
                    predictions[idx] = pred

        return predictions
    
class BucketBatchSampler(Sampler):
    """
    Custom PyTorch sampler that groups sequences of similar lengths 
    into the same batch (bucket batching). 

    Args:
        lengths (list[int]): Tokenized input length for each sample in the dataset.
        batch_size (int): Number of samples per batch.
        num_buckets (int, optional): Number of length-based buckets.
        drop_last (bool, optional): Drop final batch in a bucket if smaller than batch_size. 
    """
    def __init__(self,lengths, batch_size, num_buckets=4, drop_last=False):
        self.lengths = lengths
        self.batch_size = batch_size
        self.num_buckets = num_buckets
        self.drop_last = drop_last

        # Sort indices by sequence length
        sorted_ids = sorted(range(len(lengths)), key=lambda i: lengths[i])

        bucket_size = max(1, len(sorted_ids) // max(1, num_buckets))
        
        self.batches = []

        # Looping over buckets, adding buckets to batches
        for i in range(num_buckets):
            start = i*bucket_size
            end = min((i+1)*bucket_size, len(sorted_ids))
            bucket = sorted_ids[start:end]

            # Creating batches from buckets (skip with batch_size to fill batches)
            for j in range(0,len(bucket), batch_size):
                batch = bucket[j:j+batch_size]
                if len(batch)==batch_size or not drop_last:
                    self.batches.append(batch)

        # Collect any remaing samples not yet covered 
        remainder_start = num_buckets*bucket_size
        if remainder_start < len(sorted_ids):
            bucket = sorted_ids[remainder_start:]
            for j in range(0, len(bucket), batch_size):
                batch = bucket[j:j+batch_size]
                if len(batch) == batch_size or not drop_last:
                    self.batches.append(batch)

    def __iter__(self):
        yield from self.batches

    def __len__(self):
        return len(self.batches)
                

class InferenceCollator:
    """
    Wraps DataCollatorSeq2Seq to preserve sample indices.
    Idx field is popped before collation and reattached after.

    Args:
        tokenizer: HuggingFace tokenizer used for padding.
        model: Model used in inference.
    """
    def __init__(self,tokenizer, model=None):
        self.base_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    def __call__(self, features):
        # Remove idx before collation
        ids = [f.pop("idx") for f in features]
        batch = self.base_collator(features)

        # Reattach idx
        batch["idx"] = torch.tensor(ids, dtype=torch.long)
        return batch