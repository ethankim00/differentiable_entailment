# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This file contains code for wrapping a transformer language model and
provides convenience methods for training and inference.
"""

import copy
import jsonpickle
import os
from datetime import datetime
from typing import List, Dict

import wandb
import torch
import torch.nn as nn
import numpy as np
from tqdm import trange, tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler

# from torch.cuda.amp import autocast, GradScaler
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoConfig,
    AutoTokenizer,
    GPT2LMHeadModel,
)  # TODO

from transformers.models.roberta.modeling_roberta import RobertaClassificationHead
import logging
from data_utils import PVPS, load_task_helper, load_metrics, evaluate_results
from data_utils.pvps import ENTAILMENT_PVPS
from config import WrapperConfig, EvalConfig
from utils import InputExample, InputFeatures, DictDataset
from encoder import PromptEncoder
import transformers
transformers.logging.set_verbosity_error()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model")

CONFIG_NAME = "wrapper_config.json"


class ContinuousPrompt(nn.Module):
    """
    Class for ContinuousPrompt
    - What does it do?
    - Why is it a torch module?
    """

    def __init__(self, config: WrapperConfig, tokenizer, pvp):
        """Initialize continuous prompt object

        Args:
            config (WrapperConfig):
            tokenizer ([type]):
            pvp ([type]): Pattern Verbalizer Pair - What format?

        Raises:
            ValueError: [description]
        """
        super(ContinuousPrompt, self).__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.embed_size = config.embed_size
        self.hidden_size = self.embed_size

        # The pattern_id is supposed to indicate the number of continuous prompt tokens.
        # Determine number of continuous prompt tokens - probably something that we tune yes
        prompt_length = 0
        for idx, val in enumerate(pvp.BLOCK_FLAG):
            if val == 1:
                print(pvp.PATTERN[idx])
                print(tokenizer.encode(pvp.PATTERN[idx], add_special_tokens=False))
                prompt_length += len(tokenizer.encode(pvp.PATTERN[idx], add_special_tokens=False))
        self.prompt_length = prompt_length

        # config_class = MODEL_CLASSES[self.config.model_type]['config']
        model_config = AutoConfig.from_pretrained(
            config.model_name_or_path,
            num_labels=len(config.label_list),
            finetuning_task=config.task_name,
            cache_dir=config.cache_dir if config.cache_dir else None,
        )

        # model_class = MODEL_CLASSES[self.config.model_type]['model']
        # Load a huggingface model
        # Should mainly be our ROBERTA for MLM default
        self.model = AutoModelForMaskedLM.from_pretrained(
            config.model_name_or_path,
            config=model_config,
            cache_dir=config.cache_dir if config.cache_dir else None,
        )

        # Initialize
        self.prompt_embeddings = torch.nn.Embedding(self.prompt_length, self.embed_size)

        # Initialize modules for prompt encoding
        # LSTM is equivalent to p tuning paer
        # MLP is DART main method?, no inner is?
        if config.prompt_encoder_type == "lstm":
            self.lstm_head = torch.nn.LSTM(
                input_size=self.hidden_size,
                hidden_size=self.hidden_size,
                num_layers=2,
                bidirectional=True,
                batch_first=True,
            )
            self.mlp_head = nn.Sequential(
                nn.Linear(2 * self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
            )

        elif config.prompt_encoder_type == "mlp":
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(self.hidden_size, self.hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(self.hidden_size, self.hidden_size),
            )

        elif config.prompt_encoder_type in {"none", "inner"}:
            # Manual prompt without continuous tuning, or:
            # Use some unused tokens as prompt tokens / label tokens
            pass

        else:
            raise ValueError("unknown prompt_encoder_type.")

        if self.config.entailment:
            if "mnli" in self.config.model_name_or_path:  # TODO don't copy for roberta base
                # Initialize separate classification head
                #config.hidden_size = 1024
                #self._copy_classification_head(model_config)
                self.model.classifier = RobertaClassificationHead(model_config)
                #trainable_parameters = sum(p.numel() for p in self.model.classifier.parameters() if p.requires_grad)
                #print(trainable_parameters)
            else:
                # copy over two class entailment classification head weights
                self.model.classifier = RobertaClassificationHead(model_config)
               
                
            #self.model.classifier = RobertaClassificationHead(config)
            # add class aggregator
            self.config.num_classes = len(self.config.label_list)
            self.model.class_aggregator = torch.nn.Linear(
                self.config.num_classes, self.config.num_classes
            )

    def _copy_classification_head(self, model_config):
        """
        Load pretrained sequence classification head from _ForSequenceClassification
        and add it to the _ForMaskedLM model

        Note: Only makes sense to copy over the classifier weights if Intermediate Training
        is on 2 class NLI task. MNLI is 3 class entailment, so we can initialize a new
        2 class output head and train that

        Returns:
            adds classifier (nn.Module) to model
        """
        print("loading classifier")
        classifier = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name_or_path,
            config=model_config,
            cache_dir=self.config.cache_dir if self.config.cache_dir else None,
        )
        # Try copy.deepcopy or load_state_dict
        # for load_state_dict have to initialize R
        self.model.classifier = copy.deepcopy(classifier.classifier)
        del classifier  # don't store whole other model


class TransformerModelWrapper(object):
    """A wrapper around a Transformer-based language model."""

    def __init__(self, config: WrapperConfig):
        self.config = config

        # tokenizer_class = MODEL_CLASSES[config.model_type]['tokenizer']
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path,
            cache_dir=config.cache_dir if config.cache_dir else None,
            use_fast=False,
        )

        # Load pattern verbalizer pairs based on config
        self.pvp = PVPS[config.task_name](self, config.pattern_id)
        if self.config.entailment:
            self.pvp = ENTAILMENT_PVPS[config.task_name](
                self, config.pattern_id,
                num_trainable_tokens = config.num_trainable_tokens,
                train_verbalizer = config.train_verbalizer,
                use_prompt = config.use_prompt,
                two_sided = config.two_sided,
                train_prompt = config.train_prompt,
            ) # TODO set up PVP initialization properky
        # Initialize continuous prmpt model
        print("Prompting Pattern")
        print(self.pvp.PATTERN)
        print(self.pvp.BLOCK_FLAG)
        self.model = ContinuousPrompt(config, self.tokenizer, self.pvp)
        self.task_helper = load_task_helper(config.task_name, self)
        self.label_map = {label: i for i, label in enumerate(self.config.label_list)}

        if config.prompt_encoder_type == "inner":  # What is prompt encoder?
            self.encoder = PromptEncoder(
                self.tokenizer, self.pvp, config.label_list
            )  # Initialize prompt encoder
            # Random init prompt tokens HERE!
            #for _ in list(self.encoder.pattern_convert.keys()):
             #   print(self.tokenizer.decode(_))
            
            prompt_tokens = self.pvp.PROMPT + [self.pvp.LABEL] + ["."]
            prompt_token_ids = [self.tokenizer.encode(" " + pt)[1] for pt in prompt_tokens]
            self.model.prompt_length = len(prompt_token_ids) + config.num_trainable_tokens
            #print(prompt_token_ids)
            if self.config.entailment:
                self.encoder.init_embed(self.model.model, prompt_token_ids = prompt_token_ids)
            else:
                self.encoder.init_embed(self.model.model, random_=False)
            
                
        if config.device == "cuda":
            if (
                torch.cuda.device_count() > 1
            ):  # Should be able to easily do Data Paralellism
                self.model = torch.nn.DataParallel(self.model)
            self.model.cuda()
            # Use automatic mixed precision for faster training
            # self.scaler = GradScaler()

        # TODO add Classifcation Head Projections

    def save(self, path: str) -> None:
        """Logic to save all the addition torch nn modules to file

        Args:
            path (str):

        Raises:
            ValueError: [description]
        """
        logger.info("Saving trained model at %s..." % path)
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )

        model_to_save.model.save_pretrained(
            path
        )  
        self.tokenizer.save_pretrained(path)
        self._save_config(path)
        if self.config.entailment:
            torch.save(self.model.model.classifier.state_dict(), path + "/classification_head")


        if self.config.prompt_encoder_type == "lstm":
            state = {
                "prompt_embeddings": model_to_save.prompt_embeddings.state_dict(),
                "lstm_head": model_to_save.lstm_head.state_dict(),
                "mlp_head": model_to_save.mlp_head.state_dict(),
            }
        elif self.config.prompt_encoder_type == "mlp":
            state = {
                "prompt_embeddings": model_to_save.prompt_embeddings.state_dict(),
                "mlp": model_to_save.mlp.state_dict(),
            }
        elif self.config.prompt_encoder_type in {"none", "inner"}:
            state = {
                "word_embeddings": model_to_save.model.get_input_embeddings().state_dict()
            }
        else:
            raise ValueError("unknown prompt_encoder_type.")

        save_path_file = os.path.join(path, "embeddings.pth")
        torch.save(state, save_path_file)

    @classmethod
    def from_pretrained(cls, path: str) -> "TransformerModelWrapper":
        """Load a pretrained wrapper from a given path."""

        wrapper = TransformerModelWrapper.__new__(TransformerModelWrapper)
        wrapper.config = wrapper._load_config(path)
        wrapper.tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
        wrapper.pvp = PVPS[wrapper.config.task_name](wrapper, wrapper.config.pattern_id)
        wrapper.model = ContinuousPrompt(wrapper.config, wrapper.tokenizer, wrapper.pvp)
        wrapper.model.model = AutoModelForMaskedLM.from_pretrained(path)
        # TODO make sure classification head is loaded

        # Load prompt embeddings
        save_path_file = os.path.join(path, "embeddings.pth")
        data = torch.load(save_path_file)

        # `inner` / `none` encoder
        if "prompt_embeddings" in data:
            wrapper.model.prompt_embeddings.load_state_dict(data["prompt_embeddings"])

        if "lstm_head" in data:
            assert "mlp_head" in data
            wrapper.model.lstm_head.load_state_dict(data["lstm_head"])
            wrapper.model.mlp_head.load_state_dict(data["mlp_head"])
        if "mlp" in data:
            wrapper.model.mlp_head.load_state_dict(data["mlp"])

        if wrapper.config.prompt_encoder_type == "inner":
            wrapper.encoder = PromptEncoder(
                wrapper.tokenizer, wrapper.pvp, wrapper.config.label_list
            )

        # try:
        
        # except: 
        #     print("did not load head")
        #     pass 
        if "classification_head" in os.listdir(path):
            wrapper.pvp = ENTAILMENT_PVPS[wrapper.config.task_name](
                wrapper, wrapper.config.pattern_id,
                num_trainable_tokens = wrapper.config.num_trainable_tokens,
                train_verbalizer = wrapper.config.train_verbalizer,
                use_prompt = wrapper.config.use_prompt,
                two_sided = wrapper.config.two_sided,
                train_prompt = wrapper.config.train_prompt
            ) 
            if wrapper.config.prompt_encoder_type == "inner":
                wrapper.encoder = PromptEncoder(
                    wrapper.tokenizer, wrapper.pvp, wrapper.config.label_list
                )
            wrapper.model = ContinuousPrompt(wrapper.config, wrapper.tokenizer, wrapper.pvp)
            wrapper.model.model = AutoModelForMaskedLM.from_pretrained(path)
            prompt_length = 0
            for idx, val in enumerate(wrapper.pvp.BLOCK_FLAG):
                if val == 1:
                    prompt_length += len(wrapper.tokenizer.encode(wrapper.pvp.PATTERN[idx]))
            wrapper.model.prompt_length = prompt_length
            model_config = AutoConfig.from_pretrained(
                wrapper.config.model_name_or_path,
                num_labels=len(wrapper.config.label_list),
                finetuning_task=wrapper.config.task_name,
                cache_dir=wrapper.config.cache_dir if wrapper.config.cache_dir else None,
            )
            wrapper.model.model.classifier = RobertaClassificationHead(model_config)
            wrapper.model.model.classifier.load_state_dict(torch.load(path + "/classification_head"))
            wrapper.model.model.classifier.eval()
        wrapper.label_map = {
            label: i for i, label in enumerate(wrapper.config.label_list)
        }
        wrapper.task_helper = load_task_helper(wrapper.config.task_name, wrapper)

        if wrapper.config.device == "cuda":
            if torch.cuda.device_count() > 1:
                wrapper.model = torch.nn.DataParallel(wrapper.model)
            wrapper.model.cuda()
            # Use automatic mixed precision for faster training
            # wrapper.scaler = GradScaler()

        return wrapper

    def _save_config(self, path: str) -> None:
        with open(os.path.join(path, CONFIG_NAME), "w") as f:
            f.write(jsonpickle.encode(self.config))

    @staticmethod
    def _load_config(path: str) -> WrapperConfig:
        with open(os.path.join(path, CONFIG_NAME), "r") as f:
            return jsonpickle.decode(f.read())

    def train(
        self,
        train_data: List[InputExample],
        eval_data: List[InputExample],
        dev_data: List[InputExample],
        eval_config: EvalConfig,
        pattern_iter_output_dir,
        per_gpu_train_batch_size: int = 8,
        n_gpu: int = 1,
        num_train_epochs: int = 3,
        gradient_accumulation_steps: int = 1,
        weight_decay: float = 0.0,
        learning_rate: float = 5e-5,
        embed_learning_rate: float = 5e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps=0,
        max_grad_norm: float = 1,
        max_steps=-1,
        early_stop_epochs=5,
        **kwargs,
    ):
        """[summary]

        Args:
            train_data (List[InputExample]): [description]
            eval_data (List[InputExample]): [description]
            dev_data (List[InputExample]): [description]
            eval_config (EvalConfig): [description]
            pattern_iter_output_dir ([type]): Main output directory
            per_gpu_train_batch_size (int, optional): [description]. Defaults to 8.
            n_gpu (int, optional): [description]. Defaults to 1.
            num_train_epochs (int, optional): Number of epochs . Defaults to 3.
            gradient_accumulation_steps (int, optional): [description]. Defaults to 1.
            weight_decay (float, optional): [description]. Defaults to 0.0.
            learning_rate (float, optional): [description]. Defaults to 5e-5.
            adam_epsilon (float, optional): [description]. Defaults to 1e-8.
            warmup_steps (int, optional): [description]. Defaults to 0.
            max_grad_norm (float, optional): [description]. Defaults to 1.
            max_steps (int, optional): [description]. Defaults to -1.
            early_stop_epochs (int, optional): [description]. Defaults to 10.
        """

        def log_scalars(result_dict, set_type):
            # Write scalars with tensorboard
            # for metric, score in result_dict.items():
            #     writer.add_scalar(
            #         set_type + "-" + metric, score, global_step=global_step
            #     )
            if kwargs.get("wandb_log", False):
                # Write scalars with wandb
                wandb.log(
                    {
                        set_type + "-" + metric: score
                        for metric, score in result_dict.items()
                    }
                )

        train_batch_size = per_gpu_train_batch_size * max(1, n_gpu)
        train_dataset = self._generate_dataset(train_data)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler, batch_size=train_batch_size
        )

        if max_steps > 0:  # Can specify number of training steps instead of epochs
            t_total = max_steps
            num_train_epochs = (
                max_steps
                // (max(1, len(train_dataloader) // gradient_accumulation_steps))
                + 1
            )
        else:
            t_total = (
                len(train_dataloader) // gradient_accumulation_steps * num_train_epochs
            )

        cur_model = (
            self.model.module if hasattr(self.model, "module") else self.model
        )  # Why?

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        # Store which parameters to add weight decay to (exclude bias and layernorm weights)
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in cur_model.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in cur_model.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        embedding_parameters = None
        stage = kwargs.get("stage", 0)

        if (
            self.config.prompt_encoder_type == "lstm"
        ):  # p tuning method requires tuning extra LSTM parameters
            embedding_parameters = [
                {"params": [p for p in cur_model.lstm_head.parameters()]},
                {"params": [p for p in cur_model.mlp_head.parameters()]},
                {"params": [p for p in cur_model.prompt_embeddings.parameters()]},
            ]
        elif self.config.prompt_encoder_type == "mlp":
            embedding_parameters = [
                {"params": [p for p in cur_model.mlp.parameters()]},
                {"params": [p for p in cur_model.prompt_embeddings.parameters()]},
            ]
        elif self.config.prompt_encoder_type == "none":
            pass
        elif self.config.prompt_encoder_type == "inner":
            if stage == 1:
                # Training stage 1: only optimize prompt-related tokens
                print("Training Stage 1")
                print("Training Embedding and Classification Head Parameters with learning Rate {}".format(learning_rate))
                handle = self.encoder.add_embed_hook(
                    cur_model.model
                )  # Stage 1 set certain parameters with 0 weight decay
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for p in cur_model.model.get_input_embeddings().parameters()
                        ],
                        "weight_decay": weight_decay,
                    }
                ]
                # Also finetune the classification head if we are doing entailment
                # TODO try tuning with separate learning rate schedules
                if self.config.entailment:
                    optimizer_grouped_parameters.append(
                        {
                            "params" : [
                                p for p in cur_model.model.classifier.parameters()
                            ], 
                            "weight_decay" : weight_decay, 
                        }
                    )
            else:
                # Training stage 0 / 2: optimize all model weights with different learning rates
                # This is used when training LM ONLY!
                print("Training Stage {}".format(stage))
                print("Optimizing Model Parameters with Learning Rate {}".format(learning_rate))
                print("Optmizing Embedding Parameters with Learning Rate {}".format(embed_learning_rate))
                if self.config.entailment and stage == 2: # Only freeze trained embedding parameters in training stage 2 
                    handle = self.encoder.add_reverse_hook((cur_model.model))
                    print("Freezing Pseudotoken Embeddings")
                embedding_parameters = [
                    {
                        "params": [
                            p
                            for p in cur_model.model.get_input_embeddings().parameters()
                        ],
                        "weight_decay": 0.0,
                    }
                ]
                optimizer_grouped_parameters[0] = {
                    "params": [
                        p
                        for n, p in cur_model.model.named_parameters()
                        if not any(nd in n for nd in no_decay + ["word_embeddings"])
                    ],
                    "weight_decay": weight_decay,
                }
                # Mask out gradients of tokens unrelated with prompt / label
                if kwargs.get("fix_other_embeddings", False):
                    print("fixing other embeddings")
                    handle = self.encoder.add_embed_hook(cur_model.model)
                    # embedding_parameters[0]['weight_decay'] = 0.0

                # TODO add logic to freeze or train clasification head for entailment
        optimizer_list, scheduler_list = [], []
        optimizer_list.append(
            AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
        )
        scheduler_list.append(
            get_linear_schedule_with_warmup(
                optimizer_list[0],
                num_warmup_steps=warmup_steps,
                num_training_steps=t_total,
            )
        )

        if (
            embedding_parameters
        ):  # Use separate optimizer and learning rate schedule to train the embedding parameters
            optimizer_list.append(
                AdamW(embedding_parameters, lr=embed_learning_rate, eps=adam_epsilon)
            )
            scheduler_list.append(
                get_linear_schedule_with_warmup(
                    optimizer_list[0],
                    num_warmup_steps=warmup_steps,
                    num_training_steps=t_total,
                )
            )
        now = datetime.now()
        path_suffix = now.strftime("%m-%d_%H:%M:%S") + "stage_%d" % stage
        writer = SummaryWriter(
            log_dir=os.path.join(self.config.output_dir, "writer_logs", path_suffix)
        )

        # Statistics in training
        save_metric_name = load_metrics(self.config.task_name)[-1]
        best_dev_metric, best_loss = -1.0, 0.0
        best_global_step, early_stop_count, global_step = 0, 0, 0
        prev_loss, tr_loss = 0.0, 0.0

        # Zero Shot Test Performance
        # test_res = self.eval(
        #                     eval_data,
        #                     eval_config.per_gpu_eval_batch_size,
        #                     n_gpu,
        #                     eval_config.metrics,
        #                     )
        # eval_scores = test_res["scores"]
        # log_scalars(eval_scores, "eval")
        # logger.info("Zero Shot Performance on Test Data %s" %
        #             str(eval_scores))
        # Record dev metric scores in tensorboard
        # dev_scores = self.eval(
        #     dev_data, eval_config.per_gpu_eval_batch_size, n_gpu, eval_config.metrics)['scores']
        # logger.info("dev_data performance before training: %s" %
        #             str(dev_scores))
        # log_scalars(dev_scores, 'dev')

        # # Record dev metric scores in tensorboard
        # eval_scores = self.eval(
        #     eval_data, eval_config.per_gpu_eval_batch_size, n_gpu, eval_config.metrics)['scores']
        # logger.info("eval_data performance before training: %s" %
        #             str(eval_scores))
        # log_scalars(eval_scores, 'eval')

        # PATCH @ 2021.09.27: Record evaluation results
        if kwargs.get("record_eval", False):
            all_eval_dev, all_eval_test = [], []
        extra_mask_rate = kwargs.get("extra_mask_rate", 0.0)
        train_iterator = trange(int(num_train_epochs), desc="Epoch")
        for _ in train_iterator:  # iterate over epochs
            for step, batch in enumerate(train_dataloader):  # iterate over batches
                self.model.train()  # set model to training mode
                if extra_mask_rate > 0.0:
                    self._add_extra_mask(batch, extra_mask_rate)
                if self.config.device == "cuda":
                    batch = {
                        k: t.cuda() for k, t in batch.items()
                    }  # move the batch data to GPU
                # TODO expand input x num_lables
                # Casts operations to mixed precision
                # with torch.cuda.amp.autocast():
                #     loss = self.task_helper.train_step(
                #         batch) if self.task_helper else None
                #     if loss is None:
                #         loss = self.mlm_train_step(batch)

                if (
                    self.task_helper
                ):  # Want general EFL train step not one for each task
                    loss = self.task_helper.train_step(batch)

                elif self.config.entailment:
                    loss, accuracy = self.entailment_train_step(batch)
                    # print(cur_model.model.classifier.dense.weight)
                    # print(cur_model.model.get_input_embeddings()(torch.LongTensor([50165]).to(self.config.device))) #debug freezing embeddings
                    # print(cur_model.model.get_input_embeddings()(torch.LongTensor([24]).to(self.config.device)))
                    # print(cur_model.model.get_input_embeddings()(torch.LongTensor([0]).to(self.config.device)))
                    # print(cur_model.model.roberta.encoder.layer[0].attention.self.query.weight) # Print some random model weights
                    # [-0.0227, -0.1245, -0.1119,  ..., -0.0299, -0.1163, -0.1389]]
                    # -0.1400, -0.0104,  0.0395,  ...,  0.0513, -0.0062, -0.0361
                else:
                    loss = self.mlm_train_step(
                        batch
                    )  # main training step to compute loss

                if n_gpu > 1:
                    loss = (
                        loss.mean()
                    )  # mean() to average on multi-gpu parallel training
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                loss.backward()
                # self.scaler.scale(loss).backward()
                tr_loss += loss.item()
                if self.config.entailment:
                    avg_loss = tr_loss/(global_step +1)
                    wandb.log({"loss": avg_loss})
                    wandb.log({"total_loss": tr_loss})
                    train_iterator.set_postfix(avg_loss=avg_loss, tr_loss = tr_loss, loss = loss.item(),  accuracy = accuracy.item(), global_step = global_step)
                    if avg_loss > 0.5 and global_step > 60:
                        break

                if (step + 1) % gradient_accumulation_steps == 0:
                    writer.add_scalar(
                        "train_loss", (tr_loss - prev_loss), global_step=global_step
                    )
                    prev_loss = tr_loss

                    # Unscales the gradients of optimizer's assigned params in-place
                    # for optimizer in optimizer_list:
                    #     self.scaler.unscale_(optimizer)

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_grad_norm
                    )

                    for optimizer, scheduler in zip(optimizer_list, scheduler_list):
                        optimizer.step()
                        # self.scaler.step(optimizer)
                        # self.scaler.update()
                        scheduler.step()

                    self.model.zero_grad(set_to_none=True)
                    global_step += 1

                    # Evaluate every some steps
                    if global_step % self.config.eval_every_step == 0:
                        dev_res = self.eval(
                            dev_data,
                            eval_config.per_gpu_eval_batch_size,
                            n_gpu,
                            eval_config.metrics,
                        )
                        if kwargs.get("record_eval", False):
                            all_eval_dev.append(dev_res)
                        dev_scores = dev_res["scores"]
                        log_scalars(dev_scores, "dev")
                        # Evaluate sample and save model on best performance
                        if dev_scores[save_metric_name] > best_dev_metric:
                            if dev_scores[save_metric_name] > best_dev_metric:
                                early_stop_count = 0
                                logger.info(
                                    "Best %s on dev: %.4f | global step: %d"
                                    % (
                                        save_metric_name,
                                        best_dev_metric,
                                        best_global_step,
                                    )
                                )
                            else:
                                early_stop_count += 1
                                logger.info(
                                    "Dev scores: %.4f | early_stop_count: %d"
                                    % (dev_scores[save_metric_name], early_stop_count)
                                )
                            # Record best statistics
                            best_dev_metric = dev_scores[save_metric_name]
                            best_global_step = global_step
                            best_loss = tr_loss

                            # Perform evaluation on test
                            test_res = self.eval(
                                eval_data,
                                eval_config.per_gpu_eval_batch_size,
                                n_gpu,
                                eval_config.metrics,
                            )
                            if kwargs.get("record_eval", False):
                                all_eval_test.append(test_res)
                            eval_scores = test_res["scores"]
                            logger.info("eval_data performance: %s" % str(eval_scores))
                            log_scalars(eval_scores, "eval")

                            # TODO: can also choose to save model only on higher scores
                            # Save best model
                            self.save(pattern_iter_output_dir)
                        else:
                            early_stop_count += 1
                            if kwargs.get("record_eval", False):
                                all_eval_test.append(None)
                            logger.info(
                                "Dev scores: %.4f | early_stop_count: %d"
                                % (dev_scores[save_metric_name], early_stop_count)
                            )
                if 0 < max_steps < global_step or early_stop_count >= early_stop_epochs:
                    break
            if 0 < max_steps < global_step or early_stop_count >= early_stop_epochs:
                test_res = self.eval(
                    eval_data,
                    eval_config.per_gpu_eval_batch_size,
                    n_gpu,
                    eval_config.metrics,
                )
                eval_scores = test_res["scores"]
                logger.info("Final performance: %s" % str(eval_scores))
                log_scalars(eval_scores, "eval")
                train_iterator.close()
                break
            if avg_loss > 0.5 and global_step > 60:
                test_res = self.eval(
                    eval_data,
                    eval_config.per_gpu_eval_batch_size,
                    n_gpu,
                    eval_config.metrics,
                    )
                eval_scores = test_res["scores"]
                logger.info("Final performance: %s" % str(eval_scores))
                log_scalars(eval_scores, "eval")
                break

        try:
            handle.remove()
        except Exception:
            pass

        if kwargs.get("record_eval", False):
            return (
                best_global_step,
                (best_loss / best_global_step if best_global_step > 0 else -1),
                all_eval_dev,
                all_eval_test,
            )
        return best_global_step, (
            best_loss / best_global_step if best_global_step > 0 else -1
        )

    def eval(
        self,
        eval_data: List[InputExample],
        per_gpu_eval_batch_size: int = 8,
        n_gpu: int = 1,
        metrics: List[str] = ["acc"],
    ) -> Dict:

        eval_dataset = self._generate_dataset(eval_data)
        eval_batch_size = per_gpu_eval_batch_size * max(1, n_gpu)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size
        )

        preds = None
        all_indices, out_label_ids, question_ids = None, None, None
        all_masked_full_logits, all_masked_hidden_states = None, None
        eval_losses = [0.0]

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            self.model.eval()
            if self.config.device == "cuda":
                batch = {k: t.cuda() for k, t in batch.items()}
            labels = batch["labels"]
            indices = batch["idx"]

            with torch.no_grad():
                logits = self.task_helper.eval_step(batch) if self.task_helper else None
                if logits is None:
                    # PATCH @ 2021.09.27: add masked hidden states of each sentence
                    if not self.config.entailment:
                        (
                            logits,
                            masked_full_logits,
                            masked_hidden_states,
                        ) = self.mlm_eval_step(batch)
                    else:
                        (
                            logits,
                            masked_full_logits,
                            masked_hidden_states,
                        ) = self.entailment_eval_step(batch)

                    if all_masked_hidden_states is None:
                        all_masked_full_logits = (
                            masked_full_logits.detach().cpu().numpy()
                        )
                        all_masked_hidden_states = (
                            masked_hidden_states.detach().cpu().numpy()
                        )
                    else:
                        all_masked_full_logits = np.append(
                            all_masked_full_logits,
                            masked_full_logits.detach().cpu().numpy(),
                            axis=0,
                        )
                        all_masked_hidden_states = np.append(
                            all_masked_hidden_states,
                            masked_hidden_states.detach().cpu().numpy(),
                            axis=0,
                        )
                # Calculate evaluation loss
                prediction_scores = logits.float()
                eval_loss = nn.CrossEntropyLoss()(
                    prediction_scores.view(-1, len(self.config.label_list)),
                    labels.view(-1),
                )
                eval_losses.append(eval_loss.item())

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
                all_indices = indices.detach().cpu().numpy()
                if "question_idx" in batch:
                    question_ids = batch["question_idx"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, labels.detach().cpu().numpy(), axis=0
                )
                all_indices = np.append(
                    all_indices, indices.detach().cpu().numpy(), axis=0
                )
                if "question_idx" in batch:
                    question_ids = np.append(
                        question_ids,
                        batch["question_idx"].detach().cpu().numpy(),
                        axis=0,
                    )
        # mean loss, list of indices, predictions, labels etc
        results = {
            "eval_loss": np.mean(eval_losses),
            "indices": all_indices,
            "logits": preds,
            "labels": out_label_ids,
            "question_ids": question_ids,
            "full_logits": all_masked_full_logits,
            "masked_hidden_states": all_masked_hidden_states,
        }

        return evaluate_results(results, metrics)

    def expand_labeled_batch(
        self, labeled_batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Expand batch for entailment training

        Args:
            labeled_batch (Dict[str, torch.Tensor]): Labled Instances

        Returns:
            Dict[str, torch.Tensor]: Expanded Batch with num_classes entries for each input instance
        """
        if self.config.num_classes <= 2:
            return labeled_batch

        else:
            expanded_batch = {}
            for idx, instance in enumerate(labeled_batch["input_ids"]):
                for cls_idx, output_class in enumerate(
                    list(self.model.pvp.VERBALIZER.keys())
                ):
                    output_class_id = self.model.tokenizer.tokenize(output_class)
                    output_class_trainable_id = self.model.label_convert[
                        output_class_id
                    ]
                    # Have to identify the position of the label token -> save as verbalizer attribute
                    instance[
                        (
                            instance["input_ids"] == self.encoder.entailment_label_id
                        ).nonzero(as_tuple=True)[0]
                    ] = output_class_trainable_id # Replace "label" placeholder with actual label
                    expanded_batch["input_ids"].append(instance)
                    expanded_batch["attention_mask"].append(
                        labeled_batch["attention_mask"][idx]
                    )
                    expanded_batch["token_type_ids"].append(
                        labeled_batch["token_type_ids"][idx]
                    )
                    expanded_batch["labels"].append(labeled_batch["labels"][idx])
                    expanded_batch["mlm_labels"].append(
                        labeled_batch["mlm_labels"][idx]
                    )
                    expanded_batch["logits"].append(labeled_batch["logits"][idx])
                    expanded_batch["idx"].append(
                        idx * self.config.num_classes + cls_idx
                    )
                    expanded_batch["block_flag"].append(
                        labeled_batch["block_flag"][idx]
                    )

        # TODO check for inneficiencies transfering between devices
        for key in expanded_batch.keys():
            expanded_batch[key] = torch.tensor(expanded_batch[key])
        return DictDataset(**expanded_batch)

    def entailment_train_step(
        self, labeled_batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        # TODO expand forward pass here or in training loop

        labeled_batch = self.expand_labeled_batch(labeled_batch) # it was LABEL -> it was good, it was bad, it was neutral B x num_labels
        inputs = self._generate_default_inputs(
            labeled_batch
        )  # some additional preprocessing on batch
        mlm_labels, labels = (
            labeled_batch["mlm_labels"],
            labeled_batch["labels"],
        )
        model = self.model.module if hasattr(self.model, "module") else self.model
        outputs = model.model(**inputs, output_hidden_states = True)
        # for i, _ in enumerate(labeled_batch["input_ids"]):
        #     print(self.tokenizer.decode(_))
        #     print(labeled_batch["block_flag"][i])
        # Assume outputs[1] is the hidden states
        # outputs[1] shape B, sequence length x hidden size
        # Do pooling manually?
        # entailment_logits.shape = (B,)
        # TODO make sure we are passing correct hidden sstate to classifiner
        entailment_logits = model.model.classifier(
            outputs.hidden_states[-1]# input hidden states
        )  # # TODO use pooling instead of CLS
        # class.shape = (B/num_classes, num_classes)
        #
        if len(self.config.label_list) > 2:
            class_scores = model.model.class_aggregator(
                entailment_logits.view(-1, len(self.config.label_list))
            ) # (B, num_classes)
            loss = nn.CrossEntropyLoss()(
                class_scores.view(-1, len(self.config.label_list)), labels.view(-1)
            )
        else:
            # Binary class case
            loss = nn.CrossEntropyLoss()(entailment_logits.view(-1, 2), labels)
        # Do fluency constraint objective
        # calculate accuracy:
        predictions = entailment_logits.argmax(dim = 1, keepdim = True).squeeze()
        accuracy= (predictions == labels).sum()/len(predictions) 
        if (
            "extra_mlm_labels" in labeled_batch
        ):  # is this the fluency constraint objective? Yes
            extra_mlm_labels = labeled_batch["extra_mlm_labels"]  #
            extra_loss = nn.CrossEntropyLoss()(
                outputs[0].view(
                    -1, self.tokenizer.vocab_size
                ),  # logits over entire vocabulary
                extra_mlm_labels.view(-1),  # cross entropy over labels
            )
            _lambda = 0.1
            loss += _lambda * extra_loss

        return loss, accuracy

    def entailment_eval_step(
        self, labeled_batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Evaluation step for entailment method
        """

        labeled_batch = self.expand_labeled_batch(labeled_batch)
        inputs = self._generate_default_inputs(labeled_batch) 
        mlm_labels, labels = (
            labeled_batch["mlm_labels"],
            labeled_batch["labels"],
        )
        model = self.model.module if hasattr(self.model, "module") else self.model
        outputs = model.model(**inputs, output_hidden_states = True)
        # Likely don't need these for entailment since we aren't cloze completing a masked token
        masked_full_logits = outputs[0][labeled_batch["mlm_labels"] >= 0]
        masked_hidden_states = outputs[1][-1][
            labeled_batch["mlm_labels"] >= 0
        ]

        entailment_logits = model.model.classifier(
            outputs.hidden_states[-1] # use hidden state of last layer
        )
        if len(self.config.label_list) > 2:
            entailment_logits = model.model.class_aggregator(
                entailment_logits.view(-1, len(self.config.label_list))
            )
        return (
            entailment_logits,
            masked_full_logits,
            masked_hidden_states,
        )

    def mlm_train_step(self, labeled_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Main Training Step of Model"""
        inputs = self._generate_default_inputs(
            labeled_batch
        )  # some additional preprocessing on batch
        mlm_labels, labels = labeled_batch["mlm_labels"], labeled_batch["labels"]
        model = self.model.module if hasattr(self.model, "module") else self.model
        outputs = model.model(**inputs)  # run model on inputs
        # Should return logits over vocabulary size for each position in sequence

        # Post processing steps
        if (
            self.config.prompt_encoder_type == "inner"
        ):  # get cls logits from the masked tokens
            prediction_scores = self.encoder.convert_mlm_logits_to_cls_logits(
                mlm_labels, outputs[0]
            )
        else:
            prediction_scores = self.pvp.convert_mlm_logits_to_cls_logits(
                mlm_labels, outputs[0]
            )
        # actually calculate loss
        # is this over whole vocabulary, not, just over labels
        loss = nn.CrossEntropyLoss()(
            prediction_scores.view(-1, len(self.config.label_list)), labels.view(-1)
        )  # Prediction loss

        # Do fluency constraint objective
        if (
            "extra_mlm_labels" in labeled_batch
        ):  # is this the fluency constraint objective? Yes
            extra_mlm_labels = labeled_batch["extra_mlm_labels"]  #
            extra_loss = nn.CrossEntropyLoss()(
                outputs[0].view(
                    -1, self.tokenizer.vocab_size
                ),  # logits over entire vocabulary
                extra_mlm_labels.view(-1),  # cross entropy over labels
            )
            loss += extra_loss  # why don't I see the lambda hyperparameter mentioned in paper?

        return loss

    def mlm_eval_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform a MLM evaluation step."""
        inputs = self._generate_default_inputs(batch)
        model = self.model.module if hasattr(self.model, "module") else self.model
        # PATCH @ 2021.09.27: add masked hidden states of each sentence
        outputs = model.model(**inputs, output_hidden_states=True)

        # Get outputs of encoder in last layer
        masked_full_logits = outputs[0][batch["mlm_labels"] >= 0]
        masked_hidden_states = outputs[1][-1][
            batch["mlm_labels"] >= 0
        ]  # Why do we need the hidden states?

        if self.config.prompt_encoder_type == "inner":
            return (
                self.encoder.convert_mlm_logits_to_cls_logits(
                    batch["mlm_labels"], outputs[0]
                ),
                masked_full_logits,
                masked_hidden_states,
            )

        return (
            self.pvp.convert_mlm_logits_to_cls_logits(batch["mlm_labels"], outputs[0]),
            masked_full_logits,
            masked_hidden_states,
        )

    def _generate_dataset(self, data: List[InputExample], labelled: bool = True):
        """Generate dataset

        Args:
            data (List[InputExample]): [description]
            labelled (bool, optional): [description]. Defaults to True.

        Returns:
            [type]: [description]
        """
        features = self._convert_examples_to_features(data, labelled=labelled)
        # Convert list features to tensors
        # Huggingface format
        feature_dict = {
            "input_ids": torch.tensor(  # input text ids
                [f.input_ids for f in features], dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                [f.attention_mask for f in features], dtype=torch.long
            ),
            "token_type_ids": torch.tensor(  # sent 1, sent 2 etc - relatively important for entialment
                [f.token_type_ids for f in features], dtype=torch.long
            ),
            "labels": torch.tensor(
                [f.label for f in features], dtype=torch.long
            ),  # what is the difference between labels and mlm labels
            "mlm_labels": torch.tensor(
                [f.mlm_labels for f in features], dtype=torch.long
            ),
            "logits": torch.tensor([f.logits for f in features], dtype=torch.float),
            "idx": torch.tensor([f.idx for f in features], dtype=torch.long),
            "block_flag": torch.tensor(
                [f.block_flag for f in features], dtype=torch.long
            ),
        }

        if self.task_helper:
            self.task_helper.add_features_to_dict(features, feature_dict)

        return DictDataset(**feature_dict)

    def _convert_examples_to_features(
        self, examples: List[InputExample], labelled: bool = True
    ) -> List[InputFeatures]:
        """Convert list of input examples to list of input features

        Args:
            examples (List[InputExample]): [description]
            labelled (bool, optional): [description]. Defaults to True.

        Raises:
            ValueError: [description]

        Returns:
            List[InputFeatures]: [description]
        """
        features = []
        for example in examples:
            # Preprocessor for models pretrained using a masked language modeling objective (e.g., BERT).
            input_ids, token_type_ids, block_flag = self.pvp.encode(
                example
            )  # processing is based on PVP encode method
            attention_mask = [1] * len(
                input_ids
            )  # always use fully visible attention max
            padding_length = self.config.max_seq_length - len(
                input_ids
            )  # length to pad to

            if padding_length < 0:
                raise ValueError(
                    f"Maximum sequence length is too small, got {len(input_ids)} input ids"
                )

            input_ids = input_ids + (
                [self.tokenizer.pad_token_id] * padding_length
            )  # manually sequence
            attention_mask = attention_mask + ([0] * padding_length)  # mask padded part
            token_type_ids = token_type_ids + ([0] * padding_length)  # mask padded part
            block_flag = block_flag + ([0] * padding_length)

            assert len(input_ids) == self.config.max_seq_length
            assert len(attention_mask) == self.config.max_seq_length
            assert len(token_type_ids) == self.config.max_seq_length
            assert len(block_flag) == self.config.max_seq_length

            label = self.label_map[example.label] if example.label is not None else -100
            logits = example.logits if example.logits else [-1]

            if labelled:
                if self.config.entailment:
                    mlm_labels = [-1] * self.config.max_seq_length
                else:
                    mlm_labels = self.pvp.get_mask_positions(input_ids)
            else:
                mlm_labels = [-1] * self.config.max_seq_length

            input_features = InputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label=label,
                mlm_labels=mlm_labels,
                logits=logits,
                idx=example.idx,
                block_flag=block_flag,
            )

            # Add meta input features
            if self.task_helper:
                self.task_helper.add_special_input_features(example, input_features)
            features.append(input_features)

        return features

    def _generate_default_inputs(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Turn dataset batch into labels for model

        Args:
            batch (Dict[str, torch.Tensor]): [description]

        Raises:
            ValueError: [description]

        Returns:
            Dict[str, torch.Tensor]: [description]
        """
        input_ids = batch["input_ids"]
        bz = batch["input_ids"].shape[0]
        block_flag = batch["block_flag"]
        model = self.model.module if hasattr(self.model, "module") else self.model

        # Get word embedding from model
        # word_embeddings.shape = (vocab_size, hidden_size)
        word_embeddings = (
            model.model.get_input_embeddings()
        )  # models method of getting word embeddings
        # raw_embeds.shape = (len(input_ids), hidden_size)
        raw_embeds = word_embeddings(input_ids)

        # replace with prompt embedings from model is this overwritten by inner?
        replace_embeds = model.prompt_embeddings(
            torch.LongTensor(list(range(model.prompt_length))).to(raw_embeds.device)
        )
        # [batch_size, prompt_length, embed_size]
        replace_embeds = replace_embeds.unsqueeze(0)

        if self.config.prompt_encoder_type == "lstm":
            # [batch_size, seq_len, 2 * hidden_dim]
            # run LSTM over embeddings
            replace_embeds = model.lstm_head(replace_embeds)[
                0
            ]  # use LSTM over sequence of raw embeddings
            # use hidden states from LSTM was new embeddings
            if model.prompt_length == 1:
                replace_embeds = model.mlp_head(
                    replace_embeds
                )  # additional processing of LSTM hidden states
            else:
                replace_embeds = model.mlp_head(
                    replace_embeds
                ).squeeze()  # get rid of batch dimension

        elif (
            self.config.prompt_encoder_type == "mlp"
        ):  # Feed through MLP to get new embeddings
            replace_embeds = model.mlp(replace_embeds)

        elif self.config.prompt_encoder_type == "none":
            replace_embeds = None

        elif self.config.prompt_encoder_type == "inner":
            # assert set(self.encoder.pattern_convert.keys()) == set(input_ids[torch.where(block_flag==1)].tolist())
            replace_embeds = self.encoder.get_replace_embeds(word_embeddings)
        else:
            raise ValueError("unknown prompt_encoder_type.")

        if (
            replace_embeds is not None
        ):  # For normal cases where prompt encoder is not None
            blocked_indices = (
                (block_flag == 1)
                .nonzero(as_tuple=False)
                .reshape((bz, model.prompt_length, 2))[:, :, 1]
            )

            for bidx in range(bz):  # iterate over samples in batch
                for i in range(blocked_indices.shape[1]):
                    raw_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[
                        i, :
                    ]  # replace all the psuedotoken embeddings
                    # does this really still backpropagate to the input token embedding?
            # replace certain raw_embeds with repalce inputs

        inputs = {
            "inputs_embeds": raw_embeds,
            "attention_mask": batch["attention_mask"],
        }

        if self.config.model_type in [
            "bert"
        ]:  # not relevant for ROBERTa due to no NSP?
            inputs["token_type_ids"] = batch["token_type_ids"]

        return inputs

    def _add_extra_mask(self, batch: Dict[str, torch.Tensor], mask_rate: float) -> None:
        """Mask some random set of tokens, not including pseudotokens

        Args:
            batch (Dict[str, torch.Tensor]): [description]
            mask_rate (float): Proportion of input tokens to mask
            TODO: experiment with some sort of annealing / learning schedule for the mask rate
        """
        input_ids = batch["input_ids"]
        block_flag = batch["block_flag"]
        tokenizer = self.tokenizer
        mask_id, pad_id = tokenizer.mask_token_id, tokenizer.pad_token_id
        special_token_id_set = set(
            tokenizer.convert_tokens_to_ids(tokenizer.special_tokens_map.values())
        )  # Get set of special tokens
        extra_mlm_labels = (
            torch.ones_like(input_ids, dtype=torch.long) * -100
        )  # initially set to -1
        for idx in range(len(input_ids)):
            maskable_pos = []
            for pos in range(len(input_ids[idx])):
                if input_ids[idx][pos].item() == pad_id:  # End of actual sequence
                    break
                if input_ids[idx][pos].item() not in special_token_id_set:
                    if (
                        block_flag[idx][pos] == 0
                    ):  # Mask tokens that are not psuedotokens
                        maskable_pos.append(pos)
            mask_count = int(len(maskable_pos) * mask_rate)
            mask_pos = np.random.choice(maskable_pos, mask_count, replace=False)
            for pos in mask_pos:
                extra_mlm_labels[idx][pos] = input_ids[idx][pos]
                input_ids[idx][
                    pos
                ] = mask_id  # does this actually propagate mask to dictionary?

        batch["extra_mlm_labels"] = extra_mlm_labels
