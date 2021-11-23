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
This script can be used to search the best hyper parameters for training.
"""
#!/opt/conda/bin/python

import os
import logging
import wandb
from argparse import ArgumentParser

from cli import parser, process_args
from utils import set_seed
from model import TransformerModelWrapper
from config import load_pet_configs
from data_utils import TRAIN_SET, DEV_SET, DEV32_SET, TEST_SET, load_examples, load_metrics

logger = logging.getLogger('sweep')

def main():
    # Initialize wandb
    run = wandb.init(reinit=True, sync_tensorboard=True)

    config = wandb.config
    task, seed, encoder_type = config['task'], config['seed_split'], config['encoder_type']
    lr, wd, bs = config['learning_rate'], config['weight_decay'], config['batch_size']
    #learning_rate_stage1 = config["learning_rate_stage1"]
    embed_learning_rate = config["embed_learning_rate"]

    ########################################
    # Prepare full arguments
    data_split = '16-%d' % seed
    if task == 'MNLI-mm':
        data_dir = os.path.join('data', 'k-shot', 'MNLI', data_split)
    elif task == 'RTE-glue':
        data_dir = os.path.join('data', 'k-shot', 'RTE', data_split)
    else:
        data_dir = os.path.join('data', 'k-shot', task, data_split)
    task_dir = os.path.join('output', task, 'tune', encoder_type)
    output_dir = os.path.join(task_dir, data_split)
    learning_rate_stage1 = 0
    arguments = ['--model_type', 'roberta',
                 '--embed_size', '1024',
                 '--do_train', '--do_eval',
                 '--eval_set', 'test',
                 '--overwrite_output_dir',
                 '--task_name', task,#.lower(),
                 '--data_dir', data_dir,
                 '--pet_max_steps', '400',
                 '--model_name_or_path', 'roberta-large-mnli',
                 '--cache_dir', 'pretrain/roberta-large-mnli',
                 '--pet_per_gpu_eval_batch_size', '8',
                 '--output_dir', output_dir,
                 '--learning_rate', str(lr),
                 '--learning_rate_stage1', str(learning_rate_stage1),
                 '--embed_learning_rate', str(embed_learning_rate),
                 '--weight_decay', str(wd),
                 '--prompt_encoder_type', encoder_type,
                '--entailment', str(1.0),
                '--use_prompt', str(1),
                '--train_prompt', str(1),
                #'--two_stage_train'
                #'--parameter_efficient'
                #  '--train_verbalizer', str(1)]
    ]

    if task in ['MNLI', 'MNLI-mm', 'SNLI', 'RTE-glue']:
        arguments.extend(['--pet_max_seq_length', '256',
                          '--pet_per_gpu_train_batch_size', str(bs),
                          '--pet_gradient_accumulation_steps', '2'])
    else:
        arguments.extend(['--pet_max_seq_length', '128',
                          '--pet_per_gpu_train_batch_size', str(bs),
                          '--pet_gradient_accumulation_steps', '1'])

    args = parser.parse_args(arguments)
    process_args(args)
    logger.info(args)

    ########################################
    # Load dataset
    train_data = load_examples(args.task_name, args.data_dir, TRAIN_SET,
                               num_examples=args.train_examples, split_examples_evenly=args.split_examples_evenly)
    eval_data = load_examples(args.task_name, args.data_dir, TEST_SET if args.eval_set == 'test' else DEV_SET,
                              num_examples=args.eval_examples, split_examples_evenly=args.split_examples_evenly)
    dev_data = load_examples(args.task_name, args.data_dir, DEV32_SET,
                             num_examples=args.dev_examples, split_examples_evenly=args.split_examples_evenly)

    ########################################
    # Training process
    set_seed(args.seed)

    # Load model
    model_config, train_config, eval_config = load_pet_configs(args)
    model = TransformerModelWrapper(model_config)

    # Train model
    if not args.two_stage_train:
        stage = 1 if args.parameter_efficient else 0
        model.train(train_data=train_data,
                    dev_data=dev_data,
                    eval_data=eval_data,
                    pattern_iter_output_dir=args.output_dir,
                    eval_config=eval_config,
                    per_gpu_train_batch_size=train_config.per_gpu_train_batch_size,
                    n_gpu=train_config.n_gpu,
                    num_train_epochs=train_config.num_train_epochs,
                    max_steps=args.pet_max_steps,
                    gradient_accumulation_steps=train_config.gradient_accumulation_steps,
                    weight_decay=args.weight_decay,
                    learning_rate=args.learning_rate,
                    embed_learning_rate = args.embed_learning_rate, # TODO use train config/cli?
                    fix_other_embeddings=False,
                    stage = stage,
                    wandb_log=True)

    else:
        # stage 1 stage 2
        model.train(train_data=train_data,
                    dev_data=dev_data,
                    eval_data=eval_data,
                    pattern_iter_output_dir=args.output_dir,
                    eval_config=eval_config,
                    per_gpu_train_batch_size=train_config.per_gpu_train_batch_size,
                    n_gpu=train_config.n_gpu,
                    num_train_epochs=train_config.num_train_epochs,
                    max_steps=args.pet_max_steps,
                    gradient_accumulation_steps=train_config.gradient_accumulation_steps,
                    weight_decay=args.weight_decay,
                    learning_rate=args.learning_rate_stage1,
                    embed_learning_rate = args.embed_learning_rate, # TODO use train config/cli?
                    fix_other_embeddings=False,
                    stage = 1,
                    wandb_log=True)

        model.train(train_data=train_data,
                    dev_data=dev_data,
                    eval_data=eval_data,
                    pattern_iter_output_dir=args.output_dir,
                    eval_config=eval_config,
                    per_gpu_train_batch_size=train_config.per_gpu_train_batch_size,
                    n_gpu=train_config.n_gpu,
                    num_train_epochs=train_config.num_train_epochs,
                    max_steps=args.pet_max_steps,
                    gradient_accumulation_steps=train_config.gradient_accumulation_steps,
                    weight_decay=args.weight_decay,
                    learning_rate=args.learning_rate,
                    embed_learning_rate = args.embed_learning_rate, # TODO use train config/cli?
                    fix_other_embeddings=False,
                    stage = 2,
                    wandb_log=True)


    run.finish()


if __name__ == '__main__':
    run_parser = ArgumentParser()
    run_parser.add_argument("--task",
                            type=str,
                            choices=['SST-2', 'sst-5', 'mr', 'cr', 'mpqa', 'subj', 'trec', 'CoLA',
                                     'MNLI', 'MNLI-mm', 'SNLI', 'QNLI', 'RTE-glue', 'MRPC', 'QQP'])
    run_parser.add_argument("--encoder",
                            type=str,
                            default='inner',
                            choices=['none', 'mlp', 'lstm', 'inner', 'inner2'])
    run_parser.add_argument("--seed_split",
                            type=int,
                            default=[],
                            nargs='+',
                            choices=[13, 21, 42, 87, 100])
    run_parser.add_argument("--batch_size",
                            type=int,
                            default=[],
                            nargs='+',
                            choices=[4, 8, 16, 24, 32])
    run_parser.add_argument("--sweep_id",
                            type=str,
                            default='')
    run_args = run_parser.parse_args()

    if not run_args.seed_split:  # Default search all seed splits
        run_args.seed_split = [13, 21, 87]# 100]

    if not run_args.batch_size:  # Default search all batch sizes
        if run_args.task in ['MNLI', 'MNLI-mm', 'SNLI', 'RTE-glue']:
            # Restrict maximum batch size due to memory limit
            run_args.batch_size = [4, 8, 16]
        else:
            run_args.batch_size = [16]# 16]

    # Prepare sweep config and get sweep id
    # TODO add control for other parameters
    # - model type
    # - # shot 8, 16, 32
    # - entailment yes /no 
    # New Hyperparameters
    # - use prompt
    # - num trainable tokens
    # - two sides
    sweep_config = {
        'program': run_args.task,
        'method': 'grid',
        'metric': {
            'goal': 'maximize',
            'name': 'eval-' + load_metrics(run_args.task)[-1]
        },
        'parameters': {
            'task': {'value': run_args.task},
            'encoder_type': {'value': run_args.encoder},
            'seed_split': {'values': run_args.seed_split},
            'learning_rate': {'values': [1e-5, 5e-5]},
            #'learning_rate_stage1' : {'values': [1e-5, 2e-5]},
            'embed_learning_rate' : {'values': [1e-5, 5e-5, 1e-4]},
            'weight_decay': {'values': [0.0, 0.05, 0.1]},
            'batch_size': {'values': run_args.batch_size}
        }
    }

    if run_args.sweep_id:  # Recover from old sweep
        sweep_id = run_args.sweep_id
    else:  # Create new sweep
        sweep_id = wandb.sweep(sweep_config, project="differentiable_entailment")

    # Sweep all hyper parameters
    wandb.agent(sweep_id, function=main, project = "differentiable_entailment")
