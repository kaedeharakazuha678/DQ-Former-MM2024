import argparse
import os
import sys
import time
import torch
import wandb

from transformers import (
    HfArgumentParser,
    TrainingArguments, 
    EarlyStoppingCallback, 
    set_seed, 
    Trainer)

from utils.show_utils import print_trainable_params, print_args, get_new_filename
from utils.constant_utils import *
from utils.metrics_utils import MyComputeMetrics
# from utils.trainer import CustomTrainer as Trainer
from my_models import load_model
from my_datasets import *
from utils.args_utils import my_args
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append(os.pardir) 

os.environ["WANDB_DISABLED"]="true"
def init_wandb(args):
    wandb_run_name = f"{args.wandb_run_name}-{args.seed}-{args.formatted_time}"
    wandb.init( project = args.wandb_project_name,
                entity='1404830922',
                dir = args.output_dir + 'wandb_logs',
                name = wandb_run_name,
                group= args.wandb_group_name,
               )


def main(args): 
    #region load model and dataset
    print(">>> Loading model...")
    if args.checkpoint_path is not None:
        checkpoint = torch.load(args.checkpoint_path)
        model = load_model(args)
        model.load_state_dict(checkpoint)
    else:
        model = load_model(args)
    print_trainable_params(model)

    print(">>> Loading dataset...")
    train_dataset, dev_dataset, test_dataset = load_dataset(args)
    #endregion

    #region Set training arguments
    print(">>> Initializing the trainer...")
    wandb_run_dir = os.path.join(args.output_dir+f"checkpoints/{args.dataset_name}_{args.class_type}_{args.label_type}/{args.model_name}-{args.modalities}/{args.wandb_run_name}")
    output_dir = os.path.join(wandb_run_dir, get_new_filename(wandb_run_dir, f"seed_{args.seed}"))
    training_args = TrainingArguments(
        output_dir=output_dir,         # output directory  
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        logging_dir= args.output_dir +'logs',
        logging_steps=10,
        save_steps=100,
        eval_steps=args.log_step,
        save_total_limit=args.save_total_limit,  # Only save the last model
        save_strategy="steps",
        evaluation_strategy="steps",  # Evaluate the model every `eval_steps`
        gradient_accumulation_steps= args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type= "linear",
        warmup_ratio=args.warmup_proportion,
        push_to_hub=False,
        report_to="wandb",
        seed=args.seed,
        bf16=False,
        fp16=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_F1",
        greater_is_better= True,
        local_rank=args.local_rank,
        save_safetensors=False,
    )    
    #endregion
    
    #region Set optimizer
    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if "linear_proj" not in n and p.requires_grad]},
        {'params': [p for n, p in model.named_parameters() if "linear_proj" in n and p.requires_grad], 'lr': args.learning_rate *10}
    ], lr=args.learning_rate, weight_decay=args.weight_decay)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    #endregion
    
    #region Set Trainer
    compute_metrics = MyComputeMetrics(args)
    data_collator = DataCollator(args)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        optimizers=(optimizer, None),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]
    )
    #endregion
    
    #region training and testing
    ignore_keys = ["return_dict"]
    if args.do_train:
        trainer.train(ignore_keys_for_eval=ignore_keys)
        eval_results = trainer.evaluate(ignore_keys=ignore_keys)
        print(eval_results)

    else:
        print(">>> Starting evaluation on test dataset...")
        trainer._load_from_checkpoint(args.checkpoint_path)
        eval_results = trainer.evaluate(eval_dataset=args.test_dataset, ignore_keys=ignore_keys)
        print(eval_results)
    #endregion


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training settings")

    #region basic settings
    parser.add_argument("--wandb_project_name", type=str, default='MM-ERC',help="wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default='MM-ERC',help="wandb run name")
    parser.add_argument("--wandb_group_name", type=str, default='MM-ERC',help="wandb group name")
    #endregion

    #region training settings
    parser.add_argument("--do_train", type=bool, default=True, help="Whether to train")
    parser.add_argument("--log_step", type=int, default=100, help="log steps")
    parser.add_argument("--batch_size", type=int, default=4, help="per device batch size")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--warmup_proportion", type=float, default=0.01, help="Warmup proportion")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay")
    parser.add_argument("--save_total_limit", type=int, default=1, help="Number of checkpoints to save")
    parser.add_argument("--early_stopping_patience", type=int, default=30, help="patience for early stopping")
    parser.add_argument("--local-rank", type=int, default=-1, help="Local rank for distributed training (-1: not distributed)")
    #endregion
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to checkpoint")   
    parser.add_argument("--wandb_disable", type=bool, default=False)
    args = my_args(parser).parse_args()  
    set_seed(args.seed)
    timestamp = time.time()
    local_time = time.localtime(timestamp)
    args.formatted_time = time.strftime("%Y%m%d_%H%M%S", local_time)
    if not args.wandb_disable:
        init_wandb(args)
    print_args(args)
    main(args)
    