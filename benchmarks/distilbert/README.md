# DistilBERT
- Original code from https://github.com/huggingface/transformers


## Configure
- Prepare `data/binarized_text.bert-base-uncased.pickle` and `data/token_counts.bert-base-uncased.pickle` per [guide in the original repository](https://github.com/huggingface/transformers/tree/main/examples/research_projects/distillation).
- Setup `tspipe.yaml` : TSPipe related configuration.

## How to run
```bash
python train.py \
    --student_type=distilbert --student_config=training_configs/distilbert-large-uncased.json \
    --teacher_type=bert --teacher_name=bert-base-uncased --teacher_config=training_configs/bert-large-uncased.json \
    --alpha_ce=5.0 --alpha_mlm=2.0 --alpha_cos=1.0 --alpha_clm=0.0 --mlm --freeze_pos_embs \
    --dump_path=serialization_dir/my_first_training --data_file=data/binarized_text.bert-base-uncased.pickle \
    --token_counts=data/token_counts.bert-base-uncased.pickle --force --batch_size=24 \
    --tspipe-enable --tspipe-config=tspipe.yaml --ip=localhost --rank=0 --num-nodes=1
```