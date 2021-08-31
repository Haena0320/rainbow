#anli
#CUDA_VISIBLE_DEVICES=2 python src/run_anli.py --data_dir ./dataset/aNLI --do_train --do_eval --max_seq_length 64 --do_lower_case --adam_epsilon 1e-8 --max_grad_norm 1. --weight_decay 0.01 --warmup_proportion 0.2 --learning_rate 1e-5 --num_train_epochs 3 --eval_steps 500 --train_batch_size 2 --eval_batch_size 4 --gradient_accumulation_steps 2 --model_class bert --model_name_or_path bert-base-uncased --output_dir anil_bert --seed 42 --overwrite_output_dir

#winogrande
CUDA_VISIBLE_DEVICES=2 python src/run_winogrande.py --data_dir ./dataset/WinoGrande --do_train --do_eval --max_seq_length 80 --do_lower_case --adam_epsilon 1e-8 --max_grad_norm 1. --weight_decay 0.01 --warmup_proportion 0.1 --learning_rate 1e-5 --num_train_epochs 3 --eval_steps 500 --train_batch_size 16 --eval_batch_size 4 --gradient_accumulation_steps 1 --model_class bert --model_name_or_path bert-base-uncased --output_dir winogrande_bert --seed 42 #--overwrite_output_dir

#hellaswage
#CUDA_VISIBLE_DEVICES=2 python src/run_hellaswag.py --data_dir ./dataset/HellaSWAG --do_train --do_eval --max_seq_length 128 --do_lower_case --adam_epsilon 1e-8 --max_grad_norm 1. --weight_decay 0.01 --warmup_proportion 0.2 --learning_rate 2e-5 --num_train_epochs 10 --eval_steps 500 --train_batch_size 64 --eval_batch_size 16 --gradient_accumulation_steps 1 --model_class bert --model_name_or_path bert-base-uncased --output_dir hellaswag_bert --seed 42 --overwrite_output_dir

# CosmosQA
#CUDA_VISIBLE_DEVICES=2 python src/run_cosmosqa.py --data_dir ./dataset/CosmosQA --do_train --do_eval --max_seq_length 128 --do_lower_case --adam_epsilon 1e-8 --max_grad_norm 1. --weight_decay 0.01 --warmup_proportion 0.1 --learning_rate 5e-5 --num_train_epochs 10 --eval_steps 500 --train_batch_size 32 --eval_batch_size 8 --gradient_accumulation_steps 1 --model_class bert --model_name_or_path bert-base-uncased --output_dir cosmosqa_bert --seed 42 --overwrite_output_dir

# SocialIQA
# learning rate {1e-5, 2e-5, 3e-5}
# batch size {3,4,8}
# epochs {3,4,10}
#CUDA_VISIBLE_DEVICES=2 python src/run_socialiqa.py --data_dir ./dataset/SocialIQa --do_train --do_eval --max_seq_length 128 --do_lower_case --adam_epsilon 1e-8 --max_grad_norm 1. --weight_decay 0.01 --warmup_proportion 0.1 --learning_rate 5e-5 --num_train_epochs 10 --eval_steps 500 --train_batch_size 8 --eval_batch_size 8 --gradient_accumulation_steps 1 --model_class bert --model_name_or_path bert-base-uncased --output_dir socialiqa_bert --seed 42 --overwrite_output_dir

# PhysicalIQA
# 하이퍼파라미터 못찾음
#CUDA_VISIBLE_DEVICES=2 python src/run_physicaliqa.py --data_dir ./dataset/PhysicalIQa --do_train --do_eval --max_seq_length 128 --do_lower_case --adam_epsilon 1e-8 --max_grad_norm 1. --weight_decay 0.01 --warmup_proportion 0.1 --learning_rate 5e-5 --num_train_epochs 10 --eval_steps 500 --train_batch_size 8 --eval_batch_size 8 --gradient_accumulation_steps 1 --model_class bert --model_name_or_path bert-base-uncased --output_dir physicaliqa_bert --seed 42 --overwrite_output_dir

