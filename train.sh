python code/train.py --train_data_file_path data/GAIIC_NER/train.txt \
                     --model_name pretrain_model/nezha-cn-base \
                     --model_save_dir data/save_model \
                     --learning_rate 2e-5 \
                     --num_epoches 1 \
                     --batch_size 32 \
                     --warmup_proportion 0.1 \
                     --gradient_accumulation_steps 1 \
                     --max_grad_norm 1.0 \
                     --weight_decay 0.01 \
#                     --do_adv \
#                     --do_fake_label \
#                     --fake_train_data_path data/orther \
#                     --fake_train_data_name fake_train_data_20000.txt \
#                     --rdrop \
#                     --rdrop_rate 0.5 \
#                     --cuda_device 0 \
#                     --seed 42