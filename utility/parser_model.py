import argparse
import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Run Model.")
    parser.add_argument('--seed', type=int, default=2023, help='Random seed.')

    parser.add_argument('--model_input_data_dir', nargs='?', default='./file/mimic/',
                        help='Input model input data path')
    parser.add_argument('--task', nargs='?', default='mimic',
                        help='Task.')
    parser.add_argument('--cuda_choice', nargs='?', default='cuda:0',
                        help='GPU choice.')
    parser.add_argument('--ablation', default=[],
                        choices=['no_patient_memory'], nargs='*', help='run ablation test')

    parser.add_argument('--use_pretrain', type=int, default=0,
                        help='Whether to use pretrain word embedding.')


    parser.add_argument('--pretrain_embedding_path', nargs='?', default='pretrain_word_embedding.npy',
                        help='Pretrained word embeddinng path')
    parser.add_argument('--sequence_batch_size', type=int, default=64,
                        help='sequence batch size.')
    parser.add_argument('--pretrain_batch_size', type=int, default=64,
                        help='pretrain batch size.')


    parser.add_argument('--word_embedding_size', type=int, default=256,
                        help='Word Embedding size.')
    parser.add_argument('--text_encoder_head_num', type=int, default=8,
                        help='Text Encoder Head Num')
    parser.add_argument('--text_encoder_layer', type=int, default=1,
                        help='Text Encoder Layer')
    parser.add_argument('--text_dropout_prob', type=float, default=0.0,
                        help='Text Encoder Dropout rate')


    parser.add_argument('--hita_dropout_prob', type=float, default=0.0,
                        help='HitaNet Dropout rate')
    parser.add_argument('--hita_encoder_layer', type=int, default=1,
                        help='HitaNet Encoder layer')
    parser.add_argument('--hita_encoder_head_num', type=int, default=4,
                        help='HitaNet Encoder Head Num')
    parser.add_argument('--hita_encoder_ffn_size', type=int, default=1024,
                        help='HitaNet Encoder ffn size')
    parser.add_argument('--global_query_size', type=int, default=128,
                        help='HitaNet Global query size.')
    parser.add_argument('--hita_input_size', type=int, default=256,
                        help='HitaNet Input size')
    parser.add_argument('--hita_time_selection_layer_global_embed', type=int, default=128,
                        help='HitaNet Global time selection layer.')
    parser.add_argument('--hita_time_selection_layer_encoder_embed', type=int, default=128,
                        help='HitaNet Encoder time selection layer.')
    parser.add_argument('--hita_time_scale', type=int, default=180,
                        help='HitaNet Time Scale')


    parser.add_argument('--fusing_encoder_head_num', type=int, default=4,
                        help='Fusing Encoder Head Num')
    parser.add_argument('--fusing_encoder_layer', type=int, default=1,
                        help='Fusing Encoder Layer')
    parser.add_argument('--fusing_dropout_prob', type=float, default=0.0,
                        help='Fusing Encoder Dropout rate')


    parser.add_argument('--text_mask_prob', type=float, default=0.15,
                        help='Text mask probability.')
    parser.add_argument('--text_mask_keep_rand', nargs='?', default='[0.8,0.1,0.1]',
                        help='Probability for text mask/keep/random.')
    parser.add_argument('--visit_mask_prob', type=float, default=0.15,
                        help='Visit mask probability.')
    parser.add_argument('--visit_mask_keep_rand', nargs='?', default='[0.8,0.1,0.1]',
                        help='Probability for visit mask/keep/random.')


    parser.add_argument('--projection_dim', type=int, default=256,
                        help='Common space dim')


    parser.add_argument('--base_lr', type=float, default=2e-5,
                        help='Base Learning rate.')
    parser.add_argument('--visit_encoder_lr', type=float, default=2e-5,
                        help='Visit Encoder Learning rate.')
    parser.add_argument('--text_encoder_lr', type=float, default=2e-5,
                        help='Text Encoder Learning rate.')
    parser.add_argument('--n_epoch_pretrain_cl', type=int, default=50,
                        help='Number of pretraining epoch for cl.')
    parser.add_argument('--n_epoch_pretrain_mlm', type=int, default=100,
                        help='Number of pretraining epoch for mlm.')
    parser.add_argument('--n_epoch', type=int, default=300,
                        help='Number of epoch.')
    parser.add_argument('--clip', type=int, default=5,
                        help='Clip Value for gradient.')
    parser.add_argument('--train_dropout_rate', type=float, default=0.5,
                        help='Train Dropout rate')

    parser.add_argument('--pretrain_lr', type=float, default=1e-3,
                        help='Pretrain Learning rate.')
    parser.add_argument('--pretrain_weight_decay', type=float, default=1e-5,
                        help='Pretrain weight decay.')
    parser.add_argument('--train_weight_decay', type=float, default=1e-5,
                        help='Train weight decay.')


    parser.add_argument('--cf_print_every', type=int, default=1,
                        help='Iter interval of printing sequence loss.')
    parser.add_argument('--kg_print_every', type=int, default=1,
                        help='Iter interval of printing KG loss.')
    parser.add_argument('--evaluate_every', type=int, default=1,
                        help='Epoch interval of evaluating multi-label.')
    parser.add_argument('--unfreeze_epoch', default=3, type=int)
    parser.add_argument('--max_epochs_before_stop', default=30, type=int,
                        help='stop training if dev does not increase for N epochs')

    parser.add_argument('--gwd_tau', type=float, default=5.0,
                        help='Temperature for gwd.')
    parser.add_argument('--local_kernel_size', type=int, default=3,
                        help='Local kernel size.')
    args = parser.parse_args()

    save_dir = 'trained_model/{}/{}/seed{}_{}/'.format(
        args.task, str(datetime.datetime.now().strftime('%Y-%m-%d')),
        args.seed,args.cuda_choice)
    args.save_dir = save_dir

    return args