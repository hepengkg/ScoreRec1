import numpy as np
import pandas as pd
import argparse
import torch
from torch import nn
import torch.nn.functional as F
import os
import logging
import time as Time
from utility import calculate_hit, extract_axis_1,gini_coefficient
from collections import Counter
from Modules_ori import *


logging.getLogger().setLevel(logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description="SASRec baselines.")

    parser.add_argument('--epoch', type=int, default=500,
                        help='Number of max epochs.')
    parser.add_argument('--Model', type=str, default='SASRec',
                        help='model name.')

    parser.add_argument('--data', nargs='?', default='zhihu',
                        help='yoochoose, sports_and_outdoors,rr')

    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=64,
                        help='Number of hidden factors, i.e., embedding size.')
    parser.add_argument('--num_filters', type=int, default=16,
                        help='num_filters')
    parser.add_argument('--filter_sizes', nargs='?', default='[2,3,4]',
                        help='Specify the filter_size')
    parser.add_argument('--r_click', type=float, default=0.2,
                        help='reward for the click behavior.')
    parser.add_argument('--r_buy', type=float, default=1.0,
                        help='reward for the purchase behavior.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--model_name', type=str, default='SASRec_bce',
                        help='model name.')
    parser.add_argument('--save_flag', type=int, default=0,
                        help='0: Disable model saver, 1: Activate model saver')
    parser.add_argument('--cuda', type=int, default=0,
                        help='cuda device.')
    parser.add_argument('--l2_decay', type=float, default=1e-6,
                        help='l2 loss reg coef.')
    parser.add_argument('--alpha', type=float, default=0,
                        help='dro alpha.')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='for robust radius')
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                        help='dropout ')
    parser.add_argument('--descri', type=str, default='',
                        help='description of the work.')
    parser.add_argument('--random_seed', type=int, default=100,
                        help='random seed')
    return parser.parse_args()

args = parse_args()
import random
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(args.random_seed)

class SASRec(nn.Module):
    def __init__(self, hidden_size, item_num, state_size, dropout, device, num_heads=1):
        super(SASRec, self).__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.dropout = nn.Dropout(dropout)
        self.device = device
        self.item_embeddings = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=hidden_size,
        )
        nn.init.normal_(self.item_embeddings.weight, 0, 1)
        self.positional_embeddings = nn.Embedding(
            num_embeddings=state_size,
            embedding_dim=hidden_size
        )
        # emb_dropout is added
        self.emb_dropout = nn.Dropout(dropout)
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.ln_3 = nn.LayerNorm(hidden_size)
        self.mh_attn = MultiHeadAttention(hidden_size, hidden_size, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size, hidden_size, dropout)
        self.s_fc = nn.Linear(hidden_size, item_num)
        # self.ac_func = nn.ReLU()

    def forward(self, states, len_states):
        # inputs_emb = self.item_embeddings(states) * self.item_embeddings.embedding_dim ** 0.5
        inputs_emb = self.item_embeddings(states)
        inputs_emb += self.positional_embeddings(torch.arange(self.state_size).to(self.device))
        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1).to(self.device)
        seq *= mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        state_hidden = extract_axis_1(ff_out, len_states - 1)
        supervised_output = self.s_fc(state_hidden).squeeze()
        return supervised_output

    def forward_eval(self, states, len_states):
        # inputs_emb = self.item_embeddings(states) * self.item_embeddings.embedding_dim ** 0.5
        inputs_emb = self.item_embeddings(states)
        inputs_emb += self.positional_embeddings(torch.arange(self.state_size).to(self.device))
        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1).to(self.device)
        seq *= mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        state_hidden = extract_axis_1(ff_out, len_states - 1)
        supervised_output = self.s_fc(state_hidden).squeeze()
        test_item_emb = self.item_embeddings.weight  # 9000+1 64
        
        return supervised_output,test_item_emb



def evaluate(model, test_data, device, writer, VAL, itr):
    eval_data=pd.read_pickle(os.path.join(data_directory, test_data))
    batch_size = 100
    total_purchase = 0.0
    hit_purchase = [0, 0, 0, 0 , 0]
    ndcg_purchase = [0, 0, 0, 0, 0]

    seq, len_seq, target = list(eval_data['seq']), list(eval_data['len_seq']), list(eval_data['next'])

    num_total = len(seq)
    recommendations = {k: [] for k in topk}
    for i in range(num_total // batch_size):
        seq_b, len_seq_b, target_b = seq[i * batch_size: (i + 1)* batch_size], len_seq[i * batch_size: (i + 1)* batch_size], target[i * batch_size: (i + 1)* batch_size]
        states = np.array(seq_b)
        states = torch.LongTensor(states)
        states = states.to(device)

        prediction,test_item_emb = model.forward_eval(states, np.array(len_seq_b))
        _, topK = prediction.topk(100, dim=1, largest=True, sorted=True)
        topK = topK.cpu().detach().numpy()
        sorted_list2=np.flip(topK,axis=1)
        for k in topk:
            batch_recommendations = topK[:, :k].tolist()
            recommendations[k].extend(batch_recommendations)
        
        calculate_hit(sorted_list2,topk,target_b,hit_purchase,ndcg_purchase)

        total_purchase+=batch_size

    print('#############################################################')
    # logging.info('#############################################################')
    # print('total clicks: %d, total purchase:%d' % (total_clicks, total_purchase))
    # logging.info('total clicks: %d, total purchase:%d' % (total_clicks, total_purchase))
    hr_list = []
    ndcg_list = []
    # print('hr@{}\tndcg@{}\thr@{}\tndcg@{}\thr@{}\tndcg@{}'.format(topk[0], topk[0], topk[1], topk[1], topk[2], topk[2]))
    print('{:<10s} {:<10s} {:<10s} {:<10s} {:<10s} {:<10s} {:<10s} {:<10s}'.format('HR@' + str(topk[0]), 'NDCG@' + str(topk[0]),
                                                                   'HR@' + str(topk[1]), 'NDCG@' + str(topk[1]),
                                                                   'HR@' + str(topk[2]), 'NDCG@' + str(topk[2]),
                                                                    'HR@' + str(topk[3]), 'NDCG@' + str(topk[3])))
    
    # logging.info('#############################################################')
    for i in range(len(topk)):
        hr_purchase=hit_purchase[i]/total_purchase
        ng_purchase=ndcg_purchase[i]/total_purchase
        hr_list.append(hr_purchase)
        ndcg_list.append(ng_purchase[0,0])

        if i == 1:
            hr_20 = hr_purchase
    if VAL:
        writer.add_scalar(tag='Evaluation/hr_list5', scalar_value=hr_list[0], global_step=itr)
        writer.add_scalar(tag='Evaluation/ndcg_list5', scalar_value=(ndcg_list[0]), global_step=itr)
        writer.add_scalar(tag='Evaluation/hr_list10', scalar_value=hr_list[1], global_step=itr)
        writer.add_scalar(tag='Evaluation/ndcg_list10', scalar_value=(ndcg_list[1]), global_step=itr)
        writer.add_scalar(tag='Evaluation/hr_list20', scalar_value=hr_list[2], global_step=itr)
        writer.add_scalar(tag='Evaluation/ndcg_list20', scalar_value=(ndcg_list[2]), global_step=itr)
        writer.add_scalar(tag='Evaluation/hr_list50', scalar_value=hr_list[3], global_step=itr)
        writer.add_scalar(tag='Evaluation/ndcg_list50', scalar_value=(ndcg_list[3]), global_step=itr)
    else:
        metrics = {}
        raw_embeddings = test_item_emb.detach().cpu().numpy()
        # L2 norm
        norm = np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
        norm[norm == 0] = 1e-10  
        item_embeddings = raw_embeddings / norm
        for k in topk:
            user_lists = recommendations[k]
            if not user_lists:
                continue

            gini = gini_coefficient(user_lists, item_embeddings.shape[0])
        
            metrics[k] = {
                'Gini': gini
            }
        print("Diversity Metrics:")
        for k in topk:
            print(f"Gini={metrics[k]['Gini']:.4f}")

            writer.add_scalar(tag='Test/gini_{}'.format(k), scalar_value=(metrics[k]['Gini']), global_step=itr)
    
        writer.add_scalar(tag='Test/hr_list5', scalar_value=hr_list[0], global_step=itr)
        writer.add_scalar(tag='Test/ndcg_list5', scalar_value=(ndcg_list[0]), global_step=itr)
        writer.add_scalar(tag='Test/hr_list10', scalar_value=hr_list[1], global_step=itr)
        writer.add_scalar(tag='Test/ndcg_list10', scalar_value=(ndcg_list[1]), global_step=itr)
        writer.add_scalar(tag='Test/hr_list20', scalar_value=hr_list[2], global_step=itr)
        writer.add_scalar(tag='Test/ndcg_list20', scalar_value=(ndcg_list[2]), global_step=itr)
        writer.add_scalar(tag='Test/hr_list50', scalar_value=hr_list[3], global_step=itr)
        writer.add_scalar(tag='Test/ndcg_list50', scalar_value=(ndcg_list[3]), global_step=itr)

    print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(hr_list[0], (ndcg_list[0]), hr_list[1],
                                                                               (ndcg_list[1]), hr_list[2],
                                                                               (ndcg_list[2]), hr_list[3],
                                                                               (ndcg_list[3])))


    return hr_20


def calcu_propensity_score(buffer):
    items = list(buffer['next'])
    freq = Counter(items)
    for i in range(item_num):
        if i not in freq.keys():
            freq[i] = 0
    pop = [freq[i] for i in range(item_num)]
    pop = np.array(pop)
    ps = pop + 1
    ps = ps / np.sum(ps)
    ps = np.power(ps, 0.5)
    return ps



if __name__ == '__main__':

    args = parse_args()
    from torch.utils.tensorboard import SummaryWriter

    logdir = "./logs/Model_{}_data_{}".format(args.Model, args.data)
    logdir_pth = "./pth/Model_{}_data_{}".format(args.Model, args.data)
    if not os.path.exists(logdir_pth):
        os.mkdir(logdir_pth)

    writer = SummaryWriter(log_dir=logdir, flush_secs=5)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

    data_directory = './data/' + args.data
    data_statis = pd.read_pickle(
        os.path.join(data_directory, 'data_statis.df'))  # read data statistics, includeing seq_size and item_num
    seq_size = data_statis['seq_size'][0]  # the length of history to define the seq
    item_num = data_statis['item_num'][0]  # total number of items
    reward_click = args.r_click
    reward_buy = args.r_buy
    topk=[5, 10, 20, 50]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = SASRec(args.hidden_factor,item_num, seq_size, args.dropout_rate, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)
    bce_loss = nn.BCEWithLogitsLoss()

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total number of parameters: {total_params}')  # 646118

    model.to(device)
    # optimizer.to(device)

    train_data = pd.read_pickle(os.path.join(data_directory, 'train_data.df'))
    ps = calcu_propensity_score(train_data)
    ps = torch.tensor(ps)
    ps = ps.to(device)

    total_step=0
    hr_max = 0
    best_epoch = 0

    num_rows=train_data.shape[0]
    num_batches=int(num_rows/args.batch_size)
    for i in range(args.epoch):
        start_time = Time.time()
        for j in range(num_batches):
            batch = train_data.sample(n=args.batch_size).to_dict()
            seq = list(batch['seq'].values())
            len_seq = list(batch['len_seq'].values())
            target=list(batch['next'].values())

            target_neg = []
            for index in range(args.batch_size):
                neg=np.random.randint(item_num)
                while neg==target[index]:
                    neg = np.random.randint(item_num)
                target_neg.append(neg)
            optimizer.zero_grad()
            seq = torch.tensor(seq).long()
            len_seq = torch.tensor(len_seq).long()
            target = torch.tensor(target).long()
            target_neg = torch.LongTensor(target_neg) #256
            seq = seq.to(device)
            target = target.to(device)
            len_seq = len_seq.to(device)
            target_neg = target_neg.to(device)
            model_output = model.forward(seq, len_seq) #256 4838
            target = target.view(args.batch_size, 1)
            target_neg = target_neg.view(args.batch_size, 1)
            pos_scores = torch.gather(model_output, 1, target) #256 1
            neg_scores = torch.gather(model_output, 1, target_neg)
            pos_labels = torch.ones((args.batch_size, 1))
            neg_labels = torch.zeros((args.batch_size, 1))
            scores = torch.cat((pos_scores, neg_scores), 0)
            labels = torch.cat((pos_labels, neg_labels), 0)
            # print(labels)
            labels = labels.to(device)
            loss = bce_loss(scores, labels)
            loss_all = loss
            loss_all.backward()
            optimizer.step()

        # if i % 20 == 0:
        #     model_save_path = os.path.join(logdir_pth, f"model_epoch_{i}.pth")
        #     torch.save(model.state_dict(), model_save_path)
        #     print(f"Model saved at epoch {i} to {model_save_path}")

        if True:
            model.eval()
            # ema.set(model)
            if i % 1 == 0:
                print("Epoch {:03d}; ".format(i) + 'Train loss: {:.4f}; '.format(
                    loss_all) + "Time cost: " + Time.strftime(
                    "%H: %M: %S", Time.gmtime(Time.time() - start_time)))

            if (i + 1) % 10 == 0:
                eval_start = Time.time()
                print('-------------------------- VAL PHRASE --------------------------')
                VAL = True
                _ = evaluate(model, 'val_data.df', device, writer, VAL, i)
                print('-------------------------- TEST PHRASE -------------------------')
                VAL = False
                _ = evaluate(model, 'test_data.df', device, writer, VAL, i)
                print("Evalution cost: " + Time.strftime("%H: %M: %S", Time.gmtime(Time.time() - eval_start)))
                print('----------------------------------------------------------------')
            # ema.restore(model)