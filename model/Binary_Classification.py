import json
import os
import shutil
from copy import deepcopy
from typing import Dict
import math
from torch.optim import adagrad
import numpy as np
import torch
import torch.nn as nn
from allennlp.common import Params
from sklearn.utils import shuffle
from tqdm import tqdm
from torchcontrib.optim import SWA

from Transparency.model.modules.Decoder import AttnDecoder
from Transparency.model.modules.Encoder import Encoder

from .modelUtils import BatchHolder, get_sorting_index_with_noise_from_lengths, BatchHolderIndentity
from .modelUtils import jsd as js_divergence

file_name = os.path.abspath(__file__)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import pickle
dataset_vec = pickle.load(open('./preprocess/MIMIC/vec_diabetes.p', 'rb'))

from lime.lime_text import LimeTextExplainer

class AdversaryMulti(nn.Module) :
    def __init__(self, decoder=None) :
        super().__init__()
        self.decoder = decoder
        self.K = 5

    def forward(self, data) :
        data.hidden_volatile = data.hidden.detach()

        new_attn = torch.log(data.generate_uniform_attn()).unsqueeze(1).repeat(1, self.K, 1) #(B, 10, L)
        new_attn = new_attn + torch.randn(new_attn.size()).to(device)*3

        new_attn.requires_grad = True

        data.log_attn_volatile = new_attn
        optim = torch.optim.Adam([data.log_attn_volatile], lr=0.01, amsgrad=True)

        for _ in range(500) :
            log_attn = data.log_attn_volatile + 1 - 1
            log_attn.masked_fill_(data.masks.unsqueeze(1), -float('inf'))
            data.attn_volatile = nn.Softmax(dim=-1)(log_attn) #(B, 10, L)
            self.decoder.get_output(data)
            predict_new = data.predict_volatile #(B, 10, O)

            y_diff = torch.sigmoid(predict_new) - torch.sigmoid(data.predict.detach()).unsqueeze(1) #(B, 10, O)
            diff = nn.ReLU()(torch.abs(y_diff).sum(-1, keepdim=True) - 1e-2) #(B, 10, 1)

            jsd = js_divergence(data.attn_volatile, data.attn.detach().unsqueeze(1)) #(B, 10, 1)
            cross_jsd = js_divergence(data.attn_volatile.unsqueeze(1), data.attn_volatile.unsqueeze(2))

            loss =  -(jsd**1) + 500 * diff
            loss = loss.sum() - cross_jsd.sum(0).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()

        log_attn = data.log_attn_volatile + 1 - 1
        log_attn.masked_fill_(data.masks.unsqueeze(1), -float('inf'))
        data.attn_volatile = nn.Softmax(dim=-1)(log_attn)
        self.decoder.get_output(data)
        data.predict_volatile = torch.sigmoid(data.predict_volatile)

class Model() :
    def __init__(self, configuration, pre_embed=None) :
        configuration = deepcopy(configuration)
        self.configuration = deepcopy(configuration)

        configuration['model']['encoder']['pre_embed'] = pre_embed
        self.encoder = Encoder.from_params(Params(configuration['model']['encoder'])).to(device)

        configuration['model']['decoder']['hidden_size'] = self.encoder.output_size
        self.decoder = AttnDecoder.from_params(Params(configuration['model']['decoder'])).to(device)

        self.encoder_params = list(self.encoder.parameters())
        self.attn_params = list([v for k, v in self.decoder.named_parameters() if 'attention' in k])
        self.decoder_params = list([v for k, v in self.decoder.named_parameters() if 'attention' not in k])

        self.bsize = configuration['training']['bsize']

        weight_decay = configuration['training'].get('weight_decay', 1e-5)
        self.encoder_optim = torch.optim.Adam(self.encoder_params, lr=0.001, weight_decay=weight_decay, amsgrad=True)
        self.attn_optim = torch.optim.Adam(self.attn_params, lr=0.001, weight_decay=0, amsgrad=True)
        self.decoder_optim = torch.optim.Adam(self.decoder_params, lr=0.001, weight_decay=weight_decay, amsgrad=True)
        self.adversarymulti = AdversaryMulti(decoder=self.decoder)

        self.all_params = self.encoder_params + self.attn_params + self.decoder_params
        self.all_optim = torch.optim.Adam(self.all_params, lr=0.001, weight_decay=weight_decay, amsgrad=True)
        # self.all_optim = adagrad.Adagrad(self.all_params, weight_decay=weight_decay)


        pos_weight = configuration['training'].get('pos_weight', [1.0]*self.decoder.output_size)
        self.pos_weight = torch.Tensor(pos_weight).to(device)
        self.criterion = nn.BCEWithLogitsLoss(reduction='none').to(device)
        self.swa_settings = configuration['training']['swa']

        import time
        dirname = configuration['training']['exp_dirname']
        basepath = configuration['training'].get('basepath', 'outputs')
        self.time_str = time.ctime().replace(' ', '_')
        self.dirname = os.path.join(basepath, dirname, self.time_str)

        self.temperature = configuration['training']['temperature']
        self.train_losses = []

        if self.swa_settings[0]:
            # self.attn_optim = SWA(self.attn_optim, swa_start=3, swa_freq=1, swa_lr=0.05)
            # self.decoder_optim = SWA(self.decoder_optim, swa_start=3, swa_freq=1, swa_lr=0.05)
            # self.encoder_optim = SWA(self.encoder_optim, swa_start=3, swa_freq=1, swa_lr=0.05)
            self.swa_all_optim = SWA(self.all_optim)
            self.running_norms = []

    @classmethod
    def init_from_config(cls, dirname, **kwargs) :
        config = json.load(open(dirname + '/config.json', 'r'))
        config.update(kwargs)
        obj = cls(config)
        obj.load_values(dirname)
        return obj

    def get_param_buffer_norms(self):
        for p in self.swa_all_optim.param_groups[0]['params']:
            param_state = self.swa_all_optim.state[p]
            if 'swa_buffer' not in param_state:
                self.swa_all_optim.update_swa()

        norms = []
        # for p in np.array(self.swa_all_optim.param_groups[0]['params'])[[1, 2, 5, 6, 9]]:
        for p in np.array(self.swa_all_optim.param_groups[0]['params'])[[6, 9]]:
            param_state = self.swa_all_optim.state[p]
            buf = np.squeeze(
                param_state['swa_buffer'].cpu().numpy())
            cur_state = np.squeeze(p.data.cpu().numpy())
            norm = np.linalg.norm(buf - cur_state)
            norms.append(norm)
        if self.swa_settings[3] == 2:
            return np.max(norms)
        return np.mean(norms)

    def total_iter_num(self):
        return self.swa_all_optim.param_groups[0]['step_counter']

    def iter_for_swa_update(self, iter_num):
        return iter_num > self.swa_settings[1] \
               and iter_num % self.swa_settings[2] == 0

    def check_and_update_swa(self):
        if self.iter_for_swa_update(self.total_iter_num()):
            cur_step_diff_norm = self.get_param_buffer_norms()
            if self.swa_settings[3] == 0:
                self.swa_all_optim.update_swa()
                return
            if not self.running_norms:
                running_mean_norm = 0
            else:
                running_mean_norm = np.mean(self.running_norms)

            if cur_step_diff_norm > running_mean_norm:
                self.swa_all_optim.update_swa()
                self.running_norms = [cur_step_diff_norm]
            elif cur_step_diff_norm > 0:
                self.running_norms.append(cur_step_diff_norm)

    def train(self, data_in, target_in, train=True) :
        sorting_idx = get_sorting_index_with_noise_from_lengths([len(x) for x in data_in], noise_frac=0.1)
        data = [data_in[i] for i in sorting_idx]
        target = [target_in[i] for i in sorting_idx]

        self.encoder.train()
        self.decoder.train()
        bsize = self.bsize
        N = len(data)
        loss_total = 0

        batches = list(range(0, N, bsize))
        batches = shuffle(batches)

        for n in tqdm(batches) :
            torch.cuda.empty_cache()
            batch_doc = data[n:n+bsize]
            batch_data = BatchHolder(batch_doc)

            self.encoder(batch_data)
            self.decoder(batch_data)

            batch_target = target[n:n+bsize]
            batch_target = torch.Tensor(batch_target).to(device)

            if len(batch_target.shape) == 1 : #(B, )
                batch_target = batch_target.unsqueeze(-1) #(B, 1)

            bce_loss = self.criterion(batch_data.predict / self.temperature, batch_target)
            weight = batch_target * self.pos_weight + (1 - batch_target)
            bce_loss = (bce_loss * weight).mean(1).sum()

            loss = bce_loss
            self.train_losses.append(bce_loss.detach().cpu().numpy() + 0)

            if hasattr(batch_data, 'reg_loss') :
                loss += batch_data.reg_loss

            if train :
                if self.swa_settings[0]:
                    self.check_and_update_swa()

                    self.swa_all_optim.zero_grad()
                    loss.backward()
                    self.swa_all_optim.step()

                else:
                    # self.encoder_optim.zero_grad()
                    # self.decoder_optim.zero_grad()
                    # self.attn_optim.zero_grad()
                    self.all_optim.zero_grad()
                    loss.backward()
                    # self.encoder_optim.step()
                    # self.decoder_optim.step()
                    # self.attn_optim.step()
                    self.all_optim.step()

            loss_total += float(loss.data.cpu().item())
        if self.swa_settings[0] and self.swa_all_optim.param_groups[0][
            'step_counter'] > self.swa_settings[1]:
            print("\nSWA swapping\n")
            # self.attn_optim.swap_swa_sgd()
            # self.encoder_optim.swap_swa_sgd()
            # self.decoder_optim.swap_swa_sgd()
            self.swa_all_optim.swap_swa_sgd()
            self.running_norms = []


        return loss_total*bsize/N

    def predictor(self, inp_text_permutations):

        text_permutations = [dataset_vec.map2idxs(x.split()) for x in inp_text_permutations]
        text_permutations = BatchHolder(text_permutations)
        self.encoder(text_permutations)
        self.decoder(text_permutations)
        text_permutations.predict = torch.sigmoid(text_permutations.predict)
        pred = text_permutations.predict.cpu().data.numpy()
        for i in range(len(pred)):
            if math.isnan(pred[i][0]):
                pred[i][0] = 0.5

        ret_val = [[pred_i[0], 1-pred_i[0]] for pred_i in pred]
        ret_val = np.array(ret_val)

        return ret_val

    def evaluate(self, data) :
        self.encoder.eval()
        self.decoder.eval()
        bsize = self.bsize
        N = len(data)

        outputs = []
        attns = []

        for n in tqdm(range(0, N, bsize)) :
            torch.cuda.empty_cache()
            batch_doc = data[n:n+bsize]
            batch_data = BatchHolder(batch_doc)

            self.encoder(batch_data)
            self.decoder(batch_data)

            batch_data.predict = torch.sigmoid(batch_data.predict / self.temperature)
            if self.decoder.use_attention :
                attn = batch_data.attn.cpu().data.numpy()
                attns.append(attn)

            predict = batch_data.predict.cpu().data.numpy()
            outputs.append(predict)

        outputs = [x for y in outputs for x in y]
        if self.decoder.use_attention :
            attns = [x for y in attns for x in y]
        return outputs, attns

    def get_lime_explanations(self, data):
        explanations = []
        explainer = LimeTextExplainer(class_names=["A", "B"])
        for data_i in data:
            sentence = ' '.join(dataset_vec.map2words(data_i))
            exp = explainer.explain_instance(
                text_instance=sentence,
                classifier_fn=self.predictor,
                num_features=len(data_i),
                num_samples=5000).as_list()
            explanations.append(exp)
        return explanations

    def gradient_mem(self, data) :
        self.encoder.train()
        self.decoder.train()
        bsize = self.bsize
        N = len(data)

        grads = {'XxE' : [], 'XxE[X]' : [], 'H' : []}

        for n in tqdm(range(0, N, bsize)) :
            torch.cuda.empty_cache()
            batch_doc = data[n:n+bsize]

            grads_xxe = []
            grads_xxex = []
            grads_H = []

            for i in range(self.decoder.output_size) :
                batch_data = BatchHolder(batch_doc)
                batch_data.keep_grads = True
                batch_data.detach = True

                self.encoder(batch_data)
                self.decoder(batch_data)

                torch.sigmoid(batch_data.predict[:, i]).sum().backward()
                g = batch_data.embedding.grad
                em = batch_data.embedding
                g1 = (g * em).sum(-1)

                grads_xxex.append(g1.cpu().data.numpy())

                g1 = (g * self.encoder.embedding.weight.sum(0)).sum(-1)
                grads_xxe.append(g1.cpu().data.numpy())

                g1 = batch_data.hidden.grad.sum(-1)
                grads_H.append(g1.cpu().data.numpy())

            grads_xxe = np.array(grads_xxe).swapaxes(0, 1)
            grads_xxex = np.array(grads_xxex).swapaxes(0, 1)
            grads_H = np.array(grads_H).swapaxes(0, 1)

            import ipdb; ipdb.set_trace()
            grads['XxE'].append(grads_xxe)
            grads['XxE[X]'].append(grads_xxex)
            grads['H'].append(grads_H)

        for k in grads :
            grads[k] = [x for y in grads[k] for x in y]

        return grads

    def remove_and_run(self, data) :
        self.encoder.train()
        self.decoder.train()
        bsize = self.bsize
        N = len(data)

        outputs = []

        for n in tqdm(range(0, N, bsize)) :
            batch_doc = data[n:n+bsize]
            batch_data = BatchHolder(batch_doc)
            po = np.zeros((batch_data.B, batch_data.maxlen, self.decoder.output_size))

            for i in range(1, batch_data.maxlen - 1) :
                batch_data = BatchHolder(batch_doc)

                batch_data.seq = torch.cat([batch_data.seq[:, :i], batch_data.seq[:, i+1:]], dim=-1)
                batch_data.lengths = batch_data.lengths - 1
                batch_data.masks = torch.cat([batch_data.masks[:, :i], batch_data.masks[:, i+1:]], dim=-1)

                self.encoder(batch_data)
                self.decoder(batch_data)

                po[:, i] = torch.sigmoid(batch_data.predict).cpu().data.numpy()

            outputs.append(po)

        outputs = [x for y in outputs for x in y]

        return outputs

    def permute_attn(self, data, num_perm=100) :
        self.encoder.train()
        self.decoder.train()
        bsize = self.bsize
        N = len(data)

        permutations = []

        for n in tqdm(range(0, N, bsize)) :
            torch.cuda.empty_cache()
            batch_doc = data[n:n+bsize]
            batch_data = BatchHolder(batch_doc)

            batch_perms = np.zeros((batch_data.B, num_perm, self.decoder.output_size))

            self.encoder(batch_data)
            self.decoder(batch_data)

            for i in range(num_perm) :
                batch_data.permute = True
                self.decoder(batch_data)
                output = torch.sigmoid(batch_data.predict)
                batch_perms[:, i] = output.cpu().data.numpy()

            permutations.append(batch_perms)

        permutations = [x for y in permutations for x in y]

        return permutations

    def save_values(self, use_dirname=None, save_model=True, append_to_dir_name='') :
        if use_dirname is not None :
            dirname = use_dirname
        else :
            dirname = self.dirname + append_to_dir_name
            self.last_epch_dirname = dirname
        os.makedirs(dirname, exist_ok=True)
        shutil.copy2(file_name, dirname + '/')
        json.dump(self.configuration, open(dirname + '/config.json', 'w'))

        if save_model :
            torch.save(self.encoder.state_dict(), dirname + '/enc.th')
            torch.save(self.decoder.state_dict(), dirname + '/dec.th')

        return dirname

    def load_values(self, dirname) :
        self.encoder.load_state_dict(torch.load(dirname + '/enc.th', map_location={'cuda:1': 'cuda:0'}))
        self.decoder.load_state_dict(torch.load(dirname + '/dec.th', map_location={'cuda:1': 'cuda:0'}))

    def adversarial_multi(self, data) :
        self.encoder.eval()
        self.decoder.eval()

        for p in self.encoder.parameters() :
            p.requires_grad = False

        for p in self.decoder.parameters() :
            p.requires_grad = False

        bsize = self.bsize
        N = len(data)

        adverse_attn = []
        adverse_output = []

        for n in tqdm(range(0, N, bsize)) :
            torch.cuda.empty_cache()
            batch_doc = data[n:n+bsize]
            batch_data = BatchHolder(batch_doc)

            self.encoder(batch_data)
            self.decoder(batch_data)

            self.adversarymulti(batch_data)

            attn_volatile = batch_data.attn_volatile.cpu().data.numpy() #(B, 10, L)
            predict_volatile = batch_data.predict_volatile.cpu().data.numpy() #(B, 10, O)

            adverse_attn.append(attn_volatile)
            adverse_output.append(predict_volatile)

        adverse_output = [x for y in adverse_output for x in y]
        adverse_attn = [x for y in adverse_attn for x in y]

        return adverse_output, adverse_attn

    def logodds_attention(self, data, logodds_map:Dict) :
        self.encoder.eval()
        self.decoder.eval()

        bsize = self.bsize
        N = len(data)

        adverse_attn = []
        adverse_output = []

        logodds = np.zeros((self.encoder.vocab_size, ))
        for k, v in logodds_map.items() :
            if v is not None :
                logodds[k] = abs(v)
            else :
                logodds[k] = float('-inf')
        logodds = torch.Tensor(logodds).to(device)

        for n in tqdm(range(0, N, bsize)) :
            torch.cuda.empty_cache()
            batch_doc = data[n:n+bsize]
            batch_data = BatchHolder(batch_doc)

            self.encoder(batch_data)
            self.decoder(batch_data)

            attn = batch_data.attn #(B, L)
            batch_data.attn_logodds = logodds[batch_data.seq]
            self.decoder.get_output_from_logodds(batch_data)

            attn_volatile = batch_data.attn_volatile.cpu().data.numpy() #(B, L)
            predict_volatile = torch.sigmoid(batch_data.predict_volatile).cpu().data.numpy() #(B, O)

            adverse_attn.append(attn_volatile)
            adverse_output.append(predict_volatile)

        adverse_output = [x for y in adverse_output for x in y]
        adverse_attn = [x for y in adverse_attn for x in y]

        return adverse_output, adverse_attn

    def logodds_substitution(self, data, top_logodds_words:Dict) :
        self.encoder.eval()
        self.decoder.eval()

        bsize = self.bsize
        N = len(data)

        adverse_X = []
        adverse_attn = []
        adverse_output = []

        words_neg = torch.Tensor(top_logodds_words[0][0]).long().cuda().unsqueeze(0)
        words_pos = torch.Tensor(top_logodds_words[0][1]).long().cuda().unsqueeze(0)

        words_to_select = torch.cat([words_neg, words_pos], dim=0) #(2, 5)

        for n in tqdm(range(0, N, bsize)) :
            torch.cuda.empty_cache()
            batch_doc = data[n:n+bsize]
            batch_data = BatchHolder(batch_doc)

            self.encoder(batch_data)
            self.decoder(batch_data)
            predict_class = (torch.sigmoid(batch_data.predict).squeeze(-1) > 0.5)*1 #(B,)

            attn = batch_data.attn #(B, L)
            top_val, top_idx = torch.topk(attn, 5, dim=-1)
            subs_words = words_to_select[1 - predict_class.long()] #(B, 5)

            batch_data.seq.scatter_(1, top_idx, subs_words)

            self.encoder(batch_data)
            self.decoder(batch_data)

            attn_volatile = batch_data.attn.cpu().data.numpy() #(B, L)
            predict_volatile = torch.sigmoid(batch_data.predict).cpu().data.numpy() #(B, O)
            X_volatile = batch_data.seq.cpu().data.numpy()

            adverse_X.append(X_volatile)
            adverse_attn.append(attn_volatile)
            adverse_output.append(predict_volatile)

        adverse_X = [x for y in adverse_X for x in y]
        adverse_output = [x for y in adverse_output for x in y]
        adverse_attn = [x for y in adverse_attn for x in y]

        return adverse_output, adverse_attn, adverse_X

    def predict(self, batch_data, lengths, masks):
        batch_holder = BatchHolderIndentity(batch_data, lengths, masks)
        self.encoder(batch_holder)
        self.decoder(batch_holder)
        # batch_holder.predict = torch.sigmoid(batch_holder.predict)
        predict = batch_holder.predict
        return predict
