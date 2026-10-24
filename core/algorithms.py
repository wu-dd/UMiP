import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from . import networks

ALGORITHMS = [
    'PRODEN',
    'LWS',
    'CRDPLL',
    'PiCO',
    'DIRK',
    'ABLE',
]


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a partial-label learning algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    def __init__(self, input_shape, train_givenY, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams
        self.num_data = input_shape[0]
        self.num_classes = train_givenY.shape[1]

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError


class PRODEN(Algorithm):
    """
    PRODEN
    Reference: Progressive identification of true labels for partial-label learning, ICML 2020.
    """

    def __init__(self, input_shape, train_givenY, hparams):
        super(PRODEN, self).__init__(input_shape, train_givenY, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            self.num_classes)

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        tempY = train_givenY.sum(dim=1).unsqueeze(1).repeat(1, train_givenY.shape[1])
        label_confidence = train_givenY.float()/tempY
        self.label_confidence = label_confidence
        self.mode=self.hparams['mode']

    def update(self, minibatches,epoch=None):
        x, strong_x, partial_y, y, index = minibatches
        loss = self.rc_loss(self.predict(x), index)

        if self.mode=="UMiP":
            def UMiP_conf_proden_data(x, partialY, confidence,alpha=1):
                batch_size = x.size()[0]
                device = "cuda" if x.is_cuda else "cpu"
                lam = torch.tensor(np.random.beta(alpha, alpha, batch_size)).to(device)
                lam = torch.where(lam < 0.5, 1 - lam, lam)
                lam = torch.tensor(lam, dtype=x.dtype, device=x.device).view(batch_size, 1, 1, 1)
                index = torch.randperm(batch_size)
                mixed_x = lam * x + (1 - lam) * x[index, :]
                lam = lam.view(batch_size, 1)
                mixed_conf = lam * confidence + (1 - lam) * confidence[index, :]
                mixed_conf = mixed_conf * partialY
                base_value = mixed_conf.sum(dim=1).unsqueeze(1).repeat(1, mixed_conf.shape[1])
                mixed_confidence = mixed_conf / (base_value+1e-12)
                return mixed_x, mixed_confidence
            mixed_X, mixed_confidence = UMiP_conf_proden_data(x, partial_y, self.label_confidence[index])
            loss = loss + self.rc_loss(self.predict(mixed_X), index, mixed_confidence)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.confidence_update(x, partial_y, index)
        return {'loss': loss.item()}

    def rc_loss(self, outputs, index, batch_confidence=None):
        if batch_confidence is None:
            device = "cuda" if index.is_cuda else "cpu"
            self.label_confidence = self.label_confidence.to(device)
            batch_confidence = self.label_confidence[index]
        logsm_outputs = F.log_softmax(outputs, dim=1)
        final_outputs = logsm_outputs * batch_confidence
        average_loss = - ((final_outputs).sum(dim=1)).mean()
        return average_loss

    def predict(self, x):
        return self.network(x)

    def confidence_update(self, batchX, batchY, batch_index):
        with torch.no_grad():
            batch_outputs = self.predict(batchX)
            temp_un_conf = F.softmax(batch_outputs, dim=1)
            self.label_confidence[batch_index, :] = temp_un_conf * batchY # un_confidence stores the weight of each example
            base_value = self.label_confidence.sum(dim=1).unsqueeze(1).repeat(1, self.label_confidence.shape[1])
            self.label_confidence = self.label_confidence / (base_value+1e-12)


class LWS(Algorithm):
    """
    LWS
    Reference: Leveraged weighted loss for partial label learning, ICML 2021.
    """

    def __init__(self, input_shape, train_givenY, hparams):
        super(LWS, self).__init__(input_shape, train_givenY, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            self.num_classes)

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        label_confidence = torch.ones(train_givenY.shape[0], train_givenY.shape[1]) / train_givenY.shape[1]
        self.label_confidence = label_confidence
        self.mode = self.hparams['mode']

    def update(self, minibatches,epoch=None):
        x, strong_x, partial_y, y, index = minibatches
        device = "cuda" if index.is_cuda else "cpu"
        self.label_confidence = self.label_confidence.to(device)
        loss = self.lws_loss(self.predict(x), partial_y, index)

        if self.mode == "UMiP":
            def UMiP_conf_lw_data(x,partialY,confidence,alpha=1):
                batch_size = x.size()[0]
                lam = torch.tensor(np.random.beta(alpha, alpha, batch_size)).to(device)
                lam = torch.where(lam < 0.5, 1 - lam, lam)
                lam = torch.tensor(lam, dtype=x.dtype, device=x.device).view(batch_size, 1, 1, 1)
                index = torch.randperm(batch_size)
                mixed_x = lam * x + (1 - lam) * x[index, :]
                lam = lam.view(batch_size, 1)
                partial_confidence = confidence * partialY
                partial_mixed_partial_conf = lam * partial_confidence + (1 - lam) * partial_confidence[index, :]
                partial_mixed_conf = partial_mixed_partial_conf * partialY
                partial_base_value = partial_mixed_conf.sum(dim=1).unsqueeze(1).repeat(1, partial_mixed_conf.shape[1])
                partial_mixed_confidence = partial_mixed_conf / (partial_base_value+1e-12)
                comp_confidence = confidence * (1 - partialY)
                comp_mixed_comp_conf = lam * comp_confidence + (1 - lam) * comp_confidence[index, :]
                comp_mixed_conf = comp_mixed_comp_conf * (1 - partialY)
                comp_base_value = comp_mixed_conf.sum(dim=1).unsqueeze(1).repeat(1, comp_mixed_conf.shape[1])
                comp_mixed_confidence = comp_mixed_conf / (comp_base_value+1e-12)
                return mixed_x, partial_mixed_confidence, comp_mixed_confidence

            mixed_X, mixed_partial_confidence, mixed_comp_confidence = UMiP_conf_lw_data(x, partial_y, self.label_confidence[index])
            new_Y = torch.zeros_like(partial_y).cuda()
            new_Y[mixed_partial_confidence > 0] = 1
            average_loss1 = self.lws_loss(self.predict(mixed_X), new_Y, index, mixed_partial_confidence)
            new_Y = torch.ones_like(partial_y).cuda()
            new_Y[mixed_comp_confidence > 0] = 0
            average_loss2 = self.lws_loss(self.predict(mixed_X), new_Y, index, mixed_comp_confidence)
            loss = loss + average_loss1 + average_loss2

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.confidence_update(x, partial_y, index)
        return {'loss': loss.item()}

    def lws_loss(self, outputs, partialY, index, batch_confidence = None):
        device = "cuda" if outputs.is_cuda else "cpu"
        if batch_confidence is None:
            batch_confidence=self.label_confidence[index, :]

        onezero = torch.zeros(outputs.shape[0], outputs.shape[1])
        onezero[partialY > 0] = 1
        counter_onezero = 1 - onezero
        onezero = onezero.to(device)
        counter_onezero = counter_onezero.to(device)
        sm_outputs = F.softmax(outputs, dim=1)
        sig_loss1 = - torch.log(sm_outputs + 1e-8)
        l1 = batch_confidence * onezero * sig_loss1
        average_loss1 = torch.sum(l1) / l1.size(0)
        sig_loss2 = - torch.log(1 - sm_outputs + 1e-8)
        l2 = batch_confidence * counter_onezero * sig_loss2
        average_loss2 = torch.sum(l2) / l2.size(0)
        average_loss = average_loss1 + self.hparams["lw_weight"] * average_loss2
        return average_loss

    def predict(self, x):
        return self.network(x)

    def confidence_update(self, batchX, batchY, batch_index):
        with torch.no_grad():
            device = "cuda" if batch_index.is_cuda else "cpu"
            batch_outputs = self.predict(batchX)
            sm_outputs = F.softmax(batch_outputs, dim=1)
            onezero = torch.zeros(sm_outputs.shape[0], sm_outputs.shape[1])
            onezero[batchY > 0] = 1
            counter_onezero = 1 - onezero
            onezero = onezero.to(device)
            counter_onezero = counter_onezero.to(device)
            new_weight1 = sm_outputs * onezero
            new_weight1 = new_weight1 / (new_weight1 + 1e-8).sum(dim=1).repeat(
                self.label_confidence.shape[1], 1).transpose(0, 1)
            new_weight2 = sm_outputs * counter_onezero
            new_weight2 = new_weight2 / (new_weight2 + 1e-8).sum(dim=1).repeat(
                self.label_confidence.shape[1], 1).transpose(0, 1)
            new_weight = new_weight1 + new_weight2
            self.label_confidence[batch_index, :] = new_weight


class DIRK(Algorithm):
    """
    DIRK
    Reference: Distilling Reliable Knowledge for Instance-dependent Partial Label Learning, AAAI 2024
    """

    class tea_model(nn.Module):
        def __init__(self, num_classes,input_shape,hparams, base_encoder):
            super().__init__()
            self.encoder = base_encoder(num_classes,input_shape,hparams)
            self.register_buffer("queue_feat", torch.randn(hparams['moco_queue'], hparams['feat_dim']))
            self.register_buffer("queue_dist", torch.randn(hparams['moco_queue'], num_classes))
            self.register_buffer("queue_partY", torch.randn(hparams['moco_queue'], num_classes))
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
            self.queue_feat = F.normalize(self.queue_feat, dim=0)
            self.moco_queue = hparams['moco_queue']

        @torch.no_grad()
        def _dequeue_and_enqueue(self, keys_feat, keys_dist, keys_partY):
            batch_size = keys_feat.shape[0]
            ptr = int(self.queue_ptr)
            assert self.moco_queue % batch_size == 0
            self.queue_feat[ptr:ptr + batch_size] = keys_feat
            self.queue_dist[ptr:ptr + batch_size] = keys_dist
            self.queue_partY[ptr:ptr + batch_size] = keys_partY
            ptr = (ptr + batch_size) % self.moco_queue
            self.queue_ptr[0] = ptr

        def forward(self, img_w=None, img_s=None, img_distill=None, partY=None):
            with torch.no_grad():
                _, feat_k = self.encoder(img_w)
                output_k, _ = self.encoder(img_distill)
                output_k = torch.softmax(output_k, dim=1)
                output_k = self.get_correct_conf(output_k, partY)

            features = torch.cat((feat_k, self.queue_feat.clone().detach()), dim=0)
            partYs = torch.cat((partY, self.queue_partY.clone().detach()), dim=0)
            dists = torch.cat((output_k, self.queue_dist.clone().detach()), dim=0)
            self._dequeue_and_enqueue(feat_k, output_k, partY)
            return features, partYs, dists, output_k

        def get_correct_conf(self, un_conf, partY):
            part_confidence = un_conf * partY
            part_confidence = part_confidence / part_confidence.sum(dim=1).unsqueeze(1).repeat(1,part_confidence.shape[1])
            comp_confidence = un_conf * (1 - partY)
            comp_confidence = comp_confidence / (comp_confidence.sum(dim=1).unsqueeze(1).repeat(1, comp_confidence.shape[1]) + 1e-20)
            comp_max = comp_confidence.max(dim=1)[0].unsqueeze(1).repeat(1, partY.shape[1])
            part_min = ((1 - partY) + part_confidence).min(dim=1)[0].unsqueeze(1).repeat(1, partY.shape[1])
            fenmu = (un_conf * partY).sum(dim=1)
            M = 1.0 / fenmu
            M = M.unsqueeze(1).repeat(1, partY.shape[1])
            a = (M * comp_max) / (M * comp_max + part_min)
            a[a == 0] = 1
            rec_confidence = part_confidence * a + comp_confidence * (1 - a)
            return rec_confidence

    class stu_model(nn.Module):
        def __init__(self, num_classes,input_shape,hparams, base_encoder):
            super().__init__()
            self.encoder = base_encoder(num_classes,input_shape,hparams)
        def forward(self, img_s, img_distill, is_eval=False):
            output_s, _ = self.encoder(img_distill)
            if is_eval:
                return output_s
            _, feat_s = self.encoder(img_s)
            return output_s, feat_s

    class DIRKNet(nn.Module):
        def __init__(self, num_classes, input_shape, hparams):
            super().__init__()
            self.featurizer = networks.Featurizer(input_shape, hparams)
            self.classifier = networks.Classifier(
                self.featurizer.n_outputs,
                num_classes)
            self.head = nn.Sequential(
                nn.Linear(self.featurizer.n_outputs, self.featurizer.n_outputs),
                nn.ReLU(inplace=True),
                nn.Linear(self.featurizer.n_outputs, hparams['feat_dim']))
        def forward(self, x):
            feat = self.featurizer(x)
            feat_c = self.head(feat)
            logits = self.classifier(feat)
            return logits, F.normalize(feat_c, dim=1)

    def dirk_loss(self,output, confidence, Y=None):
        logsm_outputs = F.log_softmax(output, dim=1)
        final_outputs = logsm_outputs * confidence
        average_loss = - ((final_outputs).sum(dim=1)).mean()
        return average_loss

    class WeightedConLoss(nn.Module):
        def __init__(self, temperature=0.07, base_temperature=0.07, dist_temperature=0.07):
            super().__init__()
            self.temperature = temperature
            self.base_temperature = base_temperature
            self.dist_temperature = dist_temperature
        def forward(self, features, dist, mask=None, batch_size=-1):
            if mask is not None:
                mask = mask.float()
                anchor_dot_contrast = torch.div(torch.matmul(features[:batch_size], features.T),self.temperature)
                logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
                logits = anchor_dot_contrast - logits_max.detach()
                logits_mask = torch.scatter(torch.ones_like(anchor_dot_contrast),1,torch.arange(batch_size).view(-1, 1).cuda(),0)
                mask = logits_mask * mask
                dist_temperature = self.dist_temperature
                dist_norm = dist / torch.norm(dist, dim=-1, keepdim=True)
                anchor_dot_simi = torch.div(torch.matmul(dist_norm[:batch_size], dist_norm.T), dist_temperature)
                logits_simi_max, _ = torch.max(anchor_dot_simi, dim=1, keepdim=True)
                logits_simi = anchor_dot_simi - logits_simi_max.detach()
                exp_simi = torch.exp(logits_simi) * mask
                weight = exp_simi / exp_simi.sum(dim=1).unsqueeze(1).repeat(1, anchor_dot_simi.shape[1])
                exp_logits = torch.exp(logits) * logits_mask
                log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
                weighted_log_prob_pos = weight * log_prob
                loss = -(self.temperature / self.base_temperature) * weighted_log_prob_pos
                loss = loss.sum(dim=1).mean()
            else:
                q = features[:batch_size]
                k = features[batch_size:batch_size * 2]
                queue = features[batch_size * 2:]
                k, queue = k.detach(), queue.detach()
                l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
                l_neg = torch.einsum('nc,kc->nk', [q, queue])
                logits = torch.cat([l_pos, l_neg], dim=1)
                logits /= self.temperature
                labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
                loss = F.cross_entropy(logits, labels)
            return loss

    def __init__(self,input_shape,train_givenY,hparams):
        super(DIRK, self).__init__(input_shape,train_givenY,hparams)
        self.stu = self.stu_model(self.num_classes,input_shape,hparams,self.DIRKNet)
        self.tea=self.tea_model(self.num_classes,input_shape,hparams,self.DIRKNet)
        self.optimizer = torch.optim.Adam(
            self.stu.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.loss_cont_fn = self.WeightedConLoss(temperature=self.hparams['feat_temperature'], dist_temperature=self.hparams['dist_temperature'])
        self.curr_iter = 0
        self.mode = self.hparams['mode']

    def update(self,minibatches,epoch=None):
        x, strong_x, partial_y, y, index = minibatches

        features, partYs, dists, rec_conf_t = self.tea(x, strong_x, strong_x, partial_y)
        output_s, feat_s = self.stu(strong_x, x)
        features_cont = torch.cat((feat_s, features), dim=0)
        partY_cont = torch.cat((partial_y, partYs), dim=0)
        dist_cont = torch.cat((rec_conf_t, dists), dim=0)
        batch_size = output_s.shape[0]
        mask_partial = torch.matmul(partY_cont[:batch_size], partY_cont.T)
        mask_partial[mask_partial != 0] = 1
        _, pseudo_target = torch.max(dist_cont, dim=1)
        pseudo_target = pseudo_target.contiguous().view(-1, 1)
        mask_pseudo_target = torch.eq(pseudo_target[:batch_size], pseudo_target.T).float()
        start_upd_prot = self.curr_iter >= self.hparams['prot_start']
        if start_upd_prot:
            mask = mask_partial * mask_pseudo_target
        else:
            mask = None

        if self.hparams['weight'] != 0:
            loss_cont = self.loss_cont_fn(features=features_cont, dist=dist_cont, mask=mask, batch_size=partial_y.shape[0])
        else:
            loss_cont = torch.tensor(0.0).cuda()
        loss_dirk = self.dirk_loss(output_s, rec_conf_t)
        if self.mode == "UMiP":
            def UMiP_Conf_dirk_data(x,partialY,confidence,alpha=1):
                batch_size = x.size()[0]
                device = "cuda" if x.is_cuda else "cpu"
                lam = torch.tensor(np.random.beta(alpha, alpha, batch_size)).to(device)
                lam = torch.where(lam < 0.5, 1 - lam, lam)
                lam = torch.tensor(lam, dtype=x.dtype, device=x.device).view(batch_size, 1, 1, 1)
                index = torch.randperm(batch_size)
                mixed_x = lam * x + (1 - lam) * x[index, :]
                lam = lam.view(batch_size, 1)
                mixed_conf = lam * confidence + (1 - lam) * confidence[index, :]
                mixed_confidence = self.tea.get_correct_conf(mixed_conf, partialY)
                return mixed_x, mixed_confidence
            mixed_X, mixed_confidence=UMiP_Conf_dirk_data(x,partial_y,rec_conf_t)
            mixed_output_s,_=self.stu(strong_x, mixed_X)
            loss_dirk = loss_dirk+self.dirk_loss(mixed_output_s, mixed_confidence)
        loss = loss_dirk + self.hparams['weight'] * loss_cont

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.curr_iter += 1
        self.model_update(self.tea,self.stu,self.hparams['momentum'])
        return {'loss': loss.item()}

    def predict(self, x):
        return self.stu(None,x,is_eval=True)

    def tea_predict(self,x):
        return self.tea(x)

    def model_update(self,model_tea, model_stu, momentum=0.99):
        for param_tea, param_stu in zip(model_tea.parameters(), model_stu.parameters()):
            param_tea.data = param_tea.data * momentum + param_stu.data * (1 - momentum)


class CRDPLL(Algorithm):
    """
    CRDPLL
    Reference: Revisiting Consistency Regularization for Deep Partial Label Learning, ICML 2022.
    """

    def __init__(self, input_shape, train_givenY, hparams):
        super(CRDPLL, self).__init__(input_shape, train_givenY, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            self.num_classes)

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

        tempY = train_givenY.sum(dim=1).unsqueeze(1).repeat(1, train_givenY.shape[1])
        label_confidence = train_givenY.float() / tempY
        self.label_confidence = label_confidence
        self.mode = self.hparams['mode']
        self.consistency_criterion = nn.KLDivLoss(reduction='batchmean')
        self.train_givenY=train_givenY
        self.lam = 1

    def update(self, minibatches,epoch=None):
        x, strong_x, partial_y, y, index = minibatches
        loss = self.cr_loss(self.predict(x), self.predict(strong_x), index,epoch)

        if self.mode == "UMiP":
            def UMiP_conf_crdpll_data(x,aug_x,partialY,confidence, alpha=1):
                batch_size = x.size()[0]
                device = "cuda" if x.is_cuda else "cpu"
                lam = torch.tensor(np.random.beta(alpha, alpha, batch_size)).to(device)
                lam = torch.where(lam < 0.5, 1 - lam, lam)
                lam = torch.tensor(lam, dtype=x.dtype, device=x.device).view(batch_size, 1, 1, 1)
                index = torch.randperm(batch_size)
                mixed_x = lam * x + (1 - lam) * x[index, :]
                mixed_aug_x = lam * aug_x + (1 - lam) * aug_x[index, :]
                lam = lam.view(batch_size, 1)
                mixed_conf = lam * confidence + (1 - lam) * confidence[index, :]
                mixed_conf = mixed_conf * partialY
                base_value = mixed_conf.sum(dim=1).unsqueeze(1).repeat(1, mixed_conf.shape[1])
                mixed_confidence = mixed_conf / (base_value+1e-12)
                return mixed_x, mixed_aug_x, mixed_confidence
            mixed_X, mixed_aug_X, mixed_confidence = UMiP_conf_crdpll_data(x,strong_x, partial_y,self.label_confidence[index])
            loss = loss + self.cr_loss(self.predict(mixed_X), self.predict(mixed_aug_X), index, epoch, mixed_confidence)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.confidence_update(x,strong_x, partial_y, index)
        return {'loss': loss.item()}

    def cr_loss(self, outputs, strong_outputs, index,epoch, batch_confidence=None):
        device = "cuda" if index.is_cuda else "cpu"
        if batch_confidence is None:
            self.label_confidence = self.label_confidence.to(device)
            batch_confidence = self.label_confidence[index, :].float()
        self.consistency_criterion=self.consistency_criterion.to(device)
        self.train_givenY=self.train_givenY.to(device)
        consist_loss0 = self.consistency_criterion(F.log_softmax(outputs, dim=1), batch_confidence)
        consist_loss1 = self.consistency_criterion(F.log_softmax(strong_outputs, dim=1), batch_confidence)
        super_loss = -torch.mean(
            torch.sum(torch.log(1.0000001 - F.softmax(outputs, dim=1)) * (1 - self.train_givenY[index, :]), dim=1))
        lam = min((epoch / 100) * self.lam, self.lam)
        average_loss = lam * (consist_loss0 + consist_loss1) + super_loss
        return average_loss

    def predict(self, x):
        return self.network(x)

    def confidence_update(self,batchX,strong_batchX,batchY,batch_index):
        with torch.no_grad():
            batch_outputs = self.predict(batchX)
            strong_batch_outputs=self.predict(strong_batchX)
            temp_un_conf=F.softmax(batch_outputs,dim=1)
            strong_temp_un_conf=F.softmax(strong_batch_outputs,dim=1)
            self.label_confidence[batch_index,:]=torch.pow(temp_un_conf,1/(1+1))*torch.pow(strong_temp_un_conf,1/(1+1))*batchY
            base_value=self.label_confidence[batch_index,:].sum(dim=1).unsqueeze(1).repeat(1,self.label_confidence[batch_index,:].shape[1])
            self.label_confidence[batch_index,:]=self.label_confidence[batch_index,:]/(base_value+1e-12)


class ABLE(Algorithm):
    """
    ABLE
    Reference: Ambiguity-Induced Contrastive Learning for Instance-Dependent Partial Label Learning, IJCAI 2022
    """

    class ABLE_model(nn.Module):
        def __init__(self, num_classes,input_shape,hparams, base_encoder):
            super().__init__()
            self.encoder = base_encoder(num_classes,input_shape,hparams)
        def forward(self, hparams=None, img_w=None, images=None, partial_Y=None, is_eval=False):
            if is_eval:
                output_raw, q = self.encoder(img_w)
                return output_raw
            outputs, features = self.encoder(images)
            batch_size = hparams['batch_size']
            f1, f2 = torch.split(features, [batch_size, batch_size], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            return outputs, features

    class ABLENet(nn.Module):
        def __init__(self,num_classes,input_shape,hparams):
            super().__init__()
            self.featurizer = networks.Featurizer(input_shape, hparams)
            self.classifier = networks.Classifier(
                self.featurizer.n_outputs,
                num_classes)
            self.head = nn.Sequential(
                nn.Linear(self.featurizer.n_outputs, self.featurizer.n_outputs),
                nn.ReLU(inplace=True),
                nn.Linear(self.featurizer.n_outputs, hparams['feat_dim']))
        def forward(self,x):
            feat = self.featurizer(x)
            feat_c = self.head(feat)
            logits = self.classifier(feat)
            return logits, F.normalize(feat_c, dim=1)

    class ClsLoss(nn.Module):
        def __init__(self, predicted_score):
            super().__init__()
            self.predicted_score = predicted_score
            self.init_predicted_score = predicted_score.detach()
        def forward(self, outputs, index,batch_confidence=None):
            if batch_confidence is None:
                device = "cuda" if outputs.is_cuda else "cpu"
                self.predicted_score=self.predicted_score.to(device)
                batch_confidence=self.predicted_score[index, :]
            logsm_outputs = F.log_softmax(outputs, dim=1)
            final_outputs = batch_confidence * logsm_outputs
            cls_loss = - ((final_outputs).sum(dim=1)).mean()
            return cls_loss
        def update_target(self, batch_index, updated_confidence):
            with torch.no_grad():
                self.predicted_score[batch_index, :] = updated_confidence.detach()
            return None

    class ConLoss(nn.Module):
        def __init__(self, predicted_score, base_temperature=0.07):
            super().__init__()
            self.predicted_score = predicted_score
            self.init_predicted_score = predicted_score.detach()
            self.base_temperature = base_temperature
        def forward(self, hparams, outputs, features, Y, index):
            batch_size = hparams['batch_size']
            device = "cuda" if outputs.is_cuda else "cpu"
            self.predicted_score=self.predicted_score.to(device)
            contrast_count = features.shape[1]
            contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
            anchor_feature = contrast_feature
            anchor_count = contrast_count
            anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), hparams['temperature'])
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()
            Y = Y.float()
            output_sm = F.softmax(outputs[0: batch_size, :], dim=1).float()
            output_sm_d = output_sm.detach()
            _, target_predict = (output_sm_d * Y).max(1)
            predict_labels = target_predict.repeat(batch_size, 1).to(device)
            mask_logits = torch.zeros_like(predict_labels).float().to(device)
            pos_set = (Y == 1.0).nonzero().to(device)
            ones_flag = torch.ones(batch_size).float().to(device)
            zeros_flag = torch.zeros(batch_size).float().to(device)
            for pos_set_i in range(pos_set.shape[0]):
                sample_idx = pos_set[pos_set_i][0]
                class_idx = pos_set[pos_set_i][1]
                mask_logits_tmp = torch.where(predict_labels[sample_idx] == class_idx, ones_flag, zeros_flag).float()
                if mask_logits_tmp.sum() > 0:
                    mask_logits_tmp = mask_logits_tmp / mask_logits_tmp.sum()
                    mask_logits[sample_idx] = mask_logits[sample_idx] + mask_logits_tmp * \
                                              self.predicted_score[sample_idx][class_idx]
            mask_logits = mask_logits.repeat(anchor_count, contrast_count)
            logits_mask = torch.scatter(torch.ones_like(mask_logits),1,torch.arange(batch_size * anchor_count).view(-1, 1).to(device),0).float()
            mask_logits = mask_logits * logits_mask
            exp_logits = logits_mask * torch.exp(logits)
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
            mean_log_prob_pos = (mask_logits * log_prob).sum(1)
            loss_con_m = - (hparams['temperature'] / self.base_temperature) * mean_log_prob_pos
            loss_con = loss_con_m.view(anchor_count, batch_size).mean()
            revisedY_raw = Y.clone()
            revisedY_raw = revisedY_raw * output_sm_d
            revisedY_raw = revisedY_raw / revisedY_raw.sum(dim=1).repeat(Y.shape[1], 1).transpose(0, 1)
            new_target = revisedY_raw.detach()
            return loss_con, new_target
        def update_target(self, batch_index, updated_confidence):
            with torch.no_grad():
                self.predicted_score[batch_index, :] = updated_confidence.detach()
            return None

    def __init__(self,input_shape,train_givenY,hparams):
        super(ABLE, self).__init__(input_shape,train_givenY,hparams)
        self.network=self.ABLE_model(self.num_classes,input_shape,hparams=hparams,base_encoder=self.ABLENet)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        tempY = train_givenY.sum(dim=1).unsqueeze(1).repeat(1, train_givenY.shape[1])
        label_confidence = train_givenY.float() / tempY
        self.label_confidence = label_confidence
        self.loss_cls = self.ClsLoss(predicted_score=label_confidence.float())
        self.loss_con = self.ConLoss(predicted_score=label_confidence.float())
        self.train_givenY = train_givenY
        self.mode = self.hparams['mode']

    def update(self,minibatches,epoch=None):
        x, strong_x, partial_y, y, index = minibatches
        X_tot = torch.cat([x, strong_x], dim=0)
        batch_size = self.hparams['batch_size']

        cls_out, features = self.network(hparams=self.hparams, images=X_tot, partial_Y=partial_y, is_eval=False)
        cls_out_w = cls_out[0: batch_size, :]

        cls_loss = self.loss_cls(cls_out_w, index)
        if self.mode=="UMiP":
            def UMiP_conf_able_data(x,partialY,confidence,alpha=1):
                batch_size = x.size()[0]
                device = "cuda" if x.is_cuda else "cpu"
                lam = torch.tensor(np.random.beta(alpha, alpha, batch_size)).to(device)
                lam = torch.where(lam < 0.5, 1 - lam, lam)
                lam = torch.tensor(lam, dtype=x.dtype, device=x.device).view(batch_size, 1, 1, 1)
                index = torch.randperm(batch_size)
                mixed_x = lam * x + (1 - lam) * x[index, :]
                lam = lam.view(batch_size, 1)
                mixed_conf = lam * confidence + (1 - lam) * confidence[index, :]
                mixed_conf = mixed_conf * partialY
                base_value = mixed_conf.sum(dim=1).unsqueeze(1).repeat(1, mixed_conf.shape[1])
                mixed_confidence = mixed_conf / (base_value + 1e-12)
                return mixed_x, mixed_confidence
            mixed_X,mixed_confidence=UMiP_conf_able_data(x,partial_y,self.loss_cls.predicted_score[index])
            X_tot = torch.cat([mixed_X, strong_x], dim=0)
            cls_out_w, _ = self.network(hparams=self.hparams, images=X_tot, partial_Y=partial_y, is_eval=False)
            mixed_cls_out_w = cls_out[0: batch_size, :]
            cls_loss = cls_loss + self.loss_cls(mixed_cls_out_w, index,mixed_confidence)
        con_loss, new_target = self.loss_con(self.hparams, cls_out, features, partial_y, index)
        loss = cls_loss + self.hparams['loss_weight'] * con_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss_cls.update_target(batch_index=index, updated_confidence=new_target)
        self.loss_con.update_target(batch_index=index, updated_confidence=new_target)
        return {'loss': loss.item()}

    def predict(self,images,):
        return self.network(img_w=images,is_eval=True)


class PiCO(Algorithm):
    """
    PiCO
    Reference: PiCO: Contrastive Label Disambiguation for Partial Label Learning
    """

    class PiCO_model(nn.Module):
        def __init__(self, num_classes,input_shape,hparams, base_encoder):
            super().__init__()
            self.encoder_q = base_encoder(num_classes, input_shape, hparams)
            self.encoder_k = base_encoder(num_classes, input_shape, hparams)
            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False
            self.register_buffer("queue", torch.randn(hparams['moco_queue'], hparams['feat_dim']))
            self.register_buffer("queue_pseudo", torch.randn(hparams['moco_queue']))
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
            self.register_buffer("prototypes", torch.zeros(num_classes, hparams['feat_dim']))
            self.queue = F.normalize(self.queue, dim=0)

        @torch.no_grad()
        def _momentum_update_key_encoder(self, hparams):
            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data = param_k.data * hparams['moco_m'] + param_q.data * (1. - hparams['moco_m'])

        @torch.no_grad()
        def _dequeue_and_enqueue(self, keys, labels, hparams):
            batch_size = keys.shape[0]
            ptr = int(self.queue_ptr)
            assert hparams['moco_queue'] % batch_size == 0
            self.queue[ptr:ptr + batch_size, :] = keys
            self.queue_pseudo[ptr:ptr + batch_size] = labels
            ptr = (ptr + batch_size) % hparams['moco_queue']
            self.queue_ptr[0] = ptr

        def forward(self, img_q, im_k=None, partial_Y=None, hparams=None, is_eval=False):
            output, q = self.encoder_q(img_q)
            if is_eval:
                return output

            predicted_scores = torch.softmax(output, dim=1) * partial_Y
            max_scores, pseudo_labels_b = torch.max(predicted_scores, dim=1)
            prototypes = self.prototypes.clone().detach()
            logits_prot = torch.mm(q, prototypes.t())
            score_prot = torch.softmax(logits_prot, dim=1)
            with torch.no_grad():
                for feat, label in zip(q, pseudo_labels_b):
                    self.prototypes[label] = self.prototypes[label] * hparams['proto_m'] + (1 - hparams['proto_m']) * feat
            self.prototypes = F.normalize(self.prototypes, p=2, dim=1).detach()
            with torch.no_grad():
                self._momentum_update_key_encoder(hparams)
                _, k = self.encoder_k(im_k)
            features = torch.cat((q, k, self.queue.clone().detach()), dim=0)
            pseudo_labels = torch.cat((pseudo_labels_b, pseudo_labels_b, self.queue_pseudo.clone().detach()), dim=0)
            self._dequeue_and_enqueue(k, pseudo_labels_b, hparams)
            return output, features, pseudo_labels, score_prot

    class PiCONet(nn.Module):
        def __init__(self,num_classes,input_shape,hparams):
            super().__init__()
            self.featurizer = networks.Featurizer(input_shape, hparams)
            self.classifier = networks.Classifier(
                self.featurizer.n_outputs,
                num_classes)
            self.head = nn.Sequential(
                nn.Linear(self.featurizer.n_outputs, self.featurizer.n_outputs),
                nn.ReLU(inplace=True),
                nn.Linear(self.featurizer.n_outputs, hparams['feat_dim']))
            self.register_buffer("prototypes", torch.zeros(num_classes, hparams['feat_dim']))

        def forward(self, x):
            feat = self.featurizer(x)
            feat_c = self.head(feat)
            logits = self.classifier(feat)
            return logits, F.normalize(feat_c, dim=1)

    class partial_loss(nn.Module):
        def __init__(self, confidence, hparams, conf_ema_m=0.99):
            super().__init__()
            self.confidence = confidence
            self.init_conf = confidence.detach()
            self.conf_ema_m = conf_ema_m
            self.conf_ema_range = [float(item) for item in hparams['conf_ema_range'].split(',')]
        def set_conf_ema_m(self, epoch, total_epochs):
            start = self.conf_ema_range[0]
            end = self.conf_ema_range[1]
            self.conf_ema_m = 1. * epoch /total_epochs * (end - start) + start
        def forward(self, outputs, index,batch_confidence=None):
            if batch_confidence is None:
                device = "cuda" if outputs.is_cuda else "cpu"
                self.confidence = self.confidence.to(device)
                batch_confidence= self.confidence[index, :]
            logsm_outputs = F.log_softmax(outputs, dim=1)
            final_outputs = logsm_outputs * batch_confidence
            average_loss = - ((final_outputs).sum(dim=1)).mean()
            return average_loss
        def confidence_update(self, temp_un_conf, batch_index, batchY):
            with torch.no_grad():
                _, prot_pred = (temp_un_conf * batchY).max(dim=1)
                pseudo_label = F.one_hot(prot_pred, batchY.shape[1]).float().cuda().detach()
                self.confidence[batch_index, :] = self.conf_ema_m * self.confidence[batch_index, :] \
                                                  + (1 - self.conf_ema_m) * pseudo_label
            return None

    class SupConLoss(nn.Module):
        def __init__(self, temperature=0.07, base_temperature=0.07):
            super().__init__()
            self.temperature = temperature
            self.base_temperature = base_temperature
        def forward(self, features, mask=None, batch_size=-1):
            device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))
            if mask is not None:
                mask = mask.float().detach().to(device)
                anchor_dot_contrast = torch.div(
                    torch.matmul(features[:batch_size], features.T),
                    self.temperature)
                logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
                logits = anchor_dot_contrast - logits_max.detach()
                logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
                mask = mask * logits_mask
                exp_logits = torch.exp(logits) * logits_mask
                log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
                mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
                loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
                loss = loss.mean()
            else:
                q = features[:batch_size]
                k = features[batch_size:batch_size * 2]
                queue = features[batch_size * 2:]
                l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
                l_neg = torch.einsum('nc,kc->nk', [q, queue])
                logits = torch.cat([l_pos, l_neg], dim=1)
                logits /= self.temperature
                labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
                loss = F.cross_entropy(logits, labels)
            return loss

    def __init__(self,input_shape,train_givenY,hparams):
        super(PiCO, self).__init__(input_shape,train_givenY,hparams)
        self.network=self.PiCO_model(self.num_classes,input_shape,hparams=hparams,base_encoder=self.PiCONet)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

        tempY = train_givenY.sum(dim=1).unsqueeze(1).repeat(1, train_givenY.shape[1])
        label_confidence = train_givenY.float() / tempY
        self.label_confidence = label_confidence
        self.loss_fn = self.partial_loss(label_confidence.float(),self.hparams)
        self.loss_cont_fn = self.SupConLoss()
        self.train_givenY = train_givenY
        self.mode = self.hparams['mode']

    def update(self,minibatches,epoch=None):
        x, strong_x, partial_y, y, index = minibatches
        cls_out, features_cont, pseudo_target_cont, score_prot = self.network(x, strong_x, partial_y, self.hparams)
        batch_size = cls_out.shape[0]
        pseudo_target_cont = pseudo_target_cont.contiguous().view(-1, 1)

        start_upd_prot = epoch >= self.hparams['prot_start']
        if start_upd_prot:
            self.loss_fn.confidence_update(temp_un_conf=score_prot, batch_index=index, batchY=partial_y)
        if start_upd_prot:
            mask = torch.eq(pseudo_target_cont[:batch_size], pseudo_target_cont.T).float().cuda()
        else:
            mask = None
        loss_cont = self.loss_cont_fn(features=features_cont, mask=mask, batch_size=batch_size)
        loss_cls = self.loss_fn(cls_out, index)
        if self.mode=="UMiP":
            def UMiP_Conf_pico_data(x,partialY,confidence,alpha=1):
                batch_size = x.size()[0]
                device = "cuda" if x.is_cuda else "cpu"
                lam = torch.tensor(np.random.beta(alpha, alpha, batch_size)).to(device)
                lam = torch.where(lam < 0.5, 1 - lam, lam)
                lam = torch.tensor(lam, dtype=x.dtype, device=x.device).view(batch_size, 1, 1, 1)
                index = torch.randperm(batch_size)
                mixed_x = lam * x + (1 - lam) * x[index, :]
                lam = lam.view(batch_size, 1)
                mixed_conf = lam * confidence + (1 - lam) * confidence[index, :]
                mixed_conf = mixed_conf * partialY
                base_value = mixed_conf.sum(dim=1).unsqueeze(1).repeat(1, mixed_conf.shape[1])
                mixed_confidence = mixed_conf / (base_value + 1e-12)
                return mixed_x, mixed_confidence
            mixed_X, mixed_confidence=UMiP_Conf_pico_data(x,partial_y,self.loss_fn.confidence[index])
            mixed_cls_out, _, _, _ = self.network(mixed_X, strong_x, partial_y, self.hparams)
            loss_cls=loss_cls+self.loss_fn(mixed_cls_out,index,mixed_confidence)
        loss = loss_cls + self.hparams['loss_weight'] * loss_cont
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss_fn.set_conf_ema_m(epoch, self.hparams['epochs'])
        return {'loss': loss.item()}

    def predict(self,images,):
        return self.network(img_q=images,is_eval=True)
