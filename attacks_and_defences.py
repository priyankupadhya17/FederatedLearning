import torch
from torch.utils.data import ConcatDataset
from model import MNIST_model, Breakhis_model, vgg11, mobilenet
from collections import OrderedDict
import copy
from torch import optim
import torch.nn.functional as F
import scipy
from scipy.spatial import distance_matrix, distance
from pyod.models.copod import COPOD
import numpy as np
import random
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch, MiniBatchKMeans, BisectingKMeans, SpectralClustering
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os


def rand_cos_sim(v, r, costheta):
    """_summary_
    Given the vector v and cosine similarity costheta (a scalar between -1 and 1), compute w with same cosine similarity
    """
    # Form the unit vector parallel to v:
    u = v / torch.norm(v)

    # Pick a random vector:
    #r = np.random.multivariate_normal(np.zeros_like(v), np.eye(len(v)))

    # Form a vector perpendicular to v:
    uperp = r - r.dot(u)*u

    # Make it a unit vector:
    uperp = uperp / torch.norm(uperp)

    # w is the linear combination of u and uperp with coefficients costheta
    # and sin(theta) = sqrt(1 - costheta**2), respectively:
    w = costheta*u + torch.sqrt(1 - costheta**2)*uperp

    return w

# Hook to get the features from the middle layers
ACTIVATION = {}
def get_activation(name):
    def hook(model, input, output):
        ACTIVATION[name] = output.detach()
    return hook


class Attack():
    """
    Class that contains different attack methods
    """
    def __init__(self) -> None:
        self.mal_dataloader = None
        self.iter = None
    
    def lie_attack(self, params, n_attackers):
        
        '''assert n_attackers == 3 or n_attackers == 5 or n_attackers == 8 or n_attackers == 10 or n_attackers == 12, \
            "Please keep number of attackers as 3,5,8,10 or 12  OR  add values in z_table!"'''
        
        # z values when number of clients is 50 and num_attackers is as stated in dict
        # note that 3 and 4 key values are for when number of clients is 10 otherwise number of clients is 50
        z_values={3: 0.52, 4: 0.84, 5:0.7054, 8:0.71904, 10:0.72575, 12:0.73891}
        
        avg = torch.mean(params, dim=0)
        std = torch.std(params, dim=0)
        
        return avg + z_values[n_attackers] * std
    
    def min_max(self, all_updates, model_re, n_attackers, dev_type='std'):
        
        if dev_type == 'unit_vec':
            deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
        elif dev_type == 'sign':
            deviation = torch.sign(model_re)
        elif dev_type == 'std':
            deviation = torch.std(all_updates, 0)

        lamda = torch.Tensor([10.0]).float().cuda()
        # print(lamda)
        threshold_diff = 1e-5
        lamda_fail = lamda
        lamda_succ = 0
        
        distances = []
        for update in all_updates:
            distance = torch.norm((all_updates - update), dim=1) ** 2
            distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)
        
        max_distance = torch.max(distances)
        del distances

        while torch.abs(lamda_succ - lamda) > threshold_diff:
            mal_update = (model_re - lamda * deviation)
            distance = torch.norm((all_updates - mal_update), dim=1) ** 2
            max_d = torch.max(distance)
            
            if max_d <= max_distance:
                # print('successful lamda is ', lamda)
                lamda_succ = lamda
                lamda = lamda + lamda_fail / 2
            else:
                lamda = lamda - lamda_fail / 2

            lamda_fail = lamda_fail / 2

        mal_update = (model_re - lamda_succ * deviation)
        
        return mal_update
    
    def our_attack_grads(self, params, n_attackers, client_list, rounds, dataset_name):
        """
        DISBELIEVE ATTACK for gradients
        
        params: paremeters of clients that are attackers
        n_attackers: number of attackers
        client_list: list of clients that are attackers
        rounds: current communication round
        dataset_name: name of the dataset
        """
        # here params = grads
        
        print("OUR ATTACK:")
        
        assert n_attackers == len(client_list), 'Length of client list is not equal to number of attackers'
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        distances = []
        for param in params:
            distance = torch.norm((params - param), dim=1) ** 2
            distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)
        
        params_cpu = params.to('cpu').numpy()
        
        max_distance = torch.max(distances)

        try:
            min_distance = torch.min(distances[distances > 0])
        except:
            print('All gradients became zero')
            min_distance = torch.min(distances)
        
        avg_distance = torch.mean(distances)
        del distances
        
        avg_grad = torch.mean(params, dim=0)
        std = torch.std(params, dim=0)
        
        datasets = []
        for client in client_list:
            datasets.append(client.train_dataset)
        concat_dataset = ConcatDataset(datasets)
        
        if dataset_name == 'MNIST':
            new_model = MNIST_model()
        elif dataset_name == 'Breakhis':
            new_model = Breakhis_model()
        elif dataset_name == 'CIFAR':
            new_model = vgg11()
        elif dataset_name == 'ImageNet':
            new_model = mobilenet(n_classes=100)
        elif dataset_name == 'CIFAR-100':
            new_model = vgg11(n_classes=100)
        
        new_state_dict = OrderedDict()
        
        # Creates a new model and sets the parameters as average of all attacking client parameters
        for k in new_model.state_dict().keys():
            v = None
            for client in client_list:
                if v is None:
                    v = client.model.state_dict()[k]
                else:
                    v += client.model.state_dict()[k]
            
            new_state_dict[k] = v / n_attackers
        
        new_model.load_state_dict(new_state_dict)

        new_model.train()
        
        new_model = new_model.to(device)
        
        mal_dataloader = torch.utils.data.DataLoader(concat_dataset, batch_size=100, shuffle=True, num_workers=4)
        
        if dataset_name == 'MNIST' or dataset_name == 'CIFAR' or dataset_name == 'ImageNet' or dataset_name == 'CIFAR-100':
            opt = optim.SGD(new_model.parameters(), lr=0.1)
        elif dataset_name == 'Breakhis':
            opt = optim.SGD(new_model.parameters(), lr=0.1)
        
        grad_values = []
        grads = []
        
        try:
            data,target = next(self.iter)
        except:
            self.mal_dataloader = mal_dataloader
            self.iter = iter(self.mal_dataloader)
            data,target = next(self.iter)
        
        data = data.to(device)
        target = target.to(device)
        
        if dataset_name == 'MNIST':
            data = torch.flatten(data, start_dim=1)
        
        opt.zero_grad()
        output = new_model(data)
        
        if dataset_name == 'MNIST' or dataset_name == 'CIFAR' or dataset_name == 'ImageNet' or dataset_name == 'CIFAR-100':
            loss1 = -F.cross_entropy(output, target)
        elif dataset_name == 'Breakhis':
            loss1 = -F.cross_entropy(output, target)
        
        loss1.backward()
        # store the last updated gradient
        grad_values = []
        for new_params in new_model.parameters():
           grad_values = new_params.grad.view(-1) if not len(grad_values) else torch.cat((grad_values, new_params.grad.view(-1)))
        
        opt.step()
        
        grad_values_cpy = grad_values
        
        # Distance (D in the paper)
        d = min_distance
        
        start = 0.001
        end = 1000
        while abs(start - end) > 0.01:
            mid = (start + end) / 2
            grad_values_cpy_ = grad_values_cpy / (mid * torch.norm(grad_values_cpy))
            loss2 = torch.sum((grad_values_cpy_ - avg_grad)**2)
            if loss2 > d:
                start = mid
            else:
                end = mid
        
        grad_values = grad_values / (mid * torch.norm(grad_values))
        #grad_values = torch.nan_to_num(grad_values)
        
        loss2 = torch.sum((grad_values - avg_grad)**2)
        
        print(f'Round={rounds}, mid={mid}, loss2={loss2} and max_distance={max_distance} and min_distance={min_distance}')
        
        #return torch.mean(grads, dim=0)
        return grad_values
    
    def our_attack(self, params, n_attackers, client_list, dataset_name):
        """
        DISBELIEVE Attack for parameters

        Args:
            params (list): The parameters of each client
            n_attackers (int): The number of attackers
            client_list (client_list): List of clients that are attackers
            dataset_name: name of the dataset

        Returns:
            _type_: _description_
        """
        assert n_attackers == len(client_list), 'Length of client list is not equal to number of attackers'
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        distances = []
        for param in params:
            distance = torch.norm((params - param), dim=1) ** 2
            distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)
        
        max_distance = torch.max(distances)
        del distances
        
        datasets = []
        for client in client_list:
            datasets.append(client.train_dataset)
        concat_dataset = ConcatDataset(datasets)
        
        if dataset_name == 'MNIST':
            new_model = MNIST_model()
        elif dataset_name == 'Breakhis':
            new_model = Breakhis_model()
        elif dataset_name == 'CIFAR':
            new_model = vgg11()
        elif dataset_name == 'ImageNet':
            new_model = mobilenet(n_classes=100)
        elif dataset_name == 'CIFAR-100':
            new_model = vgg11(n_classes=100)
        
        new_state_dict = OrderedDict()
        
        # Creates a new model and sets the parameters as average of all attacking client parameters
        for k in new_model.state_dict().keys():
            v = None
            for client in client_list:
                if v is None:
                    v = client.model.state_dict()[k]
                else:
                    v += client.model.state_dict()[k]
            
            new_state_dict[k] = v / n_attackers
        
        new_model.load_state_dict(new_state_dict)
        old_model = copy.deepcopy(new_model)

        new_model.train()
        old_model.eval()
        
        new_model = new_model.to(device)
        old_model = old_model.to(device)
        
        mal_dataloader = torch.utils.data.DataLoader(concat_dataset, batch_size=100, shuffle=True, num_workers=4)
        opt = optim.SGD(new_model.parameters(), lr=0.1)
        
        for epoch in range(10):
            for batch_idx, (data, target) in enumerate(mal_dataloader):
                data = data.to(device)
                target = target.to(device)
                
                data = torch.flatten(data, start_dim=1)
                
                opt.zero_grad()
                output = new_model(data)
                
                loss1 = -F.cross_entropy(output, target)
                loss2 = 0
                for w_old, w_new in zip(old_model.parameters(), new_model.parameters()):
                    loss2 += torch.sum((w_old - w_new) ** 2)
                
                print(f'epoch={epoch}, batch_idx={batch_idx}, loss1={loss1} loss2={loss2 / (batch_idx+1)} max_distance={max_distance}')
                if loss2 >= max_distance:
                    continue

                if loss1 < -10000.0:
                    continue
                
                loss1.backward()
                opt.step()
        
        new_model.to('cpu')
        
        param_values=[]
        for param in new_model.parameters():
            param_values = param.data.view(-1) if not len(param_values) else torch.cat((param_values, param.data.view(-1)))
        
        return param_values


class Defence():
    """
    This class contains various defence/aggregation mechanisms
    """
    
    def __init__(self) -> None:
        self.plot_data = []
        for i in range(11):
            p = [[] for j in range(10)]
            self.plot_data.append(p)
    
    def trimmed_mean(self, params, beta, n_clients):
        start_index = int(n_clients * beta)
        end_index = int(n_clients - start_index)
        params, _ = torch.sort(params, dim=0)
        params = params[start_index:end_index, :]
        params = torch.mean(params, dim=0)
        
        return params
    
    def multi_krum(self, all_updates, n_attackers, multi_k=False):

        candidates = []
        candidate_indices = []
        remaining_updates = all_updates
        all_indices = np.arange(len(all_updates))

        while len(remaining_updates) > 2 * n_attackers + 2:
            torch.cuda.empty_cache()
            distances = []
            for update in remaining_updates:
                distance = []
                for update_ in remaining_updates:
                    distance.append(torch.norm((update - update_)) ** 2)
                distance = torch.Tensor(distance).float()
                distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

            distances = torch.sort(distances, dim=1)[0]
            scores = torch.sum(distances[:, :len(remaining_updates) - 2 - n_attackers], dim=1)
            indices = torch.argsort(scores)[:len(remaining_updates) - 2 - n_attackers]

            candidate_indices.append(all_indices[indices[0].cpu().numpy()])
            all_indices = np.delete(all_indices, indices[0].cpu().numpy())
            candidates = remaining_updates[indices[0]][None, :] if not len(candidates) else torch.cat((candidates, remaining_updates[indices[0]][None, :]), 0)
            remaining_updates = torch.cat((remaining_updates[:indices[0]], remaining_updates[indices[0] + 1:]), 0)
            if not multi_k:
                break
        # print(len(remaining_updates))

        aggregate = torch.mean(candidates, dim=0)

        return aggregate
    
    def krum(self, params, num_clients, max_num_attackers, mkrum_enabled=False):
        n = num_clients
        f = max_num_attackers
        k = n - f - 2
        
        cdist = torch.cdist(params, params, p=2)  # n x n
        
        # find the k+1 nbh of each point
        nbhDist, nbh = torch.topk(cdist, k + 1, largest=False)
        
        # the point closest to its nbh
        i_star = torch.argmin(nbhDist.sum(1))

        if mkrum_enabled:
            mkrum = params[nbh[i_star, :].view(-1), :].mean(0)
            return mkrum
        else:
            krum = params[i_star, :]
            print(f'Selected agent for KRUM = {i_star}')
            return krum
    
    def bulyan(self, all_updates, n_attackers):
        nusers = all_updates.shape[0]
        bulyan_cluster = []
        candidate_indices = []
        remaining_updates = all_updates
        all_indices = np.arange(len(all_updates))

        while len(bulyan_cluster) < (nusers - 2 * n_attackers):
            torch.cuda.empty_cache()
            distances = []
            for update in remaining_updates:
                distance = []
                for update_ in remaining_updates:
                    distance.append(torch.norm((update - update_)) ** 2)
                distance = torch.Tensor(distance).float()
                distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)
            # print(distances)

            distances = torch.sort(distances, dim=1)[0]

            scores = torch.sum(distances[:, :len(remaining_updates) - 2 - n_attackers], dim=1)
            indices = torch.argsort(scores)[:len(remaining_updates) - 2 - n_attackers]
            if not len(indices):
                break
            candidate_indices.append(all_indices[indices[0].cpu().numpy()])
            all_indices = np.delete(all_indices, indices[0].cpu().numpy())
            bulyan_cluster = remaining_updates[indices[0]][None, :] if not len(bulyan_cluster) else torch.cat((bulyan_cluster, remaining_updates[indices[0]][None, :]), 0)
            remaining_updates = torch.cat((remaining_updates[:indices[0]], remaining_updates[indices[0] + 1:]), 0)

        # print('dim of bulyan cluster ', bulyan_cluster.shape)

        n, d = bulyan_cluster.shape
        param_med = torch.median(bulyan_cluster, dim=0)[0]
        sort_idx = torch.argsort(torch.abs(bulyan_cluster - param_med), dim=0)
        sorted_params = bulyan_cluster[sort_idx, torch.arange(d)[None, :]]

        return torch.mean(sorted_params[:n - 2 * n_attackers], dim=0)
        
    def dos(self, params):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #device = torch.device('cpu')
        
        params_cpu = params.to('cpu').numpy()
        distance_M = torch.tensor(distance.cdist(params_cpu, params_cpu, metric='cosine'))
        distance_EU = torch.tensor(distance_matrix(params_cpu, params_cpu))
        
        clf_cs = COPOD()
        clf_eu = COPOD()
        
        clf_cs.fit(distance_M)
        clf_eu.fit(distance_EU)
        
        anomaly_scores_distance_M = clf_cs.decision_function(distance_M)
        anomaly_scores_distance_EU = clf_eu.decision_function(distance_EU)
        
        alpha = -1
        
        abnormalScores = (anomaly_scores_distance_M + anomaly_scores_distance_EU) / 2
        #abnormalScores = anomaly_scores_distance_EU
        weighted_abnormals = np.exp(alpha * abnormalScores) / np.sum(np.exp(alpha * abnormalScores))
        
        weighted_abnormals = torch.tensor(weighted_abnormals, device=device)
        weighted_abnormals = weighted_abnormals.reshape(weighted_abnormals.shape[0], 1)
        params = weighted_abnormals * params
        
        print(f'weighted_abnormals={weighted_abnormals}')
        
        dos = torch.sum(params, dim=0)
        dos = dos.to(torch.float32)
        
        return dos
    
    def new_defence_grads(self, params, nclients, lr, global_server_model, dataset_name, test_loader, img_size=(1, 3, 32, 32), plotEnabled=False, rounds=None, attack_name=None, attacker_list=[0,1,2]):
        """
        GAMMA defence for gradients as well as parameters
        params: list of all parameters
        nclients: number of clients
        global_server_model: The model at the global server (should be of similar architecture as clients)
        """
        
        output_per_layer_per_client = [[] for i in range(nclients)]
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create random input for the model
        data = torch.rand(1,3,32,32)
        data = data.to(device)
        
        model = copy.deepcopy(global_server_model)

        mapping = {}
        layers = []
        for param_tensor in model.state_dict():
            layers.append(param_tensor.rsplit('.', 1)[0])
        
        #layers = list(set(layers))  # this will sort layer names which is undesirable
        layer = []
        for layer_name in layers:
            if layer_name not in mapping:
                layer.append(layer_name)
                mapping[layer_name] = 1
        
        del mapping, layers
        
        for i in range(nclients):
            model = copy.deepcopy(global_server_model)
            opt = None
            
            # Getting the features from the middle layers
            hooks = {}
            for name, module in model.named_modules():
                hooks[name] = module.register_forward_hook(get_activation(name))
                
            if dataset_name == 'CIFAR' or dataset_name == 'ImageNet' or dataset_name == 'CIFAR-100':
                opt = torch.optim.Adam(model.parameters(), lr=lr)
            elif dataset_name == 'Breakhis':
                opt = torch.optim.Adam(model.parameters(), lr=lr)
            
            assert opt is not None, 'opt in new_defence is None'
            
            # Perform gradient descent for the new model
            opt.zero_grad()
            start_idx = 0
            for j, param in enumerate(model.parameters()):
                param_ = params[i][start_idx:start_idx+len(param.data.view(-1))].reshape(param.data.shape)
                start_idx = start_idx + len(param.data.view(-1))
                param.grad = param_
            opt.step()
            
            if dataset_name == 'MNIST':
                    data = torch.flatten(data, start_dim=1)
            
            _ = model(data)
            
            # For different clients get the features and append them for further processing
            for layer_name in layer:
                output_per_layer_per_client[i].append(ACTIVATION[layer_name].view(-1))  # always append flattened tensor
        
        n_layers = len(layer)
        
        assert n_layers == len(output_per_layer_per_client[0]), 'The sizes of layers is different, plz check'
        
        stacked_list = []
        for i in range(n_layers):
            l = []
            for j in range(nclients):
                l = output_per_layer_per_client[j][i][None, :] if len(l)==0 else torch.cat((l, output_per_layer_per_client[j][i][None,:]), 0)
            stacked_list.append(l)
        
        
        # uncomment from here ==> we were using krum like methodology and return just 1 client's params
        """
        
        outliers_cnt = []
        for i in range(nclients):
            outliers_cnt.append(0)
        
        wts = None
        
        #cnt = torch.zeros(nclients)
        #layer_importance = np.array([i for i in range(1, n_layers+1)])
        #layer_importance = np.sort(layer_importance)
        #layer_importance = layer_importance / np.sum(layer_importance)
        
        final_cnt = torch.zeros(nclients)
        
        for i in range(n_layers):
            v = stacked_list[i].detach().to('cpu').numpy()
            #print("v.shape=",v.shape)
            distance_EU = torch.tensor(distance_matrix(v, v, p=2))
            #print(f'distance_EU={distance_EU}')
            
            _, ind = torch.topk(distance_EU, k=int(nclients/2), dim=1)
            cnt = torch.zeros(nclients)
            for x in ind:
                cnt[x] += 1
            
            #clf_eu = COPOD()
            #clf_eu.fit(distance_EU)
            #anomaly_scores_distance_EU = clf_eu.decision_function(distance_EU)
            print(cnt)
            
            final_cnt[torch.argmin(cnt)] += 1
            
            #anomaly_scores_distance_EU = cnt.numpy()
            
            '''wts_ = np.exp(-1 * anomaly_scores_distance_EU) / np.sum(np.exp(-1 * anomaly_scores_distance_EU))
            for j in range(nclients):
                self.plot_data[i][j].append(wts_[j])'''
            
            #alpha = -1
            #weighted_abnormals = np.exp(alpha * anomaly_scores_distance_EU) / np.sum(np.exp(alpha * anomaly_scores_distance_EU))
            '''if i == 0:
                wts = wts_
            else:
                wts += wts_'''
         
        #wts /= n_layers
        print('final_cnt: ', final_cnt)
        #final_cnt = torch.where(final_cnt > 0, 1., 0.)  # comment this to normalize as per wts
        #print('final_cnt: ', final_cnt)
        #final_cnt = final_cnt / torch.sum(final_cnt)  
        
        client_id = torch.argmax(final_cnt)
        return params[client_id,:]
        
        wts = final_cnt.to(device)
        #wts = torch.tensor(wts, device=device)
        wts = wts.reshape(wts.shape[0], 1)
        
        #print(wts.shape)
        #print(params.shape)
        
        params = wts * params
        
        my = torch.sum(params, dim=0)
        my = my.to(torch.float32)
        
        return my
        """
        
        labels_list = []
        
        ## Initialize votes
        final_cnt = torch.zeros(nclients)
        
        for i in range(n_layers):
            v = stacked_list[i].detach().to('cpu').numpy()
            
            #################### THIS ALSO WORKS #########################################
            #agglomerative_clustering = AgglomerativeClustering(n_clusters=2).fit(v)
            #########################################################################
            
            # THIS WORKS BETTER
            specteral = SpectralClustering(n_clusters=2, affinity='rbf').fit(v)
            
            #labels = agglomerative_clustering.labels_
            labels = specteral.labels_
            
            if np.sum(labels) < int(nclients / 2):
                labels = np.where(labels == 0, 1, 0)
            print(labels)
            labels_list.append(labels)
            
            # VOTING for malicious and non-malicious 
            labels = torch.tensor(labels)
            final_cnt += labels
            
        if plotEnabled:
            """
            If plotEnabled, then plots the 2 clusters - malicious and non-malicious. Helps in determining which clients are actually malicious among the ones that are predicted to be malicious
            by comparing them to GT malicous clients    
            """
            fig = plt.figure(figsize=(10, 5))
            plt.tight_layout()
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            
            assert rounds is not None, 'Scatter plot requires round information!'
            assert attack_name is not None, 'Scatter plot requires attack name!'
            
            tsne = TSNE(n_components=2, perplexity=5.0)
            
            _, topk_final_cnt_indices = torch.topk(final_cnt, int(nclients / 2)) # top 5 clients are benign
            topk_final_cnt_indices = topk_final_cnt_indices.tolist()
            
            x_layer = stacked_list[n_layers-1].detach().to('cpu').numpy() # for last layer
            x_layer = tsne.fit_transform(x_layer)
            
            colors = []
            for j in range(nclients):
                if j in attacker_list and attack_name != 'no_attack':
                    colors.append('r')
                else:
                    colors.append('g')
            ax1.scatter(x=x_layer[:,0], y=x_layer[:,1], c=colors)
            ax1.set_title(f'Layer={i}, Round={rounds}, Actual')
            
            label_colors = []
            for j in range(nclients):
                if j in topk_final_cnt_indices:
                    label_colors.append('g')
                else:
                    label_colors.append('r')
            ax2.scatter(x=x_layer[:,0], y=x_layer[:,1], c=label_colors)
            ax2.set_title(f'Layer={i}, Round={rounds}, Pred')
            
            filename = 'new_defence_scatter_' + attack_name + '_rounds_' +str(rounds) + '.png'
            plt.savefig(os.path.join(os.getcwd(), filename))
            plt.clf()
                
        # We were using krum like methodology and return just 1 client's params based on voting mechanism
        return params[torch.argmax(final_cnt), :]

        
            
            
        
        
        
        
        
        
        
