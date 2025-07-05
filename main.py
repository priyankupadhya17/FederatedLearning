import torch
import torchvision
import torchvision.datasets as datasets
from server import Server
from model import MNIST_model, Breakhis_model, vgg11, mobilenet, VGG16
import random
import numpy as np
from math import ceil
from tqdm import tqdm
from attacks_and_defences import Attack, Defence
import copy
import pickle
from matplotlib import pyplot as plt
import os
from breakhis_dataset import BreakhisDataset
import sklearn


def download_dataset(dataset='MNIST'):
    
    """
    Download the relevant dataset

    Returns:
        trainset, val_set, testset: The train, validation (if present) and the test set.
    """
    
    if dataset == 'MNIST':
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))])
        mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        return mnist_trainset, mnist_testset

    elif dataset == 'Breakhis':
        train_transform =  torchvision.transforms.Compose([
            torchvision.transforms.Resize((32, 32)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.RandomRotation(degrees=180),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        test_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((32, 32)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        root_dir = '/data/priyank.upadhya/thesis/BreaKHis_v1/BreaKHis_v1/histology_slides/breast'  # The directory path where Breakhis dataset is stored
        
        map = {'benign':0, 'malignant':1}
        
        dataset = []
        
        for key in map.keys():
            dir = os.path.join(root_dir, key)
            img_label = map[key]
            
            for root, dirs, files in os.walk(dir):
                for file in files:
                    if '.png' in file and '400X' in root:
                        dataset.append((os.path.join(root, file), img_label))
        
        random.shuffle(dataset)
        
        n = len(dataset)
        
        train_set = dataset[:int(n * 0.68)]
        val_set = dataset[int(n * 0.68):int(n * 0.75)]
        test_set = dataset[int(n * 0.75):]
        
        train_dataset = BreakhisDataset(list_dataset=train_set, transform=train_transform)
        val_dataset = BreakhisDataset(list_dataset=val_set, transform=test_transform)
        test_dataset = BreakhisDataset(list_dataset=test_set, transform=test_transform)
        
        print(f'Train Dataset={len(train_dataset)} Val Dataset = {len(val_dataset)} Test Dataset = {len(test_dataset)}')
        
        return train_dataset, val_dataset, test_dataset
    
    elif dataset == 'CIFAR':
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
        testet = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        
        return trainset, testet
    
    elif dataset == 'CIFAR-100':
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
                                                         ])
        # Dataset
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        testet = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        
        return trainset, testet
    
    elif dataset == 'ImageNet':
        transform = torchvision.transforms.Compose([torchvision.transforms.Resize(256),
                                                    torchvision.transforms.CenterCrop(224),
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(
                                                        mean=[0.485, 0.456, 0.406],  # ImageNet mean
                                                        std=[0.229, 0.224, 0.225]    # ImageNet std
                                                    )]
                                                   )
        
        # Enter the directory for where Imagenet data is stored
        trainset = datasets.ImageFolder('/home/sysgen/Desktop/Priyank/Datasets/ImageNet100/archive/training/', transform=transform)
        testset = datasets.ImageFolder('/home/sysgen/Desktop/Priyank/Datasets/ImageNet100/archive/training/', transform=transform)
        
        return trainset, testset


def chunk_into_n(lst, n):
  size = ceil(len(lst) / n)
  return list(
    map(lambda x: lst[x * size:x * size + size],
    list(range(n)))
)


def run(dataset_name='MNIST', defence_name='no_defence', attack_name='no_attack', dir=None):
    """
    Creates clients and the server for running experiments
    
    dataset_name: String - The dataset on which all experiments can be run
    defence_name: String - The defence for which the experiment can be run
    attack_name: String - The attack name for which the experiment can be run
    dir: String: - Path where all the plots can be saved
    """
    
    assert dir is not None, 'Please enter directory where all files and png are stored'
    
    use_random_clients = False  # currently only set for CIFAR10
    
    d_ = Defence()
    
    # Set up the information for each dataset. Number of clients, number of attackers, learning rate... etc. 
    if dataset_name == 'MNIST':
        n_clients = 50
        num_communication_rounds = 150
        num_local_rounds = 1  # Number of local epochs for each client before it sends update to central server
        n_attackers = 12
        learning_rate = 0.1
        momentum = 0.9
        l2_regularization = 0.0001
        batch_size = 100
    
    elif dataset_name == 'Breakhis':
        n_clients = 10
        num_communication_rounds = 200
        num_local_rounds = 1
        n_attackers = 3
        learning_rate = 0.0001
        momentum = 0.9
        l2_regularization = 1e-6
        batch_size = 256
        stepsize = 30
    
    elif dataset_name == 'CIFAR':
        n_clients = 10
        num_communication_rounds = 100
        num_local_rounds = 1
        n_attackers = 3
        
        # For CIFAR 10, random clients were chosen as attackers otherwise by default first n = n_attackers < N clients are chosen as attackers
        use_random_clients = True
        if use_random_clients:
            attacker_list = []
            for i in range(n_attackers):
                attacker_id = random.randint(0, n_clients-1)
                while attacker_id in attacker_list:
                    attacker_id = random.randint(0, n_clients-1)
                attacker_list.append(attacker_id)
            assert len(attacker_list) == n_attackers, 'Make sure len(attacker_list) == n_attackers'
                
        learning_rate = 0.001
        momentum = 0.9
        l2_regularization = 0.0001
        batch_size = 1000
    
    elif dataset_name == 'CIFAR-100':
        n_clients = 10
        num_communication_rounds = 500
        num_local_rounds = 1
        n_attackers = 3
        
        use_random_clients = True
        if use_random_clients:
            attacker_list = []
            for i in range(n_attackers):
                attacker_id = random.randint(0, n_clients-1)
                while attacker_id in attacker_list:
                    attacker_id = random.randint(0, n_clients-1)
                attacker_list.append(attacker_id)
            assert len(attacker_list) == n_attackers, 'Make sure len(attacker_list) == n_attackers'
                
        learning_rate = 0.002
        momentum = 0.9
        l2_regularization = 0.0001
        batch_size = 1000
    
    
    elif dataset_name == 'ImageNet':
        n_clients = 6
        num_communication_rounds = 100
        num_local_rounds = 5
        n_attackers = 2
        
        use_random_clients = True
        if use_random_clients:
            attacker_list = []
            for i in range(n_attackers):
                attacker_id = random.randint(0, n_clients-1)
                while attacker_id in attacker_list:
                    attacker_id = random.randint(0, n_clients-1)
                attacker_list.append(attacker_id)
            assert len(attacker_list) == n_attackers, 'Make sure len(attacker_list) == n_attackers'
                
        learning_rate = 0.01
        momentum = 0.9
        l2_regularization = 0.0001
        batch_size = 32
        
        
    # Prepare the dataset
    if dataset_name == 'MNIST':
        train_dataset, test_dataset = download_dataset(dataset=dataset_name)
        testset_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        
        all_indices = np.arange(len(train_dataset)).tolist()
        random.shuffle(all_indices)

        # Dividing data into clients for simulation 
        chunks = chunk_into_n(all_indices, n_clients)

        client_data = []  # client_data = [(train_data_client1, val_data_client_1), ...]
        for i in range(len(chunks)):
            train_data = []
            val_data = []
            for j in range(len(chunks[i])):
                if (j+1) / len(chunks[i]) <= 0.9:
                    train_data.append(train_dataset[chunks[i][j]])
                else:
                    val_data.append(train_dataset[chunks[i][j]])
            client_data.append((train_data, val_data))
    
    elif dataset_name == 'Breakhis':
        train_dataset, val_dataset, test_dataset = download_dataset(dataset=dataset_name)
        testset_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        
        all_indices_train = np.arange(len(train_dataset)).tolist()
        random.shuffle(all_indices_train)
        
        all_indices_val = np.arange(len(val_dataset)).tolist()
        random.shuffle(all_indices_val)

        # Dividing data into clients for simulation
        chunks_train = chunk_into_n(all_indices_train, n_clients)
        chunks_val = chunk_into_n(all_indices_val, n_clients)
        
        assert len(chunks_train) == len(chunks_val), 'Length of chunks for training data != val data'
        
        n_chunks = len(chunks_train)

        client_data = []  # client_data = [(train_data_client1, val_data_client_1), ...]
        for i in range(n_chunks):
            train_data = []
            val_data = []
            for j in chunks_train[i]:
                train_data.append(train_dataset[j])
            for j in chunks_val[i]:
                val_data.append(val_dataset[j])
            client_data.append((train_data, val_data))
    
    elif dataset_name == 'CIFAR':
        train_dataset, test_dataset = download_dataset(dataset=dataset_name)
        testset_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        
        # this is for iid distribution of dataset
        all_indices = np.arange(len(train_dataset)).tolist()
        random.shuffle(all_indices)

        chunks = chunk_into_n(all_indices, n_clients)

        client_data = []  # client_data = [(train_data_client1, val_data_client_1), ...]
        for i in range(len(chunks)):
            train_data = []
            val_data = []
            for j in range(len(chunks[i])):
                if (j+1) / len(chunks[i]) <= 0.9:
                    train_data.append(train_dataset[chunks[i][j]])
                else:
                    val_data.append(train_dataset[chunks[i][j]])
            client_data.append((train_data, val_data))
        
    elif dataset_name == 'CIFAR-100':
        train_dataset, test_dataset = download_dataset(dataset=dataset_name)
        testset_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        # this is for iid
        all_indices = np.arange(len(train_dataset)).tolist()
        random.shuffle(all_indices)

        chunks = chunk_into_n(all_indices, n_clients)

        client_data = []  # client_data = [(train_data_client1, val_data_client_1), ...]
        for i in range(len(chunks)):
            train_data = []
            val_data = []
            for j in range(len(chunks[i])):
                if (j+1) / len(chunks[i]) <= 0.9:
                    #train_data.append(train_dataset[j])
                    train_data.append(train_dataset[chunks[i][j]])
                else:
                    val_data.append(train_dataset[chunks[i][j]])
            client_data.append((train_data, val_data))
        
    elif dataset_name == 'ImageNet':
        train_dataset, test_dataset = download_dataset(dataset=dataset_name)
        testset_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
        
        all_indices = np.arange(len(train_dataset)).tolist()
        random.shuffle(all_indices)

        chunks = chunk_into_n(all_indices, n_clients)

        client_data = []  # client_data = [(train_data_client1, val_data_client_1), ...]
        for i in tqdm(range(len(chunks)), disable=True):
            train_data = []
            val_data = []
            for j in tqdm(range(len(chunks[i])), disable=True):
                if (j+1) / len(chunks[i]) <= 0.9:
                    #train_data.append(train_dataset[j])
                    train_data.append(train_dataset[chunks[i][j]])
                else:
                    val_data.append(train_dataset[chunks[i][j]])
            client_data.append((train_data, val_data))
        
    # INTIALIZE GLOBAL SERVER
    global_server = Server(client_idx=0,
                           train_dataset=None, # global server doesn't need this since it is not training
                           lr=learning_rate, 
                           epochs=None, # global server doesn't need this since it is not training
                           batch_size=None, # global server doesn't need this since it is not training
                           dataset_name=dataset_name
                           )
    
    # INITIALIZE GLOBAL SERVER's MODEL and OPTIMIZER
    if dataset_name == 'MNIST':
        global_server.model = MNIST_model().to(global_server.device)
        global_server.optimizer = torch.optim.SGD(global_server.model.parameters(), 
                                                  lr=global_server.learning_rate, 
                                                  weight_decay=l2_regularization,
                                                  momentum=momentum)
    
    elif dataset_name == 'Breakhis':
        global_server.model = Breakhis_model().to(global_server.device)
        global_server.optimizer = torch.optim.Adam(global_server.model.parameters(),
                                                   lr=global_server.learning_rate)
        
        global_server.scheduler = torch.optim.lr_scheduler.StepLR(global_server.optimizer, step_size=stepsize)
    
    elif dataset_name == 'CIFAR':
        global_server.model = vgg11().to(global_server.device)
        global_server.optimizer = torch.optim.Adam(global_server.model.parameters(),
                                                   lr=global_server.learning_rate)
    
    elif dataset_name == 'CIFAR-100':
        global_server.model = vgg11(n_classes=100).to(global_server.device)
        global_server.optimizer = torch.optim.Adam(global_server.model.parameters(), lr=global_server.learning_rate)
        #global_server.optimizer = torch.optim.SGD(global_server.model.parameters(), lr=global_server.learning_rate, weight_decay=l2_regularization, momentum=momentum)
    
    elif dataset_name == 'ImageNet':
        global_server.model = mobilenet(n_classes=100).to(global_server.device)
        global_server.optimizer = torch.optim.Adam(global_server.model.parameters(),
                                                   lr=global_server.learning_rate)
    
    
    # INITIALIZE CLIENTS 
    client_list = []
    for client_idx in range(1, n_clients+1):
        client = Server(client_idx=client_idx,
                        train_dataset=client_data[client_idx-1][0], 
                        lr=learning_rate,
                        epochs=num_local_rounds, 
                        batch_size=batch_size,
                        dataset_name=dataset_name
                        )
        
        client_list.append(client)
    
    # test the accuracy before starting anything
    global_server.test(test_loader=testset_dataloader, isglobal=True)
    
    # initialize rocs
    rocs = []
    
    # START THE TRAINING
    for rounds in tqdm(range(num_communication_rounds)):
        
        print(f'Current Round:{rounds}')
        
        user_params = []
        
        for client in client_list:
            
            '''
            (1). At the start of every epoch, we copy the global model into each client.
            (2). After the aggregation, the parameters of the global model are updated.
            (3). Again back to step(1).
            '''
            
            if dataset_name == 'MNIST':
                client.model = copy.deepcopy(global_server.model).to(client.device)
                client.optimizer = torch.optim.SGD(client.model.parameters(), 
                                                   lr=client.learning_rate, 
                                                   weight_decay=l2_regularization,
                                                   momentum=momentum)
            
            elif dataset_name == 'Breakhis':
                client.model = copy.deepcopy(global_server.model).to(client.device)
                client.optimizer = torch.optim.Adam(client.model.parameters(), 
                                                   lr=client.learning_rate)
                
                client.scheduler = torch.optim.lr_scheduler.StepLR(client.optimizer, step_size=stepsize)
            
            elif dataset_name == 'CIFAR':
                client.model = copy.deepcopy(global_server.model).to(client.device)
                client.optimizer = torch.optim.Adam(client.model.parameters(), 
                                                   lr=client.learning_rate)
            
            elif dataset_name == 'CIFAR-100':
                client.model = copy.deepcopy(global_server.model).to(client.device)
                client.optimizer = torch.optim.Adam(client.model.parameters(), lr=client.learning_rate)
            
            elif dataset_name == 'ImageNet':
                client.model = copy.deepcopy(global_server.model).to(client.device)
                client.optimizer = torch.optim.Adam(client.model.parameters(), 
                                                   lr=client.learning_rate)
            
            client.train(val_dataset=client_data[client.client_idx-1][1])
            
            param_values=[]
            for param in client.model.parameters():
                # Directly work on parameters on the model
                #param_values = param.data.view(-1) if not len(param_values) else torch.cat((param_values, param.data.view(-1)))
                
                # Directly work on the gradients of the model
                param_values = param.grad.view(-1) if not len(param_values) else torch.cat((param_values, param.grad.view(-1)))
            
            user_params = param_values[None, :] if len(user_params)==0 else torch.cat((user_params, param_values[None,:]), 0)
        
        # ATTACK
        if attack_name == 'no_attack':  # If no attack, then malicious gradients/parameters are same as client gradients as we do not alter them
            malicious_grads = user_params
            del user_params
        
        elif attack_name == 'lie_attack':  # In lie attack we change the gradients/parameters of clients that are actually attackers 
            if use_random_clients:
                l = []
                for i in attacker_list:
                    l.append(user_params[i])
                mal_update = Attack().lie_attack(torch.stack(l), n_attackers)
                malicious_grads = user_params
                for i in attacker_list:
                    malicious_grads[i] = mal_update
            else:
                mal_update = Attack().lie_attack(user_params[:n_attackers], n_attackers)
                mal_updates = torch.stack([mal_update] * n_attackers)
                malicious_grads = torch.cat((mal_updates, user_params[n_attackers:]), 0)
            assert malicious_grads.shape[0] == n_clients, 'LieAttack: The number of concat params != n_clients'
            del user_params
        
        elif attack_name == 'min_max':  # In min_max attack we change the gradients/parameters of clients that are actually attackers
            if use_random_clients:
                l = []
                for i in attacker_list:
                    l.append(user_params[i])
                agg_params = torch.mean(torch.stack(l), 0)
                mal_update = Attack().min_max(all_updates=torch.stack(l), model_re=agg_params, n_attackers=n_attackers)
                malicious_grads = user_params
                for i in attacker_list:
                    malicious_grads[i] = mal_update
            else:
                agg_params = torch.mean(user_params[:n_attackers], 0)
                mal_update = Attack().min_max(all_updates=user_params[:n_attackers], model_re=agg_params, n_attackers=n_attackers)
                mal_updates = torch.stack([mal_update] * n_attackers)
                malicious_grads = torch.cat((mal_updates, user_params[n_attackers:]), 0)
            assert malicious_grads.shape[0] == n_clients, 'MinMax: The number of concat params != n_clients'
        
        elif attack_name == 'our_attack':  # In DISBELIEVE attack we change the gradients/parameters of clients that are actually attackers
            if use_random_clients:
                l = []
                c_l = []
                for i in attacker_list:
                    l.append(user_params[i])
                    c_l.append(client_list[i])
                mal_update = Attack().our_attack_grads(params=torch.stack(l), n_attackers=n_attackers, client_list=c_l, rounds=rounds, dataset_name=dataset_name)
                malicious_grads = user_params
                for i in attacker_list:
                    malicious_grads[i] = mal_update
            
            else:
                # For attacking parameters using DISBELIEVE
                #mal_update = Attack().our_attack(params=user_params[:n_attackers], n_attackers=n_attackers, client_list=client_list[:n_attackers], dataset_name=dataset_name)
                
                # For attacking gradients using DISBELIEVE
                mal_update = Attack().our_attack_grads(params=user_params[:n_attackers], n_attackers=n_attackers, client_list=client_list[:n_attackers], rounds=rounds, dataset_name=dataset_name)
                
                mal_updates = torch.stack([mal_update] * n_attackers)
                malicious_grads = torch.cat((mal_updates, user_params[n_attackers:]), 0)
            
            assert malicious_grads.shape[0] == n_clients, 'OurAttack: The number of concat params != n_clients'
        
        # DEFENCE
        if defence_name == 'no_defence':
            agg_params = torch.mean(malicious_grads, dim=0)  # if no defence then by default agg using mean
        
        elif defence_name == 'trimmed_mean':
            agg_params = Defence().trimmed_mean(params=malicious_grads, beta=0.1, n_clients=n_clients)
            
        elif defence_name == 'krum':
            #agg_params = Defence().krum(params=malicious_grads, num_clients=n_clients, max_num_attackers=n_attackers, mkrum_enabled=False)
            agg_params = Defence().multi_krum(all_updates=malicious_grads, n_attackers=n_attackers, multi_k=False)
        
        elif defence_name == 'mkrum':
            agg_params = Defence().krum(params=malicious_grads, num_clients=n_clients, max_num_attackers=n_attackers, mkrum_enabled=True)
        
        elif defence_name == 'dos':
            agg_params = Defence().dos(params=malicious_grads)
        
        elif defence_name == 'bulyan':
            agg_params = Defence().bulyan(all_updates=malicious_grads, n_attackers=n_attackers)
            
        elif defence_name == 'new_defence':  # Defend using GAMMA aggregation method
            if rounds == 0 or rounds == 100 or rounds == 200 or rounds == 300 or rounds == 400:
                p = True
            else:
                p = False
            
            if use_random_clients:
                agg_params = d_.new_defence_grads(params=malicious_grads, 
                                               nclients=n_clients, 
                                               lr=global_server.learning_rate, 
                                               global_server_model=global_server.model,
                                               dataset_name=dataset_name,
                                               test_loader=copy.deepcopy(testset_dataloader),
                                               plotEnabled=p,
                                               rounds=rounds,
                                               attack_name=attack_name,
                                               attacker_list=attacker_list)
            
            else:
                agg_params = d_.new_defence_grads(params=malicious_grads, 
                                                nclients=n_clients, 
                                                lr=global_server.learning_rate, 
                                                global_server_model=global_server.model,
                                                dataset_name=dataset_name,
                                                test_loader=copy.deepcopy(testset_dataloader),
                                                plotEnabled=p,
                                                rounds=rounds,
                                                attack_name=attack_name)
            
        
        start_idx = 0
        
        # for agg grads
        global_server.optimizer.zero_grad()
        
        # After aggregation set the parameters of the global model
        for i, param in enumerate(global_server.model.parameters()):
            param_ = agg_params[start_idx:start_idx+len(param.data.view(-1))].reshape(param.data.shape)
            start_idx = start_idx + len(param.data.view(-1))
            param.grad = param_
        
        # for agg grads
        global_server.optimizer.step()
        
        # Test the global server for accuracy
        global_server.test(test_loader=testset_dataloader, isglobal=True)
        
        gt = global_server.gt  # should be a list of Number of batches x batch size
        pred = global_server.pred  # should be a list of Number of batches x batch size
        
        assert len(gt) == len(pred), 'Make sure len of gt and pred is same'
        
        gt_l = [item for sublist in gt for item in sublist]
        pred_l = [item for sublist in pred for item in sublist]
                
        gt = np.asarray(gt_l)
        pred = np.asarray(pred_l)
        
        print(f'Shapes:: pred={pred.shape}, gt={gt.shape}')
        
        if dataset_name == 'MNIST' or dataset_name == 'CIFAR':  # 10 classes
            gt = np.eye(10)[gt]
            pred = np.eye(10)[pred]
        elif dataset_name == 'ImageNet' or dataset_name == 'CIFAR-100':  # 100 classes
            gt = np.eye(100)[gt]
            pred = np.eye(100)[pred]
        
        # Check the roc_auc score for the predicted values against the GT
        roc_val = sklearn.metrics.roc_auc_score(gt, pred)
        rocs.append(roc_val)
        
        
    #COMMUNICATION IS OVER
    acc = global_server.test_accuracy
    
    filename = os.path.join(dir, dataset_name + '_' + defence_name + '_' + attack_name + '_' + TYPE + '.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(rocs, f)
    f.close()
    
    client_list[0].plot_curves() 

    
def plot(pickle_file_name, filename):
    file = open(pickle_file_name,'rb')
    l = pickle.load(file)
    
    plt.plot(l)
    plt.savefig(filename)


DATASET_NAME_LIST = ['MNIST', 'Breakhis', 'CIFAR', 'ImageNet', 'CIFAR-100']
DATASET_NAME = DATASET_NAME_LIST[4]

# new_defence = 'GAMMA' the proposed aggregation method
# our_attack = 'DISBELIEVE' attack  

DEFENCE = ['no_defence', 'trimmed_mean', 'krum', 'dos', 'mkrum', 'bulyan', 'new_defence']
ATTACK = ['no_attack', 'lie_attack', 'min_max', 'our_attack']


torch.cuda.empty_cache()


TYPELIST = ['complete_min', 
            'noAngle_maxDist', 
            'noAngle_minDist', 
            'noAngle_maxPlusminby2Dist', 
            'dosAlgo_without_cosine', 
            'std_dev', 
            'complete_max']
TYPE = TYPELIST[2]


# CHOOSE THE DEFENCE AND ATTACKS IN PLACE THEM IN THE DEFENCE LIST AND ATTACK LIST

DEFENCE_L = DEFENCE
ATTACK_L = ATTACK

for defence in tqdm(DEFENCE_L):
    for attack in tqdm(ATTACK_L):
        print('#####################################################\n') 
        print(f'attack={attack}')
        print(f'defence={defence}')
        run(dataset_name=DATASET_NAME, defence_name=defence, attack_name=attack, dir=os.getcwd())
        plot(pickle_file_name=DATASET_NAME + '_' + defence + '_' + attack + '_' + TYPE + '.pkl' ,filename=DATASET_NAME + '_' + defence + '_' + attack + 'test_acc.png')
        print('#####################################################\n')
