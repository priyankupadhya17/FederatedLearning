# Federated Learning Research Repository

This repository contains two essential works in the federated learning paradigm, namely:  
a) [**DISBELIEVE Attack**](https://arxiv.org/abs/2308.07387)  
b) **GAMMA Aggregation Method**


## 1. DISBELIEVE Attack

<table>
  <tr>
    <td align="center" width="45%">
      <strong>Attack on Gradients</strong><br>
      <img src="Images/intuition_grad.PNG" alt="Gradient Intuition" width="75%" />
    </td>
    <td style="border-left: 2px solid #999; height: auto;" width="2%"></td>
    <td align="center" width="45%">
      <strong>Attack on Parameters</strong><br>
      <img src="Images/intuition_param.PNG" alt="Parameter Intuition" width="75%" />
    </td>
  </tr>
</table>


## 2. GAMMA Aggregation Method

<p align="center">
  <img src="Images/GAMMA.png" alt="GAMMA Method" width="80%" />
</p>


## 3. Running the Experiments

To run the experiments, simply execute:

```bash
python main.py
```

## 4. Further Information
The file `attack_and_defences.py` contains state-of-the-art model poisoning attacks such as **LIE**, **Min-Max**, and **DISBELIEVE**, along with state-of-the-art aggregation methods including **Trimmed Mean**, **DOS**, and the proposed **GAMMA** aggregation method.

The code supports experiments on various datasets including **BreakHis**, **HAM10k**, **CheXpert**, **CIFAR-10**, **CIFAR-100**, and **MNIST**.  
**Note:** Users are responsible for downloading these datasets manually and storing them locally. The appropriate data paths must then be provided in the `main.py` file.

