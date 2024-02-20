## Adaptive adversarial 
### Introduction
This project is the code for the paper Adaptive Adversarial Samples based Active Learning for Medical Image Classification 
The implementation is based on Python and Pytorch framework.
  

### Requirements  
The main package and version of the Python environment are as follows
```
# Name                    Version         
python                    3.8.5                    
pytorch                   1.10.1         
torchvision               0.11.2         
cudatoolkit               10.2.89       
cudnn                     7.6.5           
matplotlib                3.3.2              
numpy                     1.19.2        
opencv                    4.6.0.66         
pandas                    1.1.3               
scikit-learn              0.23.2                
tqdm                      4.50.2             
```  

The above environment is successful when running the code of the project. Pytorch has very good compatibility. Thus, I suggest that try to use the existing pytorch environment first.

---  
## Usage 
### 1) Download Project 

Running```git clone https://github.com/HelenMa9998/Easy_hard_AL```  
The project structure and intention are as follows : 
```
Adversarial active learning			# Source code		
    ├── seed.py			 	                                          # Set up random seed
    ├── query_strategies		                                    # All query_strategies
    │   ├── cdal_sampling.py                      # Integrating Bayesian methods with high density and diversity selection
    │   ├── cluster_margin_sampling.py                           # selects a
diverse set of examples on which the model is least confident
    │   ├── bayesian_active_learning_disagreement_dropout.py	  # Deep bayesian query method
    │   ├── entropy_sampling.py		                              # Entropy based query method
    │   ├── kcenter_greedy.py	                              # samples have the smallest maximum distance to the labeled sample set, as a traditional diversity-based method
    │   ├── least_confidence.py                             # selecting samples with the lowest confidence
    │   ├── margin_sampling.py                           # samples that are close to the decision boundary
    │   ├── entropy_sampling_dropout.py		                      # Entropy-based MC dropout query method
    │   ├── random_sampling.py		                              # Random selection
    │   ├── strategy.py                                         # Functions needed for query strategies
    ├── data.py	                                                # Prepare the dataset & initialization and update for training dataset
    ├── handlers.py                                             # Get dataloader for the dataset
    ├── main.py			                                            # An example for code utilization, including the whole process of active learning
    ├── nets.py		                                              # Training models and methods needed for query method
    ├── supervised_baseline.py	                                # An example for supervised learning traning process
    └── utils.py			                                          # Important setups including network, dataset, hyperparameters...
```
### 2) Datasets preparation 
1. Download the datasets from the official address:
   
   Messidor: https://www.adcis.net/en/third-party/messidor/
   
   Breast Cancer Diagnosis: https://iciar2018-challenge.grand-challenge.org/


<!--    BreakHis: https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/ -->
   
2. Modify the data folder path for specific dataset in `data.py`

### 3) Run Active learning process 
Please confirm the configuration information in the [`utils.py`]
```
  python main.py \
      --n_round 34 \
      --n_query 20 \
      --n_init_labeled 100 \
      --dataset_name Messidor \
      --traning_method supervised_train_acc \
      --strategy_name RandomSampling \
      --seed 2022
```
The training results will be saved to the corresponding directory(save name) in `performance.csv`.  
You can also run `supervised_baseline.py` by
```
python supervised_baseline.py
```

## Visualization
1 Active learning performance visualization  
After you got the `performance.csv`, you can run `visualization.py` to visualize the whole process

