from data import get_MSSEG
from handlers import MSSEG_Handler_2d
from nets import Net, Res_Net, Dense_Net
from query_strategies import RandomSampling, EntropySampling, EntropySamplingDropout, BALDDropout, KCenterGreedy, MarginSampling, LeastConfidence, CDALSampling, ClusterMarginSampling
from seed import setup_seed

# important settings
setup_seed()
# params = {
#     'MSSEG':
#         {'n_epoch': 200,
#          'train_args': {'batch_size':32,'shuffle':True, 'num_workers': 4,'drop_last':False},
#          'val_args': {'batch_size': 512,'shuffle':False, 'num_workers': 4,'drop_last':False},
#          'test_args': {'batch_size': 512,'shuffle':False, 'num_workers': 4,'drop_last':False},
#          'optimizer_args': {'lr': 0.001}},  
# }

params = {
    'MSSEG':
        {'n_epoch': 200,
         'train_args': {'batch_size':16,'shuffle':True, 'num_workers': 4,'drop_last':False},
         'val_args': {'batch_size': 16,'shuffle':False, 'num_workers': 4,'drop_last':False},
         'test_args': {'batch_size': 16,'shuffle':False, 'num_workers': 4,'drop_last':False},
         'optimizer_args': {'lr': 0.001}},  
}

# Get data loader
def get_handler(name,train=False,prop=False):
    if name == 'Messidor':
        return Messidor_Handler
    elif name == 'MSSEG':
        return MSSEG_Handler_2d


# Get dataset
def get_dataset(name,supervised):
    if name == 'Messidor':
        return get_Messidor(get_handler(name))
    elif name == 'MSSEG':
        if supervised == True:
            return get_MSSEG(get_handler(name),supervised = True)
        else:
            return get_MSSEG(get_handler(name))
    else:
        raise NotImplementedError


# define network for specific dataset
def get_net(name, device, prop=False):
    if name == 'Messidor':
        # return Net(Res_Net, params[name], device)
        if init==False:
            return Net(Inception_V3, params[name], device)

    elif name == 'MSSEG':
        return Net(Res_Net, params[name], device)
    else:
        raise NotImplementedError
    
# get strategies
def get_strategy(name):
    if name == "RandomSampling":
        return RandomSampling
    elif name == "EntropySampling":
        return EntropySampling
    elif name == "EntropySamplingDropout":
        return EntropySamplingDropout
    elif name == "BALDDropout":
        return BALDDropout
    elif name == "KCenterGreedy":
        return KCenterGreedy
    elif name == "MarginSampling":
        return MarginSampling
    elif name == "LeastConfidence":
        return LeastConfidence
    elif name == "CDALSampling":
        return CDALSampling
    elif name == "ClusterMarginSampling":
        return ClusterMarginSampling
    else:
        raise NotImplementedError
