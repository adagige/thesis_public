import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from datasets.aml_dataset import AMLDataset
from sampler.sampler_custom import get_LinkNeighborLoader
#from networks.gt_model import GTModel


global cfg


args = parse_args()
# Load config file
set_cfg(cfg)
load_cfg(cfg, args)
custom_set_out_dir(cfg, args.cfg_file, cfg.name_tag, args.gpu)
dump_cfg(cfg)


cfg.dataset.task_entity = ('node', 'to', 'node')
cfg.device = 'cpu'


dataset_dir = 'Documents/GitHub/thesis/fraudGT/aml_datasets'
name = 'Small-HI'
batch_size = 50 #original 2048 
shuffle = True #Always set to true from the get_loader function



dataset = AMLDataset(root=dataset_dir, name=name, reverse_mp=True, #cfg.dataset.reverse_mp,
                         add_ports= True#cfg.dataset.add_ports
                         )



#Get training sampler 
loaders = [get_LinkNeighborLoader(dataset, batch_size=batch_size, shuffle=shuffle, split='train')]

loaders.append(get_LinkNeighborLoader(dataset, batch_size=batch_size, shuffle=shuffle, split='val'))

loaders.append(get_LinkNeighborLoader(dataset, batch_size=batch_size, shuffle=shuffle, split='test'))



