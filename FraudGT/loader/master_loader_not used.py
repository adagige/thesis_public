
@register_loader('custom_master_loader')
def load_dataset_master(format, name, dataset_dir):
    """
    Master loader that controls loading of all datasets, overshadowing execution
    of any default GraphGym dataset loader. Default GraphGym dataset loader are
    instead called from this function, the format keywords `PyG` and `OGB` are
    reserved for these default GraphGym loaders.

    Custom transforms and dataset splitting is applied to each loaded dataset.

    Args:
        format: dataset format name that identifies Dataset class
        name: dataset name to select from the class identified by `format`
        dataset_dir: path where to store the processed dataset

    Returns:
        PyG dataset object with applied perturbation transforms and data splits
    """
    if format.startswith('PyG-'):
        pyg_dataset_id = format.split('-', 1)[1]
        dataset_dir = osp.join(dataset_dir, pyg_dataset_id)

        if pyg_dataset_id == 'DBLP':
            dataset = preformat_DBLP(dataset_dir)

        elif pyg_dataset_id == 'IMDB':
            dataset = preformat_IMDB(dataset_dir)
        
        elif pyg_dataset_id == 'Planetoid':
            dataset = preformat_Planetoid(dataset_dir, name)

        elif pyg_dataset_id == 'MovieLens':
            dataset = preformat_MovieLens(dataset_dir)
        
        else:
            raise ValueError(f"Unexpected PyG Dataset identifier: {format}")

    # GraphGym default loader for Pytorch Geometric datasets
    elif format == 'PyG':
        dataset = load_pyg(name, dataset_dir)

    elif format == 'OGB':
        if name.startswith('ogbn') or name in ['MAG240M', 'mag-year']:
            dataset = preformat_OGB_Node(dataset_dir, name.replace('_', '-'))

        ### Link prediction datasets.
        elif name.startswith('ogbl-'):
            # GraphGym default loader.
            dataset = load_ogb(name, dataset_dir)
            # OGB link prediction datasets are binary classification tasks,
            # however the default loader creates float labels => convert to int.
            def convert_to_int(ds, prop):
                tmp = getattr(ds.data, prop).int()
                set_dataset_attr(ds, prop, tmp, len(tmp))
            convert_to_int(dataset, 'train_edge_label')
            convert_to_int(dataset, 'val_edge_label')
            convert_to_int(dataset, 'test_edge_label')

        else:
            raise ValueError(f"Unsupported OGB(-derived) dataset: {name}")

    elif format == 'AML':
        dataset_dir = osp.join(dataset_dir, format)
        dataset = preformat_AML(dataset_dir, name)

    elif format == 'ETH':
        dataset_dir = osp.join(dataset_dir, format)
        dataset = preformat_ETH(dataset_dir)

    else:
        raise ValueError(f"Unknown data format: {format}")

    # pre_transform_in_memory(dataset, partial(task_specific_preprocessing, cfg=cfg))

    log_loaded_dataset(dataset, format, name)

    # Precompute structural encodings
    if cfg.posenc_Hetero_Node2Vec.enable:
        pe_dir = osp.join(dataset_dir, name.replace('-', '_'), 'posenc')
        if not osp.exists(pe_dir) or not check_Node2Vec(pe_dir):
            preprocess_Node2Vec(pe_dir, dataset)
        
        model = load_Node2Vec(pe_dir)
        if len(dataset) == 1:
            homo_data = dataset[0].to_homogeneous()
            for idx, node_type in enumerate(dataset.data.node_types):
                mask = homo_data.node_type == idx
                dataset.data[node_type]['pestat_Hetero_Node2Vec'] = model[mask]
        elif isinstance(dataset, TemporalDataset):
            for split_idx, split in enumerate(['train', 'val', 'test']):
                homo_data = dataset[split].to_homogeneous()
                for idx, node_type in enumerate(dataset[split].node_types):
                    mask = homo_data.node_type == idx
                    dataset[split][node_type]['pestat_Hetero_Node2Vec'] = model[split_idx][mask]
        else:
            data_list = []
            for idx in range(len(dataset)):
                data = dataset.get(idx)
                data.pestat_Hetero_Node2Vec = model[idx]
                data_list.append(data)
                # if idx >= 10:
                #     break
            data_list = list(filter(None, data_list))

            dataset._indices = None
            dataset._data_list = data_list
            dataset.data, dataset.slices = dataset.collate(data_list)

    if cfg.posenc_Hetero_Metapath.enable:
        pe_dir = osp.join(dataset_dir, name.replace('-', '_'), 'posenc')
        if not osp.exists(pe_dir) or not check_Metapath(pe_dir):
            preprocess_Metapath(pe_dir, dataset)
        
        model = load_Metapath(pe_dir)
        emb = model['model'].weight.data.detach().cpu()
        if hasattr(dataset, 'dynamicTemporal'):
            data_list = []
            for split in range(3):
                data = dataset[split]
                for node_type in dataset.data.node_types:
                    data[node_type]['pestat_Hetero_Metapath'] = \
                        emb[model['start'][node_type]:model['end'][node_type]]
                data_list.append(data)
            dataset._data, dataset.slices = dataset.collate(data_list)
        else:
            for node_type in dataset.data.node_types:
                # if hasattr(dataset.data[node_type], 'x'):
                dataset.data[node_type]['pestat_Hetero_Metapath'] = \
                    emb[model['start'][node_type]:model['end'][node_type]]
        print(dataset.data)
    if cfg.posenc_Hetero_TransE.enable:
        pe_dir = osp.join(dataset_dir, name.replace('-', '_'), 'posenc')
        if not osp.exists(pe_dir) or not check_KGE(pe_dir, 'TransE'):
            preprocess_KGE(pe_dir, dataset, 'TransE')
        
        model = load_KGE(pe_dir, 'TransE', dataset)
        for node_type in dataset.data.num_nodes_dict:
            dataset.data[node_type]['pestat_Hetero_TransE'] = model[node_type].detach().cpu()
    if cfg.posenc_Hetero_ComplEx.enable:
        pe_dir = osp.join(dataset_dir, name.replace('-', '_'), 'posenc')
        if not osp.exists(pe_dir) or not check_KGE(pe_dir, 'ComplEx'):
            preprocess_KGE(pe_dir, dataset, 'ComplEx')
        
        model = load_KGE(pe_dir, 'ComplEx', dataset)
        for node_type in dataset.data.num_nodes_dict:
            dataset.data[node_type]['pestat_Hetero_ComplEx'] = model[node_type]
    if cfg.posenc_Hetero_DistMult.enable:
        pe_dir = osp.join(dataset_dir, name.replace('-', '_'), 'posenc')
        if not osp.exists(pe_dir) or not check_KGE(pe_dir, 'DistMult'):
            preprocess_KGE(pe_dir, dataset, 'DistMult')
        
        model = load_KGE(pe_dir, 'DistMult', dataset)
        for node_type in dataset.data.num_nodes_dict:
            dataset.data[node_type]['pestat_Hetero_DistMult'] = model[node_type]
            
    print(dataset[0])

    # Precompute necessary statistics for positional encodings.
    pe_enabled_list = []
    for key, pecfg in cfg.items():
        if key.startswith('posenc_') and pecfg.enable:
            pe_name = key.split('_', 1)[1]
            pe_enabled_list.append(pe_name)
            if hasattr(pecfg, 'kernel'):
                # Generate kernel times if functional snippet is set.
                if pecfg.kernel.times_func:
                    pecfg.kernel.times = list(eval(pecfg.kernel.times_func))
                logging.info(f"Parsed {pe_name} PE kernel times / steps: "
                             f"{pecfg.kernel.times}")
    if pe_enabled_list:
        if 'LapPE' in pe_enabled_list:
            start = time.perf_counter()
            logging.info(f"Precomputing Positional Encoding statistics: "
                         f"{pe_enabled_list} for all graphs...")
            # Estimate directedness based on 10 graphs to save time.
            is_undirected = all(d.is_undirected() for d in dataset[:10])
            logging.info(f"  ...estimated to be undirected: {is_undirected}")

            pe_dir = osp.join(dataset_dir, 'LapPE')
            file_path = osp.join(pe_dir, f'{dataset.name}.pt')
            if not osp.exists(pe_dir) or not osp.exists(file_path):
                from tqdm import tqdm
                results = []
                for i in tqdm(range(len(dataset)),
                               mininterval=10,
                               miniters=len(dataset)//20):
                    data = compute_posenc_stats(dataset.get(i), 
                                                pe_types=pe_enabled_list,
                                                is_undirected=is_undirected,
                                                cfg=cfg)
                    results.append({'EigVals': data.EigVals, 'EigVecs': data.EigVecs})
                if not osp.exists(pe_dir):
                    os.makedirs(pe_dir)
                torch.save(results, file_path)
            
            from tqdm import tqdm
            results = torch.load(file_path)
            data_list = []
            for i in tqdm(range(len(dataset)),
                        mininterval=10,
                        miniters=len(dataset)//20):
                data = dataset.get(i)
                data.EigVals = results[i]['EigVals']
                data.EigVecs = results[i]['EigVecs']
                data_list.append(data)
            data_list = list(filter(None, data_list))

            dataset._indices = None
            dataset._data_list = data_list
            dataset.data, dataset.slices = dataset.collate(data_list)

            elapsed = time.perf_counter() - start
            timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed)) \
                      + f'{elapsed:.2f}'[-3:]
            logging.info(f"Done! Took {timestr}")

        # start = time.perf_counter()
        # logging.info(f"Precomputing Positional Encoding statistics: "
        #              f"{pe_enabled_list} for all graphs...")
        # # Estimate directedness based on 10 graphs to save time.
        # is_undirected = all(d.is_undirected() for d in dataset[:10])
        # logging.info(f"  ...estimated to be undirected: {is_undirected}")
        # pre_transform_in_memory(dataset,
        #                         partial(compute_posenc_stats,
        #                                 pe_types=pe_enabled_list,
        #                                 is_undirected=is_undirected,
        #                                 cfg=cfg),
        #                         show_progress=True
        #                         )
        # elapsed = time.perf_counter() - start
        # timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed)) \
        #           + f'{elapsed:.2f}'[-3:]
        # logging.info(f"Done! Took {timestr}")

    # Set standard dataset train/val/test splits
    if hasattr(dataset, 'split_idxs'):
        set_dataset_splits(dataset, dataset.split_idxs)
        delattr(dataset, 'split_idxs')

    # # Verify or generate dataset train/val/test splits
    prepare_splits(dataset)

    # # Precompute in-degree histogram if needed for PNAConv.
    # if cfg.gt.layer_type.startswith('PNA') and len(cfg.gt.pna_degrees) == 0:
    #     cfg.gt.pna_degrees = compute_indegree_histogram(
    #         dataset[dataset.data['train_graph_index']])
    #     # print(f"Indegrees: {cfg.gt.pna_degrees}")
    #     # print(f"Avg:{np.mean(cfg.gt.pna_degrees)}")

    return dataset