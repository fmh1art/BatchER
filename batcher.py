from exp import *
import argparse


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='em-wa', choices=[
                        'em-wa', 'abt-buy', 'em-ag', 'em-ds', 'em-da', 'em-fz', 'em-ia', 'em-beer'])
    parser.add_argument('--pkg_type', type=str, default='diverse',
                        choices=['random', 'similar', 'diverse'])
    parser.add_argument('--demo_selection', type=str, default='covering',
                        choices=['random', 'topk4batch', '1demo41pair', 'covering'])
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size of Batch Prompting')
    parser.add_argument('--demo_percentile', type=int, default=8,
                        help='demo percentile of thresold for covering-based demonstration selection')
    parser.add_argument('--key_file', type=str,
                        default='keys.txt', help='keys file')
    parser.add_argument('--sim_type', type=str, default='StructureAware',
                        choices=['StructureAware', 'SemanticAware'])
    parser.add_argument('--sim_func', type=str, default='ratio', choices=['ratio', 'jaro_winkler', 'SBERT'],
                        help='similarity function, for SemanticAware, you can choose SBERT, for StructureAware, you can only choose ratio/jaro_winkler')
    parser.add_argument('--dir_name', type=str,
                        default='feature_extractor/StructureAware-ratio', help='directory name of the output')
    parser.add_argument('--gpt', type=str,
                        default='gpt-3.5-turbo-0301', help='gpt model name')

    parser.add_argument('--run_all', default=False, action='store_true',
                        help='True: run all trials, False: run with args')

    return parser.parse_args()


def batcher_runall():
    trials = [
        # * you can also test other BatchPrompting framework with trials below
        {
            'dataset_name': 'em-wa',
            'pkg_type': -1,
            'demo_selection': -1
        },
        {
            'dataset_name': 'abt-buy',
            'pkg_type': -1,
            'demo_selection': -1
        },
        {
            'dataset_name': 'em-ag',
            'pkg_type': -1,
            'demo_selection': -1
        },
        {
            'dataset_name': 'em-ds',
            'pkg_type': -1,
            'demo_selection': -1
        },
        {
            'dataset_name': 'em-da',
            'pkg_type': -1,
            'demo_selection': -1
        },
        {
            'dataset_name': 'em-fz',
            'pkg_type': -1,
            'demo_selection': -1
        },
        {
            'dataset_name': 'em-ia',
            'pkg_type': -1,
            'demo_selection': -1
        },
        {
            'dataset_name': 'em-beer',
            'pkg_type': -1,
            'demo_selection': -1
        },
    ]
    for trial in trials:
        dataset_name = trial['dataset_name']
        batch_types = ['random',
                       'similar', 'diverse', ] if trial['pkg_type'] == -1 else trial['pkg_type']
        demo_selections = ['random', 'topk4batch', '1demo41pair',
                           'covering'] if trial['demo_selection'] == -1 else trial['demo_selection']

        batch_size = 8
        demo_percentile = 8
        key_file = 'keys.txt'

        # sim_func_type:          str: StructureAware/SemanticAware
        # sim_func_name:          str: ratio/jaro_winkler/SBERT
        sim_type = 'StructureAware'
        sim_func = 'ratio'

        root = f'outputs/feature_extractor/{sim_type}-{sim_func}/{dataset_name}'

        Prompt.Item_type = Item_type[dataset_name]
        gpt = GPTPOOL(key_file=key_file, temp=0.01, model='gpt-3.5-turbo-0301')

        # Question Batching
        data_b = ERDataset(dataset_name=dataset_name)
        exp_b = Experiment(gpt, data_b, dataset_name, batch_size,
                           EXP=root, sim_type=sim_type, sim_func=sim_func)

        exp_b.similar_batch = exp_b._generate_tol_similar_batch(
            exp_b.c2p, exp_b.batch_size)
        exp_b.diverse_batch = exp_b._generate_tol_diverse_batch(
            exp_b.c2p, exp_b.batch_size)
        exp_b.random_batch = exp_b._generate_tol_random_batch(exp_b.batch_size)

        save_json([[int(t) for t in lis] for lis in exp_b.random_batch],
                  f'./{root}/[random_batch]_batches.json')
        save_json([[int(t) for t in lis] for lis in exp_b.diverse_batch],
                  f'./{root}/[diverse_batch]_batches.json')
        save_json([[int(t) for t in lis] for lis in exp_b.similar_batch],
                  f'./{root}/[similar_batch]_batches.json')

        data_d = ERDataset(dataset_name=dataset_name, used=True)
        exp_d = Experiment(gpt, data_d, dataset_name, batch_size,
                           EXP=root, sim_type=sim_type, sim_func=sim_func)
        for pkg_type in batch_types:
            demo_thre = np.percentile(exp_d.dis, demo_percentile)
            exp_d.logger.info(f'demo_thre: {demo_thre}')
            exp_d.p2ds = exp_d._get_p2ds(exp_d.dis, demo_thre)
            batches = exp_b.load_batches(pkg_type)

            # Demonstration Selection
            for demo_selection in demo_selections:
                if demo_selection == 'random':
                    demos = sample(
                        list(range(len(exp_d.data.test_t1))), batch_size)
                    demos4batches = [demos for _ in range(len(batches))]
                elif demo_selection == 'topk4batch':
                    demos4batches = exp_d._topk_demo_for_batches(
                        batches, exp_d.dis, topk=batch_size)
                elif demo_selection == '1demo41pair':
                    demos4batches = exp_d._allocate_1demo_for_1pair(
                        batches, exp_d.dis, topk=1)
                elif demo_selection == 'covering':
                    demos4batches = exp_d._demo_cover_pair(
                        batches, demo_percentile, cover_cnt_all=1, cover_cnt_batch=1)
                else:
                    raise NotImplementedError

                now = datetime.datetime.now().strftime("%y-%m-%d_%Hh-%Mm-%Ss")
                exp_b.save_prompt(
                    batches, f'{pkg_type}_{demo_selection}_{now}', demos4batches, data=data_b)
                exp_b.query_batch(
                    batches, f'{pkg_type}_{demo_selection}_{now}', demos4batches, data=data_b)
                save_json([[int(t) for t in lis] for lis in demos4batches],
                          f'{root}/{pkg_type}_{demo_selection}_{now}_demos.json')

                unque_demo_cnt = len(
                    set([d for demos in demos4batches for d in demos]))
                exp_b.logger.info(
                    f'dataset: {dataset_name}, batch_type: {pkg_type}, demo_selection: {demo_selection}, unque_demo_cnt: {unque_demo_cnt}')


def batcher(args):
    dataset_name = args.dataset_name
    batch_type = args.pkg_type
    demo_selection = args.demo_selection
    batch_size = args.batch_size
    demo_percentile = args.demo_percentile
    key_file = args.key_file
    sim_type = args.sim_type
    sim_func = args.sim_func
    gpt = args.gpt
    dir_name = f'outputs/{args.dir_name}/{dataset_name}'

    Prompt.Item_type = Item_type[dataset_name]
    gpt = GPTPOOL(key_file=key_file, temp=0.01, model=gpt)

    # Question Batching
    data_b = ERDataset(dataset_name=dataset_name)
    exp_b = Experiment(gpt, data_b, dataset_name, batch_size,
                       EXP=dir_name, sim_type=sim_type, sim_func=sim_func)

    exp_b.similar_batch = exp_b._generate_tol_similar_batch(
        exp_b.c2p, exp_b.batch_size)
    exp_b.diverse_batch = exp_b._generate_tol_diverse_batch(
        exp_b.c2p, exp_b.batch_size)
    exp_b.random_batch = exp_b._generate_tol_random_batch(exp_b.batch_size)

    save_json([[int(t) for t in lis] for lis in exp_b.random_batch],
              f'./{dir_name}/[random_batch]_batches.json')
    save_json([[int(t) for t in lis] for lis in exp_b.diverse_batch],
              f'./{dir_name}/[diverse_batch]_batches.json')
    save_json([[int(t) for t in lis] for lis in exp_b.similar_batch],
              f'./{dir_name}/[similar_batch]_batches.json')

    data_d = ERDataset(dataset_name=dataset_name, used=True)
    exp_d = Experiment(gpt, data_d, dataset_name, batch_size,
                       EXP=dir_name, sim_type=sim_type, sim_func=sim_func)
    demo_thre = np.percentile(exp_d.dis, demo_percentile)
    exp_d.logger.info(f'demo_thre: {demo_thre}')
    exp_d.p2ds = exp_d._get_p2ds(exp_d.dis, demo_thre)
    batches = exp_b.load_batches(batch_type)

    # Demonstration Selection
    if demo_selection == 'random':
        demos = sample(list(range(len(exp_d.data.test_t1))), batch_size)
        demos4batches = [demos for _ in range(len(batches))]
    elif demo_selection == 'topk4batch':
        demos4batches = exp_d._topk_demo_for_batches(
            batches, exp_d.dis, topk=batch_size)
    elif demo_selection == '1demo41pair':
        demos4batches = exp_d._allocate_1demo_for_1pair(
            batches, exp_d.dis, topk=1)
    elif demo_selection == 'covering':
        demos4batches = exp_d._demo_cover_pair(
            batches, demo_percentile, cover_cnt_all=1, cover_cnt_batch=1)
    else:
        raise NotImplementedError

    now = datetime.datetime.now().strftime("%y-%m-%d_%Hh-%Mm-%Ss")
    exp_b.save_prompt(
        batches, f'{batch_type}_{demo_selection}_{now}', demos4batches, data=data_b)
    exp_b.query_batch(
        batches, f'{batch_type}_{demo_selection}_{now}', demos4batches, data=data_b)
    save_json([[int(t) for t in lis] for lis in demos4batches],
              f'{dir_name}/{batch_type}_{demo_selection}_{now}_demos.json')

    unque_demo_cnt = len(
        set([d for demos in demos4batches for d in demos]))
    exp_b.logger.info(
        f'dataset: {dataset_name}, batch_type: {batch_type}, demo_selection: {demo_selection}, unque_demo_cnt: {unque_demo_cnt}')



if __name__ == '__main__':
    args = argparser()
    if args.run_all:
        batcher_runall()
    else:
        batcher(args)
