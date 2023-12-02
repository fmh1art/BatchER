from exp import *


def bp_vs_sp():

    batch_size = 8
    pkg_type = 'random'
    demo_selection = 'random'
    tril_num = 3
    key_file = 'keys.txt'

    for dataset_name in [
        'em-wa',
        'abt-buy',
        'em-ag',
        'em-ds',
        'em-da',
        'em-fz',
        'em-ia',
        'em-beer',
    ]:

        Prompt.Item_type = Item_type[dataset_name]
        for tri in range(1, tril_num+1):
            # 设置随机种子
            random.seed(tri+42)
            np.random.seed(tri+42)

            root_batch = f'outputs/BPvsSP/{dataset_name}/trial{tri}/BatchPrompting'
            root_standard = f'outputs/BPvsSP/{dataset_name}/trial{tri}/StandardPrompting'

            gpt = GPTPOOL(key_file=key_file, temp=0.01,
                          model='gpt-3.5-turbo-0301')

            # Batch Prompting
            data = ERDataset(dataset_name=dataset_name)
            exp = Experiment(gpt, data, dataset_name,
                             batch_size, EXP=root_batch)
            exp.random_batch = exp._generate_tol_random_batch(exp.batch_size)
            save_json([[int(t) for t in lis] for lis in exp.random_batch],
                      f'./{root_batch}/[random_batch]_batches.json')
            batches = exp.random_batch
            demos = sample(list(range(len(exp.data.test_t1))), batch_size)
            demos4batches = [demos for _ in range(len(batches))]

            now = datetime.datetime.now().strftime("%y-%m-%d_%Hh-%Mm-%Ss")
            exp.query_batch(
                batches, f'{pkg_type}_{demo_selection}_{now}', demos4batches, explain=False, data=data)
            unque_demo_cnt = len(
                set([d for demos in demos4batches for d in demos]))
            exp.logger.info(
                f'dataset: {dataset_name}, batch_type: {pkg_type}, demo_selection: {demo_selection}, unque_demo_cnt: {unque_demo_cnt}')

            # Standard Prompting
            single_prop = SinglePrompt(gpt, data, root_standard, model_name='gpt-3.5-turbo-0301',
                                       predefined_demos=demos, task_description=exp.task_description)
            single_prop.run()


bp_vs_sp()
