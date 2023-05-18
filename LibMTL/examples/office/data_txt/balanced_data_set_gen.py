def get_data_stats(dataset='office-31', stats='class-data'):

    if dataset == 'office-31':
        data_domains = ['amazon', 'dslr', 'webcam']
    elif dataset == 'office-home':
        data_domains = ['Art', 'Clipart', 'Product', 'Real_World']

    data_modes = ['train', 'test', 'val']

    for data_mode in data_modes:
        for data_domain in data_domains:
            # read image list file
            data_file = f'{dataset}/{data_domain}_{data_mode}.txt'
            foo = open(data_file)
            # init class dict
            class_idx_dict = {}
            # init data count
            data_count = 0
            for k, line in enumerate(foo):
                # read line and strip trailing '\n'
                line = line.strip()
                # get data class in this line
                class_k = line.split('/')[2]
                # check if this class is already seen, and init class if not
                if class_k not in class_idx_dict:
                    class_idx_dict[class_k] = []
                # add data idx to the class
                class_idx_dict[class_k].append(k)
                # update data count
                data_count += 1

            # collect the class types in the file
            class_list = list(class_idx_dict.keys())
            # number of classes in the file
            class_count = len(class_list)
            # number of data per class in the file
            class_data_count = {class_:len(class_idx_dict[class_]) for class_ in class_list}

            # display results of scraping
            print(f'\nstats for {dataset}/{data_domain}/{data_mode}')
            if stats=="all":
                print('data count:', data_count)
                print(f'number of classes:', class_count)
                print(f'\nclass data count:\n{class_data_count}') # output too bulky and cluttered
                print(f'maximum and minimum classes')
                print(max(class_data_count, key = class_data_count.get), max(class_data_count.values()))
                print(min(class_data_count, key = class_data_count.get), min(class_data_count.values()))
            elif stats=="min-max-classes":
                print(f'maximum and minimum classes')
                print(max(class_data_count, key = class_data_count.get), max(class_data_count.values()))
                print(min(class_data_count, key = class_data_count.get), min(class_data_count.values()))
            elif stats=="class-data":
                print(f'class data count:\n{class_data_count}') # output too bulky and cluttered
            else:
                # some custom statistic/data you need to check, e.g:
                print('class index list:',class_idx_dict)

def balance_data(dataset, data_domains, data_limits):
    '''
    Example: 
    dataset='office-31'
    data_domains=['amazon'] (just one domain in this case, cinclude multiple domains)
    data_limits={'test':10, 'val':10} (just test and val datasets in this case, can include train as well)
    Here 10 is the data limit, i.e. maximum of 10 data points per class in amazon test and val datasets
    Feasible data domains for office-31:
        ['amazon', 'dslr', 'webcam']
    Feasible data domains for office-home:
        ['Art', 'Clipart', 'Product', 'Real_World']
    '''

    for data_mode in data_limits:
        for data_domain in data_domains:
            # read image list file
            data_file = f'{dataset}/{data_domain}_{data_mode}.txt'
            foo = open(data_file)
            # init class dict
            class_idx_dict = {}
            new_file_list=[]
            for k, line in enumerate(foo):
                # read line and strip trailing '\n'
                line = line.strip()
                # get data class in this line
                class_k = line.split('/')[2]
                # check if this class is already seen, and init class if not
                if class_k not in class_idx_dict:
                    class_idx_dict[class_k] = []
                    # init data count
                    class_data_count = 0
                    # init class data limit exceed flag
                    class_data_limit_exc = False
                if not class_data_limit_exc:
                    # update data count
                    class_data_count += 1
                    # collect the current line to be added to new file
                    new_file_list.append(line)
                    # check class data count has exceeded
                    if class_data_count >= data_limit [data_mode]:
                        class_data_limit_exc = True
            # close file used to read data
            foo.close()
            # create the text in new data
            new_file_text = '\n'.join(new_file_list)
            # open a new file and write data
            new_data_file = f'balanced_data/{dataset}/{data_domain}_{data_mode}.txt'
            foo = open(new_data_file, "w")
            foo.write(new_file_text)
            foo.close()
        






get_data_stats(stats='all')

