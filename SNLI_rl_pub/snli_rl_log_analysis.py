import os


def do_analyse_snli_rl(file_path, dev=True, use_loss=False, stop=None, max_prec=1.):
    results = []
    with open(file_path, 'r', encoding='utf-8') as file:
        find_entry = False
        output = [0, 0., 0., 0., 0., 0., 0.]  # loss, dev*3, test*3,
        for line in file:
            if not find_entry:
                if line.startswith('data round'):  # get step
                    output[0] = int(line.split(' ')[-6].split(':')[-1])  # loss
                    if stop is not None and output[0] > stop: break
                if line.startswith('==> for dev'):  # dev
                    output[1] = float(line.split(' ')[-1])
                    output[2] = float(line.split(' ')[-6])
                    output[3] = float(line.split(' ')[-3][:-1])

                    find_entry = True
            else:
                if line.startswith('~~> for test'):  # test
                    output[4] = float(line.split(' ')[-1])
                    output[5] = float(line.split(' ')[-6])
                    output[6] = float(line.split(' ')[-3][:-1])


                    results.append(output)
                    find_entry = False
                    output = [0, 0., 0., 0., 0., 0., 0.]

    # max step
    if len(results) > 0:
        print('max step:', results[-1][0])

    # sort
    sort = 1 if dev else 4
    if use_loss: sort += 1
    output = list(sorted(results, key=lambda elem: elem[sort], reverse=not use_loss))

    for elem in output[:20]:
        if elem[3] <= max_prec:
            print('step: %d, dev: %.4f, dev_loss: %.4f, dev_perc: %.4f,'
                  ' test: %.4f, test_loss: %.4f, test_perc: %.4f,' %
                  (elem[0], elem[1], elem[2], elem[3], elem[4] ,elem[5], elem[6]))


if __name__ == '__main__':
    type_list = ['snli', 'sst']

    type = 0
    file_path = '/Users/xxx/Desktop/tmp/file_transfer/snli_rl/Jan--1-02-05-46_log.txt'
    dev = True
    use_loss = False
    attn_rate = 1.

    do_analyse_snli_rl(file_path, dev, use_loss, None, attn_rate)

