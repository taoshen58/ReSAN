


def do_analyse_sick_rl(file_path, dev=True, delta=0, stop=None, max_prec=1.):
    results = []
    with open(file_path, 'r', encoding='utf-8') as file:
        find_entry = False
        output = [0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        # step, dev(loss_rl, loss_sl, ,prec, pearson, spearman, mse), \
        # test(oss_rl, loss_sl, ,prec,loss, pearson, spearman, mse),
        for line in file:
            if not find_entry:
                if line.startswith('data round'):  # get step
                    output[0] = int(line.split(' ')[-6].split(':')[-1])
                    if stop is not None and output[0] > stop: break
                if line.startswith('==> for dev'):  # dev
                    output[1] = float(line.split(' ')[-10])
                    output[2] = float(line.split(' ')[-9][:-1])
                    output[3] = float(line.split(' ')[-7][:-1])
                    output[4] = float(line.split(' ')[-5][:-1])  # pearson
                    output[5] = float(line.split(' ')[-3][:-1])  # spearman
                    output[6] = float(line.split(' ')[-1])  # mse
                    find_entry = True
            else:
                if line.startswith('~~> for test'):  # test
                    output[7] = float(line.split(' ')[-10])
                    output[8] = float(line.split(' ')[-9][:-1])
                    output[9] = float(line.split(' ')[-7][:-1])
                    output[10] = float(line.split(' ')[-5][:-1])
                    output[11] = float(line.split(' ')[-3][:-1])
                    output[12] = float(line.split(' ')[-1])
                    results.append(output)
                    find_entry = False
                    output = [0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    # max step
    if len(results) > 0:
        print('max step:', results[-1][0])
    # sort
    sort = 4 if dev else 10
    sort += delta
    output = list(sorted(results, key=lambda elem: elem[sort], reverse=delta in [0, 1]))

    for elem in output[:10]:
        print('step: %d, d_loss: %.4f %.4f, d_prec: %.2f, d_prsn: %.4f, d_spm: %.4f, d_mse: %.4f,'
              ' t_loss: %.4f %.4f, t_prec: %.2f, t_prsn: %.4f, t_spm: %.4f, t_mse: %.4f,' %
              (elem[0], elem[1], elem[2], elem[3], elem[4], elem[5], elem[6],
               elem[7], elem[8], elem[9], elem[10], elem[11], elem[12]))

if __name__ == '__main__':

    file_path = '/Users/xxx/Desktop/tmp/file_transfer/sick_rl/Dec-25-02-53-24_log.txt'
    dev = True
    delta = 2
    max_prec = 1.

    do_analyse_sick_rl(file_path, dev, delta, None, max_prec)