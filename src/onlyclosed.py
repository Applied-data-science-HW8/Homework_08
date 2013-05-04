import csv
import random

data_file = '/media/software/data/train.csv'
out_closed_file = '/media/software/data/processed/onlyclosed.csv'
out_open_file = '/media/software/data/processed/onlyopen.csv'

with open(data_file, 'r') as f:
    with open(out_closed_file, 'w') as g:
        with open(out_open_file, 'w') as h:
            reader = csv.reader(f)
            writer_closed = csv.writer(g)
            writer_open = csv.writer(h)
            header = reader.next()
            writer_closed.writerow(header)
            writer_open.writerow(header)
            for row_i, row in enumerate(reader):
                if row[14] != 'open':
                    writer_closed.writerow(row)
                else:
                    writer_open.writerow(row)

