import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class Data(object):
    def __init__(self, file_path, alstr, is_dev=False, batch_size=128, **args):
        self.alstr = alstr
        self.is_dev = is_dev
        self.batch_size = batch_size
        self.batch
        self.raw_data = pd.read_csv(file_path, sep='\t')

        self.alphabet = self.make_alphabet(self.alstr)
        self.encoder, self.e_dict = self.one_hot_encoder(self.alphabet)
        self.alphabet_size = len(self.alphabet)

        self.x, self.y = self.format_data(self.raw_data)
        self.input_x, self.input_num = x['desc_vecs'].values, x.drop(['desc_vecs']).values

    def shuffling(self):
        shuffle_indices = np.random.permutation(np.arange(len(self.y)))
        self.input_x = self.input_x[shuffle_indices]
        self.input_num = self.input_num[shuffle_indices]
        self.y = self.y[shuffle_indices]

    def next_batch(self):
        for i in range(len(self.y) / self.batch_size):
            start = i * self.batch_size
            end = min((i + 1) * self.batch_size, len(self.y))
            yield self.input_x[start:end], self.input_num[start:end], self.y[start, end]

    def process_full_description(self, df):
        df['desc_vecs'] = df['item_description'].apply(
                lambda x: self.doc_process(x, self.e_dict)
        )
        return df

    def categorinizer(self, df,
                      col_lists=[
                          'brand_name',
                          'general_cat',
                          'subcat_1',
                          'subcat_2'
                      ]):
        for col in col_lists:
            df[col] = \
                df[col].apply(lambda x: str(x))
            encoder = LabelEncoder()
            encoder.fit(df[col])
            df[col] = encoder.transform(df[col])
            del encoder

        return df

    def split_cat(self, text):
        try: return text.split("/")
        except: return ("No Label", "No Label", "No Label")

    def format_data(self, df):
        df['general_cat'], df['subcat_1'], df['subcat_2'] = \
            zip(*df['category_name'].apply(lambda x: self.split_cat(x)))

        df = self.categorinizer(df)
        # remove missing values in item description
        df = df[pd.notnull(df['item_description'])]
        df = self.process_full_description(df)
        df['item_description'] = df['name'] + ' ' + df['item_description']
        df = df.drop(columns=['name', 'category_name',
                              'train_id', 'item_description'])
        if self.is_dev:
            price = None
            features = df
        else:
            price = df['price']
            features = df.drop(columns=['price'])

        return features, price.values

    def one_hot_encoder(self, alphabet):
        encoder_dict = {}
        encoder = []

        encoder_dict['UNK'] = 0
        encoder.append(np.zeros(len(alphabet), dtype='float32'))

        for i, alpha in enumerate(alphabet):
            onehot = np.zeros(len(alphabet), dtype='float32')
            encoder_dict[alpha] = i + 1
            onehot[i] = 1
            encoder.append(onehot)

        encoder = np.array(encoder, dtype='float32')
        return encoder, encoder_dict

    def doc_process(self, desc, e_dict, l=500):
        desc = desc.strip().lower()
        min_len = min(l, len(desc))
        doc_vec = np.zeros(l, dtype='int64')
        for j in range(min_len):
            if desc[j] in e_dict:
                doc_vec[j] = e_dict[desc[j]]
            else:
                doc_vec[j] = e_dict['UNK']
        return doc_vec

    def make_alphabet(self, alstr):
        return [char for char in alstr]

# train = Data(train_file, config.alstr)
