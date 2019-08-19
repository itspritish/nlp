from fastai.text import *
path = untar_data(URLs.IMDB_SAMPLE)
df = pd.read_csv(path/'texts.csv')

data_lm = TextLMDataBunch.from_csv(path, 'texts.csv')
data_clas = TextClasDataBunch.from_csv(path, 'texts.csv', vocab=data_lm.train_ds.vocab, bs=32)

data_lm.save('data_lm_export.pkl')
data_clas.save('data_clas_export.pkl')

bs = 192

data_lm = load_data(path, 'data_lm_export.pkl', bs=bs)
data_clas = load_data(path, 'data_clas_export.pkl', bs=bs)


torch.cuda.set_device(0)
learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.5)
learn.fit_one_cycle(1, 1e-2)

learn.predict("This is a review about how", n_words=20)


learn.save('ft')
learn.save_encoder('ft_enc')

learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5).to_fp16()
learn.load_encoder('ft_enc')

learn.fit_one_cycle(1, 1e-2)

learn.unfreeze()
learn.fit_one_cycle(3, slice(1e-4, 1e-2))

learn.predict("This was a great movie!")

