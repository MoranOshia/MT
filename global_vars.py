import torch

# Shared vars

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_device_inf = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_hidden_size = 256

_MAX_LENGTH = 15

_SOS_token = 0

_EOS_token = 1

# Pre Process vars

_lang1 = 'heb'

_lang2 = 'arm'

_reverse = True

# Training vars

_dictionary_name = "dictionaries/heb-arm-dictionary.pickle"

_teacher_forcing_ratio = 0.5

_n_iters = 75000

_print_every= 50

_plot_every=100

_learning_rate=0.01

_dropout_p=0.1

_load_pickle = False

_pickle_name = ""

# Inference vars

_model_name = "model/fra-eng-model.pickle"
