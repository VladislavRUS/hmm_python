from helpers import *


def start(train_params):
    base_dir = './TrainingSet'
    offline_genuine = 'Offline Genuine'

    genuine_signatures = os.listdir(base_dir + '/' + offline_genuine)

    signatures_dictionary = {}

    for signature_file_name in genuine_signatures:
        key = signature_file_name.split('_')[0]

        if key not in signatures_dictionary:
            signatures_dictionary[key] = []

        path_to_signature = base_dir + '/' + offline_genuine + '/' + signature_file_name
        signatures_dictionary[key].append(path_to_signature)

    for key in signatures_dictionary:
        signatures = signatures_dictionary[key][train_params['from']:train_params['to']]

        models = []
        train_features = []

        for signature_file_name in signatures:
            train_features.append(get_image_features(signature_file_name))

        hmm_model = hmm.GMMHMM(4)
        data, lengths = get_training_data(train_features)
        hmm_model.fit(data, lengths)
        models.append(hmm_model)

        best_model = get_best_model(models, signatures)
        joblib.dump(best_model, 'models' + '/' + extract_image_name(signatures[0]) + '.pkl')
