from helpers import *


def get_signatures_of_class_name(folder, class_name):
    result = []
    files = os.listdir(folder)

    for file in files:
        if file.startswith(class_name):
            result.append(file)

    return result


def start(train_params, test_params):
    base_dir = './TrainingSet'
    offline_genuine = 'Offline Genuine'
    offline_forgeries = 'Offline Forgeries'
    models_dir = './models'

    model_files = os.listdir(models_dir)
    genuine_signatures = os.listdir(base_dir + '/' + offline_genuine)

    signatures_dictionary = {}

    for signature_file_name in genuine_signatures:
        key = signature_file_name.split('_')[0]

        if key not in signatures_dictionary:
            signatures_dictionary[key] = []

        path_to_signature = base_dir + '/' + offline_genuine + '/' + signature_file_name
        signatures_dictionary[key].append(path_to_signature)

    errors = 0
    length = 0

    for key in signatures_dictionary:
        signatures = signatures_dictionary[key][train_params['from']:train_params['to']]

        for model_file in model_files:
            if model_file.startswith(key):
                hmm_model = joblib.load(models_dir + '/' + model_file)

        scores = []

        for signature in signatures:
            scores.append(hmm_model.score(get_image_features(signature)))

        mean = np.mean(scores)

        forgeries_signatures = get_signatures_of_class_name(base_dir + '/' + offline_forgeries, key)
        genuine_signatures = get_signatures_of_class_name(base_dir + '/' + offline_genuine, key)

        length = length + len(forgeries_signatures)
        length = length + len(genuine_signatures)

        for signature in forgeries_signatures:
            score = hmm_model.score(get_image_features(base_dir + '/' + offline_forgeries + '/' + signature))

            if score > mean:
                errors = errors + 1

        for signature in genuine_signatures:
            score = hmm_model.score(get_image_features(base_dir + '/' + offline_genuine + '/' + signature))

            if score < mean:
                errors = errors + 1

    print(1 - errors / length)
