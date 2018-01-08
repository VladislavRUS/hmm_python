from helpers import *


def start(test_params):
    base_dir = './TrainingSet'
    offline_genuine = 'Offline Genuine'
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

    test_signatures = []

    for key in signatures_dictionary:
        signatures = signatures_dictionary[key][test_params['from']:test_params['to']]

        for signature in signatures:
            test_signatures.append(signature)

    errors = 0

    for signature in test_signatures:

        scores = []

        for model_file in model_files:
            hmm_model = joblib.load(models_dir + '/' + model_file)
            scores.append(hmm_model.score(get_image_features(signature)))

        highest_score_model = model_files[np.argmax(scores)]

        image_name = extract_image_name(signature)
        model_name = highest_score_model.split('.')[0]

        if image_name != model_name:
            errors = errors + 1

    print(1 - (errors / len(test_signatures)))
