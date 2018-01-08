import make_models
import test_classify
import test_recognize

train_params = {'from': 0, 'to': 5}
test_params = {'from': 5, 'to': 20}

make_models.start(train_params)
test_classify.start(test_params)
test_recognize.start(train_params, test_params)
