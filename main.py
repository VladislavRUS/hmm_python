import make_models
import test_classify
import test_recognize

train_params = {'from': 0, 'to': 10}
test_params = {'from': 10, 'to': 20}

make_models.start(train_params)
test_classify.start(test_params)
test_recognize.start(train_params, test_params)
