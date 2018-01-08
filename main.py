import make_models
import test_classify

train_params = {'from': 0, 'to': 3}
test_params = {'from': 3, 'to': 5}

make_models.start(train_params)
test_classify.start(test_params)