import make_models
import test_classify
import test_recognize

train_params = {'from': 0, 'to': 12}
test_params = {'from': 12, 'to': 24}
recognize_variance = 850

make_models.start(train_params)
test_classify.start(test_params)
test_recognize.start(train_params, test_params, recognize_variance)
