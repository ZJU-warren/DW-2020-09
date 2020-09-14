base_link = '/home/warren/Projects/Competitions/DataWhale_202009/DataSet'

raw_set_link = base_link + '/RawSet'
model_set_link = base_link + '/ModelSet'
main_set_link = base_link + '/MainSet'
result_set_link = base_link + '/ResultSet'

raw_train_link = raw_set_link + '/train.csv'
raw_test_link = raw_set_link + '/testA.csv'


clean_train_link = main_set_link + '/clean_train.csv'
clean_test_link = main_set_link + '/clean_test.csv'


model_link = model_set_link + '/%s_%s'

result_link = result_set_link + '/submit_%s.csv'
