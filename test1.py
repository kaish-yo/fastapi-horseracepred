from model.main import MainData

# MainData.test_create_model()
race_id = 202105040502
# df = MainData.scrape_pred(race_id)
# print(df)
all_data = MainData.test_get_data()
print(all_data,all_data.shape)
# pred = MainData.predict(race_id)
# print(pred)