import requests
from joblib import Parallel, delayed
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import pyrebase
import json
import os


class MainData():
    def database_connect(): #DBに接続する用の関数
        '''Firebaseにログインしてデータを引っ張ってくる'''
        ##production環境では環境変数にする
        config = {
            "apiKey": "AIzaSyAwbn5Kn_APmjjbePcurULzaGqrqC46dsg",
            "authDomain": "horse-race-pred-firebase.firebaseapp.com",
            "databaseURL": "https://horse-race-pred-firebase-default-rtdb.asia-southeast1.firebasedatabase.app",
            "projectId": "horse-race-pred-firebase",
            "storageBucket": "horse-race-pred-firebase.appspot.com",
            "messagingSenderId": "77780114055",
            "appId": "1:77780114055:web:ccecc174cedfbc77b8f622",
            "measurementId": "G-Z71C5EXW6P"
        }
        firebase = pyrebase.initialize_app(config)
        db = firebase.database()
        return db


    def scrape_train_data():
        this_year = datetime.now().year
        years = [this_year-1,this_year] #西暦 
        def scraper(year): #この関数で入力した年をIDに含むレコードをスクレイピングする
            base_URL = "https://race.netkeiba.com/race/result.html?race_id="
            row_list=[]
            '''Correct after testing'''
            id1 = range(0,12) #04~07まででほぼ確定
            id2 = range(0,12) #01~03まででほぼ確定
            id3 = range(0,13) #01~12までっぽい
            id4 = range(1,13) #01~12までであることはほぼ確定
            '''テスト用'''
            # id1 = range(4,5) #04~07まででほぼ確定
            # id2 = range(0,2) #01~03まででほぼ確定
            # id3 = range(0,3) #01~12までっぽい
            # id4 = range(1,3) #01~12までであることはほぼ確定
            ''''''
            for j in id1:
                for k in id2:
                    for l in id3:
                        for m in id4:
                            race_id = str(year) + f"{j:02d}" + f"{k:02d}" + f"{l:02d}" + f"{m:02d}"
                            r = requests.get(base_URL + race_id,
                                            headers={'User-agent':'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:61.0) Gecko/20100101 Firefox/61.0'})
                            c = r.content
                            soup = BeautifulSoup(c,"html.parser")
                            if len(soup.find_all("table",{"summary":"全着順"})) >0:
                                #あらかじめ日付情報を取得しておく
                                date_info = soup.find("dl",{"id":"RaceList_DateList"}).find("dd",{"class":"Active"}).a.text
                                date_info=date_info.split("(")[0]
                                if "月" in date_info:
                                    date_info = str(year) + '年' + date_info
                                    race_date = datetime.strptime(date_info,'%Y年%m月%d日')
                                    race_date = race_date.timestamp()*1000
                                else: #"/"の場合
                                    date_info = str(year) + '/' + date_info
                                    race_date = datetime.strptime(date_info,'%Y/%m/%d')
                                    race_date = race_date.timestamp()*1000
                                
                                print(f"This table is newer than the last record!:\n{race_id}")
                                all_rows = soup.find_all("tr",{"class":"FirstDisplay","class":"HorseList"})
                                #そのほかの共通データ
                                race_name = soup.find("div",{"class":"RaceName"}).text.rstrip("\n")
                                race_data_1 = soup.find("div",{"class":"RaceData01"}).text.split("/")
                                rd2 = soup.find("div",{"class":"RaceData02"}).find_all("span")
                                race_data_2 = [x.text for x in rd2]
                                for item in all_rows:
                                    d = {}
                                    #Race id
                                    d["Race_id"]=race_id
                                    #Race date
                                    d["Race_date"]=race_date
                                    #Race Name
                                    d["Race_Name"]=race_name
                                    #Race data 1
                                    d["Race_data_1"]=race_data_1
                                    #Race data 2
                                    d["Race_data_2"]=race_data_2
                                    #Ranking
                                    d["Ranking"]=item.find("div",{"class":"Rank"}).text
                                    #Uniform number
                                    d["Uni_Num"]=item.find("td",{"class":"Num"}).div.text
                                    #Horse number
                                    d["Hor_Num"]=item.find("td",{"class":"Num","class":"Txt_C"}).div.text
                                    #Horse Name
                                    d["Hor_name"]=item.find("span",{"class":"Horse_Name"}).a.text
                                    #Horse sex and age
                                    d["Hor_sex_and_age"]=item.find("span",{"class":"Lgt_Txt","class":"Txt_C"}).text
                                    #JockeyWeight
                                    d["JockeyWeight"]=item.find("span",{"class":"JockeyWeight"}).text
                                    #Jockey
                                    d["Jockey"]=item.find("td",{"class":"Jockey"}).a.text
                                    #Race Time
                                    d["Race_Time"]=item.find_all("span",{"class":"RaceTime"})[0].text
                                    #Arrival diff
                                    d["Arrival_diff"]=item.find_all("td",{"class":"Time"})[1].text
                                    #popularity
                                    d["Popularity"]=item.find("span",{"class":"OddsPeople"}).text
                                    #単勝人気
                                    d["Odds_popularity"]=item.find_all("td",{"class":"Odds"})[1].text
                                    #後3F
                                    d["ato_3_F"]=item.find_all("td",{"class":"Time"})[-1].text
                                    #Corner Ranking
                                    d["Corner_ranking"]=item.find("td",{"class":"PassageRate"}).text
                                    #厩舎1
                                    d["Trainer_1"]=item.find("td",{"class":"Trainer"}).span.text
                                    #厩舎2
                                    d["Trainer_2"]=item.find("td",{"class":"Trainer"}).a.text
                                    #HorseWeight
                                    d["Horse_weight"]=item.find("td",{"class":"Weight"}).text[:4]
                                    #HorseWeight flux
                                    d["Horse_weight_Flux"]=item.find("td",{"class":"Weight"}).small.text
                                    row_list.append(d)
                                print(f"craped data of {race_id}:\n{row_list}")
                                time.sleep(1)
            return row_list
        row_list = Parallel(n_jobs=-1,verbose=10)(delayed(scraper)(year) for year in years)
        # print(row_list)
        df = pd.DataFrame(row_list)
        print(f"df after scraping:\n{df}")
        
        """##取得データの整形"""
        df = df.transpose()
        df_con = pd.DataFrame()
        for column in df.columns:
            df_temp = df[column]
            df_con = pd.concat([df_con,df_temp])
        df_con.columns = ['data']
        df = df_con
        #ここでunpackの記述をするんご
        df = df.dropna().reset_index(drop=True)
        df_unpacked = []

        for row in df['data']:
            try:
                df_unpacked.append(ast.literal_eval(row)) #ast.literal_evalで文字列を辞書形式に変換している
            except ValueError:
                df_unpacked.append(row)
        df = pd.DataFrame(df_unpacked)
        print(f"df after unpacking:\n{df}")
        '''テスト用に保存する'''
        # try:
        #     with open("data.bk","wb") as f:
        #         pickle.dump(df,f)
        # except:
        #     pass
        ''''''
        #さらに列を分解する
        df['start_time'] = df['Race_data_1'].apply(lambda x: x[0].strip().replace('発走',''))
        df['field_type'] =df['Race_data_1'].apply(lambda x: x[1].strip())
        df['weather'] = df['Race_data_1'].apply(lambda x: x[2].strip().replace('天候:',''))
        try:
            df['field_condition'] =df['Race_data_1'].apply(lambda x: x[3].strip().replace('馬場:',''))
        except:
            df['field_condition'] = None
        df['competition_count'] = df['Race_data_2'].apply(lambda x: x[0].strip().replace('回',''))
        df['venue_area'] = df['Race_data_2'].apply(lambda x: x[1].strip())
        df['venue_area'] = df['Race_data_2'].apply(lambda x: x[1].strip())
        df['day_count'] = df['Race_data_2'].apply(lambda x:x[2].strip().replace('日目',''))
        df['horse_class'] = df['Race_data_2'].apply(lambda x:x[3].strip())
        df['horse_class_2'] = df['Race_data_2'].apply(lambda x:x[4].strip())
        df['race_type'] = df['Race_data_2'].apply(lambda x:x[5].strip())
        df['hadicap'] = df['Race_data_2'].apply(lambda x:x[6].strip())
        df['number_of_hourses'] = df['Race_data_2'].apply(lambda x:x[7].strip().replace('頭',''))
        #*賞金データはあまり関係なさそうだからdrop

        #Cleanse the follwing columns
        df['Hor_sex_and_age'] = df['Hor_sex_and_age'].apply(lambda x: x.strip().replace('\n',''))
        df['Arrival_diff'] = df['Arrival_diff'].apply(lambda x: x.strip().replace('\n',''))
        df['Odds_popularity'] = df['Odds_popularity'].apply(lambda x: x.strip().replace('\n',''))
        df['ato_3_F'] = df['ato_3_F'].apply(lambda x: x.strip().replace('\n',''))
        df['Corner_ranking'] = df['Corner_ranking'].apply(lambda x: x.strip().replace('\n',''))
        df['Horse_weight_Flux'] = df['Horse_weight_Flux'].apply(lambda x: x.strip().replace('(','').replace(')','').replace('+',''))
        #race_dataによっては時間情報がなくslicingがうまく行かない場合があるので、下記の関数を別途定義する
        def start_time_converter(x):
            try:
                start = int(x.strip().replace(':',''))
            except:
                start = None
            return start
        df['start_time'] = df['start_time'].apply(start_time_converter)
        df['Horse_weight'] = df['Horse_weight'].apply(lambda x: x.strip().replace(' ',''))
        df['Jockey'] = df['Jockey'].apply(lambda x: x.strip().replace("▲","").replace("△","").replace("☆","").replace("◇",""))

        # Race Timeをコンバートするには下記の関数を使う
        def get_sec(time_str):
            """Get Seconds from time."""
            try:
                m, s = time_str.split(':')
            except:
                return None #おそらくまだ順位が出ていないデータに対して
            return  int(m) * 60 + float(s)

        df['Race_Time'] = df['Race_Time'].apply(get_sec) #上記の関数を適用

        #順位の列は下記の関数を使って分解する
        def rank_spliter_2(x):
            try:
                rank = x.split('-')[1]
                return rank
            except:
                return None

        def rank_spliter_3(x):
            try:
                rank = x.split('-')[2]
                return rank
            except:
                return None

        def rank_spliter_4(x):
            try:
                rank = x.split('-')[3]
                return rank
            except:
                return None

        #さらに下記の列を分解する
        df['field_length'] = df['field_type'].apply(lambda x: None if '候' in x else x[1:5])
        df['field_type_1'] = df['field_type'].apply(lambda x: x[0])
        df['field_type_2'] = df['field_type'].apply(lambda x: x[7:].strip().replace('(','').replace(')',''))
        df['corner_rank_1'] = df['Corner_ranking'].apply(lambda x: x.split('-')[0])
        df['corner_rank_2'] = df['Corner_ranking'].apply(rank_spliter_2)
        df['corner_rank_3'] = df['Corner_ranking'].apply(rank_spliter_3)
        df['corner_rank_4'] = df['Corner_ranking'].apply(rank_spliter_4)

        #不要な列を削除する（主に分解し終えたもの）
        df = df.drop(['Race_data_1','Race_data_2','field_type','Corner_ranking','Arrival_diff'],axis=1)

        df = df[~df['Ranking'].isin(['中止', '除外', '取消'])] #失格などのレコードを削除

        # データタイプを一括で変更する
        changed_columns=list(df.columns)
        changed_columns.remove('Race_date')
        print(changed_columns)
        for column in changed_columns:
            df[column] = pd.to_numeric(df[column],errors='ignore')

        df['Ranking'].unique()

        #targetとなるRankingを2分類に分ける。エラー対策のため下記のデータ加工を施す
        df = df[df['Ranking']!='失格']
        df['Ranking'] = df['Ranking'].astype('int16')
        df['target'] = df['Ranking'].apply(lambda x: 0 if x<=3 else 1)

        # df['target'] = df['Ranking'].apply(lambda x: 0 if x <= 3 else (1 if x <= 6 else (2 if x <= 9 else 3)))
        return df
        row_list = Parallel(n_jobs=-1,verbose=10)(delayed(scraper)(year,last_date) for year in years)
        # print(row_list)
        df = pd.DataFrame(row_list)
        print(f"df after scraping:\n{df}")
        
        """##取得データの整形"""
        df = df.transpose()
        df_con = pd.DataFrame()
        for column in df.columns:
            df_temp = df[column]
            df_con = pd.concat([df_con,df_temp])
        df_con.columns = ['data']
        df = df_con
        #ここでunpackの記述をするんご
        df = df.dropna().reset_index(drop=True)
        df_unpacked = []

        for row in df['data']:
            try:
                df_unpacked.append(ast.literal_eval(row)) #ast.literal_evalで文字列を辞書形式に変換している
            except ValueError:
                df_unpacked.append(row)
        df = pd.DataFrame(df_unpacked)
        print(f"df after unpacking:\n{df}")
        
        '''テスト用に保存する'''
        # try:
        #     with open("data.bk","wb") as f:
        #         pickle.dump(df,f)
        # except:
        #     pass
        ''''''

        #さらに列を分解する
        df['start_time'] = df['Race_data_1'].apply(lambda x: x[0].strip().replace('発走',''))
        df['field_type'] =df['Race_data_1'].apply(lambda x: x[1].strip())
        df['weather'] = df['Race_data_1'].apply(lambda x: x[2].strip().replace('天候:',''))
        try:
            df['field_condition'] =df['Race_data_1'].apply(lambda x: x[3].strip().replace('馬場:',''))
        except:
            df['field_condition'] = None
        df['competition_count'] = df['Race_data_2'].apply(lambda x: x[0].strip().replace('回',''))
        df['venue_area'] = df['Race_data_2'].apply(lambda x: x[1].strip())
        df['venue_area'] = df['Race_data_2'].apply(lambda x: x[1].strip())
        df['day_count'] = df['Race_data_2'].apply(lambda x:x[2].strip().replace('日目',''))
        df['horse_class'] = df['Race_data_2'].apply(lambda x:x[3].strip())
        df['horse_class_2'] = df['Race_data_2'].apply(lambda x:x[4].strip())
        df['race_type'] = df['Race_data_2'].apply(lambda x:x[5].strip())
        df['hadicap'] = df['Race_data_2'].apply(lambda x:x[6].strip())
        df['number_of_hourses'] = df['Race_data_2'].apply(lambda x:x[7].strip().replace('頭',''))
        #*賞金データはあまり関係なさそうだからdrop

        #Cleanse the follwing columns
        df['Hor_sex_and_age'] = df['Hor_sex_and_age'].apply(lambda x: x.strip().replace('\n',''))
        df['Arrival_diff'] = df['Arrival_diff'].apply(lambda x: x.strip().replace('\n',''))
        df['Odds_popularity'] = df['Odds_popularity'].apply(lambda x: x.strip().replace('\n',''))
        df['ato_3_F'] = df['ato_3_F'].apply(lambda x: x.strip().replace('\n',''))
        df['Corner_ranking'] = df['Corner_ranking'].apply(lambda x: x.strip().replace('\n',''))
        df['Horse_weight_Flux'] = df['Horse_weight_Flux'].apply(lambda x: x.strip().replace('(','').replace(')','').replace('+',''))
        #race_dataによっては時間情報がなくslicingがうまく行かない場合があるので、下記の関数を別途定義する
        def start_time_converter(x):
            try:
                start = int(x.strip().replace(':',''))
            except:
                start = None
            return start
        df['start_time'] = df['start_time'].apply(start_time_converter)
        df['Horse_weight'] = df['Horse_weight'].apply(lambda x: x.strip().replace(' ',''))
        df['Jockey'] = df['Jockey'].apply(lambda x: x.strip().replace("▲","").replace("△","").replace("☆","").replace("◇",""))

        # Race Timeをコンバートするには下記の関数を使う
        def get_sec(time_str):
            """Get Seconds from time."""
            try:
                m, s = time_str.split(':')
            except:
                return None #おそらくまだ順位が出ていないデータに対して
            return  int(m) * 60 + float(s)

        df['Race_Time'] = df['Race_Time'].apply(get_sec) #上記の関数を適用

        #順位の列は下記の関数を使って分解する
        def rank_spliter_2(x):
            try:
                rank = x.split('-')[1]
                return rank
            except:
                return None

        def rank_spliter_3(x):
            try:
                rank = x.split('-')[2]
                return rank
            except:
                return None

        def rank_spliter_4(x):
            try:
                rank = x.split('-')[3]
                return rank
            except:
                return None

        #さらに下記の列を分解する
        df['field_length'] = df['field_type'].apply(lambda x: None if '候' in x else x[1:5])
        df['field_type_1'] = df['field_type'].apply(lambda x: x[0])
        df['field_type_2'] = df['field_type'].apply(lambda x: x[7:].strip().replace('(','').replace(')',''))
        df['corner_rank_1'] = df['Corner_ranking'].apply(lambda x: x.split('-')[0])
        df['corner_rank_2'] = df['Corner_ranking'].apply(rank_spliter_2)
        df['corner_rank_3'] = df['Corner_ranking'].apply(rank_spliter_3)
        df['corner_rank_4'] = df['Corner_ranking'].apply(rank_spliter_4)

        #不要な列を削除する（主に分解し終えたもの）
        df = df.drop(['Race_data_1','Race_data_2','field_type','Corner_ranking','Arrival_diff'],axis=1)

        df = df[~df['Ranking'].isin(['中止', '除外', '取消'])] #失格などのレコードを削除

        # データタイプを一括で変更する
        changed_columns=list(df.columns)
        changed_columns.remove('Race_date')
        print(changed_columns)
        for column in changed_columns:
            df[column] = pd.to_numeric(df[column],errors='ignore')

        df['Ranking'].unique()

        #targetとなるRankingを2分類に分ける。エラー対策のため下記のデータ加工を施す
        df = df[df['Ranking']!='失格']
        df['Ranking'] = df['Ranking'].astype('int16')
        df['target'] = df['Ranking'].apply(lambda x: 0 if x<=3 else 1)
        df = df.sort_values(ascending=True,by=['Race_date','Race_id'])
        # df['target'] = df['Ranking'].apply(lambda x: 0 if x <= 3 else (1 if x <= 6 else (2 if x <= 9 else 3)))
        return df


    @classmethod
    def save_to_db(cls): #scraping -> update db
        #First, scrape and make a dataframe of the whole race data.
        df = MainData.scrape_train_data()
        print("Scraping finished.")

        #Next, save the data as a json file to access locally.
        df.to_json("df.json",orient="records")
        print("Stored df.json locally.")
        print("Updating firebase realtime database...")
        
        #Lastly, update the database at Firebase.
        with open("df.json","r") as f:
            df = json.load(f)
        df = {
            "race_data":df,
            "user":{"name":"horse","password":"race"}
        }
        db = cls.database_connect()
        db.remove()
        db.update(df)
        print("All updates finished!!")
        return {"success":"database updated"}


    def scrape_pred(race_id): # retrieve race data by race id which is input as a get-request
        #Following code will deal with the real unseen data
        race_id = str(race_id)
        year = datetime.now().year
        base_URL = "https://race.netkeiba.com/race/shutuba.html?race_id="        
        menu = "&rf=race_submenu"
        r = requests.get(base_URL + str(race_id) + menu, headers={'User-agent':'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:61.0) Gecko/20100101 Firefox/61.0'})
        c = r.content
        soup = BeautifulSoup(c,"html.parser")
        #あらかじめ日付情報を取得しておく
        date_info = soup.find("dl",{"id":"RaceList_DateList"}).find("dd",{"class":"Active"}).a.text
        date_info=date_info.split("(")[0]
        if "月" in date_info:
            date_info = str(year) + '年' + date_info
            race_date = datetime.strptime(date_info,'%Y年%m月%d日')
        else: #"/"の場合
            date_info = str(year) + '/' + date_info
            race_date = datetime.strptime(date_info,'%Y/%m/%d')
        print(f"date info transformed: {date_info},\nrace_date:{race_date}")

        all_rows = soup.find_all("tr",{"class":"HorseList"})
        race_name = soup.find("div",{"class":"RaceName"}).text.rstrip("\n")
        race_data_1 = soup.find("div",{"class":"RaceData01"}).text.strip("\n").replace(" ","").split("/") #予測データだとフィールド情報しかない。時間、天気、馬上のコンディションは不明。直前になればわかるかも？
        rd2 = soup.find("div",{"class":"RaceData02"}).find_all("span")
        race_data_2 = [x.text for x in rd2]
        row_list=[]

        for item in all_rows:
            d = {}
            d["Race_id"] = race_id
            d["Race_date"] = race_date.timestamp()*1000
            #Race Name
            d["Race_Name"]=race_name
            #Race data 1
            d["Race_data_1"]=race_data_1
            #Race data 2
            d["Race_data_2"]=race_data_2
            #Ranking
            d["Ranking"]=None
            #Uniform number
            d["Uni_Num"]=None
            #Horse number
            d["Hor_Num"]=None
            #Horse Name
            d["Hor_name"]=item.find("span",{"class":"HorseName"}).a.text
            #Horse sex and age
            d["Hor_sex_and_age"]=item.find("td",{"class":"Barei","class":"Txt_C"}).text
            #JockeyWeight
            d["JockeyWeight"]=item.find_all("td",{"class":"Txt_C"})[1].text
            #Jockey
            try:
                d["Jockey"]=item.find("td",{"class":"Jockey"}).a.text
            except:
                d["Jockey"]=item.find("td",{"class":"Jockey"}).text.strip("\n")
            #Race Time
            d["Race_Time"]=None
            #Arrival diff
            d["Arrival_diff"]=None
            #popularity
            d["Popularity"]=None
            #単勝人気
            d["Odds_popularity"]=None
            #後3F
            d["ato_3_F"]=None
            #Corner Ranking
            d["Corner_ranking"]=None
            #厩舎1
            d["Trainer_1"]=item.find("td",{"class":"Trainer"}).span.text
            #厩舎2
            d["Trainer_2"]=item.find("td",{"class":"Trainer"}).a.text
            #HorseWeight
            try: #失格の馬がいる場合は下記でエラーが出る
                if item.find("td",{"class":"Weight"}).text == "\n":
                    d["Horse_weight"] = None
                else:
                    d["Horse_weight"]=item.find("td",{"class":"Weight"}).text[:4]
                #HorseWeight flux
                try:
                    d["Horse_weight_Flux"]=item.find("td",{"class":"Weight"}).small.text.replace("(","").replace(")","").replace("+","")
                except:
                    d["Horse_weight_Flux"]=None
                # print(f"each item: {d}")
                row_list.append(d)
            except: #エラーが出る場合はリストに追加しない
                pass
        df = pd.DataFrame(row_list)
        """##取得データの整形"""
        #さらに列を分解する
        try:
            df['Race_data_1'].apply(lambda x: x[0].strip().replace('発走',''))
        except:
            df['start_time'] = None

        df['field_type'] =df['Race_data_1'].apply(lambda x: x[1].strip())

        try:
            df['weather']=df['Race_data_1'].apply(lambda x: x[2].strip().replace('天候:',''))
        except:
            df['weather'] = None
        try:
            df['field_condition'] =df['Race_data_1'].apply(lambda x: x[3].strip().replace('馬場:',''))
        except:
            df['field_condition'] =None

        df['competition_count'] = df['Race_data_2'].apply(lambda x: x[0].strip().replace('回',''))
        df['venue_area'] = df['Race_data_2'].apply(lambda x: x[1].strip())
        df['venue_area'] = df['Race_data_2'].apply(lambda x: x[1].strip())
        df['day_count'] = df['Race_data_2'].apply(lambda x:x[2].strip().replace('日目',''))
        df['horse_class'] = df['Race_data_2'].apply(lambda x:x[3].strip())
        df['horse_class_2'] = df['Race_data_2'].apply(lambda x:x[4].strip())
        df['race_type'] = df['Race_data_2'].apply(lambda x:x[5].strip())
        df['hadicap'] = df['Race_data_2'].apply(lambda x:x[6].strip())
        df['number_of_hourses'] = df['Race_data_2'].apply(lambda x:x[7].strip().replace('頭',''))
        #*賞金データはあまり関係なさそうだからdrop
        #Cleanse the follwing columns
        df['Hor_sex_and_age'] = df['Hor_sex_and_age'].apply(lambda x: x.strip().replace('\n',''))
        df['Jockey'] = df['Jockey'].apply(lambda x: x.strip().replace("▲","").replace("△","").replace("☆","")).replace("★","")
        df['JockeyWeight'] = df['JockeyWeight'].apply(lambda x: None if x=="未定" else x)
        #フィールド情報を下記の通りに分解する
        df['field_length'] = df['field_type'].apply(lambda x: x[1:5])
        df['field_type_1'] = df['field_type'].apply(lambda x: x[0])
        df['field_type_2'] = df['field_type'].apply(lambda x: x[7:].strip().replace('(','').replace(')',''))
        #不要な列を削除する（主に分化し終えたもの）
        df = df.drop(['Race_data_1','Race_data_2','field_type','Corner_ranking','Arrival_diff'],axis=1)
        # データタイプを一括で変更する
        changed_columns = list(df.columns)
        changed_columns.remove('Race_date')
        for column in changed_columns:
            df[column] = pd.to_numeric(df[column],errors='ignore')
        return df
    
    @classmethod
    def get_data(cls): #query all data from the database
        '''Firebaseにログインしてデータを引っ張ってくる'''
        db = cls.database_connect()
        data = db.child("race_data").get()
        data = data.val()
        df = pd.DataFrame(data) #dataframeとして取り込む
        return df


    def transform_data(df): #transform train data to input it into the model
        #Label encodifng of Race_id
        # df['Race_id'] = df['Race_id'].apply(lambda x: str(x))
        le = LabelEncoder()
        df['Race_id'] = le.fit_transform(df['Race_id'])
        """馬別に統計量を取得"""
        horse_df_1 = df[['Hor_name','Odds_popularity','Horse_weight','corner_rank_1','corner_rank_2', 'corner_rank_3', 'corner_rank_4','field_length','Horse_weight_Flux','JockeyWeight']].groupby('Hor_name').mean()
        horse_df_1 = horse_df_1.sort_values('Hor_name')
        horse_df_1.columns = [i+('_mean') for i in horse_df_1.columns]
        horse_df_1 = horse_df_1.reset_index()
        horse_df_2 = df[['Hor_name','Race_id']].groupby('Hor_name').count()
        horse_df_2.columns = ['Race_experience_count']
        horse_df_2 = horse_df_2.reset_index()
        """## 騎手別に統計量を取得"""
        ## As it is
        """## フィールドデータなどの取得"""
        ## As it is
        """##各dfを結合して特徴量を作成する"""
        df = df.merge(horse_df_1,how='left',on='Hor_name')
        df = df.merge(horse_df_2,how='left',on='Hor_name')
        df = df.sort_values('Race_id')
        #カテゴリ変数をlabel encodingで変換する
        cat_cols = ['Race_Name','Race_date','Hor_name', 'Hor_sex_and_age', 'Jockey', 'Trainer_1', 'Trainer_2', 'weather', 'field_condition', 'venue_area', 'horse_class', 'horse_class_2', 'race_type', 'hadicap', 'field_type_1', 'field_type_2']
        for col in cat_cols:
            le = LabelEncoder()
            le.fit(df[col])
            df[col] = le.transform(df[col])
        return df


    @classmethod
    def transform_data_pred(cls,test_df):
        with open("df.json",'r') as f:
            df = json.load(f)
            df = pd.DataFrame(df)  #データベースからすべてのtraining dataを引っこ抜く
        print(f"Race_date:\n{test_df['Race_date']}")
        #encoding
        # le = LabelEncoder()
        # df['Race_id'] = le.fit_transform(df['Race_id'])
        """馬別に統計量を取得"""
        horse_df_1 = df[['Hor_name','Odds_popularity','Horse_weight','corner_rank_1','corner_rank_2', 'corner_rank_3', 'corner_rank_4','field_length','Horse_weight_Flux','JockeyWeight']].groupby('Hor_name').mean()
        horse_df_1 = horse_df_1.sort_values('Hor_name')
        horse_df_1.columns = [i+('_mean') for i in horse_df_1.columns]
        horse_df_1 = horse_df_1.reset_index()
        horse_df_2 = df[['Hor_name','Race_id']].groupby('Hor_name').count()
        horse_df_2.columns = ['Race_experience_count']
        horse_df_2 = horse_df_2.reset_index()
        """## 騎手別に統計量を取得"""
        ## As it is
        """## フィールドデータなどの取得"""
        ## As it is

        """##各dfを結合して特徴量を作成する"""
        test_df = test_df.merge(horse_df_1,how='left',on='Hor_name')
        test_df = test_df.merge(horse_df_2,how='left',on='Hor_name')
        test_df = test_df.sort_values('Race_id')
        # test_df['Race_id'] = le.transform(test_df['Race_id']) #Race_idをtraining dataと合わせて変換する
        #カテゴリ変数をlabel encodingで変換する
        cat_cols = ['Race_Name', 'Hor_name', 'Hor_sex_and_age', 'Jockey', 'Trainer_1', 'Trainer_2', 'weather', 'field_condition', 'venue_area', 'horse_class', 'horse_class_2', 'race_type', 'hadicap', 'field_type_1', 'field_type_2']
        for col in cat_cols:
            le = LabelEncoder()
            le.fit(test_df[col])
            test_df[col] = le.transform(test_df[col])
        return test_df
        

    @classmethod
    def create_model(cls): 
        """transform the data from db for training"""
        print("Querying data from the database...")
        df = cls.get_data()
        df.to_json("df.json",orient="records") #予測時のデータ加工に使用
        len_df = df.shape
        print(f"The number of records before transformation is {len_df}")
        print("Transforming the data...")
        df = cls.transform_data(df)
        len_df = len(df)
        """## Training with cross-validation"""
        print("Model building has started...")
        # データの並び順を元に分割する
        folds = TimeSeriesSplit(n_splits=5)

        X = df.drop(['Ranking','target','Race_Time','corner_rank_1',
            'corner_rank_2', 'corner_rank_3', 'corner_rank_4'],axis=1) #Leakになりそうなものも含めて除外する
        y = df['target']


        #評価結果格納用
        oof_predictions = np.zeros(df.shape[0])

        for i, (train_index, val_index) in enumerate(folds.split(df)):
            params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'binary',
            # 'predict_disable_shape_check':'true'
            # 'metric': {'multi_error'},
            }
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]
            
            train_dataset = lgb.Dataset(X_train, y_train, categorical_feature=['Race_id'])
            val_dataset = lgb.Dataset(X_val, y_val, reference=train_dataset, categorical_feature=['Race_id'])

            model = lgb.train(params = params,
                            train_set=train_dataset,
                            valid_sets= [train_dataset,val_dataset],
                            num_boost_round =5000,
                            early_stopping_rounds=50,
                            verbose_eval = 50
                            )
            oof_predictions[val_index] = model.predict(X_val)
            
            with open(f'pred_model_{i}.pkl', mode='wb') as f: #save each model
                pickle.dump(model,f)
        
        result = pd.DataFrame(oof_predictions)[0].apply(lambda x: 0 if x < 0.5 else 1)

        
        print("All processes finished successfully.")
        return model, confusion_matrix(df['target'],result), classification_report(df['target'],result) #confusion matrix is in ndarray, and classification report is in string
    

    @classmethod
    def predict(cls,race_id):  #The format is same as the one in database.
        #transform the df
        df = cls.scrape_pred(race_id)
        # if df == None: #when testing table was not scaraped
        #     return {"Error":"No table found."}
        Hor_df = df['Hor_name']
        df = cls.transform_data_pred(df)
        try: #すでに結果が出ているデータの場合
            df = df.drop(['Ranking','target','Race_Time','corner_rank_1',
                'corner_rank_2', 'corner_rank_3', 'corner_rank_4'],axis=1)
        except KeyError: #まだ結果が出ていない場合
            df = df.drop(['Ranking'],axis=1)
        col_num = len(df.columns)
        row_num = len(df)
        print(f"Columns: {col_num}")
        print(f"Rows: {row_num}")
        print(f"dataframe scaraped and transformed:{df}")
        
        #Read each model
        model = []
        for i in range(0,5):
            with open(f'pred_model_{i}.pkl',mode='rb') as f:
                each_model = pickle.load(f)
                model.append(each_model)
        #predict
        oof_pred = np.zeros(df.shape[0])
        for i in range(0,5):
            model_num = i
            pred_model = model[model_num]
            oof_pred += pred_model.predict(df)/5
        oof_pred = pd.DataFrame(oof_pred,columns=['Prediction'])
        oof_pred = pd.concat([oof_pred,Hor_df],axis=1)
        oof_pred = oof_pred.sort_values(by=['Prediction'])
        oof_pred = oof_pred.values.tolist()
        return oof_pred
    

