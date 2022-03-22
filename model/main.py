import os
from typing import Optional
from sqlmodel import Field, SQLModel, Session, select
from sqlalchemy.schema import Column
from sqlalchemy.types import BigInteger
from db import engine
from functools import total_ordering
import requests
from joblib import Parallel, delayed
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import pickle
import datetime
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import json
import jpholiday
import lxml
import traceback
import chromedriver_binary  # パスを通すため
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException
from tqdm import tqdm
import time
import ast
import concurrent.futures
import threading
from env_vars import blob_conn_str
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient



class MainData(SQLModel,table=True):
    record_id: Optional[int] = Field(default=None,primary_key=True)
    Race_id: Optional[int] = Field(default_factory="next_val", sa_column=Column(BigInteger(), primary_key=False, autoincrement=False))
    Race_date: int
    Race_name: str
    Ranking: int
    Uni_num: int
    Hor_Num: int
    Hor_name: str
    Hor_sex_and_age: str	
    JockeyWeight: float	
    Jockey: str
    Race_Time: float
    Odds_popularity: float
    ato_3_F: float
    Trainer: str
    Horse_weight_Flux: Optional[float] = 0
    single_odds: float
    start_time: float	
    weather: Optional[str] = None
    field_condition: Optional[str] = None
    competition_count: int
    venue_area: str
    day_count: int
    horse_class: str
    number_of_horses: int
    Corner_rank_1: int
    Corner_rank_2: Optional[int] = None
    Corner_rank_3: Optional[int] = None
    Corner_rank_4: Optional[int] = None
    Horse_weight: float
    field_length: float
    field_type_1: str
    field_type_2: str
    
    def database_connect(): #DBに接続する用の関数
        session = Session(bind=engine)
        return session
    
    @classmethod
    def scrape_train_data(cls,test_mode=False,test_len=5):
        '''関数1(休日に開催されると仮定して開催日のリストを作成する)'''
        session = cls.database_connect()
        statement = select(cls).order_by(cls.Race_date.desc())
        results = session.exec(statement)
        results = results.first()
        if results != None:
            date_after = str(results.Race_date) #int情報
            date_after = datetime.date(int(date_after[0:4]), int(date_after[4:6]), int(date_after[6:8])) #datetime情報
            print(f"The latest data is dated: {results.Race_date}")
            date_until = datetime.date.today()        
        else:
            date_after = datetime.date(1999,1,1)
            date_until = datetime.date.today()
        print(f'date_after: {date_after}')
        print(f'date_until:{date_until}')
        def race_day_list(date_after):
            this_year = datetime.datetime.now().year
            years = [this_year -1 ,this_year]
            months = range(1,13)
            days = range(1,32)

            def isHoliday(DATE):
                Date = datetime.date(int(DATE[0:4]), int(DATE[4:6]), int(DATE[6:8]))
                if Date.weekday() >= 5 or jpholiday.is_holiday(Date):
                    if Date > date_after and date_until >= Date:
                        return 1 #休日かつDBに保存済みの日付よりも後かつ今日以前の日付
                    else:
                        return 0
                else:
                    return 0 #平日
            
            date_list = []
            for year in years:
                for month in months:
                    try:
                        for day in days:
                            date = f"{year}{month:02}{day:02}"
                            if isHoliday(date) == 1:
                                date_list.append(date)
                    except ValueError:
                        pass
            return date_list
        #上記の関数を実行
        race_date_list = race_day_list(date_after=date_after)
        if len(race_date_list) == 0: #該当の日付がない場合
            return {"Result": "Database is already updated!"}
        
        #テストモードの場合は少な目に
        if test_mode==True:
            race_date_list = race_date_list[:test_len]
        '''
        関数2
        (上記の日付リストに基づいて、各日付ごとに開催されているレースのIDを一覧ページから取得する)
        '''
        #ドライバー
        def scrape_race_ids(race_date_list):
            options = Options()
            #下記のオプションを追加しないとなぜかエラーになる
            options.add_argument("start-maximized")
            options.add_argument("enable-automation")
            options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-infobars")
            options.add_argument('--disable-extensions')
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-browser-side-navigation")
            options.add_argument("--disable-gpu")
            options.add_argument("--dns-prefetch-disable")
            options.add_argument('--ignore-certificate-errors')
            options.add_argument('--ignore-ssl-errors')
            options.add_argument('--proxy-server="direct://"')
            options.add_argument('--proxy-bypass-list=*')
            options.add_argument('--start-maximized')
            options.add_experimental_option('excludeSwitches', ['enable-logging'])
            options.add_experimental_option("excludeSwitches", ["enable-automation"])
            options.add_experimental_option('useAutomationExtension', False)
            options.add_argument("window-size=1920,1080")
            options.headless = True
            prefs = {"profile.default_content_setting_values.notifications" : 2}
            options.add_experimental_option("prefs",prefs)
            options.page_load_strategy = 'normal'
            base_url = "https://race.netkeiba.com/top/race_list.html?kaisai_date="
            
            race_id_list = []

            for race_date in tqdm(race_date_list):
                link_list = []
                # try:
                driver = webdriver.Chrome(options=options)
                driver.implicitly_wait(60)
                # driver.set_page_load_timeout(600)
                i = 0
                while i < 3:
                    try:
                        driver.get(base_url+race_date)
                        break
                    except TimeoutException as ex:
                        i = i + 1
                        time.sleep(5)
                        continue
                else:
                    print("Reload has failed three times.")
                    break
                driver.implicitly_wait(60)
                page_source = driver.page_source
                soup = BeautifulSoup(page_source, features='lxml')
                race_list = soup.find_all('div',{'id':'race_list'})[0]
                race_list = race_list.find_all('a',{'class':''})
                for item in race_list:
                    link_list.append(item.attrs['href'])

                target = 'race_id='
                for item in link_list:
                    idx = item.find(target)
                    try:
                        result = item[idx+8:idx+20]
                        result = int(result) #数字以外がある場合にはじくため
                        race_id = {'race_id':str(result),
                                'race_date':race_date}
                        race_id_list.append(race_id)
                    except:
                        pass
                # print(race_id_list)
                time.sleep(5)
                
            return race_id_list
        
        #関数2の実行
        race_id_list = scrape_race_ids(race_date_list)
        
        # print(f"race_id_list:\n{race_id_list}")
        '''
        関数3
        (関数2で取得したrace_idに基づきレースの結果をスクレイピング&データ整形する)
        '''
        def scrape_results(race_id_list):
            columns = ['Race_id','Race_date','Race_name','Race_data_1','Race_data_2','Ranking','Uni_num','Hor_Num',
                    'Hor_name','Hor_sex_and_age','JockeyWeight','Jockey','Race_Time','Arrival_diff','Odds_popularity',
                    'single_odds','ato_3_F','Corner_ranking','Trainer','Horse_weight_Flux'
                    ]
            df = pd.DataFrame(columns=columns)
            # df = pd.DataFrame()
            for race_id_data in tqdm(race_id_list):
                print(f"race_id:{race_id_data['race_id']}")
                #既存のスクリプトをもとにスクレピングのコードを書く
                ##テーブルデータをまず引っ張る
                base_URL = 'https://race.netkeiba.com/race/result.html?race_id='
                race_id = race_id_data['race_id']
                race_date = race_id_data['race_date']
                try:
                    dfs = pd.read_html(base_URL+str(race_id))
                    main_table = dfs[0]
                    main_table['Race_id'] = race_id
                    main_table['Race_date'] = race_date
                    ##天候などの共通データなどを引っ張る
                    r = requests.get(base_URL + str(race_id), headers={'User-agent':'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:61.0) Gecko/20100101 Firefox/61.0'})
                    c = r.content
                    soup = BeautifulSoup(c,"lxml")
                    race_name = soup.find("div",{"class":"RaceName"}).text.rstrip("\n")
                    race_data_1 = soup.find("div",{"class":"RaceData01"}).text.split("/")
                    rd2 = soup.find("div",{"class":"RaceData02"}).find_all("span")
                    race_data_2 = [x.text for x in rd2]
                    ##共通データをdfに反映
                    main_table['Race_name'] = race_name
                    # main_table['Race_data_1'] = race_data_1
                    main_table['Race_data_1'] = ''
                    main_table['Race_data_1'] = main_table['Race_data_1'].apply(lambda x: race_data_1)
                    # main_table['Race_data_2'] = race_data_2
                    main_table['Race_data_2'] = ''
                    main_table['Race_data_2'] = main_table['Race_data_2'].apply(lambda x: race_data_2)

                    ##main_tableのcolumn名を変更する
                    column_map = {'着順':'Ranking','枠':'Uni_num','馬番':'Hor_Num','馬名':'Hor_name','性齢':'Hor_sex_and_age',
                                '斤量':'JockeyWeight','騎手':'Jockey','タイム':'Race_Time','着差':'Arrival_diff','人気':'Odds_popularity',
                                '単勝オッズ':'single_odds','後3F':'ato_3_F','コーナー通過順':'Corner_ranking','厩舎':'Trainer',
                                '馬体重(増減)':'Horse_weight_Flux'}
                    main_table = main_table.rename(columns=column_map)
                    main_table = main_table.dropna(subset=['Race_Time'])
                    
                    ##データを整形
                    main_table['start_time'] = main_table['Race_data_1'].apply(lambda x: x[0].strip().replace('発走',''))
                    main_table['field_type'] = main_table['Race_data_1'].apply(lambda x: x[1].strip())
                    try:
                        main_table['weather'] = main_table['Race_data_1'].apply(lambda x: x[2].strip().replace('天候:',''))
                    except IndexError: #存在しない場合
                        main_table['weather'] = None
                    try:
                        main_table['field_condition'] =main_table['Race_data_1'].apply(lambda x: x[3].strip().replace('馬場:',''))
                    except IndexError:
                        main_table['field_condition'] = None

                    main_table['competition_count'] = main_table['Race_data_2'].apply(lambda x: x[0].strip().replace('回',''))
                    main_table['venue_area'] = main_table['Race_data_2'].apply(lambda x: x[1].strip())
                    main_table['day_count'] = main_table['Race_data_2'].apply(lambda x:x[2].strip().replace('日目',''))
                    main_table['horse_class'] = main_table['Race_data_2'].apply(lambda x:x[3].strip())
                    main_table['number_of_horses'] = main_table['Race_data_2'].apply(lambda x:x[4].strip().replace('頭','') if len(x)<6 else x[7].strip().replace('頭',''))
                    ##*賞金データはあまり関係なさそうだからdrop

                    ##Cleanse the follwing columns
                    main_table['Hor_sex_and_age'] = main_table['Hor_sex_and_age'].apply(lambda x: x.strip().replace('\n',''))
                    # main_table['Arrival_diff'] = main_table['Arrival_diff'].apply(lambda x: x.strip().replace('\n',''))
                    # main_table['Odds_popularity'] = main_table['Odds_popularity'].apply(lambda x: x.strip().replace('\n',''))
                    # main_table['ato_3_F'] = main_table['ato_3_F'].apply(lambda x: x.strip().replace('\n',''))
                    
                    try:
                        main_table['Corner_rank_1'] = main_table['Corner_ranking'].apply(lambda x: x.split('-')[0])
                    except AttributeError: #コーナーが一つしかないときに発生する
                        main_table['Corner_rank_1'] = main_table['Corner_ranking']
                    try:
                        main_table['Corner_rank_2'] = main_table['Corner_ranking'].apply(lambda x: x.split('-')[1])
                    except:
                        main_table['Corner_rank_2'] = None
                    try:
                        main_table['Corner_rank_3'] = main_table['Corner_ranking'].apply(lambda x: x.split('-')[2])
                    except:
                        main_table['Corner_rank_3'] = None
                    try:
                        main_table['Corner_rank_4'] = main_table['Corner_ranking'].apply(lambda x: x.split('-')[3])
                    except:
                        main_table['Corner_rank_4'] = None
                    main_table['Horse_weight_Flux'] = main_table['Horse_weight_Flux'].apply(lambda x: x.strip().replace('(','').replace(')','').replace('+',''))
                    #race_dataによっては時間情報がなくslicingがうまく行かない場合があるので、下記の関数を別途定義する
                    def start_time_converter(x):
                        try:
                            start = int(x.strip().replace(':',''))
                        except:
                            start = None
                        return start
                    main_table['start_time'] = main_table['start_time'].apply(start_time_converter)
                    main_table['Horse_weight'] = main_table['Horse_weight_Flux'].apply(lambda x: x[:3])
                    main_table['Horse_weight_Flux'] = main_table['Horse_weight_Flux'].apply(lambda x: x[3:].replace('+','').replace('(','').replace(')',''))
                    main_table['Jockey'] = main_table['Jockey'].apply(lambda x: x.strip().replace("▲","").replace("△","").replace("☆","").replace("◇",""))
                    #フィールド情報を下記の通りに分解する
                    main_table['field_length'] = main_table['field_type'].apply(lambda x: x[1:5])
                    main_table['field_type_1'] = main_table['field_type'].apply(lambda x: x[0])
                    main_table['field_type_2'] = main_table['field_type'].apply(lambda x: x[7:].strip().replace('(','').replace(')',''))

                    # Race Timeをコンバートするには下記の関数を使う
                    def get_sec(time_str):
                        """Get Seconds from time."""
                        try:
                            m, s = time_str.split(':')
                        except:
                            return None #おそらくまだ順位が出ていないデータに対して
                        return  int(m) * 60 + float(s)

                    main_table['Race_Time'] = main_table['Race_Time'].apply(get_sec) #上記の関数を適用
                    
                    ##不要な列を削除する（主に分解し終えたもの）
                    main_table = main_table.drop(['Race_data_1','Race_data_2','field_type','Arrival_diff'],axis=1)
                    main_table = main_table[~main_table['Ranking'].isin(['中止', '除外', '取消', '失格'])] #失格などのレコードを削除
                    
                    ## データタイプを一括で変更する
                    changed_columns=list(main_table.columns)
                    for column in changed_columns:
                        main_table[column] = pd.to_numeric(main_table[column],errors='ignore')
                    ##エラー対策のため下記のデータ加工を施す
                    main_table['Ranking'] = main_table['Ranking'].astype('int16')
                    #列の順番を並べ替え&要らない列を除外
                    main_table = main_table[['Race_id','Race_date','Race_name','Ranking','Uni_num',
                    'Hor_Num','Hor_name','Hor_sex_and_age','JockeyWeight',
                    'Jockey','Race_Time','Odds_popularity','ato_3_F',
                    'Trainer','Horse_weight_Flux','single_odds','start_time','weather',
                    'field_condition','competition_count','venue_area','day_count',
                    'horse_class','number_of_horses','Corner_rank_1','Corner_rank_2',
                    'Corner_rank_3','Corner_rank_4','Horse_weight','field_length',
                    'field_type_1','field_type_2',
                    ]]
                    #最後にdfに結合する
                    df = pd.concat([df,main_table])
                except ValueError: #when no table was found
                    pass
                time.sleep(1)
            return df
        
        #関数3の実行(concurrent.future使って並行処理)
        unit_num = os.cpu_count() * 2 #cpu*2の値を取得
        race_id_list_split = list(np.array_split(race_id_list,unit_num)) #上記の値をもとにrace_id_listを複数のリストに分割
        race_id_list_split = [list(i) for i in race_id_list_split] #形式をarrayからlistに変更
        with concurrent.futures.ThreadPoolExecutor() as executor: #並行処理によって関数3を実行
            futures = [executor.submit(scrape_results, race_id_list) for race_id_list in race_id_list_split]
            result = [f.result() for f in futures]
        df = pd.DataFrame() #threadごとに作成したテーブルを結合してこの変数に格納する
        for table in result:
            df = pd.concat([df,table])
        #Corner_rankがないものについてはDrop.おそらくまだ更新されていないだけ
        df = df.dropna(subset=['Corner_rank_1','Corner_rank_2','Corner_rank_3','Corner_rank_4'],how='all')
        return df

    @classmethod
    def save_to_db(cls,test_mode=False,test_len=5): #scraping -> update db
        df = MainData.scrape_train_data(test_mode=test_mode,test_len=test_len)
        print("Scraping finished.")
        if type(df) == dict :
            return {"Result":"Update testing has successfully finished!!"}  
        if test_mode == True: #テストモード
            records = []
            for _, rec in df.iterrows():
                # print(f"count:{index}")
                adding_rec = MainData(**rec)
                records.append(adding_rec)
            # session = cls.database_connect()
            # print("status: db connected")
            # print(f'records to be stored:\n{records}')
            # session.add_all(records)
            # print("status: session created")
            # session.commit()
            # print("Update testing has successfully finished!!")
            return {"Result":"Update testing has successfully finished!!"}  
        else:
            records = []
            for _, rec in df.iterrows():
                # print(f"count:{index}")
                adding_rec = MainData(**rec)
                records.append(adding_rec)
            # session = cls.database_connect()
            session = Session(bind=engine)
            print("status: db connected")
            session.add_all(records)
            print("status: session created")
            session.commit()
            print("All updates finished!!")
            return {"success":"database updated"}

    @classmethod
    def scrape_pred(cls,race_id): 
        #既存のスクリプトをもとにスクレピングのコードを書く
        ##テーブルデータをまず引っ張る
        base_URL = 'https://race.netkeiba.com/race/shutuba.html?race_id='
        dfs = pd.read_html(base_URL+str(race_id))
        main_table = dfs[0]
        columns = ['Uni_num','Hor_Num','印','Hor_name','Hor_sex_and_age','JockeyWeight','Jockey','Trainer','Horse_weight_Flux','Odds','人気','登録','メモ']
        main_table = pd.read_html('https://race.netkeiba.com/race/shutuba.html?race_id='+ str(race_id))[0]
        main_table.columns= columns
        main_table = main_table.drop(['印','人気','登録','メモ','Odds'],axis=1)
        # main_table = main_table.dropna(subset=['馬体重(増減)'])
        main_table['Race_id'] = race_id
        ##天候などの共通データなどを引っ張る
        r = requests.get(base_URL + str(race_id), headers={'User-agent':'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:61.0) Gecko/20100101 Firefox/61.0'})
        c = r.content
        soup = BeautifulSoup(c,"lxml")
        ###まずは日付データ
        year = str(race_id)[:4]
        date_info = soup.find("dl",{"id":"RaceList_DateList"}).find("dd",{"class":"Active"}).a.text
        date_info = date_info.split("(")[0]
        if "月" in date_info:
            date_info = str(year) + '年' + date_info
            race_date = datetime.datetime.strptime(date_info,'%Y年%m月%d日')
            race_date = race_date.timestamp()*1000
        else: #"/"の場合
            date_info = str(year) + '/' + date_info
            race_date = datetime.datetime.strptime(date_info,'%Y/%m/%d')
            race_date = race_date.timestamp()*1000
        main_table['Race_date'] = race_date
        race_name = soup.find("div",{"class":"RaceName"}).text.rstrip("\n")
        race_data_1 = soup.find("div",{"class":"RaceData01"}).text.split("/")
        rd2 = soup.find("div",{"class":"RaceData02"}).find_all("span")
        race_data_2 = [x.text for x in rd2]
        ##共通データをdfに反映
        main_table['Race_name'] = race_name
        main_table['Race_data_1'] = ''
        main_table['Race_data_1'] = main_table['Race_data_1'].apply(lambda x: race_data_1)
        main_table['Race_data_2'] = ''
        main_table['Race_data_2'] = main_table['Race_data_2'].apply(lambda x: race_data_2)

        ##main_tableのcolumn名を変更する
        # column_map = {'着順':'Ranking','枠':'Uni_num','馬番':'Hor_Num','馬名':'Hor_name','性齢':'Hor_sex_and_age',
        #             '斤量':'JockeyWeight','騎手':'Jockey','タイム':'Race_Time','着差':'Arrival_diff','人気':'Odds_popularity',
        #             '単勝オッズ':'single_odds','後3F':'ato_3_F','コーナー通過順':'Corner_ranking','厩舎':'Trainer',
        #             '馬体重(増減)':'Horse_weight_Flux'}
        # main_table = main_table.rename(columns=column_map)
        columns =['Race_id','Race_date','Race_name','Ranking','Uni_num',
                    'Hor_Num','Hor_name','Hor_sex_and_age','JockeyWeight',
                    'Jockey','Race_Time','Odds_popularity','ato_3_F',
                    'Trainer','Horse_weight_Flux','single_odds','start_time','weather',
                    'field_condition','competition_count','venue_area','day_count',
                    'horse_class','number_of_horses','Corner_rank_1','Corner_rank_2',
                    'Corner_rank_3','Corner_rank_4','Horse_weight','field_length',
                    'field_type_1','field_type_2',
                    ]
        df = pd.DataFrame(columns=columns)
        df = pd.concat([df,main_table])
        # print(df)
        
        ##データを整形
        df['start_time'] = df['Race_data_1'].apply(lambda x: x[0].strip().replace('発走',''))
        df['field_type'] = df['Race_data_1'].apply(lambda x: x[1].strip())
        df['weather'] = df['Race_data_1'].apply(lambda x: x[2].strip().replace('天候:',''))
        try:
            df['field_condition'] =df['Race_data_1'].apply(lambda x: x[3].strip().replace('馬場:',''))
        except:
            df['field_condition'] = None

        df['competition_count'] = df['Race_data_2'].apply(lambda x: x[0].strip().replace('回',''))
        df['venue_area'] = df['Race_data_2'].apply(lambda x: x[1].strip())
        df['day_count'] = df['Race_data_2'].apply(lambda x:x[2].strip().replace('日目',''))
        df['horse_class'] = df['Race_data_2'].apply(lambda x:x[3].strip())
        df['number_of_horses'] = df['Race_data_2'].apply(lambda x:x[4].strip().replace('頭','') if len(x)<6 else x[7].strip().replace('頭',''))
        ##*賞金データはあまり関係なさそうだからdrop

        ##Cleanse the follwing columns
        df['Hor_sex_and_age'] = df['Hor_sex_and_age'].apply(lambda x: x.strip().replace('\n',''))
        # df['Arrival_diff'] = df['Arrival_diff'].apply(lambda x: x.strip().replace('\n',''))
        # df['Odds_popularity'] = df['Odds_popularity'].apply(lambda x: x.strip().replace('\n',''))
        # df['ato_3_F'] = df['ato_3_F'].apply(lambda x: x.strip().replace('\n',''))
        try:
            df['Corner_rank_1'] = df['Corner_ranking'].apply(lambda x: x.split('-')[0])
        except:
            df['Corner_rank_1'] = None
        try:
            df['Corner_rank_2'] = df['Corner_ranking'].apply(lambda x: x.split('-')[1])
        except:
            df['Corner_rank_2'] = None
        try:
            df['Corner_rank_3'] = df['Corner_ranking'].apply(lambda x: x.split('-')[2])
        except:
            df['Corner_rank_3'] = None
        try:
            df['Corner_rank_4'] = df['Corner_ranking'].apply(lambda x: x.split('-')[3])
        except:
            df['Corner_rank_4'] = None
        df['Horse_weight_Flux'] = df['Horse_weight_Flux'].apply(lambda x: x.strip().replace('(','').replace(')','').replace('+',''))
        #race_dataによっては時間情報がなくslicingがうまく行かない場合があるので、下記の関数を別途定義する
        def start_time_converter(x):
            try:
                start = int(x.strip().replace(':',''))
            except:
                start = None
            return start
        df['start_time'] = df['start_time'].apply(start_time_converter)
        df['Horse_weight'] = df['Horse_weight_Flux'].apply(lambda x: x[:3])
        df['Horse_weight_Flux'] = df['Horse_weight_Flux'].apply(lambda x: x[3:].replace('+','').replace('(','').replace(')',''))
        df['Jockey'] = df['Jockey'].apply(lambda x: x.strip().replace("▲","").replace("△","").replace("☆","").replace("◇",""))
        #フィールド情報を下記の通りに分解する
        df['field_length'] = df['field_type'].apply(lambda x: x[1:5])
        df['field_type_1'] = df['field_type'].apply(lambda x: x[0])
        df['field_type_2'] = df['field_type'].apply(lambda x: x[7:].strip().replace('(','').replace(')',''))

        # Race Timeをコンバートするには下記の関数を使う
        def get_sec(time_str):
            """Get Seconds from time."""
            try:
                m, s = time_str.split(':')
            except:
                return None #おそらくまだ順位が出ていないデータに対して
            return  int(m) * 60 + float(s)

        df['Race_Time'] = df['Race_Time'].apply(get_sec) #上記の関数を適用
        ##不要な列を削除する（主に分解し終えたもの）
        df = df.drop(['Race_data_1','Race_data_2','field_type'],axis=1)
        df = df[~df['Ranking'].isin(['中止', '除外', '取消', '失格'])] #失格などのレコードを削除
        
        ## データタイプを一括で変更する
        changed_columns=list(df.columns)
        for column in changed_columns:
            df[column] = pd.to_numeric(df[column],errors='ignore')
        ##エラー対策のため下記のデータ加工を施す
        
        return df
        

    @classmethod
    def get_data(cls): #query all data from the database
        '''Firebaseにログインしてデータを引っ張ってくる'''
        session = cls.database_connect()
        statement = select(cls)
        results = session.exec(statement).all()
        # data = db.child("race_data").get()
        # data = data.val()
        records = []
        for i in results:
            record = i.__dict__
            records.append(record)
        df = pd.DataFrame(records)
        df = df.drop(['_sa_instance_state','record_id'],axis=1)
        return df
    
    @classmethod
    def get_latest_record_date(cls):
        session = cls.database_connect()
        result = session.query(cls).order_by(cls.Race_date.desc()).first()
        latest_date = result.Race_date
        return latest_date      

    def transform_data(df): #transform train data to input it into the model
        #Label encodifng of Race_id
        # df['Race_id'] = df['Race_id'].apply(lambda x: str(x))
        le = LabelEncoder()
        df['Race_id'] = le.fit_transform(df['Race_id'])
        """馬別に統計量を取得"""
        horse_df_1 = df[['Hor_name','Odds_popularity','Horse_weight','Corner_rank_1','Corner_rank_2', 'Corner_rank_3', 'Corner_rank_4','field_length','Horse_weight_Flux','JockeyWeight']].groupby('Hor_name').mean()
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
        cat_cols = ['Race_name','Race_date','Hor_name', 'Hor_sex_and_age', 'Jockey', 'Trainer', 'weather', 'field_condition', 'venue_area', 'horse_class', 'field_type_1', 'field_type_2']
        for col in cat_cols:
            le = LabelEncoder()
            le.fit(df[col])
            df[col] = le.transform(df[col])
        return df


    @classmethod
    def transform_data_pred(cls,test_df):
        df = cls.get_data()
        print(f"Race_date:\n{test_df['Race_date']}")
        #encoding
        # le = LabelEncoder()
        # df['Race_id'] = le.fit_transform(df['Race_id'])
        """馬別に統計量を取得"""
        horse_df_1 = df[['Hor_name','Odds_popularity','Horse_weight','Corner_rank_1','Corner_rank_2', 'Corner_rank_3', 'Corner_rank_4','field_length','Horse_weight_Flux','JockeyWeight']].groupby('Hor_name').mean()
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
        cat_cols = ['Race_name','Race_date','Hor_name', 'Hor_sex_and_age', 'Jockey', 'Trainer', 'weather', 'field_condition', 'venue_area', 'horse_class', 'field_type_1', 'field_type_2']
        
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
        # df.to_json("df.json",orient="records") #予測時のデータ加工に使用
        len_df = df.shape
        print(f"The number of records before transformation is {len_df}")
        print("Transforming the data...")
        df = cls.transform_data(df)
        len_df = len(df)
        """## Training with cross-validation"""
        print("Model building has started...")
        # データの並び順を元に分割する
        folds = TimeSeriesSplit(n_splits=15)

        X = df.drop(['Ranking','Race_Time','Corner_rank_1',
            'Corner_rank_2', 'Corner_rank_3', 'Corner_rank_4'],axis=1) #Leakageになりそうなものも含めて除外する
        y = df['Ranking']

        #評価結果格納用
        oof_predictions = np.zeros(df.shape[0])

        for i, (train_index, val_index) in enumerate(folds.split(df,groups=df['Race_id'])):
            print(f"Fold:{i+1} has started." )
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

            #group引数用
            train_data = pd.concat([X_train,y_train],axis=1)
            train_groups = train_data.groupby('Race_id').size().to_frame('size')['size'].to_numpy()
            val_data = pd.concat([X_val,y_val],axis=1)
            val_groups = val_data.groupby('Race_id').size().to_frame('size')['size'].to_numpy()
            # print(train_groups)
            # print(val_groups)

            train_dataset = lgb.Dataset(X_train, y_train, group=train_groups)
            val_dataset = lgb.Dataset(X_val, y_val, reference=train_dataset, group=val_groups)

            params = {'task': 'train',
                    'boosting_type': 'gbdt',
                    'objective': 'lambdarank',
                    'metric': 'ndcg',
                    'nucg_eval_at': [5],
                    'feature_pre_filter': False,
                    'lambda_l1': 0.0, 'lambda_l2': 0.0,
                    'num_leaves': 229,
                    'feature_fraction': 0.7,
                    'bagging_fraction': 1.0,
                    'bagging_freq': 0,
                    'min_child_samples': 20,
                    'early_stopping_round':100}

            model = lgb.train(params = params,
                            train_set=train_dataset,
                            valid_sets= [train_dataset,val_dataset],
                            num_boost_round =5000,
                            )
            oof_predictions[val_index] = model.predict(X_val,num_iteration=model.best_iteration)
            # try:
            with open(f'pred_model_{i}.pkl', mode='wb') as f: #save each model
                pickle.dump(model,f)
                saved_file_name = f'pred_model_{i}.pkl'
                ## Add some code to upload the files to Azure Storage here
                # Create a blob client using the local file name as the name for the blob
                blob_service_client = BlobServiceClient.from_connection_string(blob_conn_str)
                container_name = 'modelbinarydata'
                blob_client = blob_service_client.get_blob_client(container=container_name,blob=saved_file_name)
                print("\nUploading to Azure Storage as blob:\n\t" + saved_file_name)
                # Upload the created file
                with open(saved_file_name,'rb') as data:
                    blob_client.upload_blob(data,overwrite=True)
            # except:
            #     print("Warning: Updates to Azure Storage has failed")
            #     with open(f'pred_model_{i}.pkl', mode='wb') as f: #save each model
            #         pickle.dump(model,f)

        result = pd.DataFrame(oof_predictions,columns=['predicted_class'])

        df_result = pd.concat([df,result],axis=1)
        
        print("All processes finished successfully.")
        
        return {'Result':"All processes finished successfully."}

    @classmethod
    def predict(cls,race_id):  #The format is same as the one in database.
        #transform the df
        df = cls.scrape_pred(race_id)
        # if df == None: #when testing table was not scaraped
        #     return {"Error":"No table found."}
        Hor_df = df['Hor_name']
        df = cls.transform_data_pred(df)
        df = df.drop(['Ranking','Race_Time','Corner_rank_1',
            'Corner_rank_2', 'Corner_rank_3', 'Corner_rank_4'],axis=1)
        col_num = len(df.columns)
        row_num = len(df)
        print(f"Columns: {col_num}")
        print(f"Rows: {row_num}")
        print(f"dataframe scaraped and transformed:{df}")
        
        #Read each model
        model = []
        for i in range(0,15):
            ## Add some code to download the files to Azure Storage here
            downloaded_file_name = f'pred_model_{i}.pkl'
            # try:
            # Connect to Azure Storage and the targeted file
            blob_service_client = BlobServiceClient.from_connection_string(blob_conn_str)
            container_name = 'modelbinarydata'
            blob_client = blob_service_client.get_blob_client(container=container_name,blob= downloaded_file_name)    
            # Download and overwrite each model's binary file in local
            with open(downloaded_file_name, 'wb') as f:
                f.write(blob_client.download_blob().readall())
            print(f'Retrived from Azure and updated model file: {downloaded_file_name}')
            # Load the updated local file and added to the list to be used for prediction later
            with open(downloaded_file_name,mode='rb') as f:
                each_model = pickle.load(f)
                model.append(each_model)
            # except:
            #     print('Warning: Access to Azure Storage has failed')
            #     with open(downloaded_file_name,mode='rb') as f:
            #         each_model = pickle.load(f)
            #         model.append(each_model)
        
        #predict
        oof_pred = np.zeros(df.shape[0])
        for i in range(0,15):
            model_num = i
            pred_model = model[model_num]
            oof_pred += pred_model.predict(df)*i
        oof_pred = pd.DataFrame(oof_pred,columns=['Prediction'])
        oof_pred = pd.concat([oof_pred,Hor_df],axis=1)
        oof_pred = oof_pred.sort_values(by=['Prediction'])
        oof_pred = oof_pred.values.tolist()
        return oof_pred