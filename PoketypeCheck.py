import pandas as pd
from pandas import plotting  
import matplotlib.pyplot as plt
import codecs
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

type_list = ["くさ", "ほのお", "みず", "むし", "ノーマル", "あく", "いわ", "はがね",
             "でんき", "ゴースト", "ドラゴン", "エスパー", "かくとう", "どく", "フェアリー", "じめん", "ひこう", "こおり"]

def type_to_num(p_type,test,typ):
    if p_type == test:
        return 0
    elif p_type == typ:
        return 2
    else:
        return 1

def lrRegress():
    with codecs.open("pokemon_status.csv", "r", "utf-8", "ignore") as file:
        df = pd.read_table(file, delimiter=",")
    
    result_type1 = ""
    result_type2 = ""
    result_type3 = ""
    result_test = 0
    result_train = 0

    for ctype1 in type_list:
        for ctype2 in type_list:
            if ctype1 == ctype2:
                continue
            for ctype3 in type_list:
                if ctype1 == ctype3 or ctype2 == ctype3:
                    continue                
                    
                poke1_type1 = df[df['タイプ１'] == ctype1]
                poket1_ype2 = df[df['タイプ２'] == ctype1]
                poke1_type = pd.concat([poke1_type1, poket1_ype2])

                poke2_type1 = df[df['タイプ１'] == ctype2]
                poke2_type2 = df[df['タイプ２'] == ctype2]
                poke2_type = pd.concat([poke2_type1,poke2_type2])

                poke3_type1 = df[df['タイプ１'] == ctype3]
                poke3_type2 = df[df['タイプ２'] == ctype3]
                poke3_type = pd.concat([poke3_type1,poke3_type2])
                
                pokemon_types = pd.concat([poke1_type, poke2_type, poke3_type], ignore_index=True)
                type1 = pokemon_types["タイプ１"].apply(type_to_num,test=ctype1,typ=ctype3)
                type2 = pokemon_types["タイプ２"].apply(type_to_num,test=ctype1,typ=ctype3)
                pokemon_types["type_num"] = type1*type2
                pokemon_types.head()

                X = pokemon_types.iloc[:, 7:13].values
                y = pokemon_types["type_num"].values

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
                #lr = KNeighborsClassifier(n_neighbors = 6)
                lr = LogisticRegression(C=1.0, max_iter=500)
                lr.fit(X_train, y_train)
            


                if lr.score(X_test, y_test) > result_test:
                    result_type1 = ctype1
                    result_type2 = ctype2
                    result_type3 = ctype3                 
                    
                    result_test = lr.score(X_test, y_test)
                    result_train = lr.score(X_train, y_train)
                    print("-------------------------------------------------")
                    print("type1:" + ctype1 + " type2:" + ctype2 + " type3:" + ctype3)
                    print("trainデータに対するscore: %.3f" % lr.score(X_train, y_train))
                    print("testデータに対するscore: %.3f" % lr.score(X_test, y_test))
                    print("-------------------------------------------------")
    
    print("Highest Score is ...")
    print("========================================================")
    print("type1 is " + result_type1 + " type2 is " + result_type2 + " type3 is " + result_type3)
    print("result score is "+ str(result_test))
    print("result score(train) is "+ str(result_train))
    print("========================================================")

poketype1 = "でんき"
poketype2 = "ノーマル"
poketype3 = "はがね"
def metal_normal():
    with codecs.open("pokemon_status.csv", "r", "utf-8", "ignore") as file:
        df = pd.read_table(file, delimiter=",")  
    
    metal1 = df[df['タイプ１'] == poketype1]
    metal2 = df[df['タイプ２'] == poketype1]
    metal = pd.concat([metal1, metal2])

    normal1 = df[df['タイプ１'] == poketype2]
    normal2 = df[df['タイプ２'] == poketype2]
    normal = pd.concat([normal1,normal2])

    water1 = df[df['タイプ１'] == poketype3]
    water2 = df[df['タイプ２'] == poketype3]
    water = pd.concat([water1,water2])
    
    pokemon_m_n = pd.concat([metal, normal, water], ignore_index=True)
    type1 = pokemon_m_n["タイプ１"].apply(type_to_num,test=poketype1,typ=poketype3)
    type2 = pokemon_m_n["タイプ２"].apply(type_to_num,test=poketype1,typ=poketype3)
    pokemon_m_n["type_num"] = type1*type2
    pokemon_m_n.head()

    X = pokemon_m_n.iloc[:, 7:13].values
    y = pokemon_m_n["type_num"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
    #lr = LogisticRegression(max_iter=500, C=1.0)
    lr = KNeighborsClassifier(n_neighbors = 8)
    lr.fit(X_train, y_train)
            
    print("-------------------------------------------------")
    print("trainデータに対するscore: %.3f" % lr.score(X_train, y_train))
    print("testデータに対するscore: %.3f" % lr.score(X_test, y_test))
    print("-------------------------------------------------")
    
    
    with codecs.open("pokemon.csv", "r", "utf-8", "ignore") as file:
        check = pd.read_table(file, delimiter=",")  
    
    metal11 = check[check['type1'] == poketype1]
    metal22 = check[check['type2'] == poketype1]
    metal = pd.concat([metal11, metal22])

    elec1 = check[check['type1'] == poketype2]
    elec2 = check[check['type2'] == poketype2]
    elec = pd.concat([elec1,elec2])
    
    water1 = check[check['type1'] == poketype3]
    water2 = check[check['type2'] == poketype3]
    water = pd.concat([water1,water1])

    pokemon_check = pd.concat([metal,elec,water], ignore_index=True)
    type11 = pokemon_check["type1"].apply(type_to_num,test=poketype1,typ=poketype3)
    type22 = pokemon_check["type2"].apply(type_to_num,test=poketype1,typ=poketype3)
    pokemon_check["type_num"] = type11*type22
    pokemon_check.head()

    X = pokemon_check.iloc[:, 1:7].values
    y = pokemon_check["type_num"].values

    i = 0
    error1 = 0
    success1 = 0
    error2 = 0
    success2 = 0
    error3 = 0
    success3 = 0
    print("[はがねタイプと判断したポケモン一覧]")
    print("----------------------------------------")
    print("")

    while i < len(pokemon_check):
        y_pred = lr.predict(X[i].reshape(1, -1))
        if y_pred == 0:
            if pokemon_check.loc[i, ["type_num"]].values == 0:
                success1 += 1
            else:
                error1 += 1
                print(pokemon_check.loc[i, "name"])
                print(str(poketype1)+"タイプではないです")
                print("")
        elif y_pred == 1:
            if pokemon_check.loc[i, ["type_num"]].values == 0:
                error2 += 1
                print(pokemon_check.loc[i, "name"])
                print(str(poketype1)+"タイプです")
                print("")
            elif pokemon_check.loc[i, ["type_num"]].values == 2:
                error2 += 1
                print(pokemon_check.loc[i, "name"])
                print(str(poketype3)+"タイプです")
                print("")             
            elif pokemon_check.loc[i, ["type_num"]].values == 1:
                success2 += 1
        elif y_pred == 2:
            if pokemon_check.loc[i, ["type_num"]].values == 2:
                success3 += 1
            else:
                error3 += 1
                print(pokemon_check.loc[i, "name"])
                print(str(poketype3)+"タイプではないです")
                print("")

        else:
            print("意味不エラー")
        i += 1
    print("----------------------------------------")
    print("合計ポケモン数：%d匹" % len(pokemon_check))
    print("正しく" + str(poketype1) + "タイプと判断したポケモンの数: %d匹" % success1)
    print("正しく" + str(poketype3) + "タイプと判断したポケモンの数: %d匹" % success3)
    print("正しく" + str(poketype1) + "タイプではないと判断ポケモンの数: %d匹" % success2)
    print("誤って" + str(poketype1) + "タイプと判断したポケモンの数: %d匹" % error1)
    print("誤って" + str(poketype3) + "タイプと判断したポケモンの数: %d匹" % error2)
    print("誤ってタイプ判定したポケモンの数: %d匹" % error2)
    print("検査合計結果：%d匹" % (error1+error2+error3+success1+success2+success3))
    print("識別率：%.3f%%" % ((success1+success2+success3)/(error1+error2+error3+success1+success2+success3)*100))

#lrRegress()
metal_normal()