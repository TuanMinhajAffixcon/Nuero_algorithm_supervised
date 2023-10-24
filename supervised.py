import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import joblib
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE 
from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,confusion_matrix
import os

# st.set_page_config(layout="wide")
st.set_page_config(page_title='NEURO ENGINE',page_icon=':man_and_woman_holding_hands:',layout='wide')
custom_css = """
<style>
body {
    background-color: #22222E; 
    secondary-background {
    background-color: #FA55AD; 
    padding: 10px; 
}
</style>
"""
st.write(custom_css, unsafe_allow_html=True)
st.markdown(custom_css, unsafe_allow_html=True)

st.title('NEURO ENGINE - With Labels')

### making a takens for same vectorization
usecols=['Segment Name']
unsupervised_tokens=[]
affix_seg=pd.read_csv('Affixcon_Segmentation.csv',encoding='latin-1',usecols=usecols).dropna()['Segment Name'].tolist()

income=["Under $20,799","$20,800 - $41,599","$41,600 - $64,999","$65,000 - $77,999","$78,000 - $103,999","$104,000 - $155,999","$156,000+"]
age=["<20","20-24","25-29","30-34","35-39","40-44","45-49","50-54","55-59","60-64","65-69","70-74","75-79","80-84",">84"]
gender=['Female','Male']
features=[affix_seg,income,age,gender]
for item in features:
    unsupervised_tokens.extend(item)

# st.write(unsupervised_tokens)
#-----------------------------------------------------------------------------------------------------------------------------

def vectorizer(ds,vocabulary):
    vectorizer_list=[]
    for sentence in ds['Concatenated']:
        sentence_lst=np.zeros(len(vocabulary))
        for i in range(len(vocabulary)):
            if vocabulary[i] in sentence.split('|'):
                sentence_lst[i]=1
        vectorizer_list.append(sentence_lst)
    return vectorizer_list

col1,col2=st.columns((0.75,0.25))

with col1:
    file_uploader = st.file_uploader(" :file_folder: Upload the sample file", type=["csv"])
    file_name = file_uploader.name.rsplit(".", 1)[0]
    if not file_name.endswith(".pkl"):
        # Append the .pkl extension if it's not present
        file_name += ".pkl"

if file_uploader is not None:
    df_matched=pd.read_csv(file_uploader)
    st.write('Sample data record count',len(df_matched))
    df_matched.drop('maid',axis=1,inplace=True)
    df_matched['Income']=df_matched['Income'].fillna(df_matched['Income'].mode()[0])
    df_matched['age_range']=df_matched['age_range'].fillna(df_matched['age_range'].mode()[0])
    df_matched['Gender']=df_matched['Gender'].fillna(df_matched['Gender'].mode()[0])
    df_matched['Flag']=df_matched['Flag'].fillna(0)
    df_matched.Flag=df_matched.Flag.replace('Yes',1)
    df_matched=df_matched.fillna("")
    
    col1,col2=st.columns((0.75,0.25))

    with col1:
        with st.expander('Sample Data Analysis'):
            st.write(df_matched)
    
    df_matched['Concatenated'] = df_matched[['interests', 'brands_visited', 'place_categories','geobehaviour', 'Income', 'age_range', 'Gender']].apply(lambda row: '|'.join(row), axis=1)


    if not os.path.exists('vectorized_sample_'+file_name):
        vectorized_sample=vectorizer(df_matched,unsupervised_tokens)
        joblib.dump(vectorized_sample,'vectorized_sample_'+file_name)
        st.write("save data: Run Again ")
        rerun_button = st.button("Rerun App")

        if rerun_button:
            st.experimental_rerun()

    else:
        vectorized_sample = joblib.load('vectorized_sample_'+file_name)
# np.set_printoptions(threshold=np.inf)


# #-----------------------------------------------------------------------------------------------------------------------------

    if not os.path.exists('train_test_'+file_name):

        X=pd.DataFrame(vectorized_sample)
        joblib.dump(X, 'train_test_'+file_name)
    else:
        X = joblib.load('train_test_'+file_name)
    y=df_matched['Flag']
    
    # st.write(X)
    # st.write(y.value_counts())

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    # st.write(X_train.shape)
    # st.write(X_test.shape)
    # st.write(y_train.shape)
    # st.write(df_matched)


    vectorized_X_train_smote=X_train
    y_train_smote=y_train
    # st.write(vectorized_X_train_smote)
    # st.write(y_train_smote)


    # plt.pie(np.array([y_train.value_counts()[0],y_train.value_counts()[1]]),labels=['No','Yes'])
    # smote=SMOTE()
    # vectorized_X_train_smote,y_train_smote=smote.fit_resample(X_train,y_train)

    # st.write(y_train_smote.value_counts())
    # plt.pie(np.array([y_train_smote.value_counts()[0],y_train_smote.value_counts()[1]]),labels=['Yes','No'])
    # # st.pyplot(plt)

    def training_score(y_act,y_pred):
        acc=round(accuracy_score(y_act,y_pred)*100,3)
        f1=round(f1_score(y_act,y_pred)*100,3)
        ps=round(precision_score(y_act,y_pred)*100,3)
        rs=round(recall_score(y_act,y_pred)*100,3)
        st.write(f'Training Scores: \n\tAccuracy = {acc}\n\tPrecision = {ps}\n\tF1_score = {f1}\n\tRecall = {rs}')

    def testing_score(y_act,y_pred):
        acc=round(accuracy_score(y_act,y_pred)*100,3)
        f1=round(f1_score(y_act,y_pred)*100,3)
        ps=round(precision_score(y_act,y_pred)*100,3)
        rs=round(recall_score(y_act,y_pred)*100,3)
        st.write(f'Testing Scores: \n\tAccuracy = {acc}\n\tPrecision = {ps}\n\tF1_score = {f1}\n\tRecall = {rs}')


    if not os.path.exists("best_model_"+file_name):

        lr = LogisticRegression()
        lr.fit(vectorized_X_train_smote,y_train_smote)

        y_train_pred=lr.predict(vectorized_X_train_smote)
        training_score(y_train_smote,y_train_pred)

        y_test_pred=lr.predict(X_test)
        testing_score(y_test,y_test_pred)
        st.write('LogisticRegression')
        st.write('confusion Matrix is:')
        st.write(confusion_matrix(y_test,y_test_pred))


        mnb=MultinomialNB()
        mnb.fit(vectorized_X_train_smote,y_train_smote)
        y_train_pred=mnb.predict(vectorized_X_train_smote)
        training_score(y_train_smote,y_train_pred)
        y_test_pred=mnb.predict(X_test)
        testing_score(y_test,y_test_pred)
        st.write('MultinomialNB')
        st.write('confusion Matrix is:')
        st.write(confusion_matrix(y_test,y_test_pred))


        dt=DecisionTreeClassifier()
        dt.fit(vectorized_X_train_smote,y_train_smote)
        y_train_pred=dt.predict(vectorized_X_train_smote)
        training_score(y_train_smote,y_train_pred)
        y_test_pred=dt.predict(X_test)
        testing_score(y_test,y_test_pred)
        st.write('DecisionTreeClassifier')
        st.write('confusion Matrix is:')
        st.write(confusion_matrix(y_test,y_test_pred))

        rf=RandomForestClassifier()
        rf.fit(vectorized_X_train_smote,y_train_smote)
        y_train_pred=rf.predict(vectorized_X_train_smote)
        training_score(y_train_smote,y_train_pred)
        y_test_pred=rf.predict(X_test)
        testing_score(y_test,y_test_pred)
        st.write('RandomForestClassifier')
        st.write('confusion Matrix is:')
        st.write(confusion_matrix(y_test,y_test_pred))

        joblib.dump(rf,"best_model_"+file_name)

    else:
        loaded_model=joblib.load("best_model_"+file_name)
        predictions=loaded_model.predict_proba(X_test)

        if not os.path.exists(file_name+'_model prediction.csv'):
        
            df_prediction_prob=pd.DataFrame(predictions,columns=['prob_0','prob_1'])
            df_test_dataset=pd.DataFrame(np.array(y_test),columns=['Actual_outcome'])
            # df_x_test=pd.DataFrame(X_test)
            dfx=pd.concat([df_test_dataset,df_prediction_prob],axis=1)
            dfx.to_csv(file_name+'_model prediction.csv')
        
        
        else:
            new_columns_probable=['Actual_Index','Actual_Outcome','Probability_Non_Buyers','Probability_Most_Probabale_Buyers']
            with col2:
                with st.expander('Probability Prediction'):
                    Probability_Prediction=pd.read_csv(file_name+'_model prediction.csv').sort_values('prob_1',ascending=False).reset_index(drop=True)
                    Probability_Prediction.columns=new_columns_probable
                    st.write(Probability_Prediction)
            
            def model_prediction_csv(csv_name):

                model_prediction=pd.read_csv(csv_name).sort_values('prob_1',ascending=False)
                total_rows = len(model_prediction)
                records_per_decile=total_rows//10
                decile_labels = [i for i in range(1, (total_rows // records_per_decile) + 2) for _ in range(records_per_decile)]
                # decile_labels += [i for i in range(1, (total_rows % records_per_decile) + 2)]
                model_prediction['deciles'] = np.array(decile_labels[:total_rows])
                # st.write(model_prediction)

                new_columns=['each_decile_data_count','Probability_Threshold','Success','Unsuccess',
                             'Cumulative_Success','Cumulative_Unsuccess','% Cumulative_Success',
                             '% Cumulative_Unsuccess','% Total Non Buyers Avoided','Total_inflow',
                             'Total_Expenditure','Profit(*1000)']

                model_prediction_pivot=model_prediction.copy()
                agg_dict = {
                    'deciles': 'count',  
                    'prob_1': 'min',
                    'Actual_outcome':'sum'      
                }


                model_prediction_pivot = model_prediction_pivot.groupby('deciles').agg(agg_dict)
                model_prediction_pivot['bad']=model_prediction_pivot['deciles']-model_prediction_pivot['Actual_outcome']
                model_prediction_pivot['cum_good']=model_prediction_pivot['Actual_outcome'].cumsum()
                model_prediction_pivot['cum_bad']=model_prediction_pivot['bad'].cumsum()
                # Create a new column with the division
                model_prediction_pivot['% Cumm Good'] = round(model_prediction_pivot['cum_good'] / model_prediction_pivot['cum_good'].max()*100)
                model_prediction_pivot['% Cumm bad'] = round(model_prediction_pivot['cum_bad'] / model_prediction_pivot['cum_bad'].max()*100)
                model_prediction_pivot['% Cumm Bad Avoided']=100-model_prediction_pivot['% Cumm bad']
                model_prediction_pivot['total_inflow']=int(revenue)*model_prediction_pivot['cum_good']
                model_prediction_pivot['total_expenditure']=(model_prediction_pivot['cum_good']+model_prediction_pivot['cum_bad'])*int(cost)
                model_prediction_pivot['profit (*1000)']=(int(revenue)*model_prediction_pivot['cum_good']-((model_prediction_pivot['cum_good']+model_prediction_pivot['cum_bad'])*int(cost)))/1000
                model_prediction_pivot = model_prediction_pivot.rename_axis('No_of_Decile')



                model_prediction_pivot.columns=new_columns

                return model_prediction_pivot
            

            with col2:
                revenue=st.text_input('Revenue from a successful buyer: ')
                cost=st.text_input('Cost of promotional sample kit: ')


            with col1:
                with st.expander('Sample Data Profit Analysis'):
                    st.write(model_prediction_csv(file_name+'_model prediction.csv'))


                master_uploader = st.file_uploader(" :file_folder: Upload the master file", type=["csv"])
            if master_uploader is not None:
                master=pd.read_csv(master_uploader)

                master.drop('maid',axis=1,inplace=True)
                master['Income']=master['Income'].fillna(master['Income'].mode()[0])
                master['Age_Range']=master['Age_Range'].fillna(master['Age_Range'].mode()[0])
                master['Gender']=master['Gender'].fillna(master['Gender'].mode()[0])
                master=master.fillna("")
                master['Concatenated'] = master[['interests', 'brands_visited', 'place_categories','geobehaviour', 'Income', 'Age_Range', 'Gender']].apply(lambda row: '|'.join(row), axis=1)
                
                file_name_master = master_uploader.name.rsplit(".", 1)[0]
                if not file_name_master.endswith(".pkl"):
                    # Append the .pkl extension if it's not present
                    file_name_master += ".pkl"
                if not os.path.exists('vectorized_sample_'+file_name_master):
                    vectorized_sample=vectorizer(master,unsupervised_tokens)
                    joblib.dump(vectorized_sample,'vectorized_sample_'+file_name_master)
                    st.write("save data: Run Again ")
                    rerun_button = st.button("Rerun App")

                    if rerun_button:
                        st.experimental_rerun()
            
                else:
                    vectorized_sample = joblib.load('vectorized_sample_'+file_name_master)
                    # st.write(len(vectorized_sample))

                # np.set_printoptions(threshold=np.inf)
                model = joblib.load('best_model_wine-samples-500_test.pkl')

                y_pred = model.predict(vectorized_sample)
                predictions = model.predict_proba(vectorized_sample)

                # # writing model output file
                df_prediction_prob = pd.DataFrame(predictions, columns = ['prob_0', 'prob_1'])
                y_pred=pd.DataFrame(y_pred,columns=['Actual_outcome'])


                dfx=pd.concat([y_pred,df_prediction_prob], axis=1)
                if not os.path.exists('master_data_predictor.csv'):

                    dfx.to_csv("master_data_predictor.csv")
                else:
                    col1,col2=st.columns((0.75,0.25))
                    with col2:
                        with st.expander('Master Data Probable Buyers Prediction'):
                            master_df=pd.read_csv("master_data_predictor.csv").sort_values('prob_1',ascending=False).reset_index(drop=True)
                            master_df.columns=new_columns_probable
                            st.write(master_df)
                    with col1:
                        with st.expander('Master Data Profit Analysis'):
                            model_prediction_csv=model_prediction_csv('master_data_predictor.csv')
                            st.write(model_prediction_csv)
                    
                        with st.expander('Strategic marketing options '):
                    
                            decile_selection=st.multiselect("select_Deciles",[100,90,80,70,60,50,40,30,20,10])


                            dfs=[]
                            for i in range(len(decile_selection)):
                                
                                Participants_Covered=model_prediction_csv.each_decile_data_count.sum()*decile_selection[i]/100
                                Cumulative_Success=model_prediction_csv[model_prediction_csv.index == decile_selection[i]/10]['Cumulative_Success'].values[0]
                                Cumulative_Unsuccess=model_prediction_csv[model_prediction_csv.index == decile_selection[i]/10]['Cumulative_Unsuccess'].values[0]
                                Success_Percentage=Cumulative_Success/(Cumulative_Success+Cumulative_Unsuccess)*100
                                total_buyers=model_prediction_csv[model_prediction_csv.index == decile_selection[i]/10]['% Cumulative_Success'].values[0]
                                buyers_avoided=model_prediction_csv[model_prediction_csv.index == decile_selection[i]/10]['% Total Non Buyers Avoided'].values[0]
                                Probability_Threshold=model_prediction_csv[model_prediction_csv.index == decile_selection[i]/10]['Probability_Threshold'].values[0]
                                inflow=Participants_Covered*Success_Percentage/100*int(revenue)
                                outflow=Participants_Covered*int(cost)
                                profit=(inflow-outflow)/1000

                                data={'% Strategic_Option':[decile_selection[i]],'Participants_Covered':[Participants_Covered],
                                    '% Success_Percentage':[Success_Percentage],'% Total Buyers Reached':[total_buyers],
                                    '% Total Non Buyers Avoided':[buyers_avoided],'Probability Threshold':[Probability_Threshold],
                                    'Profit(*1000)':[profit]}
                                df=pd.DataFrame(data)
                                dfs.append(df)
                            result_df = pd.concat(dfs, axis=0, ignore_index=True).set_index("% Strategic_Option")

                                
                            
                            st.write(result_df)
