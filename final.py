import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor

def PrecessingData(filename):
    td=pd.read_csv(filename)
    td['departure_date'] = pd.to_datetime(td['departure_date'], errors='coerce')
    td['booking_date'] = pd.to_datetime(td['booking_date'], errors='coerce')
    td['prior_day'] = td['departure_date'] - td['booking_date']

    prior_days = []
    for i in range(0, len(td['departure_date'])):
        prior_days.append(td.iloc[i, -1].days)
    
    td['prior_days'] = prior_days
    td.drop('prior_day', axis=1, inplace=True)

    temp = td.loc[td['departure_date'] == td['booking_date']][['departure_date', 'cum_bookings']]
    temp.rename(columns={'cum_bookings': 'total'}, inplace=True)
    td = pd.merge(td, temp, on='departure_date', how='inner')
    td['delta'] = td['total'] - td['cum_bookings']
    
    dow = []
    for i in range(0, len(td['departure_date'])):
        dow.append(datetime.weekday(td.iloc[i, 0]))
    td['dow'] = dow
    return td



def Method_1(training, verification):
    
    deltaDemands = np.array(training['delta'])
    feature = np.array(training[['prior_days', 'dow']])
    regressor = RandomForestRegressor(n_estimators=100, random_state=0)
    regressor.fit(feature, deltaDemands)
        
    feature = np.array(verification[['prior_days', 'dow']])
    delta = regressor.predict(feature)
    verification['pred_delta'] = delta
    verification['pred_demand'] = verification['pred_delta'] + verification['cum_bookings']
    verification.drop('pred_delta', axis=1, inplace=True)
    verification.drop('prior_days', axis=1, inplace=True)
    verification.drop('delta', axis=1, inplace=True)
    verification.drop('dow', axis=1, inplace=True)
        
    verification['forecast_error']=abs(verification['final_demand']-verification['pred_demand'])
    verification['naive_error']=abs(verification['final_demand']-verification['naive_forecast'])
    MASE=np.nansum(verification['forecast_error'])/np.nansum(verification['naive_error'])
    forecast=pd.DataFrame(verification.loc[:,('departure_date','booking_date','pred_demand')])
    return MASE,forecast


def Method_2(training, verification):
    
    td_sub=training[['prior_days','dow','delta']]
    avg_delta=td_sub.groupby(['prior_days','dow'],as_index=False).delta.mean()
    pd.DataFrame(avg_delta)
    verification.drop('delta', axis=1, inplace=True)
    verification = pd.merge(avg_delta,verification, left_on=['prior_days','dow'],right_on=['prior_days','dow'],how='inner')
    verification['pred_demand']=verification['delta']+verification['cum_bookings']
    verification['forecast_error']=abs(verification['final_demand']-verification['pred_demand'])
    verification['naive_error']=abs(verification['final_demand']-verification['naive_forecast'])
    MASE=np.nansum(verification['forecast_error'])/np.nansum(verification['naive_error'])
    forecast=pd.DataFrame(verification.loc[:,('departure_date','booking_date','pred_demand')])
    return MASE,forecast
    
def Method_3(training, verification):
    
    training['delta_rate'] = (training['total'] - training['cum_bookings'])/training['total']
    training_sub=training[['prior_days','dow','delta_rate']]
    avg_delta_rate=training_sub.groupby(['prior_days','dow'],as_index=False).delta_rate.mean()
    pd.DataFrame(avg_delta_rate)
   
    verification.drop('delta', axis=1, inplace=True)
    
    verification = pd.merge(avg_delta_rate,verification, left_on=['prior_days','dow'],right_on=['prior_days','dow'],how='inner')
    verification['pred_demand']=verification['cum_bookings']/(1-verification['delta_rate'])
    
    verification['forecast_error']=abs(verification['final_demand']-verification['pred_demand'])
    verification['naive_error']=abs(verification['final_demand']-verification['naive_forecast'])
    MASE=np.nansum(verification['forecast_error'])/np.nansum(verification['naive_error'])
    forecast=pd.DataFrame(verification.loc[:,('departure_date','booking_date','pred_demand')])
    return MASE,forecast
        
    
def airlineForecast():
    training = PrecessingData("airline_booking_trainingData.csv")
    verification = PrecessingData("airline_booking_validationData_revised.csv")

    MASE1, forecast1 = Method_1(training.copy(), verification.copy())
    MASE2, forecast2 = Method_2(training.copy(), verification.copy())
    MASE3, forecast3 = Method_3(training.copy(), verification.copy())
    print "MASE for model1: {}\nMASE for model2: {}\nMASE for model3: {}".format(MASE1,MASE2,MASE3)
    output = []
    if MASE1 < MASE2 and MASE1 < MASE3:
        output.append(MASE1)
        output.append(forecast1)
    elif MASE2 < MASE1 and MASE2 < MASE3:
        output.append(MASE2)
        output.append(forecast2)
    else:
        output.append((MASE3))
        output.append((forecast3))
    
    return output

airlineForecast()
