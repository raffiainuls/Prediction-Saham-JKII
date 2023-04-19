import streamlit as st
from prediction import load_models, result,  predict, evaluation
import plotly.express  as px
import plotly.graph_objects  as go
import streamlit as st



# Tambahkan konten Streamlit di sini

st.markdown("<h1 style = 'text-align : center; color : black; font_size : 40 px; font-family : Arial'><b>Simple Stock Prediction<b></h1>", unsafe_allow_html= True)
st.markdown("------")
st.markdown("Created by [Raffi Ainul Afif](https://www.linkedin.com/in/raffi-ainul-afif-9811a411b/)")

option = st.selectbox(
    'Select Model',
    ('Xgboost', 'Gradient Boost', 'LSTM', 'Linear Regression', 'Support Vector Regression')

)
model_lstm, model_gradient, model_linear, model_svr, model_xgboost = load_models()
mse_loss, bce_loss, hubber_loss, hinge_loss,mape,rmse, df = result(option)

fig = go.Figure()
fig.add_trace(go.Scatter(go.Scatter(x=df.index, y=df['predict_train'],
                    mode='lines',
                    name='Train prediction')))
fig.add_trace(go.Scatter(x=df.index, y=df['predict_test'],
                    mode='lines',
                    name='Test prediction'))
fig.add_trace(go.Scatter(go.Scatter(x=df.index, y=df['Close'],
                    mode='lines',
                    name='Actual Value')))
fig.update_layout(
            xaxis=dict(
            showline=True,
            showgrid=True,
            showticklabels=False,
            linecolor='black',
            linewidth=2
            ),
            yaxis=dict(
            title_text='Close (USD)',
            titlefont=dict(
                family='Rockwell',
                size=18,
                color='black',
                ),
            showline=True,
            showgrid=True,
            showticklabels=True,
            linecolor='black',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Rockwell',
                size=18,
                color='black',
                ),
            ),
            showlegend=True,
            

        )

annotations = []
annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                              xanchor='left', yanchor='bottom',
                              text='Results (LSTM)',
                              font=dict(family='Rockwell',
                                        size=26,
                                        color='black'),
                              showarrow=False))
fig.update_layout(annotations=annotations)

st.plotly_chart(fig, width= 800, height=100)
  

with st.container():
    col1,col2,col3= st.columns(3)
    with col1 :
        
        st.write("<h1 style='font-align : center; font-size : 40; color :black'> MAPE</h1>",unsafe_allow_html=True)
        st.write("<h1 style='font-align : center; font-size : 40; color :#2F58CD'>{:.2f}%</h1>".format(mape), unsafe_allow_html=True)

        st.write("<h1 style='font-align : center; font-size : 40; color :black'> RMSE</h1>",unsafe_allow_html=True)
        st.write("<h1 style='font-align : center; font-size : 40; color :#2F58CD'>{:.2f}</h1>".format(rmse), unsafe_allow_html=True)

        
    with col2:
        st.write("<h1 style='font-align : center; font-size : 40; color :black'> BCE</h1>",unsafe_allow_html=True)
        st.write("<h1 style='font-align : center; font-size : 40; color :#2F58CD'>{:.2f}</h1>".format(bce_loss), unsafe_allow_html=True)

        st.write("<h1 style='font-align : center; font-size : 40; color :black'> Hubeer Loss</h1>",unsafe_allow_html=True)
        st.write("<h1 style='font-align : center; font-size : 40; color :#2F58CD'>{:.2f}</h1>".format(hubber_loss), unsafe_allow_html=True)

       
    with col3:
        st.write("<h1 style='font-align : center; font-size : 50; color :black'> MSE</h1>",unsafe_allow_html=True)
        st.write("<h1 style='font-align : center; font-size : 50; color :#2F58CD'>{:.2f}</h1>".format(mse_loss), unsafe_allow_html=True)

        




        
        

            