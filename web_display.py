import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from jupyter_dash import JupyterDash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import stress_level_real_time
import multiprocessing as mp
import dash

pd.options.plotting.backend = "plotly"
countdown = 20
cols = ['stress_level']
X = [0]  
df=pd.DataFrame(X, columns=cols)
df.iloc[0]=0;
fig = df.plot(template = 'plotly_dark')
X1=[0,0,0,0,0,0,0]
cols2=['angry','disgust', 'fear', 'happy', 'neutral', 'sad','surprise']
df4 = pd.DataFrame({'value':X1, 'emotion':cols2})
fig2 = px.pie(df4, values="value", names="emotion",template = 'plotly_dark')
X2=[0,0,0,0,0,0]
cols3=['crossing_arms', 'crossing_fingers', 'netural&others', 'touching_faces',
       'touching_jaw', 'touching_neck']
df5 = pd.DataFrame({'value':X2, 'gesture':cols3})
fig3 = df5.plot(kind='bar',x='gesture',y='value',template = 'plotly_dark')
q_score = mp.Queue()
q_emotion = mp.Queue()
q_gesture = mp.Queue()

app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("Stress Level Detection System Dashboard"),
            dcc.Interval(
            id='interval-component',
            interval=1*1000,
            n_intervals=0
        ),
        html.Div([
            html.Button('Start', id='start', n_clicks=0),
            html.Button('Stop', id='stop', n_clicks=0),
        ]),
        html.Div(id='model',
             children="Start Stress Level Detedtion by clicking 'Start' button"),
        html.Div([
            dcc.Graph(id='graph',figure=fig,style={'height':500})],style={
                "width": "65%",
                "display": "inline-block",
                "margin-top": "`10px",}
            ),
        html.Div([
            dcc.Graph(id='histogram',figure=fig2,style={'height':250}),
            dcc.Graph(id='pie',figure=fig3,style={'height':250}),
            ],style={
                "width": "35%",
                "display": "inline-block",
                "margin-top": "10px",}
            ),    
])

# Define callback to update graph
@app.callback(
    Output('graph','figure'),
    [Input('interval-component', "n_intervals")]
)
def streamFig(value):
    
    global df
    if not q_score.empty():
        Y=[q_score.get(False)]
        df2 = pd.DataFrame(Y, columns = cols)
        df = pd.concat([df,df2], ignore_index=True)
    df3=df.copy()
    df3 = df3.tail(20)
    fig = df3.plot(y="stress_level",template = 'plotly_dark')
    
    return(fig)

@app.callback(
    Output('histogram','figure'),
    [Input('interval-component', "n_intervals")]
)
def streamFig(value):
    global X1
    if not q_emotion.empty():
        X1[q_emotion.get(False)]+=1 
    Y1=X1.copy()
    z=sum(Y1)
    count=0
    if(z!=0):
        for i in Y1:
            Y1[count]=i/z
            count+=1
    df4 = pd.DataFrame({'value':Y1, 'emotion':cols2})
    fig2 = px.pie(df4, values="value", names="emotion",template = 'plotly_dark')
    return(fig2)
@app.callback(
    Output('pie','figure'),
    [Input('interval-component', "n_intervals")]
)
def streamFig(value):
    global X2
    if not q_gesture.empty():
        X2[q_gesture.get(False)]+=1 
    df5 = pd.DataFrame({'value':X2, 'gesture':cols3})
    fig3 = df5.plot(kind='bar',x='gesture',y='value',template = 'plotly_dark')
    return(fig3)

@app.callback(
    Output("model", "is_open"),
    [Input("start", "n_clicks"), Input("stop", "n_clicks")],
    [State("model", "is_open")],
)
def activate_model(n1,n2,is_open):
    ctx = dash.callback_context
    if ctx.triggered:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == "start":
            r.start()
            return True
        else:
            r.terminate()
            return False
    return is_open

if __name__ == '__main__':
    r = mp.Process(target=stress_level_real_time.main, args=(q_score,q_emotion,q_gesture,))
    app.run_server( port = 8069)