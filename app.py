import dash
from dash import dcc, html, Input,Output,State
import plotly.express as px
import pandas as pd
from sklearn.metrics import r2_score
from ucimlrepo import fetch_ucirepo 
import os
import ssl
from dash.dependencies import Input, Output

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

color_palette = px.colors.qualitative.Pastel

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
file_path = 'adult_features.csv'
if not os.path.exists(file_path):
    ssl._create_default_https_context = ssl._create_unverified_context
    adult = fetch_ucirepo(id=2) 
    X = adult.data.features 
    y = adult.data.targets 
    merged_data = pd.concat([X, y], axis=1)
    merged_data.rename(columns={'income': 'income_label'}, inplace=True)
    merged_data.to_csv('adult_features.csv', index=False)
df = pd.read_csv(file_path)

df['income_label'] = df['income_label'].apply(lambda x: 0 if x == '<=50K' else 1)

income_proportion_by_education = df.groupby('education')['income_label'].mean()

income_proportion_df = pd.DataFrame({
    'Education Level': income_proportion_by_education.index,
    'Proportion Earning >50K': income_proportion_by_education.values
})

fig = px.bar(
    income_proportion_df,
    x='Education Level',
    y='Proportion Earning >50K',
    labels={'Education Level': 'Education Level', 'Proportion Earning >50K': 'Proportion Earning >50K'},
    title='Proportion Earning >50K by Education Level',
    color_discrete_sequence=color_palette
)

fig.update_traces(marker_color='skyblue', marker_line_color='black', marker_line_width=1.5)
fig.update_layout(plot_bgcolor='darkgrey', paper_bgcolor='lightgrey')


gender_distribution = df['sex'].value_counts()

gender_distribution_df = pd.DataFrame({
    'Gender': gender_distribution.index,
    'Count': gender_distribution.values
})

pie_fig = px.pie(
    gender_distribution_df,
    names='Gender',
    values='Count',
    title='Gender Distribution',
    color_discrete_sequence=color_palette
)

app.layout = html.Div([
    html.Br(),
    html.H1(html.B("Income Insights Explorer"), style={'textAlign': 'center'}),
    html.Div([
        dcc.Graph(figure=fig),
    ], id='bar-chart-container', style={'width': '90%', 'height':'50%', 'display': 'inline-block', 'margin':'20px'}),
    html.Div([
        html.Div([
            dcc.RadioItems(
                id='gender-radioitems',
                options=[{'label': i, 'value': i} for i in df['sex'].unique()],
                value='Male',
                labelStyle={'display': 'inline-block'}
            ),
            dcc.Graph(id='pie-chart'),
        ],id='pie-chart-container'),
        html.Div([
            html.Div([
                dcc.Dropdown(
                    id='income-dropdown',
                    options=[{'label': '>50K' if i == 1 else '<=50K', 'value': i} for i in df['income_label'].unique()],
                    value='<=50K',
                    multi=False
                ),
            ], style={'display': 'inline-block', 'width': '49%'}),
            html.Div([
                dcc.Dropdown(
                    id='sex-dropdown',
                    options=[{'label': i, 'value': i} for i in df['sex'].unique()],
                    value='Male',
                    multi=False
                ),
            ], style={'display': 'inline-block', 'width': '49%'}),
            dcc.Graph(id='heatmap'),
        ],id='heatmap-container')
    ], style={'width': '100%', 'display': 'inline-block'}),
    html.Div([
        dcc.Checklist(
                id='occupation-checklist',
                options=[{'label': i, 'value': i} for i in df['occupation'].dropna().unique()],
                value=['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners', 'Prof-specialty'],
                labelStyle={'display': 'inline-block'}
        ),
        dcc.Graph(id='grouped-bar-chart'),
    ],id='grouped-bar-chart-container'),
    html.Div([
        dcc.Dropdown(
            id='entry-interval-dropdown',
            options=[{'label': f'{i},000 Entries', 'value': i*1000} for i in range(5, 50, 5)],
            value=5000
        ),

        dcc.Dropdown(
            id='parallel-dropdown',
            options=[{'label': i, 'value': i} for i in ['age', 'capital-loss', 'capital-gain', 'education-num', 'income_label', 'fnlwgt', 'hours-per-week']],
            value=['age', 'capital-loss', 'capital-gain', 'education-num', 'income_label', 'fnlwgt', 'hours-per-week'],
            multi=True
        ),
        dcc.Graph(id='parallel-coordinates'),
    ],id='parallel-coordinates-container'),
])

@app.callback(Output('pie-chart', 'figure'),[Input('gender-radioitems', 'value')])
def update_pie_chart(selected_gender):
    filtered_df = df[df['sex'] == selected_gender]
    gender_distribution = filtered_df['income_label'].value_counts()
    gender_distribution_df = pd.DataFrame({
        'Income': ['>50K' if i == 1 else '<=50K' for i in gender_distribution.index],
        'Count': gender_distribution.values
    })
    pie_fig = px.pie(
        gender_distribution_df,
        names='Income',
        values='Count',
        title=f'Income Distribution for {selected_gender}',
        color_discrete_sequence=color_palette
    )
    pie_fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='lightgrey'
    )
    return pie_fig

@app.callback(
    Output('grouped-bar-chart', 'figure'),
    [Input('occupation-checklist', 'value')]
)
def update_grouped_bar_chart(selected_occupations):
    filtered_df = df[df['occupation'].isin(selected_occupations)]
    income_by_occupation_and_education = filtered_df.groupby(['occupation', 'education'])['income_label'].mean().reset_index()
    grouped_bar_fig = px.bar(
        income_by_occupation_and_education,
        x='education',
        y='income_label',
        color='occupation',
        barmode='group',
        labels={'education': 'Education Level', 'income_label': 'Proportion Earning >50K', 'occupation': 'Occupation'},
        title='Income by Occupation and Education',
        color_discrete_sequence=color_palette,
    )
    grouped_bar_fig.update_layout(
        plot_bgcolor='darkgrey',
        paper_bgcolor='lightgrey'
    )
    return grouped_bar_fig


@app.callback(
    Output('heatmap', 'figure'),
    [Input('income-dropdown', 'value'),
     Input('sex-dropdown', 'value')]
)
def update_heatmap(selected_income, selected_sex):
    filtered_df = df[(df['income_label'] == selected_income) & (df['sex'] == selected_sex)]
    avg_hours_per_week = filtered_df.groupby(['education', 'occupation'])['hours-per-week'].mean().reset_index()
    fig = px.density_heatmap(avg_hours_per_week, x="education", y="occupation", z="hours-per-week", nbinsx=20, nbinsy=20, color_continuous_scale=px.colors.diverging.BrBG)
    fig.update_layout(title='Average Hours per Week by Education and Occupation based on Income Level', xaxis_title='Education', yaxis_title='Occupation', coloraxis_colorbar_title='Avg # of hrs/week', plot_bgcolor='darkgrey', paper_bgcolor='lightgrey')
    return fig

@app.callback(
    Output('parallel-coordinates', 'figure'),
    [Input('parallel-dropdown', 'value')]
)
def update_parallel_coordinates(selected_features):
    # Ensure 'income_label' is always included in the selected features
    selected_features = list(set(selected_features + ['income_label']))
    filtered_df = df[selected_features]
    fig = px.parallel_coordinates(filtered_df, color='income_label', color_continuous_scale=px.colors.diverging.Tealrose, labels={column: column for column in filtered_df.columns})
    fig.update_layout(title='Parallel Coordinates Plot\n', plot_bgcolor='darkgrey', paper_bgcolor='lightgrey')
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
