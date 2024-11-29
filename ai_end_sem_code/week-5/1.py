
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from hmmlearn.hmm import GaussianHMM
from yahoo_fin.stock_info import get_data






cscoi= get_data("CSCO", start_date="12/04/2014", end_date="12/04/2024", index_as_date = True, interval="1d")



MP = pd.DataFrame(cscoi)

MP


MP['daily_return'] = MP['adjclose'].pct_change()

# Drop the first row as it will have NaN in the return
MP.dropna(subset=['daily_return'], inplace=True)


MP.to_csv('MSData.csv', index=True)

# Read the data with correct date parsing
df = pd.read_csv('MSData.csv')

df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

# Reshape the daily returns to fit the model
returns = df['daily_return'].values.reshape(-1, 1)

# Initialize and fit the HMM model
model = GaussianHMM(n_components=4, covariance_type="diag", n_iter=50000)
model.fit(returns)

# Predict hidden states
hidden_states = model.predict(returns)
df['hidden_state'] = hidden_states

# Get the most recent state and transition matrix
most_recent_state = hidden_states[-1]
transition_matrix = model.transmat_
future_state = np.argmax(transition_matrix[most_recent_state])

# Define distinct colors for each state
state_colors = {
    0: '#FF8C8C',  
    1: '#A890FE',  # bright green
    2: '#38ADAE',  # bright blue
    3: '#F9C449'   # golden yellow
}

# Create state labels based on volatility
state_volatilities = [np.sqrt(cov[0]) for cov in model.covars_]
state_means = [mean[0] for mean in model.means_]
state_labels = {
    np.argmin(state_volatilities): 'Low Volatility',
    np.argmax(state_volatilities): 'High Volatility'
}
for i in range(4):
    if i not in state_labels:
        if state_means[i] > 0:
            state_labels[i] = 'Bullish'
        else:
            state_labels[i] = 'Bearish'

# Create combined state labels with numbers
state_labels_with_numbers = {
    state: f'State {state} - {label}' for state, label in state_labels.items()
}

# Simulate next 10 days
n_days = 10
future_hidden_states = []
future_returns = []
current_state = most_recent_state

for _ in range(n_days):
    current_state = np.random.choice(range(model.n_components), p=transition_matrix[current_state])
    future_hidden_states.append(current_state)
    predicted_return = float(np.random.normal(
        loc=model.means_[current_state][0], 
        scale=np.sqrt(model.covars_[current_state][0])
    ))
    future_returns.append(predicted_return)

# Create future DataFrame
last_date = df['Date'].iloc[-1]
future_dates = pd.date_range(start=last_date, periods=n_days + 1, freq='D')[1:]
future_df = pd.DataFrame({
    'Date': future_dates,
    'daily_return': future_returns,
    'hidden_state': future_hidden_states
})

# Create subplots - one for returns, one for transition matrix
fig = make_subplots(rows=2, cols=1, 
                    subplot_titles=("Stock Returns with 10-Day Prediction", 
                                  "State Transition Probabilities Matrix"),
                    vertical_spacing=0.2,
                    row_heights=[0.7, 0.3])

# Add continuous grey line for all historical data
fig.add_trace(go.Scatter(
    x=df['Date'],
    y=df['daily_return'],
    mode='lines',
    line=dict(color='grey', width=1),
    showlegend=False,
    hoverinfo='skip'
), row=1, col=1)

# Add historical data split by state
for state in range(model.n_components):
    mask = df['hidden_state'] == state
    fig.add_trace(go.Scatter(
        x=df.loc[mask, 'Date'],
        y=df.loc[mask, 'daily_return'],
        mode='markers',
        name=f'Historical - {state_labels_with_numbers[state]}',
        marker=dict(
            color=state_colors[state],
            size=8
        ),
        legendgroup=f'state_{state}'
    ), row=1, col=1)

# Add grey line connecting the last historical point to predicted points
fig.add_trace(go.Scatter(
    x=[df['Date'].iloc[-1], future_df['Date'].iloc[0]],
    y=[df['daily_return'].iloc[-1], future_df['daily_return'].iloc[0]],
    mode='lines',
    line=dict(color='grey', width=1),
    showlegend=False,
    hoverinfo='skip'
), row=1, col=1)

# Add grey lines connecting predicted points
fig.add_trace(go.Scatter(
    x=future_df['Date'],
    y=future_df['daily_return'],
    mode='lines',
    line=dict(color='grey', width=1),
    showlegend=False,
    hoverinfo='skip'
), row=1, col=1)

# Add predicted returns with states
for state in range(model.n_components):
    mask = future_df['hidden_state'] == state
    if mask.any():  # Only add trace if there are predictions for this state
        fig.add_trace(go.Scatter(
            x=future_df.loc[mask, 'Date'],
            y=future_df.loc[mask, 'daily_return'],
            mode='markers',
            name=f'Predicted - {state_labels_with_numbers[state]}',
            marker=dict(
                color=state_colors[state],
                size=12,
                symbol='star'
            ),
            legendgroup=f'state_{state}'
        ), row=1, col=1)

# Add confidence intervals for predictions
std_dev = np.sqrt(model.covars_[future_hidden_states]).flatten()
upper_bound = np.array(future_returns) + std_dev
lower_bound = np.array(future_returns) - std_dev

fig.add_trace(go.Scatter(
    x=future_df['Date'],
    y=upper_bound,
    mode='lines',
    line=dict(width=0),
    showlegend=False,
    hoverinfo='skip'
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=future_df['Date'],
    y=lower_bound,
    mode='lines',
    line=dict(width=0),
    fillcolor='rgba(128, 128, 128, 0.2)',
    fill='tonexty',
    showlegend=True,
    name='Prediction Interval'
), row=1, col=1)

# Add transition matrix heatmap without colorbar
fig.add_trace(go.Heatmap(
    z=transition_matrix,
    x=[f'To State {i}<br>{state_labels[i]}' for i in range(model.n_components)],
    y=[f'From State {i}<br>{state_labels[i]}' for i in range(model.n_components)],
    hoverongaps=False,
    colorscale='YlOrRd',
    text=np.round(transition_matrix, 3),
    texttemplate='%{text}',
    textfont={"size": 14},
    showscale=False  # Remove colorbar
), row=2, col=1)

# Update layout
fig.update_layout(
    height=1200,
    width=1000,
    showlegend=True,
    template='plotly_white',
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=1.05
    ),
    # Add a vertical line to separate historical and predicted data
    shapes=[dict(
        type="line",
        x0=df['Date'].iloc[-1],
        x1=df['Date'].iloc[-1],
        y0=df['daily_return'].min(),
        y1=df['daily_return'].max(),
        line=dict(color="black", width=2, dash="dash"),
    )]
)

# Update axes
fig.update_xaxes(showgrid=True, gridcolor='lightgrey', gridwidth=1)
fig.update_yaxes(showgrid=True, gridcolor='lightgrey', gridwidth=1)

# Show the plot
fig.show()

# Analyze state transitions
state_changes = []
for i in range(1, len(df['hidden_state'])):
    if df['hidden_state'].iloc[i] != df['hidden_state'].iloc[i-1]:
        state_changes.append({
            'date': df['Date'].iloc[i],
            'from_state': f"State {df['hidden_state'].iloc[i-1]} - {state_labels[df['hidden_state'].iloc[i-1]]}",
            'to_state': f"State {df['hidden_state'].iloc[i]} - {state_labels[df['hidden_state'].iloc[i]]}",
            'return': df['daily_return'].iloc[i]
        })

# Print predictions in a clean format
print("\nPredicted Returns for Next 10 Days:")
print("Date            | Return   | State")
print("-" * 50)
for date, ret, state in zip(future_df['Date'], future_df['daily_return'], future_df['hidden_state']):
    print(f"{date.strftime('%Y-%m-%d')} | {float(ret):8.4f} | {state_labels_with_numbers[state]}")

# Print state characteristics
print("\nState Characteristics:")
print("-" * 50)
for state in range(model.n_components):
    mean_return = float(model.means_[state][0])
   
    volatility = float(np.sqrt(model.covars_[state][0]))
    print(f"{state_labels_with_numbers[state]}:")
    print(f"  Mean Return: {mean_return:8.4f}")
    print(f"  Volatility: {volatility:8.4f}")
    

# Print state transition analysis
print("\nState Transition Analysis:")
print("-" * 80)
transitions_df = pd.DataFrame(state_changes)
if not transitions_df.empty:
    transition_counts = transitions_df.groupby(['from_state', 'to_state']).size().sort_values(ascending=False)
    print("\nTop 5 Most Frequent Transitions:")
    print(transition_counts.head())

    print("\nAverage Returns During State Transitions:")
    avg_transition_returns = transitions_df.groupby(['from_state', 'to_state'])['return'].agg(['mean', 'count']).round(4)
    print(avg_transition_returns.sort_values('count', ascending=False).head())
else:
    print("\nNo state transitions found in the data")

# Calculate average duration of each state
state_durations = []
current_state = df['hidden_state'].iloc[0]
current_duration = 1

for i in range(1, len(df['hidden_state'])):
    if df['hidden_state'].iloc[i] == current_state:
        current_duration += 1
    else:
        state_durations.append({
            'state': f"State {current_state} - {state_labels[current_state]}",
            'duration': current_duration
        })
        current_state = df['hidden_state'].iloc[i]
        current_duration = 1

# Add the last state duration
state_durations.append({
    'state': f"State {current_state} - {state_labels[current_state]}",
    'duration': current_duration
})

# Calculate and print average state durations
durations_df = pd.DataFrame(state_durations)
avg_durations = durations_df.groupby('state')['duration'].agg(['mean', 'max', 'min']).round(2)
print("\nState Duration Analysis (in days):")
print("-" * 80)
print(avg_durations)

# Count the occurrences of each state in historical data
historical_state_counts = df['hidden_state'].value_counts()

# Count the occurrences of each state in predicted data
predicted_state_counts = pd.Series(future_df['hidden_state']).value_counts()

# Print state counts
print("\nState Counts (Historical):")
print("-" * 50)
for state, count in historical_state_counts.items():
    print(f"{state_labels_with_numbers[state]}: {int(count)} occurrences")

print("\nState Counts (Predicted):")
print("-" * 50)
for state, count in predicted_state_counts.items():
    print(f"{state_labels_with_numbers[state]}: {int(count)} occurrences")

# Overall state counts (historical + predicted)
overall_state_counts = historical_state_counts.add(predicted_state_counts, fill_value=0)

# Print overall state counts
print("\nOverall State Counts:")
print("-" * 50)
for state, count in overall_state_counts.items():
    print(f"{state_labels_with_numbers[state]}: {int(count)} occurrences")



# Plot bar chart for state counts
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# State labels and colors for the plots
state_labels = [state_labels_with_numbers[state] for state in range(model.n_components)]
state_colors = [state_colors[state] for state in range(model.n_components)]

# Create subplots with specific types for the pie chart
fig = make_subplots(
    rows=2, cols=2,
    specs=[[{"type": "xy"}, {"type": "domain"}], [{"type": "xy"}, {"type": "xy"}]],
    subplot_titles=(
        "Overall State Counts (Historical + Predicted)", 
        "Percentage Distribution of States", 
        "Mean Returns by State", 
        "Volatility by State"
    )
)

# Bar Chart: Overall State Counts
fig.add_trace(
    go.Bar(
        x=state_labels,
        y=overall_state_counts.values,
        marker_color=state_colors,
        name="State Counts"
    ),
    row=1, col=1
)

# Pie Chart: State Distribution
fig.add_trace(
    go.Pie(
        labels=state_labels,
        values=overall_state_counts.values,
        marker_colors=state_colors,
        name="State Distribution"
    ),
    row=1, col=2
)

# Bar Chart: Mean Returns by State
mean_returns = [float(model.means_[state][0]) for state in range(model.n_components)]
fig.add_trace(
    go.Bar(
        x=state_labels,
        y=mean_returns,
        marker_color=state_colors,
        name="Mean Returns"
    ),
    row=2, col=1
)

# Bar Chart: Volatility by State
volatilities = [np.sqrt(float(model.covars_[state][0])) for state in range(model.n_components)]
fig.add_trace(
    go.Bar(
        x=state_labels,
        y=volatilities,
        marker_color=state_colors,
        name="Volatility"
    ),
    row=2, col=2
)

# Update layout
fig.update_layout(
    title="State Analysis (Counts, Distribution, Mean Returns, Volatility)",
    title_x=0.5,
    height=800,
    width=1000,
    showlegend=True
)

# Show plot
fig.show()
C:\Users\Abhi Patel\AppData\Local\Temp\ipykernel_17880\950991757.py:88: DeprecationWarning:

Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future. Ensure you extract a single element from your array before performing this operation. (Deprecated NumPy 1.25.)

C:\Users\Abhi Patel\AppData\Local\Temp\ipykernel_17880\950991757.py:264: DeprecationWarning:

Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future. Ensure you extract a single element from your array before performing this operation. (Deprecated NumPy 1.25.)

Predicted Returns for Next 10 Days:
Date            | Return   | State
--------------------------------------------------
2024-11-28 |  -0.0001 | State 1 - Low Volatility
2024-11-29 |  -0.0075 | State 0 - Bullish
2024-11-30 |   0.0021 | State 1 - Low Volatility
2024-12-01 |  -0.0091 | State 0 - Bullish
2024-12-02 |   0.0102 | State 1 - Low Volatility
2024-12-03 |   0.0149 | State 0 - Bullish
2024-12-04 |   0.0077 | State 1 - Low Volatility
2024-12-05 |  -0.0024 | State 0 - Bullish
2024-12-06 |   0.0063 | State 1 - Low Volatility
2024-12-07 |  -0.0243 | State 0 - Bullish

State Characteristics:
--------------------------------------------------
State 0 - Bullish:
  Mean Return:   0.0001
  Volatility:   0.0108
State 1 - Low Volatility:
  Mean Return:   0.0020
  Volatility:   0.0098
State 2 - Bearish:
  Mean Return:  -0.0009
  Volatility:   0.0216
State 3 - High Volatility:
  Mean Return:  -0.0034
  Volatility:   0.0540

State Transition Analysis:
--------------------------------------------------------------------------------

Top 5 Most Frequent Transitions:
from_state                 to_state                 
State 1 - Low Volatility   State 0 - Bullish            986
State 0 - Bullish          State 1 - Low Volatility     962
State 3 - High Volatility  State 1 - Low Volatility      23
State 0 - Bullish          State 3 - High Volatility     17
                           State 2 - Bearish             14
dtype: int64

Average Returns During State Transitions:
                                                       mean  count
from_state                to_state                                
State 1 - Low Volatility  State 0 - Bullish          0.0003    986
State 0 - Bullish         State 1 - Low Volatility   0.0020    962
State 3 - High Volatility State 1 - Low Volatility   0.0046     23
State 0 - Bullish         State 3 - High Volatility -0.0205     17
                          State 2 - Bearish         -0.0117     14

State Duration Analysis (in days):
--------------------------------------------------------------------------------
                            mean  max  min
state                                     
State 0 - Bullish           1.00    1    1
State 1 - Low Volatility    1.01    2    1
State 2 - Bearish          31.93   66   10
State 3 - High Volatility   2.00   15    1

State Counts (Historical):
--------------------------------------------------
State 0 - Bullish: 994 occurrences
State 1 - Low Volatility: 991 occurrences
State 2 - Bearish: 479 occurrences
State 3 - High Volatility: 48 occurrences

State Counts (Predicted):
--------------------------------------------------
State 1 - Low Volatility: 5 occurrences
State 0 - Bullish: 5 occurrences

Overall State Counts:
--------------------------------------------------
State 0 - Bullish: 999 occurrences
State 1 - Low Volatility: 996 occurrences
State 2 - Bearish: 479 occurrences
State 3 - High Volatility: 48 occurrences
C:\Users\Abhi Patel\AppData\Local\Temp\ipykernel_17880\950991757.py:398: DeprecationWarning:

Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future. Ensure you extract a single element from your array before performing this operation. (Deprecated NumPy 1.25.)

 
