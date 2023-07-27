import pandas as pd
import plotly.graph_objects as go
import numpy as np
import seaborn as sns
from scipy.stats import f_oneway
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

df = pd.read_csv('decision_science_dataset.csv')

def create_heatmap(df):
    # Calculating the correlation matrix
    correlacao = df.corr()

    # Creating the Plotly heatmap
    fig = go.Figure(data=go.Heatmap(
        z=correlacao.values,
        x=correlacao.index,
        y=correlacao.columns,
        colorscale='Blues',  # You can choose other colorscales from Plotly
        zmin=-1, zmax=1,    # Setting the range of the color scale (-1 to 1 for correlation)
        colorbar=dict(title='Correlation')  # Label for the color bar
    ))

    # Adding annotations to the heatmap
    for i in range(len(correlacao.index)):
        for j in range(len(correlacao.columns)):
            fig.add_annotation(
                x=correlacao.index[i],
                y=correlacao.columns[j],
                text=f'{correlacao.iloc[i, j]:.2f}',
                showarrow=False,
                font=dict(color='black', size=12)
            )

    # Customize the layout of the heatmap
    fig.update_layout(
        title='Correlation Heatmap',
        xaxis=dict(title='Features'),
        yaxis=dict(title='Features'),
    )

    return fig

import plotly.graph_objects as go

def create_histogram(df, column_name):
    column_data = df[column_name]

    # Creating a histogram using Plotly
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=column_data,
        nbinsx=10,
        marker_color='deepskyblue',
        opacity=0.7,
        marker_line_color='black',  # Adding black borders to columns
        marker_line_width=1,        # Setting the border width
    ))

    # Defining labels for x and y
    fig.update_layout(
        xaxis_title=f'Variable Values {column_name}',
        yaxis_title='Frequency',
        title=f'Variable Histogram {column_name}',
    )

    #fig.show()

    return fig

def create_box_plot(df):
    # Creating the quantiles of tag7
    df1 = df[df['tag7_resp'] < df['tag7_resp'].quantile(0.25)]
    df2 = df[(df['tag7_resp'] >= df['tag7_resp'].quantile(0.25)) & (df['tag7_resp'] < df['tag7_resp'].quantile(0.75))]
    df3 = df[df['tag7_resp'] >= df['tag7_resp'].quantile(0.75)]

    # Creating the quantiles of tag1
    df4 = df[df['tag1'] < df['tag1'].quantile(0.25)]
    df5 = df[(df['tag1'] >= df['tag1'].quantile(0.25)) & (df['tag1'] < df['tag1'].quantile(0.75))]
    df6 = df[df['tag1'] >= df['tag1'].quantile(0.75)]

    # Calculating the means for each group
    mean_tag7_resp_df4 = df4['tag7_resp'].mean()
    mean_tag7_resp_df5 = df5['tag7_resp'].mean()
    mean_tag7_resp_df6 = df6['tag7_resp'].mean()

    # Adding a column to identify the origin of each dataframe
    df4['tag1'] = 'df4'
    df5['tag1'] = 'df5'
    df6['tag1'] = 'df6'

    # Concatenate all the dataframes
    df_concatenated = pd.concat([df4, df5, df6])

    # Create the box plot using Plotly
    fig = go.Figure()

    # Updated colors: df4 - pink, df5 - light blue, df6 - fuchsia
    for group, color in zip(['df4', 'df5', 'df6'], ['hotpink', 'deepskyblue', 'fuchsia']):
        fig.add_trace(go.Box(
            x=df_concatenated[df_concatenated['tag1'] == group]['tag1'],
            y=df_concatenated[df_concatenated['tag1'] == group]['tag7_resp'],
            name=group,
            marker_color=color
        ))

    # Customize the layout of the box plot
    fig.update_layout(
        xaxis_title='tag1',
        yaxis_title='tag7_resp',
        title='Box plot to compare tag7_resp between tag1 groups',
    )

    #fig.show()

    return fig

def create_violin_plot(df):
    # Creating the quantiles of tag7
    df1 = df[df['tag7_resp'] < df['tag7_resp'].quantile(0.25)]
    df2 = df[(df['tag7_resp'] >= df['tag7_resp'].quantile(0.25)) & (df['tag7_resp'] < df['tag7_resp'].quantile(0.75))]
    df3 = df[df['tag7_resp'] >= df['tag7_resp'].quantile(0.75)]

    # Creating the quantiles of tag1
    df4 = df[df['tag1'] < df['tag1'].quantile(0.25)]
    df5 = df[(df['tag1'] >= df['tag1'].quantile(0.25)) & (df['tag1'] < df['tag1'].quantile(0.75))]
    df6 = df[df['tag1'] >= df['tag1'].quantile(0.75)]

    # Calculating the means for each group
    mean_tag7_resp_df4 = df4['tag7_resp'].mean()
    mean_tag7_resp_df5 = df5['tag7_resp'].mean()
    mean_tag7_resp_df6 = df6['tag7_resp'].mean()

    # Adding a column to identify the origin of each dataframe
    df4['tag1_group'] = 'df4'
    df5['tag1_group'] = 'df5'
    df6['tag1_group'] = 'df6'

    # Concatenate all the dataframes
    df_concatenated = pd.concat([df4, df5, df6])

    # Define a custom palette with the desired colors
    custom_palette = {
        'df4': 'hotpink',
        'df5': 'deepskyblue',
        'df6': 'fuchsia'
    }

    # Create the violin plot using Plotly
    fig = go.Figure()

    for group in ['df4', 'df5', 'df6']:
        fig.add_trace(go.Violin(
            x=df_concatenated[df_concatenated['tag1_group'] == group]['tag1_group'],
            y=df_concatenated[df_concatenated['tag1_group'] == group]['tag7_resp'],
            name=group,
            box_visible=True,
            line_color=custom_palette[group],
            fillcolor=custom_palette[group],
            opacity=0.6,
            meanline_visible=True,
            line_width=2,
            points='all'
        ))

    # Customize the layout of the violin plot
    fig.update_layout(
        xaxis_title='tag1',
        yaxis_title='tag7_resp',
        title='Violin Plot to compare tag7_resp between tag1 groups',
    )

    #fig.show()

    return fig

def create_line_chart(df, column_name):
    # Create the line chart using Plotly
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[column_name],
        mode='lines',
        line=dict(color='fuchsia', width=2),
        name=column_name,
    ))

    # Customize the layout of the line chart
    fig.update_layout(
        xaxis_title='Index',
        yaxis_title=f'Values from {column_name}',
        title=f'Line Chart of {column_name}',
        showlegend=True,
        legend=dict(x=0, y=1),
        margin=dict(l=50, r=50, t=50, b=50),
    )

    #fig.show()

    return fig

def elbow_method(df):
    from sklearn.cluster import KMeans

    # Select only the 'tag1' and 'tag7_resp' columns for clustering
    selected_cols = ['tag1', 'tag7_resp']
    data_for_clustering = df[selected_cols]

    # Initialize an empty list to store the SSE values for each value of K
    sse = []

    # Calculate the SSE for different values of K (e.g., 1 to 10)
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data_for_clustering)
        sse.append(kmeans.inertia_)

    # Create the Elbow Method graph using Plotly
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(range(1, 11)),
        y=sse,
        mode='lines+markers',
        line=dict(color='deepskyblue', width=2),
        marker=dict(color='deepskyblue', size=8),
    ))

    # Customize the layout of the Elbow Method graph
    fig.update_layout(
        xaxis_title='Number of Clusters (K)',
        yaxis_title='SSE (Sum of Squared Errors)',
        title='Elbow Method for K-means (tag1 and tag7_resp)',
        xaxis=dict(tickvals=list(range(1, 11))),
        margin=dict(l=50, r=50, t=50, b=50),
    )

    #fig.show()

    return fig

def create_cluster(df):
    from sklearn.cluster import KMeans

    # Select only the 'tag1' and 'tag7_resp' columns for clustering
    selected_cols = ['tag1', 'tag7_resp']
    data_for_clustering = df[selected_cols]

    # Defining the number of clusters (K)
    K = 3

    # Creating the KMeans object and fitting the data
    kmeans = KMeans(n_clusters=K)
    kmeans.fit(data_for_clustering)

    # Adding the cluster labels to the original DataFrame
    df['cluster_labels'] = kmeans.labels_

    # Creating a dictionary to store each cluster DataFrame
    clusters_dict = {}
    for cluster_num in range(K):
        cluster_df = df[df['cluster_labels'] == cluster_num].copy()
        clusters_dict[cluster_num] = cluster_df

    # Checking the number of instances in each cluster
    for cluster_num, cluster_df in clusters_dict.items():
        print(f'Cluster {cluster_num}: {cluster_df.shape[0]} instances')

    # Saving each cluster DataFrame to a CSV file
    for cluster_num, cluster_df in clusters_dict.items():
        cluster_df.to_csv(f'cluster_{cluster_num}.csv', index=False)

    # Custom color map for scatter plot
    cluster_colors = {0: 'deeppink', 1: 'dodgerblue', 2: 'fuchsia'}
    cluster_labels_color = [cluster_colors[label] for label in df['cluster_labels']]

    # Create the scatter plot with Plotly
    fig = go.Figure()

    for cluster_num in range(K):
        cluster_df = df[df['cluster_labels'] == cluster_num]
        fig.add_trace(go.Scatter(
            x=cluster_df['tag1'],
            y=cluster_df['tag7_resp'],
            mode='markers',
            marker=dict(color=cluster_colors[cluster_num], size=8, line=dict(width=0)),  # Remove the border line
            name=f'Cluster {cluster_num}'
        ))

    # Plot the cluster centers
    fig.add_trace(go.Scatter(
        x=kmeans.cluster_centers_[:, 0],
        y=kmeans.cluster_centers_[:, 1],
        mode='markers',
        marker=dict(color='red', symbol='x', size=10, line=dict(width=1, color='Black')),
        name='Cluster Centers'
    ))

    # Customize the layout of the scatter plot
    fig.update_layout(
        xaxis_title='tag1',
        yaxis_title='tag7_resp',
        title='K-means Clustering',
        showlegend=True,
    )

    #fig.show()

    return fig

def plot_clusters(cluster_dataframes, cluster_colors):
    fig = go.Figure()

    for cluster_num, cluster_df in cluster_dataframes.items():
        color = cluster_colors[cluster_num]
        fig.add_trace(go.Scatter(
            x=cluster_df['tag7_resp'],
            y=cluster_df['tag1'],
            mode='markers',
            marker=dict(color=color, size=8, opacity=0.7),
            name=f'Cluster {cluster_num}',
        ))

    # Customize the layout of the cluster plot
    fig.update_layout(
        xaxis_title='tag7_resp',
        yaxis_title='tag1',
        title='Relationship between Cluster and tag7_resp',
        showlegend=True,
    )

    #fig.show()

    return fig

def perform_anova_test(df, selected_cols, K):
    """
    Perform the ANOVA test on a DataFrame after clustering using KMeans.

    Parameters:
    - df: DataFrame containing the data for clustering and ANOVA test.
    - selected_cols: List of column names to use for clustering.
    - K: Number of clusters to create using KMeans.

    Returns:
    - F-statistic and p-value obtained from the ANOVA test.
    - A string indicating the result of the test: significant or not significant.
    """

    # Selecting only the specified columns for clustering
    data_for_clustering = df[selected_cols]

    # Creating the KMeans object and fitting the data
    kmeans = KMeans(n_clusters=K)
    kmeans.fit(data_for_clustering)

    # Adding the cluster labels to the original DataFrame
    df['cluster_labels'] = kmeans.labels_

    # Performing the ANOVA test to check the difference between clusters and the 'tag7_resp' variable
    clusters = [df[df['cluster_labels'] == cluster_num]['tag7_resp'] for cluster_num in range(K)]

    f_statistic, p_value = f_oneway(*clusters)

    # Checking the results of the ANOVA test
    if p_value < 0.05:
        result = "There is a significant difference between the clusters and the 'tag7_resp' variable."
    else:
        result = "There is no significant difference between the clusters and the 'tag7_resp' variable."

    return f_statistic, p_value, result

def train_and_evaluate_mlp_regression():
    # Load the DataFrames of each cluster from CSV files
    cluster_0_df = pd.read_csv('cluster_0.csv')
    cluster_1_df = pd.read_csv('cluster_1.csv')
    cluster_2_df = pd.read_csv('cluster_2.csv')

    # Concatenate the DataFrames of each cluster into a single DataFrame
    # with columns 'tag1', 'tag7_resp', and 'cluster_labels'
    merged_df = pd.concat([cluster_0_df, cluster_1_df, cluster_2_df], ignore_index=True)

    # Separate the features (clusters) and the target (tag7_resp)
    X = merged_df.drop(['tag7_resp'], axis=1)
    y = merged_df['tag7_resp']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the MLPRegressor model
    mlp_regressor = MLPRegressor(hidden_layer_sizes=(256, 128, 64, 32), random_state=42)
    mlp_regressor.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = mlp_regressor.predict(X_test)

    # Evaluate the model's performance
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error (MSE): {mse}')
    
    # Return y_pred and y_test
    return y_pred, y_test, mlp_regressor, X, y, X_test

def plot_predictions(y_test, y_pred):
    fig = go.Figure()

    # Scatter plot of model predictions against the actual values
    fig.add_trace(go.Scatter(
        x=y_test,
        y=y_pred,
        mode='markers',
        marker=dict(color='lightpink', opacity=0.7),
        name='Predictions vs. Actual Values',
    ))

    # Plotting a reference line
    fig.add_trace(go.Scatter(
        x=[min(y_test), max(y_test)],
        y=[min(y_test), max(y_test)],
        mode='lines',
        line=dict(color='fuchsia', dash='dash'),
        name='Reference Line',
    ))

    # Set labels for the x and y axes
    fig.update_layout(
        xaxis_title='Actual Values',
        yaxis_title='Model Predictions',
        title='MLP Regression Model - Predictions vs. Actual Values',
    )

    # Add a legend
    fig.update_layout(showlegend=True)

    # Show the grid
    fig.update_layout(xaxis=dict(showgrid=True, zeroline=False),
                      yaxis=dict(showgrid=True, zeroline=False))

    return fig

def plot_residuos(y_test, y_pred):
    fig = go.Figure()

    # Calculate residuals (differences between real values and model predictions)
    residuals = y_test - y_pred

    # Create a scatter plot of model predictions vs. residuals
    fig.add_trace(go.Scatter(
        x=y_pred,
        y=residuals,
        mode='markers',
        marker=dict(color='lightpink', opacity=0.7),
        name='Model Predictions vs. Residuals',
    ))

    # Add a reference line at y=0 to show where the residuals are centered
    fig.add_shape(type='line',
                  x0=min(y_pred), x1=max(y_pred),
                  y0=0, y1=0,
                  line=dict(color='fuchsia', dash='dash'),
                  name='Reference Line')

    # Set labels for the x and y axes
    fig.update_layout(
        xaxis_title='Model Predictions',
        yaxis_title='Residuals (Real Values - Predictions)',
        title='Residuals Plot',
    )

    # Add a legend to the plot
    fig.update_layout(showlegend=True)

    # Add a grid to the plot
    fig.update_layout(xaxis=dict(showgrid=True, zeroline=False),
                      yaxis=dict(showgrid=True, zeroline=False))

    return fig


def plot_learning_curve(estimator, X, y, cv=5, scoring='neg_mean_squared_error'):
    """
    Plot the learning curve to check how the model's performance varies with the size of the training set.
    This helps identify whether the model is suffering from underfitting or overfitting.

    Args:
        estimator: The machine learning model.
        X (array-like): The feature matrix.
        y (array-like): The target vector.
        cv (int, cross-validation generator, or an iterable): Determines the cross-validation splitting strategy.
        scoring (str or callable): The scoring method used for evaluation.

    Returns:
        None
    """

    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, scoring=scoring)
    train_scores_mean = -np.mean(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=train_sizes,
        y=train_scores_mean,
        mode='markers+lines',
        marker=dict(color='fuchsia'),
        name='Training',
    ))

    fig.add_trace(go.Scatter(
        x=train_sizes,
        y=test_scores_mean,
        mode='markers+lines',
        marker=dict(color='deeppink'),
        name='Validation',
    ))

    # Set labels for the x and y axes
    fig.update_layout(
        xaxis_title='Training Set Size',
        yaxis_title='Mean MSE',
        title='Learning Curve',
    )

    # Add a legend to the plot
    fig.update_layout(showlegend=True)

    # Show the grid
    fig.update_layout(xaxis=dict(showgrid=True, zeroline=False),
                      yaxis=dict(showgrid=True, zeroline=False))


    return fig

def plot_temporal_series(X_test, y_test, y_pred):
    """
    Create a temporal DataFrame with the predicted values and tag7_resp, then plot the temporal series
    with the colors 'deeppink' and 'lightpink' with transparency.

    Args:
        X_test (pd.DataFrame): DataFrame containing the feature matrix (with the 'data' column as index).
        y_test (pd.Series): Series containing the real values (tag7_resp).
        y_pred (pd.Series): Series containing the predicted values.

    Returns:
        None
    """
    # Create a DataFrame with the predicted values and tag7_resp
    df_temporal = pd.DataFrame({'data': X_test.index, 'tag7_resp': y_test, 'y_pred': y_pred})

    # Sort the DataFrame by date
    df_temporal.sort_values(by='data', inplace=True)

    fig = go.Figure()

    # Plot the temporal series with the colors 'deeppink' and 'lightpink' with transparency
    fig.add_trace(go.Scatter(
        x=df_temporal['data'],
        y=df_temporal['tag7_resp'],
        mode='lines',
        marker=dict(color='lightpink'),
        name='tag7_resp',
    ))

    fig.add_trace(go.Scatter(
        x=df_temporal['data'],
        y=df_temporal['y_pred'],
        mode='lines',
        marker=dict(color='deeppink', opacity=0.1),
        name='Predictions',
    ))

    # Set labels for the x and y axes
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Values',
        title='Temporal Series - Predicted Values vs. tag7_resp',
    )

    # Add a legend to the plot
    fig.update_layout(showlegend=True)

    # Show the grid
    fig.update_layout(xaxis=dict(showgrid=True, zeroline=False),
                      yaxis=dict(showgrid=True, zeroline=False))


    return fig