import matplotlib.pyplot as plt
import pandas as pd

PLOT_WIDTH = 8
PLOT_HIGHT = 3


def plot_hist_and_boxplot(df: pd.DataFrame, column: str, y_label: str) -> None:
    _, ax1 = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HIGHT))

    ax1.hist(df[column], bins=min(100, df[column].nunique()))
    ax1.set_ylabel("Occurrences (frequency)")
    ax1.set_xlabel(y_label)

    ax2 = ax1.twinx()

    ax2.boxplot(df[column], vert=False)
    ax2.set_title(f"Histogram and Box Plot of {y_label}")

    plt.tight_layout()
    plt.show()


def plot_timeseries(df: pd.DataFrame, column: str, y_label: str, **kwargs) -> None:
    date_time_column = kwargs.get("date_time_column", "date_time")
    df_plot = df.copy(deep=True)
    df_plot.set_index(date_time_column, inplace=True)

    # Plot the time series
    plt.figure(figsize=(PLOT_WIDTH, PLOT_HIGHT))
    plt.plot(df_plot.index, df_plot[column])
    plt.xlabel("Date")
    plt.ylabel(y_label)
    plt.title(f"{y_label} Time Series")

    plt.show()


def plot_gantt_of_categories(
    df: pd.DataFrame, category_column: str, y_label: str, **kwargs
) -> None:
    date_time_column = kwargs.get("date_time_column", "date_time")
    _, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HIGHT))

    for _, row in df.iterrows():
        start = row[date_time_column]
        end = start + pd.DateOffset(hours=1)
        weather = row[category_column]
        ax.barh(weather, end - start, left=start, height=0.5)

    ax.set_ylabel(y_label)
    ax.set_xlabel("Date Time")
    ax.set_title(f"Time Series of {y_label} Categories (Gantt Chart)")
    ax.invert_yaxis()
    plt.show()
