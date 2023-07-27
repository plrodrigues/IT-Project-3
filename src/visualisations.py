import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
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


def add_daylight_shadows_to_time_series(df: pd.DataFrame, **kwargs) -> None:
    date_time_column = kwargs.get("date_time_column", "date_time")
    # Define daylight start and end times
    sr_date = df[date_time_column].dt.date
    # The day_start and day_end arrays depend on the definition of daylight and night times.
    # If we simply define daylight as, for example, from 6:00 to 18:00, and night time as
    # from 18:00 to 6:00 the next day. However, this depends on the location of the
    # data acquisition and on the time of the year.
    # For simplicity, we are limiting to hard coded values. Libraries, such as sunlac
    # can be used as an automatic format for identifying the daylight and dawn times.
    day_start_time = pd.to_datetime("06:00:00").time()
    day_end_time = pd.to_datetime("18:00:00").time()
    day_starts = [
        pd.to_datetime(str(date) + " " + day_start_time.strftime("%H:%M:%S")) for date in sr_date
    ]
    day_ends = [
        pd.to_datetime(str(date) + " " + day_end_time.strftime("%H:%M:%S")) for date in sr_date
    ]

    # Add shadowing for daylight and night times
    for i in range(1, len(day_ends)):
        plt.axvspan(day_ends[i - 1], day_starts[i], facecolor="moccasin", alpha=0.1)


def plot_timeseries(df: pd.DataFrame, column: str, y_label: str, **kwargs) -> None:
    date_time_column = kwargs.get("date_time_column", "date_time")
    rotation = kwargs.get("rotation", 0)
    do_add_daily_shadows = kwargs.get("do_add_daily_shadows", False)
    df_plot = df.copy(deep=True)
    df_plot.set_index(date_time_column, inplace=True)

    # Plot the time series
    plt.figure(figsize=(PLOT_WIDTH, PLOT_HIGHT))
    plt.plot(df_plot.index, df_plot[column])

    if do_add_daily_shadows:
        add_daylight_shadows_to_time_series(df, **kwargs)

    plt.xlabel("Date")
    plt.ylabel(y_label)
    plt.xticks(rotation=rotation)
    plt.title(f"{y_label} Time Series")

    plt.show()


def plot_gantt_of_categories(
    df: pd.DataFrame, category_column: str, y_label: str, **kwargs
) -> None:
    # Constants for Gantt chart only
    PLOT_WIDTH = 10
    PLOT_HEIGHT = 3
    AXIS_FONTSIZE = 10
    LABELS_FONTSIZE = 8
    # key word arguments
    date_time_column = kwargs.get("date_time_column", "date_time")
    rotation = kwargs.get("rotation", 0)
    hight = kwargs.get("hight", PLOT_HEIGHT)
    date_interval_days = kwargs.get("date_interval_days", 2)
    # Remove duplicates
    df_plot = df.copy(deep=True)
    df_plot = df_plot[[date_time_column, category_column]]
    df_plot[f"{category_column}_segment"] = (
        df_plot[category_column] != df_plot[category_column].shift()
    ).cumsum()
    df_plot = df_plot.drop_duplicates(
        subset=[category_column, f"{category_column}_segment"], keep="first"
    )
    df_plot.set_index(date_time_column, inplace=True)

    # Create a color map for each unique category
    categories = df_plot[category_column].unique()
    colors = [plt.cm.tab20b(i) for i in range(len(categories))]
    color_map = dict(zip(categories, colors))

    _, ax = plt.subplots(figsize=(PLOT_WIDTH, hight))
    # Set the locator and formatter to show dates at a daily interval
    date_format = mdates.DateFormatter("%Y-%m-%d")
    ax.xaxis.set_major_locator(
        mdates.DayLocator(interval=date_interval_days)
    )  # Interval=1 means show every day
    ax.xaxis.set_major_formatter(date_format)

    for i in range(len(df_plot) - 1):
        start = df_plot.index[i]
        end = df_plot.index[i + 1]
        weather = df_plot.iloc[i, df_plot.columns.get_loc(category_column)]

        # Use the color map to set the color for each category
        color = color_map[weather]

        ax.barh(y=weather, width=(end - start), left=start, height=0.5, color=color)

    ax.set_ylabel(y_label, fontsize=AXIS_FONTSIZE)
    ax.set_xlabel("Date Time", fontsize=AXIS_FONTSIZE)
    ax.set_title(f"Time Series of {y_label} Categories (Gantt Chart)")
    ax.tick_params(axis="both", which="major", labelsize=LABELS_FONTSIZE)
    ax.invert_yaxis()

    # Set y-axis tick labels color
    for tick_label in ax.get_yticklabels():
        category = tick_label.get_text()
        color = color_map[category]
        tick_label.set_color(color)

    plt.tight_layout()
    plt.xticks(rotation=rotation)
    plt.show()
