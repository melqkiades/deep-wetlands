import math
import os
import time

import imageio
import matplotlib
import numpy
import pandas
import seaborn
from dotenv import load_dotenv
from matplotlib import pyplot as plt


def plot_scatter(results_2018_file, results_2020_file, study_area):

    data_dir = os.getenv('DATA_DIR') + '/'
    model_name = os.getenv('MODEL_NAME')
    # study_area = os.getenv('STUDY_AREA')

    # model_2018_results = '/tmp/descending_Orebro lan_mosaic_2018-07-04_sar_VH_20-epochs_new_water_estimates_filtered.csv'
    # model_2020_results = '/tmp/descending_Orebro lan_mosaic_2020-06-23_sar_VH_20-epochs_new_water_estimates_filtered.csv'
    # model_2018_results = '/tmp/descending_Orebro lan_mosaic_2018-07-04_sar_VH_20-epochs_0.00005-lr_42-rand_new_water_estimates_filtered.csv'
    # model_2020_results = '/tmp/descending_Orebro lan_mosaic_2020-06-23_sar_VH_20-epochs_0.00005-lr_42-rand_new_water_estimates_filtered.csv'
    # model_2018_results = f'/tmp/descending_Orebro lan_mosaic_2018-07-04_sar_VH_20-epochs_0.00005-lr_42-rand_{study_area}_new_water_estimates_filtered.csv'
    # model_2020_results = f'/tmp/descending_Orebro lan_mosaic_2020-06-23_sar_VH_20-epochs_0.00005-lr_42-rand_{study_area}_new_water_estimates_filtered.csv'

    df_2018 = pandas.read_csv(results_2018_file, header=0, names=['Index', 'Extension', 'Date'])
    df_2018.drop('Index', axis=1, inplace=True)
    df_2018.sort_values(by='Date', inplace=True)
    df_2018['Date'] = pandas.to_datetime(df_2018['Date'])
    df_2018 = df_2018.drop(df_2018[df_2018['Date'] < '2018-01-01'].index)
    df_2018 = df_2018.drop(df_2018[df_2018['Date'] > '2019-12-31'].index)
    print(df_2018.head())
    print(df_2018.tail())

    df_2020 = pandas.read_csv(results_2020_file, header=0, names=['Index', 'Extension', 'Date'])
    df_2020.drop('Index', axis=1, inplace=True)
    df_2020.sort_values(by='Date', inplace=True)
    df_2020['Date'] = pandas.to_datetime(df_2020['Date'])
    df_2020 = df_2020.drop(df_2020[df_2020['Date'] < '2020-01-01'].index)
    print(df_2020.head())
    print(df_2020.tail())

    df_2020 = df_2020.reset_index(drop=True)
    # full_df = df_2018.append(df_2020)
    full_df = pandas.concat([df_2018, df_2020])
    # full_df = pandas.concat([df_2020])

    print('2018 DF', len(df_2018))
    print('2020 DF', len(df_2020))
    print('Full DF', len(full_df))
    # full_df = full_df.drop(full_df[full_df['Date'] < '2016-01-01'].index)
    print(full_df.head())
    print(full_df.dtypes)
    ax = full_df.plot(x='Date', y='Extension', kind='scatter', title=f'{study_area} water extension in time')
    ax.set_ylabel("Water extension in m2")
    plt.tight_layout()
    # plt.scatter(['2020-08-04', '2018-04-24', '2021-11-21'], [62330, 134579, 86235], c='#ff0000')

    # Remove all special characters from study area name
    study_area_ascii = study_area.lower().replace('å', 'a').replace('ä', 'a').replace('ö', 'o')

    full_df.to_csv(f'/tmp/paper_water_estimates_{study_area_ascii}.csv')
    # plt.scatter(['2017-04-16', '2020-08-04', '2018-04-24', '2021-11-21'], [115044, 62330, 134579, 86235], c='#ff0000', s=35)
    # plt.scatter(['2017-04-16', '2020-08-04', '2018-04-24', '2021-11-21'], [115044, 62330, 134579, 86235], c='#ff0000', s=80)
    plt.savefig(f'/tmp/{study_area_ascii}_water_estimates.pdf')
    # seaborn.regplot(data=full_df, x='Date', y='Extension')

    plt.show()


# Plot a scatter plot from three CSV of the water estimates. Each CSV contains a different study area.
# The dotts of the scatter plot are colored according to the study area.
def plot_scatter_three_areas():
    study_areas = [
        'Svårtadalen',
        'Hjälstaviken',
        'Hornborgasjön',
        # 'Mossaträsk',
    ]

    df_all = pandas.DataFrame(columns=['Extension', 'Date', 'Study_area'])

    for study_area in study_areas:
        study_area_ascii = study_area.lower().replace('å', 'a').replace('ä', 'a').replace('ö', 'o')
        file_name = f'/tmp/paper_water_estimates_{study_area_ascii}.csv'
        df = pandas.read_csv(file_name, usecols=['Extension', 'Date'])
        df['Study_area'] = study_area
        df_all = pandas.concat([df_all, df])

    # Change the data type of the Date column to datetime
    df_all['Date'] = pandas.to_datetime(df_all['Date'])

    # Multiply the water extension by 55 to get the extension in m2
    df_all['Extension'] = df_all['Extension'] * 55 / 1000000

    print(df_all.head())
    print(df_all.dtypes)

    # ax = seaborn.scatterplot(data=df_all, x='Date', y='Extension', hue='Study_area', palette='deep')
    ax = seaborn.scatterplot(data=df_all, x='Date', y='Extension', hue='Study_area', palette='deep', s=20)
    # ax = seaborn.scatterplot(data=df_all, x='Date', y='Extension', hue='Study_area', palette='deep', s=10)

    # Set plot title
    ax.set_title("Water extension in time")
    ax.set_ylabel("Water extension in km$^2$")

    # Format Y axis to separate thousands with a space
    ax.get_yaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

    # Set the y axis to log scale
    # ax.set_yscale('log')
    # plt.yscale('log')

    # Set legend title
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, title='Study area')
    # Move the legend to the bottom
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=3)

    plt.tight_layout()
    plt.savefig(f'/tmp/water_estimates_per_area.pdf')
    plt.show()





def count_ndwi_water():
    study_area = os.getenv('STUDY_AREA')
    folder = f'/tmp/bulk_export_{study_area}_ndwi/'
    # file_name = '20150813T101020_20160319T132810_T33VWG.png'

    results_list = []

    if not os.path.exists(folder):
        raise FileNotFoundError(f'The folder contaning the TIFF files does not exist: {folder}')

    filenames = next(os.walk(folder), (None, None, []))[2]  # [] if no file
    for file_name in filenames:
        if file_name.endswith('.png'):
            continue
        ndwi_mask = imageio.imread(folder + file_name)
        unique, counts = numpy.unique(ndwi_mask, return_counts=True)
        results = dict(zip(unique, counts))
        # print('Date', image_date, 'Counts = ', results)

        has_nan = False
        for key in results.keys():
            if math.isnan(key):
                print('Found a NaN')
                has_nan = True

        if has_nan:
            continue

        image_date = file_name[0:8]
        results['Date'] = image_date
        results['Date'] = image_date
        results['File_name'] = file_name
        results_list.append(results)

    data_frame = pandas.DataFrame(results_list)
    data_frame['Date'] = data_frame['Date'].apply(pandas.to_datetime)
    data_frame.rename(columns={0.5: 'No_water', 1.0: 'Water'}, inplace=True)
    print(data_frame.head())
    print(data_frame.columns.values)

    data_frame = data_frame[data_frame['Date'].dt.month.isin([4, 5, 6, 7, 8, 9, 10, 11])]
    data_frame = data_frame.drop(data_frame[data_frame['Date'] < '2018-01-01'].index)
    data_frame = data_frame.drop(data_frame[data_frame['Date'] > '2022-12-31'].index)

    if study_area == 'ojesjon':
        bad_images = ['2018-04-02', '2018-11-25', '2018-11-26', '2020-11-20', '2020-11-27', '2021-11-29', '2022-10-08', '2022-10-30']
        data_frame = data_frame.drop(data_frame[data_frame['Date'].isin(bad_images)].index)

    data_frame.to_csv(f'/tmp/paper_ndwi_water_estimates_{study_area}.csv')

    ax = data_frame.plot(x='Date', y='Water', kind='scatter', title=f'{study_area} water extension in time22', c='red')


    #
    #
    #
    #
    #
    #
    # model_2018_results = '/tmp/descending_Orebro lan_mosaic_2018-07-04_sar_VH_20-epochs_new_water_estimates_filtered.csv'
    # model_2020_results = '/tmp/descending_Orebro lan_mosaic_2020-06-23_sar_VH_20-epochs_new_water_estimates_filtered.csv'
    # model_2018_results = '/tmp/descending_Orebro lan_mosaic_2018-07-04_sar_VH_20-epochs_0.00005-lr_42-rand_new_water_estimates_filtered.csv'
    # model_2020_results = '/tmp/descending_Orebro lan_mosaic_2020-06-23_sar_VH_20-epochs_0.00005-lr_42-rand_new_water_estimates_filtered.csv'
    model_2018_results = f'/tmp/descending_Orebro lan_mosaic_2018-07-04_sar_VH_20-epochs_0.00005-lr_42-rand_{study_area}_new_water_estimates_filtered.csv'
    model_2020_results = f'/tmp/descending_Orebro lan_mosaic_2020-06-23_sar_VH_20-epochs_0.00005-lr_42-rand_{study_area}_new_water_estimates_filtered.csv'

    df_2018 = pandas.read_csv(model_2018_results, header=0, names=['Index', 'Extension', 'Date'])
    df_2018.drop('Index', axis=1, inplace=True)
    df_2018.sort_values(by='Date', inplace=True)
    df_2018['Date'] = pandas.to_datetime(df_2018['Date'])
    df_2018 = df_2018.drop(df_2018[df_2018['Date'] < '2018-01-01'].index)
    df_2018 = df_2018.drop(df_2018[df_2018['Date'] > '2019-12-31'].index)
    print(df_2018.head())
    print(df_2018.tail())

    df_2020 = pandas.read_csv(model_2020_results, header=0, names=['Index', 'Extension', 'Date'])
    df_2020.drop('Index', axis=1, inplace=True)
    df_2020.sort_values(by='Date', inplace=True)
    df_2020['Date'] = pandas.to_datetime(df_2020['Date'])
    df_2020 = df_2020.drop(df_2020[df_2020['Date'] < '2020-01-01'].index)
    print(df_2020.head())
    print(df_2020.tail())

    df_2020 = df_2020.reset_index(drop=True)
    # full_df = df_2018.append(df_2020)
    full_df = pandas.concat([df_2018, df_2020])

    print('2018 DF', len(df_2018))
    print('2020 DF', len(df_2020))
    print('Full DF', len(full_df))
    # full_df = full_df.drop(full_df[full_df['Date'] < '2016-01-01'].index)
    print(full_df.head())
    print(full_df.dtypes)
    full_df.plot(x='Date', y='Extension', kind='scatter', title=f'{study_area} water extension in time', ax=ax)

    # Add lines to the plot to connect the scatter dots
    # data_frame.plot(x='Date', y='Water', c='red', ax=ax)
    # full_df.plot(x='Date', y='Extension', ax=ax)

    # plt.scatter(['2020-08-04', '2018-04-24', '2021-11-21'], [62330, 134579, 86235], c='#1e8449')

    # plt.savefig('/tmp/open_vegetated_water.pdf')
    plt.show()


def main():
    load_dotenv()

    study_areas = [
        'Svårtadalen',
        'Hjälstaviken',
        'Hornborgasjön',
        'Mossaträsk',
    ]
    for study_area in study_areas:
        print('before', study_area)
        # Convert special characters to ASCII equivalents
        study_area_ascii = study_area.lower().replace('å', 'a').replace('ä', 'a').replace('ö', 'o')
        print('after', study_area_ascii)
        results_2018_file = f'/tmp/scatter_plot/prime-2018_{study_area_ascii}_new_water_estimates_filtered.csv'
        results_2020_file = f'/tmp/scatter_plot/solar-jazz-436_{study_area_ascii}_new_water_estimates_filtered.csv'
        plot_scatter(results_2018_file, results_2020_file, study_area)
    # count_ndwi_water()

    # plot_scatter_three_areas()


start = time.time()
main()
end = time.time()
total_time = end - start
print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))
