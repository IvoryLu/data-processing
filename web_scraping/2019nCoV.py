import time
import json
import requests
from datetime import datetime
import numpy as np
import matplotlib
import matplotlib.figure
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Set font
plt.rcParams['font.sans-serif'] = ['Helvetica']
# Solve the problem when save figures
plt.rcParams['axes.unicode_minus'] = False

# Catch Daily Counts of patients


def catch_daily():

    url = 'https://view.inews.qq.com/g2/getOnsInfo?name=wuwei_ww_cn_day_counts&callback=&_=%d' % int(
        time.time() * 1000)
    data = json.loads(requests.get(url=url).json()['data'])
    data.sort(key=lambda x: x['date'])

    # Initialize list
    # Date
    date_list = list()
    # Counts of confirmation
    confirm_list = list()
    # Counts of suspects
    suspect_list = list()
    # Counts of death
    dead_list = list()
    # Counts of recovered patients
    heal_list = list()

    # Save data in the list
    for item in data:
        month, day = item['date'].split('.')
        date_list.append(datetime.strptime('2020-%s-%s' %
                                           (month, day), '%Y-%m-%d'))
        confirm_list.append(int(item['confirm']))
        suspect_list.append(int(item['suspect']))
        dead_list.append(int(item['dead']))
        heal_list.append(int(item['heal']))

    return date_list, confirm_list, suspect_list, dead_list, heal_list

# Catch distribution of patiens in different area


def catch_distribution():

    #    data = {'西藏': 0}
    data = {}
    url = 'https://view.inews.qq.com/g2/getOnsInfo?name=wuwei_ww_area_counts&callback=&_=%d' % int(
        time.time() * 1000)
    for item in json.loads(requests.get(url=url).json()['data']):
        if item['area'] not in data:
            data.update({item['area']: 0})
        data[item['area']] += int(item['confirm'])

    return data

# Plot Daily Counts Line Chart


def plot_daily():

    # Get data
    date_list, confirm_list, suspect_list, dead_list, heal_list = catch_daily()

    # print(confirm_list)

    plt.figure('2019-nCoV Outbreaks Line Chart',
               facecolor='#f4f4f4', figsize=(10, 8))
    plt.title('2019-nCoV Outbreaks Curve', fontsize=20)

    plt.plot(date_list, confirm_list, label='Confirmed')
    plt.plot(date_list, suspect_list, label='Suspected')
    plt.plot(date_list, dead_list, label='Death')
    plt.plot(date_list, heal_list, label='Recovered')
    #
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))  # 格式化时间轴标注
    plt.gcf().autofmt_xdate()  # 优化标注（自动倾斜）
    # Enable Grid
    plt.grid(linestyle=':')
    # Data legend
    plt.legend(loc='best')
    plt.savefig('2019-nCoV Outbreaks Curve.png')  # 保存为文件
    plt.show()

# Plot Counts of patients on the Map


def plot_distribution():

    data = catch_distribution()

    font = FontProperties(fname='res/Roboto-Black.ttf', size=14)
    lat_min = 0
    lat_max = 60
    lon_min = 70
    lon_max = 140

    handles = [
        matplotlib.patches.Patch(color='#ffaa85', alpha=1, linewidth=0),
        matplotlib.patches.Patch(color='#ff7b69', alpha=1, linewidth=0),
        matplotlib.patches.Patch(color='#bf2121', alpha=1, linewidth=0),
        matplotlib.patches.Patch(color='#7f1818', alpha=1, linewidth=0),
    ]
    labels = ['1-9 people', '10-99 people', '100-999 people', '>1000 people']

    fig = matplotlib.figure.Figure()
    # Set size of the canvas
    fig.set_size_inches(10, 8)
    axes = fig.add_axes((0.1, 0.12, 0.8, 0.8))  # rect = l,b,w,h
    m = Basemap(projection='lcc', width=5000000, height=5000000,
                lat_0=36, lon_0=102, resolution='l', ax=axes)
    # m = Basemap(llcrnrlon=lon_min, urcrnrlon=lon_max,
    #            llcrnrlat=lat_min, urcrnrlat=lat_max, resolution='l', ax=axes)
    m.readshapefile('res/china-shapefiles/china',
                    'province', drawbounds=True)
    m.readshapefile(
        'res/china-shapefiles/china_nine_dotted_line', 'section', drawbounds=True)

    # Draw Continental Boundaries
    m.drawcoastlines(color='black')
    # Draw Border Line between countries
    m.drawcountries(color='black')
    # Draw Latitude
    m.drawparallels(np.arange(lat_min, lat_max, 10),
                    labels=[1, 0, 0, 0])
    # Draw Longitude
    m.drawmeridians(np.arange(lon_min, lon_max, 10),
                    labels=[0, 0, 0, 1])

    for info, shape in zip(m.province_info, m.province):
        pname = info['OWNER'].strip('\x00')
        fcname = info['FCNAME'].strip('\x00')
        # No Island
        if pname != fcname:
            continue

        for key in data.keys():
            if key in pname:
                if data[key] == 0:
                    color = '#f0f0f0'
                elif data[key] < 10:
                    color = '#ffaa85'
                elif data[key] < 100:
                    color = '#ff7b69'
                elif data[key] < 1000:
                    color = '#bf2121'
                else:
                    color = '#7f1818'
                break

        poly = Polygon(shape, facecolor=color, edgecolor=color)
        axes.add_patch(poly)

    axes.legend(handles, labels, bbox_to_anchor=(0.5, -0.11),
                loc='lower center', ncol=4, prop=font)
    axes.set_title("2019-nCoV Outbreaks Map", fontproperties=font)
    FigureCanvasAgg(fig)
    fig.savefig('2019-nCoV Outbreaks Map.png')


if __name__ == '__main__':
    plot_daily()
    plot_distribution()
