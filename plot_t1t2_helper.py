
# 核磁谱绘制


# # 导入依赖库


import os

import warnings
warnings.filterwarnings('ignore')
# 加入上述两条屏蔽警告信息，调试的时候应该去掉这两行

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle, islice
import matplotlib as mpl


from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['font.sans-serif']=['Microsoft YaHei']

plt.rcParams['axes.unicode_minus'] = False
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable



# # 设置数据路径


# 项目路径
# project_dir = os.getcwd()
# data_dir = "data"
# fig_dir_path = "figure"


'''
plot_line = True
show_sw = True
'''
plot_line = True



# file_t1 = os.path.join(project_dir,data_dir,'T1_point1_01.txt')
# file_t2 = os.path.join(project_dir, data_dir,'T2_point1_01.txt')
# file_fvol =  os.path.join(project_dir,data_dir,'T1T2_point_01.txt')
# file_t1_spectrum = os.path.join(project_dir, data_dir,'T1_spectrum_01.txt')
# file_t2_spectrum = os.path.join(project_dir, data_dir,'T2_spectrum_01.txt')


# t1_domain = np.loadtxt(file_t1) # 1d minimum points
# t2_domain = np.loadtxt(file_t2) # 1d minimum points
# t1_spectrum = np.loadtxt(file_t1_spectrum) # 1d minimum points
# t2_spectrum = np.loadtxt(file_t2_spectrum) # 1d minimum points
# f_grid = np.loadtxt(file_fvol)
# print("after loadtxt")


# # 绘图参考
# https://zhajiman.github.io/post/matplotlib_colorbar/


# https://zhajiman.github.io/post/matplotlib_colorbar/
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms

def add_box(ax):
    '''用红框标出一个ax的范围.'''
    axpos = ax.get_position()
    rect = mpatches.Rectangle(
        (axpos.x0, axpos.y0), axpos.width, axpos.height,
        lw=3, ls='--', ec='r', fc='none', alpha=0.5,
        transform=ax.figure.transFigure
    )
    ax.patches.append(rect)

def add_right_cax(ax, pad, width):
    '''
    在一个ax右边追加与之等高的cax.
    pad是cax与ax的间距,width是cax的宽度.
    '''
    axpos = ax.get_position()
    caxpos = mtransforms.Bbox.from_extents(
        axpos.x1 + pad,
        axpos.y0,
        axpos.x1 + pad + width,
        axpos.y1
    )
    cax = ax.figure.add_axes(caxpos)

    return cax


# # 绘制T1、T2联合分布图
def plot_T1T2_joint(t1_domain, t2_domain, f_grid):
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    '''控制两列间隔'''
    fig.subplots_adjust(hspace=0.2, wspace=0.4)


    # 绘制第1个图
    t1_x = range(len(t1_domain))
    im1 = axes[0, 0].plot(t1_x,t1_domain,label='T1')
    axes[0, 0].set_title('t1_domain',fontsize = 20)
    axes[0, 0].set_ylabel('T1', fontdict={'family' : 'Times New Roman', 'size'   : 18})
    axes[0, 0].set_xlabel('n', fontdict={'family' : 'Times New Roman', 'size'   : 18})
    axes[0, 0].tick_params(labelcolor='black', labelsize='large', width=3)
    axes[0, 0].grid()
    axes[0, 0].legend(loc='upper center')

    '''绘制第3个图'''
    # 显示T1-T2图表
    axes[0, 1].contourf(t2_domain, t1_domain, f_grid,cmap=plt.get_cmap('jet'), extend='both',levels=200)    # 显示图表

    # 以10为底的对数，只显示正数
    log_axi_array = [0.01,0.1,1,10,100,1000,10000,100000]
    axes[0, 1].semilogx(base=10,subs=log_axi_array,nonpositive='mask')
    axes[0, 1].semilogy(base=10,subs=log_axi_array,nonpositive='mask')

    '''绘制三条斜线'''
    if plot_line:
        # plt.loglog([np.log10(0.3),np.log10(3000)],[np.log10(0.3),np.log10(3000)],'k--',color='y')
        # plt.loglog([np.log10(0.3),np.log10(3000)-1],[1+np.log10(0.3),np.log10(3000)],'k--',color='y')
        # plt.loglog([np.log10(0.3),np.log10(3000)-2],[2+np.log10(0.3),np.log10(3000)],'k--',color='y')
        # 不取对数的值
        axes[0, 1].loglog([0.03,30000],[0.03,30000],'k--',color='y')
        axes[0, 1].loglog([0.03,30000/10],[0.3,30000],'k--',color='y')
        axes[0, 1].loglog([0.03,30000/100],[3,30000],'k--',color='y')

    '''设置横纵坐标轴范围'''
    axes[0, 1].set_xlim(0.1,10000)
    axes[0, 1].set_ylim(0.1,10000)
    axes[0, 1].set_ylabel('T1', fontdict={'family' : 'Times New Roman', 'size'   : 18})
    axes[0, 1].set_xlabel('T2', fontdict={'family' : 'Times New Roman', 'size'   : 18})
    axes[0, 1].tick_params(labelcolor='black', labelsize='large', width=3)
    axes[0, 1].set_title('T1 vs T2',fontsize = 20)

    cax = add_right_cax(axes[0, 1], pad=0.02, width=0.02)
    # cbar = axes[0, 1].colorbar(axes[0, 1], cax=cax)
    # # ax= plt.axes()
    cmp = plt.get_cmap('jet')
    #
    # # https://zhajiman.github.io/post/matplotlib_colorbar/
    #
    norm = mcolors.LogNorm(vmin=1E0, vmax=1E3)
    # # norm = mpl.colors.BoundaryNorm([0,1,2,3], cmp.N)
    # # create an Axes on the right side of ax. The width of cax will be 5%
    # # of ax and the padding between cax and ax will be fixed at 0.05 inch.

    # ax.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmp), ax=cax,location='right')
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmp), cax=cax)

    t2_x = range(len(t2_domain))
    '''绘制第3个图T1 vs T2'''
    '''https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html'''
    '''https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_masked.html#sphx-glr-gallery-lines-bars-and-markers-scatter-masked-py'''
    # 散点图
    # T2截止值
    T2_threshold = 33
    area1 = np.ma.masked_where(t2_domain < T2_threshold, t1_domain)
    area2 = np.ma.masked_where(t2_domain >= T2_threshold, t1_domain)
    # T1值控制面积，T2值控制颜色
    im31 = axes[1, 0].scatter(t2_domain,t2_domain, s=area1, marker='^', c=t2_domain)
    im32 = axes[1, 0].scatter(t2_domain,t2_domain, s=area2, marker='o', c=t2_domain)
    # im3 = axes[1, 0].scatter(t1_domain,t2_domain,marker=None, cmap=cmp,label='T1 vs T2')

    # im3 = axes[1, 0].hist2d(t1_domain,t2_domain,bins=100, cmap=cmp,label='T1 vs T2')
    axes[1, 0].semilogx(base=10,subs=log_axi_array,nonpositive='mask')
    axes[1, 0].semilogy(base=10,subs=log_axi_array,nonpositive='mask')
    axes[1, 0].set_title('T1值控制面积，T2值控制颜色',fontsize = 20)
    axes[1, 0].set_ylabel('T2', fontdict={'family' : 'Times New Roman', 'size'   : 18})
    axes[1, 0].set_xlabel('T2', fontdict={'family' : 'Times New Roman', 'size'   : 18})
    axes[1, 0].tick_params(labelcolor='black', labelsize='large', width=3)
    axes[1, 0].grid()
    axes[1, 0].legend(loc='upper center')
    cax2 = add_right_cax(axes[1, 0], pad=0.02, width=0.02)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmp), cax=cax2)

    '''绘制第4个图'''

    im4 = axes[1, 1].plot(t2_x,t2_domain,label='T2')
    axes[1, 1].set_title('t2_domain',fontsize = 20)
    axes[1, 1].set_ylabel('T1', fontdict={'family' : 'Times New Roman', 'size'   : 18})
    axes[1, 1].set_xlabel('n', fontdict={'family' : 'Times New Roman', 'size'   : 18})
    axes[1, 1].tick_params(labelcolor='black', labelsize='large', width=3)
    axes[1, 1].grid()
    axes[1, 1].legend(loc='upper center')

    # plt.savefig(os.path.join(project_dir,fig_dir_path, 'all_T1T2map.png'), bbox_inches='tight')
    plt.show()
    
    return fig, axes


# # 参考


# https://matplotlib.org/stable/gallery/index

# # 绘制T1、T2联合分布图
def scatter_hist(x, y, z, figure, ax, ax_histx, ax_histy, plot_line= True):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    # 显示T1-T2图表
    ax.contourf(y, x, z,cmap=plt.get_cmap('jet'), extend='both',levels=200)    # 显示图表

    # 以10为底的对数，只显示正数
    log_axi_array = [0.001,0.01,0.1,1,10,100,1000,10000,100000]
    ax.semilogx(base=10,subs=log_axi_array,nonpositive='mask')
    ax.semilogy(base=10,subs=log_axi_array,nonpositive='mask')

    '''绘制三条斜线'''
    if plot_line:
        # ax.loglog([np.log10(0.3),np.log10(30000)],[np.log10(0.3),np.log10(30000)],'k--',color='y')
        # ax.loglog([np.log10(0.3),np.log10(30000)-1],[1+np.log10(0.3),np.log10(30000)],'k--',color='y')
        # ax.loglog([np.log10(0.3),np.log10(30000)-2],[2+np.log10(0.3),np.log10(30000)],'k--',color='y')
        # 不取对数的值
        ax.loglog([0.003,30000],[0.003,30000],'k--',color='y')
        ax.loglog([0.003,30000/10],[0.03,30000],'k--',color='y')
        ax.loglog([0.003,30000/100],[0.3,30000],'k--',color='y')
        # ax.loglog([0.03,30000/100],[3,30000],'k--',color='y')
        # ax.loglog([0.003,30000/1000],[3,30000],'k--',color='y')

    '''设置横纵坐标轴范围'''
    ax.set_xlim(0.01,10000)
    ax.set_ylim(0.01,10000)
    ax.set_ylabel('T1(s)', fontdict={'family' : 'Times New Roman', 'size'   : 18})
    ax.set_xlabel('T2(s)', fontdict={'family' : 'Times New Roman', 'size'   : 18})
    ax.tick_params(labelcolor='black', labelsize='large', width=3)
    ax.set_title('T1 vs T2',fontsize = 20)

    cax = add_right_cax(ax, pad=0.02, width=0.02)
    # cbar = axes[0, 1].colorbar(axes[0, 1], cax=cax)
    # # ax= plt.axes()
    cmp = plt.get_cmap('jet')
    #
    # # https://zhajiman.github.io/post/matplotlib_colorbar/
    #
    norm = mcolors.LogNorm(vmin=1E0, vmax=1E3)
    # # norm = mpl.colors.BoundaryNorm([0,1,2,3], cmp.N)
    # # create an Axes on the right side of ax. The width of cax will be 5%
    # # of ax and the padding between cax and ax will be fixed at 0.05 inch.

    # ax.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmp), ax=cax,location='right')
    figure.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmp), cax=cax)

    # 将矩阵的所有行叠加到第一行
    matrix_horzon = np.sum(z, axis=0)

    matrix_vertical = np.sum(z, axis=-1)
    ax_histx.plot(x,matrix_horzon)
    ax_histx.grid(axis='x',linestyle='-.')
    ax_histy.plot(matrix_vertical, y)
    ax_histy.grid(axis='y',linestyle='-.')
    
    
def plot_T1T2_distribution(t1_domain, t2_domain, f_grid):
    # Start with a square Figure.
    fig = plt.figure(figsize=(10, 10))
    # Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
    # the size of the marginal axes and the main axes in both directions.
    # Also adjust the subplot parameters for a square plot.
    gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                        left=0.1, right=0.9, bottom=0.1, top=0.9,
                        wspace=0.3, hspace=0.2)
    # Create the Axes.
    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
    # Draw the scatter plot and marginals.
    scatter_hist(t1_domain, t2_domain, f_grid, fig, ax, ax_histx, ax_histy,True)
    return fig, ax, ax_histx, ax_histy


def plot_T1T2(t1_domain, t2_domain, f_grid, plot_line=False):
    # Start with a square Figure.
    fig = plt.figure(figsize=(6, 6))
    # Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
    # the size of the marginal axes and the main axes in both directions.
    # Also adjust the subplot parameters for a square plot.
    gs = fig.add_gridspec(1, 1,left=0.1, right=0.9, bottom=0.1, top=0.9,
                        wspace=0.3, hspace=0.2)
    # Create the Axes.
    ax = fig.add_subplot(gs[0, 0])

    # Draw the scatter plot and marginals.
    # the scatter plot:
    # 显示T1-T2图表
    ax.contourf(t2_domain, t1_domain, f_grid,cmap=plt.get_cmap('jet'), extend='both',levels=200)    # 显示图表
    # ax.contourf(t2_domain, t1_domain, f_grid, cmap='gray', extend='both',levels=200)    # 显示图表

    # 以10为底的对数，只显示正数
    log_axi_array = [0.001,0.01,0.1,1,10,100,1000,10000,100000]
    ax.semilogx(base=10,subs=log_axi_array,nonpositive='mask')
    ax.semilogy(base=10,subs=log_axi_array,nonpositive='mask')

    '''绘制三条斜线'''
    if plot_line:
        # ax.loglog([np.log10(0.3),np.log10(30000)],[np.log10(0.3),np.log10(30000)],'k--',color='y')
        # ax.loglog([np.log10(0.3),np.log10(30000)-1],[1+np.log10(0.3),np.log10(30000)],'k--',color='y')
        # ax.loglog([np.log10(0.3),np.log10(30000)-2],[2+np.log10(0.3),np.log10(30000)],'k--',color='y')
        # 不取对数的值
        ax.loglog([0.003,30000],[0.003,30000],'k--',color='y')
        ax.loglog([0.003,30000/10],[0.03,30000],'k--',color='y')
        ax.loglog([0.003,30000/100],[0.3,30000],'k--',color='y')
        # ax.loglog([0.03,30000/100],[3,30000],'k--',color='y')
        # ax.loglog([0.003,30000/1000],[3,30000],'k--',color='y')

    '''设置横纵坐标轴范围'''
    # ax.set_xlim(0.01,10000)
    # ax.set_ylim(0.01,10000)
    # ax.set_ylabel('T1(s)', fontdict={'family' : 'Times New Roman', 'size'   : 18})
    # ax.set_xlabel('T2(s)', fontdict={'family' : 'Times New Roman', 'size'   : 18})
    # ax.tick_params(labelcolor='black', labelsize='large', width=3)
    # ax.set_title('T1 vs T2',fontsize = 20)

    # cax = add_right_cax(ax, pad=0.02, width=0.02)

    # cmp = plt.get_cmap('jet')
    #
    # # https://zhajiman.github.io/post/matplotlib_colorbar/
    #
    # norm = mcolors.LogNorm(vmin=1E0, vmax=1E3)
    # # # norm = mpl.colors.BoundaryNorm([0,1,2,3], cmp.N)
    # # # create an Axes on the right side of ax. The width of cax will be 5%
    # # # of ax and the padding between cax and ax will be fixed at 0.05 inch.

    # # ax.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmp), ax=cax,location='right')
    # fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmp), cax=cax)
    plt.axis('off')   # 去坐标轴
    plt.xticks([])    # 去 x 轴刻度
    plt.yticks([])    # 去 y 轴刻度
    return fig, ax