import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.patches as mpatches
import seaborn as sns
import matplotlib.pyplot as plt


def app():
    st.write("## Topological Space for PD Subtypes using Unsupervised Approach")
    original_data = pd.read_csv("data/PDBP_PPMI_replication_progression_space_3d.csv")
    colorable_columns_maps = {
        'Subtypes': "Subtypes",
    }
    colorable_columns = list(colorable_columns_maps)
    # st.write("### Select a factor to color according to the factor")
    select_color = "Subtypes" # st.selectbox('', [colorable_columns_maps[i] for i in colorable_columns], index=0)
    cols = st.columns(2)
    if cols[0].checkbox("Show Replication Cohort (PDBP)"):
        pass
    else:
        original_data = original_data[(original_data['dataset'] == 'ppmi')]
    # st.write("#### Select the duration from baseline to visualize progression")
    year_list = ['Baseline', 'Year1', 'Year2', 'Year3', 'Year4', 'Year5']
    if cols[1].checkbox("Show odd years"):
        end_select_year = st.select_slider('Select the duration from baseline to visualize progression', ['Year4', 'Year5'])
        itera = 2
    else:
        end_select_year = st.select_slider('Select the duration from baseline to visualize progression', ['Year2', 'Year3', 'Year4', 'Year5'])
        itera = 1


    year_mapping = {'Baseline': 'BL', 'Year1': 'V04', 'Year2': 'V06', 'Year3': 'V08', 'Year4': 'V10', 'Year5': 'V12'}
    # original_data = original_data[original_data['visit'] == year_mapping[select_year]]
    #
    palette_progression = {'PDvec3': 'orange', 'PDvec2': 'blue', 'PDvec1': 'green', 'Non-PD': 'red'}
    subtype_order = ['PDvec3', 'PDvec2', 'PDvec1', 'Non-PD']
    subtype_replace = {'PD_h': 'PDvec3', 'PD_m': 'PDvec2', 'PD_l': 'PDvec1', 'HC': 'Non-PD', 'Control': 'Non-PD'}
    subtype_column = {'GMM': 'Subtypes'}
    color_patch = []
    for lab, color in palette_progression.items():
        color_patch.append(mpatches.Patch(color=color, label=lab))


    # st.write(ppmi.head())
    # st.write(pdbp.head())
    cols = st.columns(3)

    @st.cache(hash_funcs={"MyUnhashableClass": lambda _: None}, allow_output_mutation=True, ttl=24 * 3600)
    def get_plot(select_year):
        ppmi = original_data[
            (original_data['dataset'] == 'ppmi') & (original_data['visit'] == year_mapping[select_year])]
        pdbp = original_data[
            (original_data['dataset'] == 'pdbp') & (original_data['visit'] == year_mapping[select_year])]
        y_axis = 'Motor dimension'
        z_axis = 'Sleep dimension'
        x_axis = 'Cognitive dimension'
        label_name = {'y_axis': 'Motor disturbance', 'x_axis': 'Cognitive impairment', 'z_axis': 'Sleep disturbance'}
        my_dpi = 96

        width = 11
        size_variation = 3
        fig = plt.figure(figsize=(4 * size_variation, 3 * size_variation))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(ppmi[x_axis], ppmi[y_axis], ppmi[z_axis], c=ppmi['Subtypes'].map(lambda x: palette_progression[x]),
                   marker='o', alpha=0.6 if len(pdbp) > 0 else 1, s=20 if len(pdbp) > 0 else 30)
        if len(pdbp) > 0:
            ax.scatter(pdbp[x_axis], pdbp[y_axis], pdbp[z_axis],
                       c=pdbp['Subtypes'].map(lambda x: palette_progression[x]),
                       marker='s', edgecolors='black', s=30)
        # for subtype, color in palette_progression.items():
        #    x_meanpt = pdbp.groupby('Subtypes').agg('mean').loc[subtype][x_axis]
        #    y_meanpt = pdbp.groupby('Subtypes').agg('mean').loc[subtype][y_axis]
        #    z_meanpt = pdbp.groupby('Subtypes').agg('mean').loc[subtype][z_axis]
        #    mean_line = [(0, x_meanpt), (0, y_meanpt), (0, z_meanpt)]

        # label the axes
        ax.set_xlabel(r'{}$\rightarrow$'.format(label_name['x_axis']), labelpad=10, fontsize=18)
        ax.set_ylabel(r'{}$\rightarrow$'.format(label_name['y_axis']), labelpad=10, fontsize=18)
        ax.set_zlabel(r'{}$\rightarrow$'.format(label_name['z_axis']), labelpad=10, fontsize=18)
        ax.grid(True, alpha=0.2)
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.set_xlim(0, 0.6)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        ax.view_init(elev=30., azim=330)
        square_patch = []
        circle_patch = []
        for lab, color in palette_progression.items():
            # color_patch.append(mpatches.Patch(color=color, label=lab))
            circle_patch.append(
                plt.plot([], [], marker="o", ms=16, ls="", mec=None, color=color, label='PPMI-' + lab,
                         alpha=0.6 if len(pdbp) > 0 else 1)[0])
            if len(pdbp) > 0:
                square_patch.append(
                    plt.plot([], [], marker="s", ms=16, ls="", mec=None, color=color, label='PDBP-' + lab)[0])
        ax.legend(handles=circle_patch + square_patch, bbox_to_anchor=(0.1, -0.3), loc='lower left', ncol=4,
                  numpoints=1, title='Subtypes', fontsize='large', title_fontsize='large')

        ax.set_title(select_year, fontsize=24)
        return fig

    for enm in range(3):
        select_year = year_list[year_list.index(end_select_year) - 2*itera + enm*itera]




        with cols[enm]:
            # st.header("A cat")
            # st.image("https://static.streamlit.io/examples/cat.jpg")
            st.pyplot(get_plot(select_year), height=300, width=500)

        #  ax.can_zoom(True)
        # plt.show()