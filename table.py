import pandas as pd
import numpy as np
import plotnine
from plotnine import *
from plotnine.themes.elements import margin

# ['Age', 'Accessibility', 'EdLevel', 'Employment', 'Gender',
#    'MentalHealth', 'MainBranch', 'YearsCode', 'YearsCodePro', 'Country',
#    'PreviousSalary', 'ComputerSkills', 'Employed', 'Citizenship']


def organize_data():
    jobs = pd.read_csv('stackoverflow_full.csv')


    print('before:\n',jobs.columns)
    print()
    print(jobs)

    ### RENAME VALUES TO SOMETHING MORE MEANINGFUL ###
    jobs['Employed'] = jobs['Employed'].replace({0.0: 'No', 1.0: 'Yes'})

    jobs['Employment'] = jobs['Employment'].replace({0.0: 'No', 1.0: 'Yes'})

    jobs['Country'] = jobs['Country'].apply( lambda x: 'Yes' if x == 'United States of America' else 'No')
    jobs = jobs.rename(columns={'Country': 'Citizenship'})


    bins = [    0, 
                30000, 
                60000,
                90000,
                120000,
                150000,
                180000,
                210000,
                float('inf')
            ]  
    labels = [  '0-30k',
                '30k-60k',
                '60k-90k',
                '90k-120k',
                '120k-150k',
                '150k-180k',
                '180k-210k',
                '210k+'
            ]
    jobs['PreviousSalary'] = pd.cut(jobs['PreviousSalary'],bins=bins, labels=labels, right=False)

    bins = [    0, 
                5, 
                10,
                15,
                20,
                25,
                30,
                35,
                40,
                float('inf')
            ]  
    labels = [  '0-5',
                '5-10',
                '10-15',
                '15-20',
                '20-25',
                '25-30',
                '30-35',
                '35-40',
                '40+'
            ]
    jobs['ComputerSkills'] = pd.cut(jobs['ComputerSkills'],bins=bins, labels=labels, right=False)

    bins = [    0, 
                2, 
                5,
                10,
                15,
                20,
                25,
                30,
                35,
                40,
                45,
                float('inf')
            ]  
    labels = [  '0-2',
                '2-5',
                '5-10',
                '10-15',
                '15-20',
                '20-25',
                '25-30',
                '30-35',
                '35-40',
                '40-45',
                '45+'
            ]
    jobs['YearsCode'] = pd.cut(jobs['YearsCode'],bins=bins, labels=labels, right=False)
    jobs['YearsCodePro'] = pd.cut(jobs['YearsCodePro'],bins=bins, labels=labels, right=False)



    jobs = jobs.drop(columns=[jobs.columns[0],'HaveWorkedWith'])

    jobs = jobs[['Employed','Age', 'Accessibility', 'EdLevel', 'Employment', 'Gender',
        'MentalHealth', 'MainBranch', 'YearsCode', 'YearsCodePro',
        'Citizenship', 'PreviousSalary', 'ComputerSkills']]

    print('after:\n',jobs.columns)
    print()
    print(jobs)

    jobs.to_csv('job_applicants_plots/jobs.csv', index=False)
    


# def att_prop_label(label_df, tag):
#     print('ATTRIBUTE VALUES PROPORTIONS IN EACH TARGET LABEL', tag + ':')

#     for col in label_df.columns:
#         print(label_df[col].value_counts())



jobs = pd.read_csv('job_applicants_plots/jobs.csv')

# hired = jobs[jobs['Employed'] == 'Yes']
# # att_prop_label(hired,'"Yes"')

# not_hired = jobs[jobs['Employed'] == 'No']
# # att_prop_label(not_hired,'"No"')



# # Convert dictionary into a DataFrame
# def plot_proportion(data, name):
#     df = pd.concat(data).reset_index()
#     df.columns = ['Attribute', 'Category', 'Count']

#     # Plot grouped bar chart
#     plot = (
#         ggplot(df, aes(x='Attribute', y='Count', fill='Category'))
#         + geom_bar(stat='identity', position=position_dodge())
#         + labs(
#             title="Value Counts of Attributes",
#             x="Attributes",
#             y="Count"
#         )
#         + theme(axis_text_x=element_text(angle=45, hjust=1))
#     )

#     plot.save('job_applicants_plots/att_distribution_label/' + name + '_attributes.png')

hired_data_tech = {
    'MainBranch': pd.Series({'Dev': 30120, 
                             'NotDev': 3950
                            }),
    'YearsCodePro': pd.Series({'0-2': 3321,
                               '5-10': 9643, 
                               '2-5': 8902, 
                               '10-15': 5160, 
                               '15-20': 2675, 
                               '20-25': 2115, 
                               '25-30': 1035, 
                               '30-35': 607,
                               '35-40': 360, 
                               '40-45': 190, 
                               '45+': 62
                            }),
    'PreviousSalary': pd.Series({'0-30k': 8712,
                                 '30k-60k': 8991,  
                                 '60k-90k': 6898, 
                                 '90k-120k': 3949,
                                 '120k-150k': 2491, 
                                 '150k-180k': 1747, 
                                 '180k-210k': 1021, 
                                 '210k+': 261
                                }),
    'ComputerSkills': pd.Series({'0-5': 5115, 
                                 '5-10': 14722, 
                                 '10-15': 10542, 
                                 '15-20': 3131,
                                 '20-25': 496, 
                                 '25-30': 61, 
                                 '30-35': 3
                                }),
    'YearsCode': pd.Series({'0-2': 317, 
                            '2-5': 3057, 
                            '5-10': 9641, 
                            '10-15': 7824, 
                            '15-20': 4503, 
                            '20-25': 3443,
                            '25-30': 1803, 
                            '30-35': 1432, 
                            '35-40': 1043,
                            '40-45': 737, 
                            '45+': 270
                            })
}
hired_data_personal = {
    'Age': pd.Series({'<35': 21649, 
                      '>35': 12421
                    }),
    'Accessibility': pd.Series({'Yes': 912,
                                'No': 33158
                            }),
    'EdLevel': pd.Series({'Undergraduate': 16460, 
                          'Master': 9718, 
                          'PhD': 1861, 
                          'NoHigherEd': 1525,
                          'Other': 4506,
                        }),
    'Employment': pd.Series({'Yes': 30230, 
                             'No': 3840
                            }),
    'Gender': pd.Series({'Man': 31491, 
                         'Woman': 1939, 
                         'NonBinary': 640
                        }),
    'MentalHealth': pd.Series({'Yes': 7316, 
                               'No': 26754
                            }),
    'Citizenship': pd.Series({'Yes': 6600,
                              'No': 27470
                            }),
}




def prop_per_x(x, count):
    """
    Compute the proportion of the counts for each value of x
    """
    df = pd.DataFrame({"x": x, "count": count})
    prop = df["count"] / df.groupby("x")["count"].transform("sum")
    return prop

def plot_proportions(data,attribute,label):
    plot = (
        ggplot(data, aes(attribute, fill=label))
        + geom_bar(position="dodge2")
        + labs(
            title= (attribute + ' Proportions Across ' + label + ' Labels')
        )
        + geom_text(
            aes(
                label=after_stat("prop_per_x(x, count) * 100"),
                y=stage(after_stat="count", after_scale="y+.25"),
            ),
            stat="count",
            # nudge_y=0.2,
            va='bottom',
            position=position_dodge2(width=0.9),
            format_string="{:.1f}%",
            size=9,
        )
    )
    plot.save('job_applicants_plots/label_distribution_att/labels_in_' + attribute.lower() + '.png')

    plot = (
        ggplot(jobs, aes(label, fill=attribute))
        + geom_bar(position="dodge2")
        + labs(
            title= (label + ' Proportions Across ' + attribute + ' Attribute')
        )
        + geom_text(
            aes(
                label=after_stat("prop_per_x(x, count) * 100"),
                y=stage(after_stat="count", after_scale="y+.25"),
            ),
            stat="count",
            # nudge_y=0.2,
            va='bottom',
            position=position_dodge2(width=0.9),
            format_string="{:.1f}%",
            size=9,
        )
    )
    plot.save('job_applicants_plots/label_distribution_att/' + attribute.lower() + '_in_labels.png')


attributes = ['Age', 'Accessibility', 'EdLevel', 'Employment', 'Gender',
   'MentalHealth', 'MainBranch', 'YearsCode', 'YearsCodePro', 'Citizenship',
   'PreviousSalary', 'ComputerSkills']

for a in attributes:
    plot_proportions(jobs, a,'Employed')

