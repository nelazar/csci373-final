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
    


def att_prop_label(label_df, tag):
    print('ATTRIBUTE VALUES PROPORTIONS IN EACH TARGET LABEL', tag + ':')

    for col in label_df.columns:
        print(label_df[col].value_counts())




def label_prop_att(att_df, label_value):
    pass



jobs = pd.read_csv('job_applicants_plots/jobs.csv')

hired = jobs[jobs['Employed'] == 'Yes']
# att_prop_label(hired,'"Yes"')

not_hired = jobs[jobs['Employed'] == 'No']
att_prop_label(not_hired,'"No"')


# print(not_hired)

# print(jobs['Employed'])


# citizens = jobs[['Employed', 'Country']]




# jobs['Citizenship'] = citizens['Citizenship']

# jobs = jobs.drop(columns=['HaveWorkedWith'])


# print(jobs.columns)
# print(jobs)
# print(jobs.value_counts())

