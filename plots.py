import pandas as pd
import plotnine
from plotnine import *
from plotnine.themes.elements import margin

jobs = pd.read_csv('stackoverflow_full.csv')
jobs = jobs.drop(columns=jobs.columns[0])
print(jobs)


### LABEL DISTRIBUTION ###
# bar = (
#     ggplot()
#     + geom_bar(
#         jobs,
#         aes('Employed', fill='Employed'))

#     + geom_text(
#         aes(label=after_stat('count')),
#         # stat='Count',

#     #     nudge_x=-0.14,
#     #     nudge_y=0.125,
#     #     va="bottom",
#     )
#     + geom_text(
#         aes(label=after_stat('prop*100'), group=1),
#         stat='count',
#         nudge_x=0.14,
#         nudge_y=0.125,
#         va='bottom',
#         format_string="({:.1f}%)",
#     )
# )
def bar_chart(data, x_axis, y_axis, title, file):
    dataframe = pd.DataFrame(data)
    chart = (
        ggplot(dataframe) 
        + aes(x=x_axis, y=y_axis)
        + geom_col(position="dodge")
        + ggtitle(title)
        + scale_fill_grey()
    )

    chart.save(filename=file)

    return chart


def value_distribution(data,x_axis, title, x_name, y_name, file_name):
    bar = (
        ggplot(data, 
            aes(x=x_axis, fill=x_axis)) 
        + geom_bar()
        + labs(
            title=title,
            x=x_name,
            y=y_name
        )
        # + geom_col(position="dodge")
        + geom_text(
            aes(label=after_stat('prop*100'), group=1),
            stat='count',
            # nudge_x=0.14,
            nudge_y=1,
            va='bottom',
            format_string="({:.1f}%)",
        )
        + scale_fill_grey()
    )

    bar.save(filename=file_name)

def value_distribution_no_percentage(data,x_axis, title, x_name, y_name, file_name):
    bar = (
        ggplot(data, 
            aes(x=x_axis, fill=x_axis)) 
        + geom_bar()
        + labs(
            title=title,
            x=x_name,
            y=y_name
        )
        # + geom_col(position="dodge")
        # + geom_text(
        #     aes(label=after_stat('prop*100'), group=1),
        #     stat='count',
        #     # nudge_x=0.14,
        #     nudge_y=1,
        #     va='bottom',
        #     format_string="({:.1f}%)",
        # )
        + scale_fill_grey()
    )

    bar.save(filename=file_name)




def plot_histogram(dataset, x_axis, plot_title,x_name, y_name, filename):
    plot = (
    ggplot(dataset, aes(x=x_axis)) +
    geom_histogram(binwidth=5, fill='blue', color='black', alpha=0.7) +
    labs(
        title=plot_title,
        x=x_name,
        y=y_name
    )
    )


    plot.save(filename=filename)


print(jobs.columns)
jobs['Employed'] = jobs['Employed'].replace({0.0: 'No', 1.0: 'Yes'})

print(jobs['Employed'])

value_distribution(jobs, 'Employed', 
                   'Target Label Values Distribution Across Dataset', 
                   'Employment Status', 'Number of Applicants', 
                   'dataset_plots/job_applicants/label_proportions.png')


value_distribution(jobs, 'Age', 
                   'Age Attribute Values Distribution Across Dataset', 
                   'Age', 'Number of Applicants', 
                   'dataset_plots/job_applicants/age_proportions.png')

value_distribution(jobs, 'Gender', 
                   'Gender Attribute Values Distribution Across Dataset', 'Gender', 'Number of Applicants', 
                   'dataset_plots/job_applicants/gender_proportions.png')

value_distribution(jobs, 'EdLevel', 
                   'Education Level Attribute Values Distribution Across Dataset', 
                   'Education Level', 'Number of Applicants', 
                   'dataset_plots/job_applicants/edlevel_proportions.png')

value_distribution(jobs, 'Accessibility', 
                   'Accessibility Attribute Values Distribution Across Dataset', 
                   'Accessibility', 'Number of Applicants', 
                   'dataset_plots/job_applicants/accessibility_proportions.png')


jobs['Employment'] = jobs['Employment'].replace({0.0: 'No', 1.0: 'Yes'})
value_distribution(jobs, 'Employment', 
                   'Employment Attribute Values Distribution Across Dataset', 
                   'Employment Compatibility', 'Number of Applicants', 
                   'dataset_plots/job_applicants/employment_proportions.png')

value_distribution(jobs, 'MentalHealth', 
                   'Mental Health Attribute Values Distribution Across Dataset', 
                   'MentalHealth', 'Number of Applicants', 
                   'dataset_plots/job_applicants/mentalhealth_proportions.png')

value_distribution(jobs, 'MainBranch', 
                   'Main Branch Attribute Values Distribution Across Dataset', 
                   'MainBranch', 'Number of Applicants', 
                   'dataset_plots/job_applicants/mainbranch_proportions.png')

value_distribution_no_percentage(jobs, 'YearsCode', 
                   'Coding Experience Attribute Values Distribution Across Dataset', 
                   'Years of Coding Experience', 'Number of Applicants', 
                   'dataset_plots/job_applicants/yearscode_proportions.png')

value_distribution_no_percentage(jobs, 'YearsCodePro', 
                   'Professional Coding Experience Attribute Values Distribution Across Dataset', 
                   'Years of Professional Coding Experience', 'Number of Applicants', 
                   'dataset_plots/job_applicants/yearscodepro_proportions.png')


citizens = jobs[['Employed', 'Country']]
citizens = citizens.rename(columns={'Country': 'Citizenship'})

citizens['Citizenship'] = citizens['Citizenship'].apply( lambda x: 'Yes' if x == 'United States of America' else 'No')

print(citizens)

value_distribution(citizens, 'Citizenship', 
                   'Citizenship Attribute Values Distribution Across Dataset', 
                   'Citizenship Status', 'Number of Applicants', 
                   'dataset_plots/job_applicants/citizens_proportions.png')

plot = (
    ggplot(jobs, aes(x='PreviousSalary')) +
    geom_histogram(binwidth=10000, fill='blue', color='black', alpha=0.7) +  # Wider bins
    labs(
        title='Previous Salary Attribute Values Distribution Across Dataset',
        x='Previous Salary',
        y='Number of Applicants'
    )
)

plot.save('dataset_plots/job_applicants/prevsalary_proportions2.png')


# plot_histogram(jobs, 'PreviousSalary', 
#                    'Previous Salary Attribute Values Distribution Across Dataset', 
#                    'Previous Salary', 'Number of Applicants', 
#                    'dataset_plots/job_applicants/prevsalary_proportions.png')


plot_histogram(jobs, 'ComputerSkills', 
                   'Computer Skills Attribute Values Distribution Across Dataset', 
                   'Number of Computer Skills', 'Number of Applicants', 
                   'dataset_plots/job_applicants/compskills_proportions.png')



# attributes = {}

# print('\ncols:')
# print(jobs.columns)

# print('\nfilter:')
# print(jobs['Age'])

# print('\nunique of filtered:')
# print(jobs['Age'].unique()) # 

# print('\ntotal count:')
# print(jobs['Age'].count()) # total instances

# print('\nspecific attribute value count:')
# print(jobs[jobs['Age'] == '<35']['Age'].count()) 

# age_rep_label = jobs[['Employed', 'Age']]

# print(age_rep_label)

# print(jobs[jobs['Age']=='<35'])

# for col in jobs.columns:
#     print('--------------------------------')
#     print(str(col) + ':')

#     labels = col.unique

#     for label in labels:
#         print('     ' + label + str(jobs[label].count()))

'''
Age: 
    <35 (65%)
    >35 (35%)

Accessibility: if they require accomodations or not (?)
    yes
    no

EdLevel:
    undergrad (51%)
    master (26%)
    other (23%)

Employment: level of qualification that they have for a given role

Gender:


MentalHealth

MainBranch: if they are a profesional developer or not

YearsCode: how long they've been coding for

YearsCodePro: how long they've been coding for in professional context

Country

PreviousSalary:

HaveWorkedWith:

ComputerSkills: number of tech skills that they have

Employed: target variable, were they hired or not

'''

### PLOTS ###

### RELEVANT STUFF PROBABLY ###
# age       | combine
# gender    |

# edlevel                       | combine
# employment (qualification)    |

# acessibility (if requires acomodations probs)     | combine
# mental health                                     |

# country (analyze international vs USA hires)

# yearscode         | combine
# yearcodepro       |
# computerskills    |

# employed (obv bc its target)

def prop_per_x(x, count):
    """
    Compute the proportion of the counts for each value of x
    """
    df = pd.DataFrame({"x": x, "count": count})
    prop = df["count"] / df.groupby("x")["count"].transform("sum")
    return prop


(    ggplot(mtcars, aes("factor(cyl)", fill="factor(am)"))
    + geom_bar(position="dodge2")
    + geom_text(
        aes(
            label=after_stat("prop_per_x(x, count) * 100"),
            y=stage(after_stat="count", after_scale="y+.25"),
        ),
        stat="count",
        position=position_dodge2(width=0.9),
        format_string="{:.1f}%",
        size=9,
    )
)





# general attribute distribution
    # bar chart where x index is the target, colored by each attribute value
    # so i guess may be best to do one bar chart per attribute lowkey
    # then do one with everything just bc what if


# for each attribute
    # out of this attribute value, how many were hired vs not
    # one bar chart per attribute where x=values and color=target

# also for each attribute
    # bar chart where axis=value and color=values of every other attributes except label
