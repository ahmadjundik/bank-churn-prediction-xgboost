import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

st.title("Bank Customer Churn Visualization")

def load_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    data['Churn'] = data['Attrition_Flag'].apply(lambda x: 1 if x == 'Attrited Customer' else 0)
    return data

uploaded_file = st.file_uploader("Choose a file", type=["csv", "txt"])

if uploaded_file:
    df = load_data(uploaded_file)

    attribute = st.sidebar.selectbox("Choose an attribute for analysis:", (
        "Gender", "Age", "Education Level", "Marital Status", "Income Category", "Card Category",
        "Months on Book", "Total Relationship Count", "Months Inactive 12 Mon", "Contacts Count 12 mon",
        "Credit Limit", "Total Revolving Balance", "Total Transaction Amount", "Total Transaction Count", "Avg Utilization Ratio"
    ))

    if attribute == 'Gender' and 'Gender' in df.columns:
        gender = st.sidebar.multiselect('Select Gender:', options=df['Gender'].unique(), default=df['Gender'].unique())
        df_filtered = df[df['Gender'].isin(gender)]
        st.title("Dashboard Customer Churn by Gender")
        
        gender_counts = df_filtered['Gender'].value_counts().reindex(['M', 'F']).fillna(0).astype(int)
        churn_counts = df_filtered[df_filtered['Churn'] == 1]['Gender'].value_counts().reindex(['M', 'F']).fillna(0).astype(int)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Male", value=gender_counts.get('M', 0))
        with col2:
            st.metric(label="Female", value=gender_counts.get('F', 0))
        with col3:
            st.metric(label="Total Churn", value=churn_counts.sum())

        st.subheader("Customer Churn by Gender")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=gender_counts.index, y=gender_counts.values, color='lightblue', label='Total')
        sns.barplot(x=churn_counts.index, y=churn_counts.values, color='darkblue', label='Churn')
        
        plt.title('Customer Churn by Gender')
        plt.ylabel('Count')
        plt.legend()
        st.pyplot(fig)

    elif attribute == 'Age' and 'Customer_Age' in df.columns:
        df['Churn Status'] = df['Attrition_Flag'].replace({
            'Existing Customer': 'Non-Churn',
            'Attrited Customer': 'Churn'
        })
        age_selection = st.sidebar.slider('Select Age Range:', min_value=int(df['Customer_Age'].min()), max_value=int(df['Customer_Age'].max()), value=(int(df['Customer_Age'].min()), int(df['Customer_Age'].max())))
        df_filtered = df[(df['Customer_Age'] >= age_selection[0]) & (df['Customer_Age'] <= age_selection[1])]
        st.title("Dashboard Customer Churn by Age")
        
        hist_fig = px.histogram(df_filtered, x='Customer_Age', color='Churn Status',
                                labels={'Customer_Age': 'Age', 'Churn Status': 'Churn Status'},
                                title='Customer Churn by Age',
                                barmode='stack',
                                nbins=int(df['Customer_Age'].max() - df['Customer_Age'].min())
                               )
        hist_fig.update_layout(xaxis_title='Age', yaxis_title='Count of Customers',
                               legend_title="Churn Status")
        st.plotly_chart(hist_fig, use_container_width=True)

    elif attribute == 'Education Level' and 'Education_Level' in df.columns:
        education_levels = st.sidebar.multiselect('Select Education Level:', options=df['Education_Level'].unique(), default=df['Education_Level'].unique())
        df_filtered = df[df['Education_Level'].isin(education_levels)]
        st.title("Dashboard Customer Churn by Education Level")
        
        education_counts = df_filtered['Education_Level'].value_counts().fillna(0).astype(int)
        churn_counts = df_filtered[df_filtered['Churn'] == 1]['Education_Level'].value_counts().fillna(0).astype(int)

        st.subheader("Customer Churn by Education Level")
        fig, ax = plt.subplots(figsize=(8, 4))
        non_churn_color = '#B0C4DE'
        churn_color = '#DC143C'
        sns.barplot(x=education_counts.index, y=education_counts.values, color=non_churn_color, label='Non-Churn')
        sns.barplot(x=churn_counts.index, y=churn_counts.values, color=churn_color, label='Churn')
        
        for p, label in zip(ax.patches[len(education_counts):], churn_counts.values):
            ax.annotate(label, (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')

        plt.title('Customer Churn by Education Level')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.legend()
        st.pyplot(fig)

    elif attribute == 'Marital Status' and 'Marital_Status' in df.columns:
        marital_statuses = df['Marital_Status'].unique()
        df['Churn Status'] = df['Churn'].map({0: 'Non-Churn', 1: 'Churn'})
        st.title("Dashboard Customer Churn by Marital Status")

        source = []
        target = []
        value = []
        label = list(marital_statuses) + ['Non-Churn', 'Churn']

        for i, status in enumerate(marital_statuses):
            non_churn_count = df[(df['Marital_Status'] == status) & (df['Churn'] == 0)].shape[0]
            churn_count = df[(df['Marital_Status'] == status) & (df['Churn'] == 1)].shape[0]

            source.extend([i, i])
            target.extend([len(marital_statuses), len(marital_statuses) + 1])
            value.extend([non_churn_count, churn_count])

        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=label,
                color="blue"
            ),
            link=dict(
                source=source,
                target=target,
                value=value,
                color=["rgba(0,176,246,0.5)", "rgba(246,78,139,0.6)"] * int(len(source) / 2)
            ))])

        fig.update_layout(title_text="Customer Churn Distribution by Marital Status", font_size=10)
        st.plotly_chart(fig)

    elif attribute == 'Income Category' and 'Income_Category' in df.columns:
        income_categories = st.sidebar.multiselect('Select Income Category:', options=df['Income_Category'].unique(), default=df['Income_Category'].unique())
        df_filtered = df[df['Income_Category'].isin(income_categories)]
        st.title("Dashboard Customer Churn by Income Category")
        
        pie_fig = px.pie(df_filtered, names='Income_Category', title='Distribution of Income Categories')
        st.plotly_chart(pie_fig, use_container_width=True)
        
        churn_data = df_filtered.groupby('Income_Category')['Churn'].value_counts().unstack().fillna(0)
        churn_data.columns = ['Non-Churn', 'Churn']
        bar_fig = px.bar(churn_data, barmode='group', title="Customer Churn by Income Category")
        st.plotly_chart(bar_fig, use_container_width=True)

    elif attribute == 'Card Category' and 'Card_Category' in df.columns:
        card_categories = st.sidebar.multiselect('Select Card Category:', options=df['Card_Category'].unique(), default=df['Card_Category'].unique())
        df_filtered = df[df['Card_Category'].isin(card_categories)]
        st.title("Dashboard Customer Churn by Card Category")
        
        pie_fig = px.pie(df_filtered, names='Card_Category', title='Distribution of Card Categories')
        st.plotly_chart(pie_fig, use_container_width=True)
        
        churn_data = df_filtered.groupby('Card_Category')['Churn'].value_counts().unstack().fillna(0)
        churn_data.columns = ['Non-Churn', 'Churn']
        bar_fig = px.bar(churn_data, barmode='group', title="Customer Churn by Card Category",
                         labels={'value':'Number of Customers', 'variable':'Churn Status'},
                         log_y=True, 
                         text_auto='.2s',
                         color_discrete_sequence=['#636EFA', '#EF553B'])  
        st.plotly_chart(bar_fig, use_container_width=True)

    elif attribute == 'Months on Book' and 'Months_on_book' in df.columns:
        months_selection = st.sidebar.slider('Select Months on Book Range:', min_value=int(df['Months_on_book'].min()), max_value=int(df['Months_on_book'].max()), value=(int(df['Months_on_book'].min()), int(df['Months_on_book'].max())))
        df_filtered = df[(df['Months_on_book'] >= months_selection[0]) & (df['Months_on_book'] <= months_selection[1])]
        st.title("Dashboard Customer Churn by Months on Book")
    
        hist_fig = px.histogram(df_filtered, x='Months_on_book', color='Attrition_Flag',
                            labels={'Months_on_book': 'Months on Book', 'Attrition_Flag': 'Attrition Flag'},
                            title='Customer Churn by Months on Book',
                            barmode='stack',
                            nbins=int(df['Months_on_book'].max() - df['Months_on_book'].min())
                           )
        hist_fig.update_layout(xaxis_title='Months on Book', yaxis_title='Count of Customers',
                           legend_title="Attrition Flag")
        st.plotly_chart(hist_fig, use_container_width=True)

    elif attribute == 'Total Relationship Count' and 'Total_Relationship_Count' in df.columns:
        relationship_count_selection = st.sidebar.slider('Select Total Relationship Count Range:', min_value=int(df['Total_Relationship_Count'].min()), max_value=int(df['Total_Relationship_Count'].max()), value=(int(df['Total_Relationship_Count'].min()), int(df['Total_Relationship_Count'].max())))
        df_filtered = df[(df['Total_Relationship_Count'] >= relationship_count_selection[0]) & (df['Total_Relationship_Count'] <= relationship_count_selection[1])]
        st.title("Dashboard Customer Churn by Total Relationship Count")
    
        hist_fig = px.histogram(df_filtered, x='Total_Relationship_Count', color='Attrition_Flag',
                            labels={'Total_Relationship_Count': 'Total Relationship Count', 'Attrition_Flag': 'Attrition Flag'},
                            title='Customer Churn by Total Relationship Count',
                            barmode='stack',
                            nbins=int(df['Total_Relationship_Count'].max() - df['Total_Relationship_Count'].min())
                           )
        hist_fig.update_layout(xaxis_title='Total Relationship Count', yaxis_title='Count of Customers',
                           legend_title="Attrition Flag")
        st.plotly_chart(hist_fig, use_container_width=True)

    elif attribute == 'Months Inactive 12 Mon' and 'Months_Inactive_12_mon' in df.columns:
        months_inactive_selection = st.sidebar.slider('Select Months Inactive in Last 12 Months Range:', min_value=int(df['Months_Inactive_12_mon'].min()), max_value=int(df['Months_Inactive_12_mon'].max()), value=(int(df['Months_Inactive_12_mon'].min()), int(df['Months_Inactive_12_mon'].max())))
        df_filtered = df[(df['Months_Inactive_12_mon'] >= months_inactive_selection[0]) & (df['Months_Inactive_12_mon'] <= months_inactive_selection[1])]
        st.title("Dashboard Customer Churn by Months Inactive in Last 12 Months")
    
        hist_fig = px.histogram(df_filtered, x='Months_Inactive_12_mon', color='Attrition_Flag',
                            labels={'Months_Inactive_12_mon': 'Months Inactive in Last 12 Months', 'Attrition_Flag': 'Attrition Flag'},
                            title='Customer Churn by Months Inactive in Last 12 Months',
                            barmode='stack',
                            nbins=int(df['Months_Inactive_12_mon'].max() - df['Months_Inactive_12_mon'].min())
                           )
        hist_fig.update_layout(xaxis_title='Months Inactive in Last 12 Months', yaxis_title='Count of Customers',
                           legend_title="Attrition Flag")
        st.plotly_chart(hist_fig, use_container_width=True)

    elif attribute == 'Contacts Count 12 mon' and 'Contacts_Count_12_mon' in df.columns:
        contacts_count_selection = st.sidebar.slider('Select Contacts Count in Last 12 Months Range:', min_value=int(df['Contacts_Count_12_mon'].min()), max_value=int(df['Contacts_Count_12_mon'].max()), value=(int(df['Contacts_Count_12_mon'].min()), int(df['Contacts_Count_12_mon'].max())))
        df_filtered = df[(df['Contacts_Count_12_mon'] >= contacts_count_selection[0]) & (df['Contacts_Count_12_mon'] <= contacts_count_selection[1])]
        st.title("Dashboard Customer Churn by Contacts Count in Last 12 Months")
    
        hist_fig = px.histogram(df_filtered, x='Contacts_Count_12_mon', color='Attrition_Flag',
                            labels={'Contacts_Count_12_mon': 'Contacts Count in Last 12 Months', 'Attrition_Flag': 'Attrition Flag'},
                            title='Customer Churn by Contacts Count in Last 12 Months',
                            barmode='stack',
                            nbins=int(df['Contacts_Count_12_mon'].max() - df['Contacts_Count_12_mon'].min())
                           )
        hist_fig.update_layout(xaxis_title='Contacts Count in Last 12 Months', yaxis_title='Count of Customers',
                           legend_title="Attrition Flag")
        st.plotly_chart(hist_fig, use_container_width=True)

    elif attribute == 'Credit Limit' and 'Credit_Limit' in df.columns:
        credit_limit_selection = st.sidebar.slider('Select Credit Limit Range:', min_value=int(df['Credit_Limit'].min()), max_value=int(df['Credit_Limit'].max()), value=(int(df['Credit_Limit'].min()), int(df['Credit_Limit'].max())))
        df_filtered = df[(df['Credit_Limit'] >= credit_limit_selection[0]) & (df['Credit_Limit'] <= credit_limit_selection[1])]
        st.title("Dashboard Customer Churn by Credit Limit")
    
        hist_fig = px.histogram(df_filtered, x='Credit_Limit', color='Attrition_Flag',
                            labels={'Credit_Limit': 'Credit Limit', 'Attrition_Flag': 'Attrition Flag'},
                            title='Customer Churn by Credit Limit',
                            barmode='stack',
                            nbins=int(df['Credit_Limit'].max() - df['Credit_Limit'].min())
                           )
        hist_fig.update_layout(xaxis_title='Credit Limit', yaxis_title='Count of Customers',
                           legend_title="Attrition Flag")
        st.plotly_chart(hist_fig, use_container_width=True)


    elif attribute == 'Total Revolving Balance' and 'Total_Revolving_Bal' in df.columns:
        revolving_balance_selection = st.sidebar.slider('Select Total Revolving Balance Range:', min_value=int(df['Total_Revolving_Bal'].min()), max_value=int(df['Total_Revolving_Bal'].max()), value=(int(df['Total_Revolving_Bal'].min()), int(df['Total_Revolving_Bal'].max())))
        df_filtered = df[(df['Total_Revolving_Bal'] >= revolving_balance_selection[0]) & (df['Total_Revolving_Bal'] <= revolving_balance_selection[1])]
        st.title("Dashboard Customer Churn by Total Revolving Balance")
    
        hist_fig = px.histogram(df_filtered, x='Total_Revolving_Bal', color='Attrition_Flag',
                            labels={'Total_Revolving_Bal': 'Total Revolving Balance', 'Attrition_Flag': 'Attrition Flag'},
                            title='Customer Churn by Total Revolving Balance',
                            barmode='stack',
                            nbins=int(df['Total_Revolving_Bal'].max() - df['Total_Revolving_Bal'].min())
                           )
        hist_fig.update_layout(xaxis_title='Total Revolving Balance', yaxis_title='Count of Customers',
                           legend_title="Attrition Flag")
        st.plotly_chart(hist_fig, use_container_width=True)


    elif attribute == 'Total Transaction Amount' and 'Total_Trans_Amt' in df.columns:
        transaction_amount_selection = st.sidebar.slider('Select Total Transaction Amount Range:', min_value=int(df['Total_Trans_Amt'].min()), max_value=int(df['Total_Trans_Amt'].max()), value=(int(df['Total_Trans_Amt'].min()), int(df['Total_Trans_Amt'].max())))
        df_filtered = df[(df['Total_Trans_Amt'] >= transaction_amount_selection[0]) & (df['Total_Trans_Amt'] <= transaction_amount_selection[1])]
        st.title("Dashboard Customer Churn by Total Transaction Amount")
    
        hist_fig = px.histogram(df_filtered, x='Total_Trans_Amt', color='Attrition_Flag',
                            labels={'Total_Trans_Amt': 'Total Transaction Amount', 'Attrition_Flag': 'Attrition Flag'},
                            title='Customer Churn by Total Transaction Amount',
                            barmode='stack',
                            nbins=int(df['Total_Trans_Amt'].max() - df['Total_Trans_Amt'].min())
                           )
        hist_fig.update_layout(xaxis_title='Total Transaction Amount', yaxis_title='Count of Customers',
                           legend_title="Attrition Flag")
        st.plotly_chart(hist_fig, use_container_width=True)

    elif attribute == 'Total Transaction Count' and 'Total_Trans_Ct' in df.columns:
        transaction_count_selection = st.sidebar.slider('Select Total Transaction Count Range:', min_value=int(df['Total_Trans_Ct'].min()), max_value=int(df['Total_Trans_Ct'].max()), value=(int(df['Total_Trans_Ct'].min()), int(df['Total_Trans_Ct'].max())))
        df_filtered = df[(df['Total_Trans_Ct'] >= transaction_count_selection[0]) & (df['Total_Trans_Ct'] <= transaction_count_selection[1])]
        st.title("Dashboard Customer Churn by Total Transaction Count")
    
        hist_fig = px.histogram(df_filtered, x='Total_Trans_Ct', color='Attrition_Flag',
                            labels={'Total_Trans_Ct': 'Total Transaction Count', 'Attrition_Flag': 'Attrition Flag'},
                            title='Customer Churn by Total Transaction Count',
                            barmode='stack',
                            nbins=int(df['Total_Trans_Ct'].max() - df['Total_Trans_Ct'].min())
                           )
        hist_fig.update_layout(xaxis_title='Total Transaction Count', yaxis_title='Count of Customers',
                           legend_title="Attrition Flag")
        st.plotly_chart(hist_fig, use_container_width=True)

   elif attribute == 'Average Utilization Ratio' and 'Avg_Utilization_Ratio' in df.columns:
        utilization_ratio_selection = st.sidebar.slider('Select Average Utilization Ratio Range:', min_value=float(df['Avg_Utilization_Ratio'].min()), max_value=float(df['Avg_Utilization_Ratio'].max()), value=(float(df['Avg_Utilization_Ratio'].min()), float(df['Avg_Utilization_Ratio'].max())))
        df_filtered = df[(df['Avg_Utilization_Ratio'] >= utilization_ratio_selection[0]) & (df['Avg_Utilization_Ratio'] <= utilization_ratio_selection[1])]
        st.title("Dashboard Customer Churn by Average Utilization Ratio")
    
        hist_fig = px.histogram(df_filtered, x='Avg_Utilization_Ratio', color='Attrition_Flag',
                            labels={'Avg_Utilization_Ratio': 'Average Utilization Ratio', 'Attrition_Flag': 'Attrition Flag'},
                            title='Customer Churn by Average Utilization Ratio',
                            barmode='stack',
                            nbins=int(df['Avg_Utilization_Ratio'].max() - df['Avg_Utilization_Ratio'].min())
                            )
        hist_fig.update_layout(xaxis_title='Average Utilization Ratio', yaxis_title='Count of Customers',
                            legend_title="Attrition Flag")
        st.plotly_chart(hist_fig, use_container_width=True)


    else:
        st.title("Customer Churn Dashboard")
        st.write("Please select a valid attribute to visualize.")
else:
    st.title("Customer Churn Dashboard")
    st.write("Please select an attribute to visualize.")



