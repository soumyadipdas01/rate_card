import streamlit as st
from io import StringIO
import pandas as pd
from main import geminiAI
from pandasai import SmartDataframe
import plotly.express as px
st.set_page_config(layout="wide")


MESSAGES = "messages"
DFS = "df"
USER = "user"
BOT = "assistant"
COUNTER = "counter"



st.header("Rate card analysis")

col1, br, col2 = st.columns([0.3, 0.1, 0.6])


# Initialize chat history
if MESSAGES not in st.session_state:
    st.session_state[MESSAGES] = []

# initialize dfs state
if DFS not in st.session_state:
    st.session_state[DFS] = pd.DataFrame()


# initialize dfs state
if COUNTER not in st.session_state:
    st.session_state[COUNTER] = 100000



def plot_graph(dataframe):
    cats = dataframe.select_dtypes(include=['object']).columns
    conts = dataframe.select_dtypes(include=['number']).columns
    groups = []
    for i in cats:
        for j in conts:
            groups.append(px.bar(dataframe, x=i, y=j))
    return groups


def upload_handler():
    uploaded_files = st.session_state["uploader"]
    for i in uploaded_files:
        data = StringIO(i.getvalue().decode("utf-8")).read()
        l = [x.split(",") for x in data.split("\n")]
        header = l[0]
        body = l[1:]
        df = pd.DataFrame(columns=header, data = body)
        df['supplier'] = i.name.split('.')[0]
        df = df.drop(["Monthly Billable Hours\r"], axis=1)
        st.session_state[DFS] = pd.concat([st.session_state[DFS], df], axis=0)
    st.session_state[DFS] = st.session_state[DFS].replace("NA","$0.00")
    #st.session_state[DFS].to_csv("out.csv")


def query_handler():
    with cont:
        prompt = st.session_state["ask"]
        # Display user message in chat message container
        st.chat_message(USER).markdown(prompt)
        # Add user message to chat history
        st.session_state[MESSAGES].append({"role": USER, "content": prompt})

        llm = geminiAI()
        pandas_ai = SmartDataframe(st.session_state[DFS], config={"llm": llm, "verbose": False})
        response = pandas_ai.chat(prompt, "dataframe")

        # Display assistant response in chat message container
        with st.chat_message(BOT):
            if isinstance(response, pd.DataFrame):
                if (response.shape[1] == 1):
                    response.reset_index(inplace=True)
                counter = 0
                for graphs in plot_graph(response):
                    counter += 1
                    st.plotly_chart(graphs,key=f"{len(st.session_state[MESSAGES])}_{str(counter)}")

                #st.plotly_chart(px.bar(response, x=list(response.columns)[0], y=list(response.columns)[1]))

            st.write(response)    
        # Add assistant response to chat history
        st.session_state[MESSAGES].append({"role": BOT, "content": response})


def faq_handler(args):
    print(args)
    st.session_state["ask"] = args



with col1:
    st.file_uploader("upload csv files", type=['csv'], accept_multiple_files=True, on_change = upload_handler, key = "uploader")
    #st.button("sort the suppliers based on average hourly rate", key = "btn1", on_click=faq_handler, args = ["sort the suppliers based on average hourly rate"])



with col2:
    # Display chat messages from history on app rerun
    if len(st.session_state[MESSAGES]) > 0:
        for message in st.session_state[MESSAGES]:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if isinstance(message["content"], pd.DataFrame):
                    for graphs in plot_graph(message["content"]):
                        st.session_state[COUNTER] += 1
                        st.plotly_chart(graphs, key = f"{len(st.session_state[MESSAGES])}_{str(st.session_state[COUNTER])}")
                    #st.plotly_chart(px.bar(message["content"], x=list(message["content"].columns)[0], y=list(message["content"].columns)[1]))

    cont = st.container()

    # React to user input
    st.chat_input("Ask anything...", key = "ask", on_submit=query_handler)