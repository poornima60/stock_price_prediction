# pip install streamlit fbprophet yfinance plotly
import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import finnhub
import base64


@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

  
with st.sidebar.container():
    st.write("This is the sidebar content.")
                
                    # Initialize Finnhub client
    finnhub_client = finnhub.Client(api_key="cigib2hr01qsmg0cltu0cigib2hr01qsmg0cltug")

    # Fetch stock news
    news = finnhub_client.general_news('general', min_id=0)

    def display_news_with_images(news_list):
        for item in news_list:
            st.markdown(f"### {item['headline']}")
            st.image(item['image'], use_column_width=True)
            st.markdown(f"*Source:* {item['source']}")
            st.markdown(f"*Summary:* {item['summary']}")
            st.markdown(f"*URL:* {item['url']}")
            st.markdown("---")

    def main():
        st.title("Stock News and Information")
        
        # Create a sidebar to display the stock news with images
        st.sidebar.title("Stock News")
        display_news_with_images(news)
        
        # Main content in the main area of the web page
        st.write("Welcome to our stock information page!")
        # Add other components or data visualization as needed
            
    if __name__ == "__main__":
        main()

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
selected_stock = st.selectbox('Select dataset for prediction', stocks)
n_years = st.slider('Years of prediction:', 1, 4)
period = n_years*365
data_load_state = st.text('Load data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')
# printing stock change value

st.subheader('Raw data')
st.write(data.tail())
# Plot raw data

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(
        x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(
        title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
plot_raw_data()
# Predict forecast with Prophet.
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)
# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())
st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)
st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)