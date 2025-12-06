# 필요한 라이브러리 임포트
import streamlit as st
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from io import StringIO
import requests
import warnings
from bs4 import BeautifulSoup
import os 
import tempfile
import base64
from gtts import gTTS

# 경고 메시지 무시 설정
warnings.filterwarnings("ignore")

# ========== Yahoo Finance 직접 호출 함수들 ==========

def get_etf_history(symbol, period="max", interval="1mo"):
    """Yahoo Finance API 직접 호출 - 가격 데이터"""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {
        'range': period,
        'interval': interval
    }
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 200:
            data = response.json()
            result = data['chart']['result'][0]
            timestamps = result['timestamp']
            ohlcv = result['indicators']['quote'][0]
            
            df = pd.DataFrame({
                'Open': ohlcv['open'],
                'High': ohlcv['high'],
                'Low': ohlcv['low'],
                'Close': ohlcv['close'],
                'Volume': ohlcv['volume']
            }, index=pd.to_datetime(timestamps, unit='s'))
            
            return df
    except Exception as e:
        st.error(f"가격 데이터 수집 에러: {e}")
    
    return pd.DataFrame()


def get_dividend_data(symbol):
    """Yahoo Finance API 직접 호출 - 배당 데이터"""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {
        'range': '10y',
        'interval': '1mo',
        'events': 'div'
    }
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 200:
            data = response.json()
            result = data['chart']['result'][0]
            
            if 'events' in result and 'dividends' in result['events']:
                divs = result['events']['dividends']
                div_data = [(pd.to_datetime(int(k), unit='s'), v['amount']) for k, v in divs.items()]
                df = pd.DataFrame(div_data, columns=['Date', 'Dividend'])
                df = df.set_index('Date').sort_index()
                return df['Dividend']
    except:
        pass
    
    return pd.Series(dtype=float)


def get_etf_info(symbol):
    """Yahoo Finance API 직접 호출 - ETF 정보"""
    url = f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{symbol}"
    params = {
        'modules': 'summaryProfile,summaryDetail,defaultKeyStatistics,fundProfile,price'
    }
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 200:
            data = response.json()
            result = data['quoteSummary']['result'][0]
            
            info = {}
            
            # price 모듈에서 정보 추출
            if 'price' in result:
                price = result['price']
                info['longName'] = price.get('longName', '정보 없음')
            
            # fundProfile 모듈에서 정보 추출
            if 'fundProfile' in result:
                fund = result['fundProfile']
                info['fundFamily'] = fund.get('family', '정보 없음')
                
                if 'feesExpensesInvestment' in fund:
                    fees = fund['feesExpensesInvestment']
                    info['expenseRatio'] = fees.get('annualReportExpenseRatio', {}).get('raw', None)
            
            # summaryDetail 모듈에서 정보 추출
            if 'summaryDetail' in result:
                summary = result['summaryDetail']
                info['dividendYield'] = summary.get('dividendYield', {}).get('raw', None)
                info['totalAssets'] = summary.get('totalAssets', {}).get('raw', None)
            
            # defaultKeyStatistics 모듈에서 정보 추출
            if 'defaultKeyStatistics' in result:
                stats = result['defaultKeyStatistics']
                info['category'] = stats.get('category', '정보 없음')
                info['fundInceptionDate'] = stats.get('fundInceptionDate', {}).get('fmt', '정보 없음')
            
            return info
    except Exception as e:
        st.warning(f"ETF 정보 수집 에러: {e}")
    
    return {}


def calculate_dividend_frequency(dividends):
    """배당 주기를 계산하는 함수"""
    if len(dividends) < 2:
        return "배당 데이터가 부족합니다."

    date_diffs = dividends.index.to_series().diff().dt.days.dropna()
    avg_diff = date_diffs.mean()

    if avg_diff <= 32:
        return "월별 배당"
    elif avg_diff <= 95:
        return "분기별 배당"
    elif avg_diff <= 365:
        return "연간 배당"
    else:
        return "불규칙한 배당"


@st.cache_data(ttl=3600)
def get_blog_content(web_url):
    """블로그 정보를 가져오는 함수"""
    blog_url = web_url.replace("blog", "m.blog")
    try:
        response = requests.get(blog_url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            blog_title = soup.find("meta", property="og:title")['content']
            blog_content = soup.find("div", class_="se-main-container").get_text("\n", strip=True)
            return blog_title, blog_content
        else:
            return None, f"HTTP 요청 실패: {response.status_code}"
    except Exception as e:
        return None, f"블로그 크롤링 에러: {e}"


def tts(response_text):
    """TTS: 텍스트를 음성으로 변환하여 Streamlit 페이지에 표시"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts_obj = gTTS(text=response_text, lang="ko")
            tts_obj.save(fp.name)
            
            with open(fp.name, "rb") as f:
                data = f.read()
                b64 = base64.b64encode(data).decode()
                audio_html = f"""
                    <audio autoplay="True" controls>
                    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                    </audio>
                """
                st.markdown(audio_html, unsafe_allow_html=True)
            
            os.unlink(fp.name)
    except Exception as e:
        st.error(f"음성 변환 에러: {e}")


# ========== 메인 앱 시작 ==========

# 나눔 폰트 설정
font_path = os.path.join(os.path.dirname(__file__), 'NanumGothic.ttf')
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

# Streamlit 설정
st.set_page_config(layout="centered")
st.markdown("<h1 style='font-size:32px; text-align: center;'>ETF 분석 및 ChatGPT 투자 조언</h1>", unsafe_allow_html=True)
st.markdown("### ETF List 종목 확인")

# Alpha Vantage API 키
api_key = st.secrets["AV_API_KEY"]

# Alpha Vantage Symbol 검색 API URL
url = 'https://www.alphavantage.co/query'
params = {
    'function': 'LISTING_STATUS',
    'apikey': api_key
}

response = requests.get(url, params=params)

if response.status_code == 200:
    data = StringIO(response.text)
    df = pd.read_csv(data)
else:
    st.error(f"API 요청 실패: {response.status_code}")
    st.stop()

# API 요청 및 데이터 수집
if response.status_code == 200:
    data = StringIO(response.text)
    df = pd.read_csv(data)
    
    # 디버깅: 데이터 확인
    st.write("--- API 응답 디버깅 ---")
    st.write(f"컬럼: {df.columns.tolist()}")
    st.write(f"데이터 shape: {df.shape}")
    st.write(df.head())
else:
    st.error(f"API 요청 실패: {response.status_code}")
    st.stop()

# ETF 데이터 필터링
etf_df = df[df['assetType'] == 'ETF']
active_etf_df = etf_df[etf_df['status'] == 'Active']

# ETF 선택
etf_symbol = st.selectbox("ETF 종목을 선택하세요:", active_etf_df['symbol'])

# Alpha Vantage ETF 프로필
url = f'https://www.alphavantage.co/query?function=ETF_PROFILE&symbol={etf_symbol}&apikey={api_key}'
r = requests.get(url)
av_data = r.json()

# ETF 가격 데이터 가져오기 (Yahoo Finance 직접 호출)
etf_data = get_etf_history(etf_symbol, period="max", interval="1mo")

if etf_data.empty:
    st.warning(f"⚠️ {etf_symbol}의 데이터를 가져올 수 없습니다. 다른 ETF를 선택해 주세요.")
    st.stop()

etf_data['YM'] = etf_data.index.to_period('M').astype(str).str.replace('-', '')
df_lists = etf_data.groupby('YM')['Close'].last().reset_index().rename(columns={'YM': 'YM', 'Close': 'INDEX'})

# 종가 추이 그래프
plt.figure(figsize=(8, 4))
plt.plot(df_lists['YM'], df_lists['INDEX'], marker='o', linestyle='-', color='b')
plt.xticks(df_lists['YM'][::36], rotation=45, fontsize=8)
plt.title('ETF 종가 추이', font=font_prop, fontsize=16)
plt.xlabel("Year-Month", fontsize=10)
plt.ylabel("Closing Price", fontsize=10)
st.pyplot(plt)
plt.close()

# 수익률 계산 함수
def calculate_returns(df, periods):
    for period in periods:
        col_name = f'return_{period}m'
        df[col_name] = df['INDEX'] / df['INDEX'].shift(period) - 1
    return df

# 특정 시점으로부터의 수익률 계산
periods = [1, 2, 3, 6, 12, 24, 36, 48, 60]
df_with_returns = calculate_returns(df_lists.copy(), periods)

# 수익률 그래프
plt.figure(figsize=(8, 4))
for period in periods:
    plt.plot(
        df_with_returns['YM'],
        df_with_returns[f'return_{period}m'],
        label=f'Return {period} months',
        marker='o',
        markersize=4
    )

plt.xlabel('Year-Month', fontsize=10)
plt.ylabel('Return', fontsize=10)
plt.title('Returns over Different Periods (1 to 5 years)', fontsize=14)
plt.legend(fontsize=8)
plt.grid(True)
plt.xticks(df_with_returns['YM'][::36], rotation=45, fontsize=8)
st.pyplot(plt)
plt.close()

# 연간화 수익률 계산 함수
def calculate_annualized_returns(df, periods):
    for period in periods:
        col_name = f'return_{period}m'
        annualized_col_name = f'annualized_return_{period}m'
        df[col_name] = df['INDEX'] / df['INDEX'].shift(period) - 1
        df[annualized_col_name] = (1 + df[col_name]) ** (12 / period) - 1
    return df

df_with_annualized_returns = calculate_annualized_returns(df_lists.copy(), periods)

# 통계치 계산
annualized_returns_stats = []
for period in periods:
    col_name = f'annualized_return_{period}m'
    period_data = df_with_annualized_returns[col_name].dropna()
    average = round(period_data.mean(), 3)
    std_dev = round(period_data.std(), 3)
    count = len(period_data)
    negative_count = (period_data < 0).sum()
    non_annualized_average = round(df_with_annualized_returns[f'return_{period}m'].dropna().mean(), 3)

    annualized_returns_stats.append({
        'Period (months)': period,
        'Annualized Average Return': average,
        'Standard Deviation': std_dev,
        'Data Count': count,
        'Negative Count': negative_count,
        'Non-Annualized Average Return': non_annualized_average
    })

annualized_returns_df = pd.DataFrame(annualized_returns_stats)

st.write("Annualized Returns Table")
st.dataframe(annualized_returns_df)

# ETF 정보 수집 (Alpha Vantage 데이터 활용)
etf_info = get_etf_info(etf_symbol)
dividends = get_dividend_data(etf_symbol)

# Alpha Vantage 데이터에서 정보 추출
etf_name = av_data.get('name', etf_symbol)  # ETF 이름
expense_ratio = av_data.get('net_expense_ratio', None)
dividend_yield = av_data.get('dividend_yield', None)
net_assets = av_data.get('net_assets', None)
inception_date = av_data.get('inception_date', '정보 없음')

etf_info_yf = {
    "ETF 이름": etf_name if etf_name else etf_symbol,
    "운용사": av_data.get('asset_class', '정보 없음'),
    "운용 보수(Expense Ratio)": f"{float(expense_ratio) * 100:.2f}%" if expense_ratio else "정보 없음",
    "배당 수익률": f"{float(dividend_yield) * 100:.2f}%" if dividend_yield else "정보 없음",
    "배당 주기": calculate_dividend_frequency(dividends),
    "총 자산": f"${int(net_assets):,}" if net_assets else "정보 없음",
    "카테고리": av_data.get('asset_class', '정보 없음'),
    "설립 연도": inception_date
}
  
# Alpha Vantage에서 상위 보유 종목 및 섹터 분포
if 'holdings' in av_data and av_data['holdings']:
    top_10_data = list(av_data['holdings'][:10])
    filtered_top_10 = [{item['description']: item['weight']} for item in top_10_data]
    etf_info_yf["상위 보유 종목"] = filtered_top_10
else:
    etf_info_yf["상위 보유 종목"] = []

if 'sectors' in av_data and av_data['sectors']:
    sector_info = [{item['sector']: item['weight']} for item in av_data['sectors']]
    etf_info_yf["섹터 분포"] = sector_info
else:
    etf_info_yf["섹터 분포"] = []

# 최근 5년간 월별 종가 데이터
etf_5y_data = get_etf_history(etf_symbol, period="5y", interval="1mo")
if not etf_5y_data.empty:
    price_text = etf_5y_data["Close"].tail(5).to_string(index=False)
else:
    price_text = "데이터 없음"

# 텍스트 구성
etf_info_text = "\n".join([f"{key}: {value}" for key, value in etf_info_yf.items() if key not in ['상위 보유 종목', '섹터 분포']])

if etf_info_yf["상위 보유 종목"]:
    holdings_text = "\n".join([f"{list(item.keys())[0]}: {list(item.values())[0]}" for item in etf_info_yf["상위 보유 종목"]])
else:
    holdings_text = "정보 없음"

if etf_info_yf["섹터 분포"]:
    sector_text = "\n".join([f"{list(item.keys())[0]}: {list(item.values())[0]}" for item in etf_info_yf["섹터 분포"]])
else:
    sector_text = "정보 없음"

# 블로그 정보 가져오기
account_url = "https://blog.naver.com/jung2598123/223613727928"
account_title, account_content = get_blog_content(account_url)

time_url = "http://blog.naver.com/jung2598123/223613727928"
time_title, time_content = get_blog_content(time_url)

tax_url = "http://blog.naver.com/jung2598123/223613736933"
tax_title, tax_content = get_blog_content(tax_url)

fee_url = "https://blog.naver.com/jung2598123/223613746967"
fee_title, fee_content = get_blog_content(fee_url)

integrated_margin_url = "http://blog.naver.com/jung2598123/223613746967"
integrated_margin_title, integrated_margin_content = get_blog_content(integrated_margin_url)

fractional_trading_url = "https://blog.naver.com/jung2598123/223613721912"
fractional_trading_title, fractional_trading_content = get_blog_content(fractional_trading_url)

margin_ratio_url = "http://blog.naver.com/jung2598123/223613721912"
margin_ratio_title, margin_ratio_content = get_blog_content(margin_ratio_url)

# GPT 프롬프트 생성
prompt = f"""
당신은 금융 전문가입니다. 다음은 ETF {etf_info_yf['ETF 이름']} ({etf_symbol})에 대한 정보입니다.
이 ETF에 대해 초보 투자자도 이해할 수 있도록 간단하게 소개해 주세요.

- ETF 기본 정보:
{etf_info_text}

- 최근 5년간 월별 종가 데이터 (최신 5개):
{price_text}

- 상위 보유 종목:
{holdings_text}

- 섹터 분포:
{sector_text}

- 기간별 수익률: 
{annualized_returns_df}

- 매매방식 
{account_title}: {account_content}
{time_title}: {time_content}
{tax_title}: {tax_content}
{fee_title}: {fee_content}
{integrated_margin_title}: {integrated_margin_content}
{fractional_trading_title}: {fractional_trading_content}
{margin_ratio_title}: {margin_ratio_content}

위의 정보를 바탕으로, {etf_info_yf['ETF 이름']} ETF가 어떤 상품인지 기본정보, 상위보유종목, 섹터분포로 설명하고, 
기간별 수익률을 활용하여 투자 전략과 장단점을 그리고 마지막으로 매매방식을 아나운서의 대본 형태이며, 
세개의 문단으로 간단하게 존댓말로 설명해 주세요. 중요한 건 특정 증권사를 언급하면 안됩니다. 
"""

# OpenAI API Key
cgpt_api_key = st.secrets["OPENAI_API_KEY"]

# ChatGPT 함수
def ask_chatgpt(prompt):
    headers = {
        "Authorization": f"Bearer {cgpt_api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "gpt-4o-mini-2024-07-18",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 1000,
        "top_p": 0.9,
        "frequency_penalty": 0,
        "presence_penalty": 0
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)

    if response.status_code == 200:
        content = response.json()
        return content['choices'][0]['message']['content'].strip()
    else:
        error_message = response.json().get('error', {}).get('message', '알 수 없는 오류')
        return f"Error: {response.status_code}, {error_message}"

# ChatGPT 응답 생성
chatgpt_response = ask_chatgpt(prompt)
st.subheader("ChatGPT 응답")
st.write(chatgpt_response)

# ChatGPT 응답을 음성으로 재생
tts(chatgpt_response)







