# 필요한 라이브러리 임포트
import streamlit as st
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
import pandas as pd
from datetime import datetime
from io import StringIO
import requests
import warnings
from bs4 import BeautifulSoup

# 경고 메시지 무시 설정
warnings.filterwarnings("ignore")

# 나눔 폰트 설정
font_path = os.path.join(os.path.dirname(__file__), 'NanumGothic.ttf')
# font_path = 'C:/Windows/Fonts/NanumGothic.ttf'  # 폰트 파일 경로
font_prop = fm.FontProperties(fname=font_path)  # 폰트 속성 설정
plt.rcParams['font.family'] = 'NanumGothic'  # 그래프의 기본 폰트 설정
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# Streamlit 설정
st.set_page_config(layout="centered")  # 레이아웃을 중앙 정렬로 설정
st.markdown("<h1 style='font-size:32px; text-align: center;'>ETF 분석 및 ChatGPT 투자 조언</h1>", unsafe_allow_html=True)

st.markdown("### ETF List 종목 확인")

# 인터넷이 안되는 경우
# file_path='active_etf_df.xlsx'
# active_etf_df = pd.read_excel(file_path)

# Alpha Vantage API 키 입력
api_key = st.secrets["AV_API_KEY"]

# Alpha Vantage Symbol 검색 API URL
url = 'https://www.alphavantage.co/query'

# 요청 파라미터 설정
params = {
    'function': 'LISTING_STATUS',
    'apikey': api_key
}

response = requests.get(url, params=params)

# API 요청 및 데이터 수집
if response.status_code == 200:
    data = StringIO(response.text)
    df = pd.read_csv(data)
else:
    st.write("API 요청 실패:", response.status_code)

# ETF 데이터 필터링
etf_df = df[df['assetType'] == 'ETF']
active_etf_df = etf_df[etf_df['status'] == 'Active']

# ETF 선택
etf_symbol = st.selectbox("ETF 종목을 선택하세요:", active_etf_df['symbol'])

url = f'https://www.alphavantage.co/query?function=ETF_PROFILE&symbol={etf_symbol }&apikey={api_key}'
r = requests.get(url)
data = r.json()

# 배당 데이터를 수집하는 함수
def get_dividend_data(etf_symbol):
    etf = yf.Ticker(etf_symbol)
    dividends = etf.dividends
    return dividends

# 배당 주기를 계산하는 함수
def calculate_dividend_frequency(dividends):
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

# ETF 티커 데이터 가져오기
ticker = yf.Ticker(etf_symbol)
etf_data = ticker.history(period="max", interval='1mo')
etf_data.index = etf_data.index.tz_localize(None)
etf_data['YM'] = etf_data.index.to_period('M').astype(str).str.replace('-', '')
df_lists = etf_data.groupby('YM')['Close'].last().reset_index().rename(columns={'YM': 'YM', 'Close': 'INDEX'})

# 그림 크기와 글자 크기 조정
plt.figure(figsize=(8, 4))
plt.plot(df_lists['YM'], df_lists['INDEX'], marker='o', linestyle='-', color='b')
plt.xticks(df_lists['YM'][::36], rotation=45, fontsize=8)  # X축 라벨 간격과 폰트 크기 조정
plt.title('ETF 종가 추이', font=font_prop, fontsize=16)
plt.xlabel("Year-Month", fontsize=10)
plt.ylabel("Closing Price", fontsize=10)
st.pyplot(plt)

# 수익률 계산 함수 정의
def calculate_returns(df, periods):
    for period in periods:
        col_name = f'return_{period}m'
        df[col_name] = df['INDEX'] / df['INDEX'].shift(period) - 1
    return df

# 특정 시점으로부터의 수익률 계산
periods = [1, 2, 3, 6, 12, 24, 36, 48, 60]
df_with_returns = calculate_returns(df_lists, periods)

# 수익률 그래프 생성
plt.figure(figsize=(8, 4))
for period in periods:
    plt.plot(
        df_with_returns['YM'],
        df_with_returns[f'return_{period}m'],
        label=f'Return {period} months',
        marker='o',  # 선 위에 점 표시
        markersize=4  # 점 크기 조절
    )

plt.xlabel('Year-Month', fontsize=10)
plt.ylabel('Return', fontsize=10)
plt.title('Returns over Different Periods (1 to 5 years)', fontsize=14)
plt.legend(fontsize=8)
plt.grid(True)
plt.xticks(df_with_returns['YM'][::36], rotation=45, fontsize=8)
st.pyplot(plt)

# 연간화 수익률 계산 함수 정의
def calculate_annualized_returns(df, periods):
    for period in periods:
        col_name = f'return_{period}m'
        annualized_col_name = f'annualized_return_{period}m'
        df[col_name] = df['INDEX'] / df['INDEX'].shift(period) - 1
        df[annualized_col_name] = (1 + df[col_name]) ** (12 / period) - 1
    return df

df_with_annualized_returns = calculate_annualized_returns(df_lists, periods)

# 각 기간에 대한 통계치 계산
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

# Streamlit에서 테이블로 표시
st.write("Annualized Returns Table")
st.dataframe(annualized_returns_df)

# Yahoo Finance에서 ETF 정보 수집
etf = yf.Ticker(etf_symbol)
etf_info_yf = {
    "ETF 이름": etf.info.get("longName", "정보 없음"),
    "운용사": etf.info.get("fundFamily", "정보 없음"),
    "운용 보수(Expense Ratio)": f"{etf.info.get('expenseRatio', 0) * 100:.2f}%" if etf.info.get("expenseRatio") else "정보 없음",
    "배당 수익률": f"{etf.info.get('dividendYield', 0) * 100:.2f}%" if etf.info.get("dividendYield") else "정보 없음",
    "배당 주기": calculate_dividend_frequency(get_dividend_data(etf_symbol)),
    "총 자산": f"{etf.info.get('totalAssets', '정보 없음'):,}" if etf.info.get("totalAssets") else "정보 없음",
    "카테고리": etf.info.get("category", "정보 없음"),
    "설립 연도": etf.info.get("fundInceptionDate", "정보 없음")
}

# 최신 데이터에 따른 상위 보유 종목 및 섹터 분포 추가
top_10_data = list(data['holdings'][:10])  # 슬라이싱 후 명시적으로 리스트로 변환
filtered_top_10 = [{ item['description']: item['weight']} for item in top_10_data]
etf_info_yf["상위 보유 종목"] = filtered_top_10
sector_info = [{ item['sector']: item['weight']} for item in data['sectors']]
etf_info_yf["섹터 분포"] = sector_info

# 최근 5년간 월별 종가 데이터 수집 (5개만 표시)
etf_prices = etf.history(period="5y", interval="1mo")["Close"]
price_text = etf_prices.tail(5).to_string(index=False)

# ETF 기본 정보 텍스트 구성
etf_info_text = "\n".join([f"{key}: {value}" for key, value in etf_info_yf.items() if key not in ['상위 보유 종목', '섹터 분포']])

# 상위 보유 종목 텍스트 구성
holdings_text = "\n".join([f"{list(item.keys())[0]}: {list(item.values())[0]}" for item in etf_info_yf["상위 보유 종목"]])

# 섹터 분포 텍스트 구성
sector_text = "\n".join([f"{list(item.keys())[0]}: {list(item.values())[0]}" for item in etf_info_yf["섹터 분포"]])

# 블로그 정보를 가져오는 함수
def get_blog_content(web_url):
    blog_url = web_url.replace("blog", "m.blog")  # 모바일 버전으로 변환
    response = requests.get(blog_url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        blog_title = soup.find("meta", property="og:title")['content']
        blog_content = soup.find("div", class_="se-main-container").get_text("\n", strip=True)  # '\n'을 사용해 가독성 높임
        return blog_title, blog_content
    else:
        return None, f"HTTP 요청 실패: {response.status_code}"

# 각 블로그 항목별로 내용 가져오기

# 1. 매매 가능 계좌
account_url = "https://blog.naver.com/jung2598123/223613727928"
account_title, account_content = get_blog_content(account_url)

# 2. 매매 시간
time_url = "http://blog.naver.com/jung2598123/223613727928"
time_title, time_content = get_blog_content(time_url)

# 3. 세금
tax_url = "http://blog.naver.com/jung2598123/223613736933"
tax_title, tax_content = get_blog_content(tax_url)

# 4. 보수수수료
fee_url = "https://blog.naver.com/jung2598123/223613746967"
fee_title, fee_content = get_blog_content(fee_url)

# 5. 통합증거금 서비스
integrated_margin_url = "http://blog.naver.com/jung2598123/223613746967"
integrated_margin_title, integrated_margin_content = get_blog_content(integrated_margin_url)

# 6. 소수점 매매
fractional_trading_url = "https://blog.naver.com/jung2598123/223613721912"
fractional_trading_title, fractional_trading_content = get_blog_content(fractional_trading_url)

# 7. 증거금 비율
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
세개의 문단으로 간단하게 존댓말로 설명해 주세요. 
"""

print("\nGPT 프롬프트:\n", prompt)

# OpenAI API Key 설정 (환경 변수에서 불러오기)
cgpt_api_key = st.secrets["OPENAI_API_KEY"]  # 환경 변수에 API Key 저장 필요

# ChatGPT에게 질문을 요청하는 함수
def ask_chatgpt(prompt):
    headers = {
        "Authorization": f"Bearer {cgpt_api_key }",
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

chatgpt_response = ask_chatgpt(prompt)
st.subheader("ChatGPT 응답")
st.write(chatgpt_response)

# TTS 라이브러리 임포트
from gtts import gTTS
import base64
import os

# TTS: 텍스트를 음성으로 변환하여 Streamlit 페이지에 표시
def tts(response_text):
    filename = "output.mp3"
    tts = gTTS(text=response_text, lang="ko")
    tts.save(filename)

    # mp3 파일을 base64로 인코딩하여 Streamlit에 표시
    with open(filename, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        audio_html = f"""
            <audio autoplay="True" controls>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)

    # 사용이 끝난 파일 삭제
    os.remove(filename)

# ChatGPT 응답을 음성으로 재생
tts(chatgpt_response)
