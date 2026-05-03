import pandas as pd
import matplotlib.pyplot as plt
import io
import requests

# 한글 폰트 설정 (환경에 따라 수정이 필요할 수 있습니다)
plt.rcParams['font.family'] = 'Malgun Gothic' # Windows 기준
plt.rcParams['axes.unicode_minus'] = False

def load_data(url):
    """1. 데이터 수집 함수 (자동 인코딩 처리)"""
    response = requests.get(url)
    
    response.encoding = response.apparent_encoding 
    
    csv_data = io.StringIO(response.text)
    df = pd.read_csv(csv_data)
    return df
def process_timeseries(df):
    """2. 시계열 핸들링 및 트렌드 분석 함수"""
    # 날짜 변환
    df['일시'] = pd.to_datetime(df['일시'])
    # 7일 이동 평균(Moving Average) 생성
    df['평균기온_7일_이동평균'] = df['평균기온'].rolling(window=7).mean()
    # 인덱스 설정
    df.set_index('일시', inplace=True)
    return df

def min_max_scaling(df, columns):
    """3. 데이터 정규화 함수 (Min-Max Scaling)"""
    df_scaled = df.copy()
    for col in columns:
        min_val = df[col].min()
        max_val = df[col].max()
        df_scaled[f'{col}_정규화'] = (df[col] - min_val) / (max_val - min_val)
    return df_scaled

def restructure_data(df):
    """4. 데이터 재구조화 (pivot_table, groupby) 함수"""
    # 계절 분류 함수
    def get_season(month):
        if month in [3, 4, 5]: return '봄'
        elif month in [6, 7, 8]: return '여름'
        elif month in [9, 10, 11]: return '가을'
        else: return '겨울'

    # 분석용 임시 컬럼 생성
    temp_df = df.reset_index()
    temp_df['연도'] = temp_df['일시'].dt.year
    temp_df['월'] = temp_df['일시'].dt.month
    temp_df['계절'] = temp_df['월'].apply(get_season)

    # 4-1. 계절별 통계 (groupby)
    seasonal_stats = temp_df.groupby('계절')[['평균기온', '최대풍속', '평균풍속']].mean()

    # 4-2. 연도별/월별 평균기온 피벗 테이블
    yearly_pivot = temp_df.pivot_table(values='평균기온', index='연도', columns='월', aggfunc='mean')
    
    return seasonal_stats, yearly_pivot, temp_df

def plot_weather_trends(df):
    """5. 시각화 그래프 생성 함수"""
    plt.figure(figsize=(15, 7))
    
    # 원본 데이터와 7일 이동 평균선 시각화
    plt.plot(df.index, df['평균기온'], label='일간 평균기온', alpha=0.3, color='gray')
    plt.plot(df.index, df['평균기온_7일_이동평균'], label='7일 이동 평균선', color='red', linewidth=2)
    
    plt.title('기상 데이터 기온 추세 분석 (7일 이동 평균)')
    plt.xlabel('날짜')
    plt.ylabel('기온 (℃)')
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.show()

def export_to_excel(df_pre, seasonal_stats, yearly_pivot, filename):
    """6. 엑셀 내보내기 (다중 시트) 함수"""
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        df_pre.to_excel(writer, sheet_name='전처리_데이터')
        seasonal_stats.to_excel(writer, sheet_name='계절별_통계')
        yearly_pivot.to_excel(writer, sheet_name='연도별_피벗테이블')
    print(f"보고서 저장 완료: {filename}")

def main():
    """핵심 파이프라인 제어 (Main 함수)"""
    url = "https://github.com/dongupak/DataML/raw/main/csv/weather.csv"
    
    # 파이프라인 단계별 실행
    print("1. 데이터 로드 중...")
    raw_df = load_data(url)
    
    print("2. 시계열 처리 및 정규화 수행 중...")
    ts_df = process_timeseries(raw_df)
    normalized_df = min_max_scaling(ts_df, ['평균기온', '최대풍속'])
    
    print("3. 통계 분석 및 데이터 재구조화 중...")
    seasonal_stats, yearly_pivot, full_processed_df = restructure_data(normalized_df)
    
    print("4. 시각화 결과 생성 중...")
    plot_weather_trends(normalized_df)
    
    print("5. 최종 보고서(Excel) 생성 중...")
    export_to_excel(full_processed_df, seasonal_stats, yearly_pivot, "weather_analysis_report.xlsx")

if __name__ == "__main__":
    main()