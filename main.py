import sys  
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QLineEdit, 
                             QCheckBox, QPushButton, QMessageBox)
from PyQt5.QtCore import Qt
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Plotting functions
def close_plot(stocks, df_stock, ax=None): #繪製收盤價圖函數
    if ax is None or df_stock is None or df_stock.empty: 
        return #判斷子圖有無繪製，是否有股票dataframe，是否有股票資料，若無結束函數
    stock = stocks[0] #提取股票列表中的第一個股票
    if ('Close', stock) in df_stock.columns: #判斷收盤價是否在資料中
        df_stock['Close', stock].plot(ax=ax, label=stock)
        #繪製收盤價折線，標籤設為股票代碼
    ax.legend() #子圖上新增圖例，顯示股票代碼
    ax.set_xlabel('Date') #x軸標籤為Date
    ax.set_ylabel('Closing Price') #y軸標籤為Closing Price
    ax.set_title('Closing Prices') #標題設為Closing Price

def means_plot(stocks, df_stock, ax=None): #繪製均線圖函數
    if ax is None or df_stock is None or df_stock.empty:
        return #判斷子圖有無繪製，是否有股票dataframe，是否有股票資料，若無結束函數
    stock = stocks[0] #提取股票列表中的第一個股票
    if ('Close', stock) in df_stock.columns:  #判斷收盤價是否在資料中
        close_prices = df_stock['Close', stock] #索取收盤價資料
        ma5 = close_prices.rolling(window=5).mean() #計算5日均線(週線)
        ma20 = close_prices.rolling(window=20).mean() #計算20均線(月線)
        ma60 = close_prices.rolling(window=60).mean() #計算60均線(季線)
        close_prices.plot(ax=ax, label=f'{stock} Close', alpha=0.5)
        #繪製收盤價折線標籤為"股票代碼 Close"，透明度50%(避免影響觀看均線)
        ma5.plot(ax=ax, label=f'{stock} 5-day MA', linestyle='--')
        #繪製5日均線折線標籤為"股票代碼 5-day MA"折線形式'--'
        ma20.plot(ax=ax, label=f'{stock} 20-day MA', linestyle='-.')
        #繪製20日均線折線標籤為"股票代碼 20-day MA"折線形式'-.'
        ma60.plot(ax=ax, label=f'{stock} 60-day MA', linestyle=':')
        #繪製60日均線折線標籤為"股票代碼 60-day MA"折線形式':'
    ax.legend() #子圖上新增圖例，顯示股票代碼
    ax.set_xlabel('Date') #x軸標籤為Date
    ax.set_ylabel('Price') #y軸標籤為Price
    ax.set_title('Moving Averages') #標題設為Moving Averages

def rsi_plot(stocks, df_stock, ax=None): #繪製rsi值函數
    def calculate_rsi(data, periods=14): #計算rsi函數
        delta = data.diff() #計算每日價格變化
        gain = delta.where(delta > 0, 0) #尋找漲幅資料
        loss = -delta.where(delta < 0, 0) #尋找跌幅資料
        avg_gain = gain.rolling(window=periods).mean() #計算漲幅平均
        avg_loss = loss.rolling(window=periods).mean() #計算跌幅平均
        rs = avg_gain / avg_loss #計算rs
        rsi = 100.0 - (100.0 / (1.0 + rs)) #計算rsi
        return rsi #回傳rsi值

    if ax is None or df_stock is None or df_stock.empty:
        return #判斷子圖有無繪製，是否有股票dataframe，是否有股票資料，若無結束函數
    stock = stocks[0] #提取股票列表中的第一個股票
    if ('Close', stock) in df_stock.columns:
        close_prices = df_stock['Close', stock] #索取收盤價資料
        rsi = calculate_rsi(close_prices) #索取rsi資料
        rsi.plot(ax=ax, label=stock) #繪製rsi折線標籤為股票代碼
    ax.axhline(y=70, color='r', linestyle='--') #rsi=70繪製一條紅色水平線
    ax.axhline(y=30, color='g', linestyle='--') #rsi=30繪製一條綠色水平線
    ax.set_ylim(0, 100) #y軸設置為0
    ax.legend() #子圖新增圖例，顯示股票代碼
    ax.set_xlabel('Date') #x軸標籤為Date
    ax.set_ylabel('RSI') #y軸標籤為RSI
    ax.set_title('Relative Strength Index (RSI)') #標題設為Relative Strength Index (RSI)

def macd_plot(stocks, df_stock, ax=None): #繪製macd圖函數
    def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
        #計算macd函數
        ema_fast = data.ewm(span=fast_period, adjust=False).mean()
        #計算12ema
        ema_slow = data.ewm(span=slow_period, adjust=False).mean()
        #計算26ema
        macd = ema_fast - ema_slow #計算macd
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        #計算9訊號線
        return macd, signal #回傳macd，signal值

    if ax is None or df_stock is None or df_stock.empty:
        return #判斷子圖有無繪製，是否有股票dataframe，是否有股票資料，若無結束函數
    stock = stocks[0] #提取股票列表中的第一個股票
    if ('Close', stock) in df_stock.columns:
        close_prices = df_stock['Close', stock] #索取收盤價資料
        macd, signal = calculate_macd(close_prices) #索取macd，signal值
        macd.plot(ax=ax, label=f'{stock} MACD', alpha=0.7) 
        #繪製macd折線，透明度70%
        signal.plot(ax=ax, label=f'{stock} Signal', linestyle='--', alpha=0.7)
        #繪製signal線，透明度70%
    ax.legend() #子圖新增圖例，顯示股票代碼
    ax.set_xlabel('Date') #x軸標籤為Date
    ax.set_ylabel('MACD') #y軸標籤為MACD
    ax.set_title('Moving Average Convergence Divergence (MACD)') 
    #標題設為Moving Average Convergence Divergence (MACD)


def candlestick_plot(stocks, df_stock, ax=None): #繪製蠟燭圖函數
    if ax is None or df_stock is None or df_stock.empty:
        print("Error: df_stock is None or empty")
        return #判斷子圖有無繪製，是否有股票dataframe，是否有股票資料，若無結束函數
    if len(stocks) > 1:
        raise ValueError("K線圖一次只能顯示一支股票！請選擇單一股票。")
    #判斷股票數量
    stock = stocks[0] #提取股票列表中的第一個股票
    required_cols = [('Open', stock), ('High', stock), ('Low', stock), ('Close', stock)]
    #索取開盤價，最高價，最低價，收盤價
    if not all(col in df_stock.columns for col in required_cols):
        raise ValueError(f"股票 {stock} 的數據不完整，缺少以下欄位之一：Open, High, Low, Close")
    #判斷資料是否有所欠缺
    stock_data = pd.DataFrame({
        'Open': df_stock['Open', stock],
        'High': df_stock['High', stock],
        'Low': df_stock['Low', stock],
        'Close': df_stock['Close', stock]
    }, index=df_stock.index)
    #創建新的dataframe包含k線所需資料
    print(f"Generating K-line chart for {stock} with {len(stock_data)} data points")
    #列印調試訊息，顯示股票代碼與資料點數
    width = 0.6 #k柱寬度0.6
    up = stock_data['Close'] >= stock_data['Open'] 
    down = stock_data['Open'] > stock_data['Close']
    #判斷漲跌
    ax.bar(stock_data.index[up], stock_data['Close'][up] - stock_data['Open'][up], 
           width, bottom=stock_data['Open'][up], color='r')
    ax.bar(stock_data.index[down], stock_data['Open'][down] - stock_data['Close'][down], 
           width, bottom=stock_data['Close'][down], color='g')
    #若為漲繪製紅k柱，若為跌繪製綠k柱
    for i in range(len(stock_data)):
        ax.vlines(stock_data.index[i], stock_data['Low'].iloc[i], stock_data['High'].iloc[i], 
                  color='black', linewidth=1)
    #繪製上下影線
    ax.set_xlabel('Date') #x軸標籤為Date
    ax.set_ylabel('Price') #y軸標籤為Price
    ax.set_title(f'Candlestick Chart - {stock}') #標題設為標題設為Candlestick Chart - 股票代碼

# PyQt5 GUI
class StockAnalyzerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stock Analysis")
        self.setGeometry(100, 100, 500, 400)

        widget = QWidget()
        self.setCentralWidget(widget)
        layout = QVBoxLayout()
        widget.setLayout(layout)

        layout.addWidget(QLabel("輸入股票代碼(例如 2330.TW):"))
        self.stock_entry = QLineEdit()
        layout.addWidget(self.stock_entry)

        layout.addWidget(QLabel("輸入開始日期(年 月 日，例如 2023 1 1):"))
        self.start_date_entry = QLineEdit()
        layout.addWidget(self.start_date_entry)

        layout.addWidget(QLabel("輸入結束日期(年 月 日，留空為當前日期):"))
        self.end_date_entry = QLineEdit()
        layout.addWidget(self.end_date_entry)

        layout.addWidget(QLabel("選擇要生成的圖表:"))
        self.chart_vars = {
            "收盤價": QCheckBox("收盤價"),
            "日均線": QCheckBox("日均線"),
            "MACD": QCheckBox("MACD"),
            "RSI": QCheckBox("RSI"),
            "K線": QCheckBox("K線")
        }
        for checkbox in self.chart_vars.values():
            layout.addWidget(checkbox)

        generate_button = QPushButton("生成圖表")
        generate_button.clicked.connect(self.plot_charts)
        layout.addWidget(generate_button)

    def plot_charts(self):
        stocks_input = self.stock_entry.text().strip()
        start_input = self.start_date_entry.text().strip()
        end_input = self.end_date_entry.text().strip()

        try:
            # Input validation
            if not stocks_input:
                raise ValueError("請輸入股票代碼！")
            stocks = stocks_input.split()

            if not start_input:
                raise ValueError("請輸入開始日期！")
            year, month, day = map(int, start_input.split())
            if not (1 <= month <= 12 and 1 <= day <= 31):
                raise ValueError("開始日期無效！")
            start = pd.Timestamp(year, month, day)

            if end_input:
                eyear, emonth, eday = map(int, end_input.split())
                if not (1 <= emonth <= 12 and 1 <= eday <= 31):
                    raise ValueError("結束日期無效！")
                end = pd.Timestamp(eyear, emonth, eday)
            else:
                end = pd.Timestamp.now()

            if start >= end:
                raise ValueError("開始日期必須早於結束日期！")

            QApplication.setOverrideCursor(Qt.WaitCursor)

            # Download stock data
            df_stock = yf.download(stocks, start=start, end=end, progress=False)
            if df_stock is None or df_stock.empty:
                raise ValueError("未獲取到數據，請檢查股票代碼或日期範圍！")

            print(f"Downloaded data shape: {df_stock.shape}")
            print(f"Columns: {df_stock.columns.tolist()}")

            # Define plotting functions
            plot_options = {
                "收盤價": close_plot,
                "日均線": means_plot,
                "MACD": macd_plot,
                "RSI": rsi_plot,
                "K線": candlestick_plot
            }
            
            # Get selected plots
            selected_plots = [name for name, cb in self.chart_vars.items() if cb.isChecked()]
            if not selected_plots:
                raise ValueError("請至少選擇一個圖表類型！")

            if "K線" in selected_plots and len(stocks) > 1:
                raise ValueError("K線圖一次只能顯示一支股票！請輸入單一股票代碼。")

            # Display each plot in its own figure
            for plot_name in selected_plots:
                print(f"Creating individual display figure for {plot_name}")
                fig, ax = plt.subplots(figsize=(10, 6))
                plot_options[plot_name](stocks, df_stock, ax)
                plt.title(plot_name)
                plt.tight_layout()
                plt.show()  # Displays immediately; may overlap on some systems

            # Save plots: Combined into one file (vertical layout)
            fig_save, axes_save = plt.subplots(len(selected_plots), 1, figsize=(12, 6 * len(selected_plots)))

            # Ensure axes_save is iterable (handle single plot case)
            if len(selected_plots) == 1:
                axes_save = [axes_save]

            # Plot each chart in the combined figure for saving
            for i, plot_name in enumerate(selected_plots):
                print(f"Plotting {plot_name} on saved subplot (row {i})")
                plot_options[plot_name](stocks, df_stock, axes_save[i])
                axes_save[i].set_title(plot_name)  # Add title to each subplot

            plt.tight_layout()
            plt.savefig("static/combined_plot.png")
            plt.close(fig_save)  # Free memory

            QMessageBox.information(self, "成功", "圖表已生成並顯示！")

        except ValueError as e:
            QMessageBox.critical(self, "錯誤", str(e))
        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"發生未知錯誤：{str(e)}")
            print(f"詳細錯誤訊息：{str(e)}")
        finally:
            QApplication.restoreOverrideCursor()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StockAnalyzerGUI()
    window.show()
    sys.exit(app.exec_())
