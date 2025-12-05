from trend_tool import analyze_trend_from_excel

analyze_trend_from_excel("my_data.xlsx") # subtitute your excel file name


analyze_trend_from_excel(
    xlsx_path="experiment_results.xlsx",
    sheet_name="Sheet1",
    start_date="2023-10-01",     
    end_date="2023-10-15",
    alpha=0.05
)
