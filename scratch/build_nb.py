import json
import os

with open("3_final_model/enhanced-v7.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

markdown_cell = {
    'cell_type': 'markdown',
    'metadata': {},
    'source': [
        '# ============================================================\n',
        '# NEW FEATURE: Symbol Analysis & June Recomposition Tracker\n',
        '# ============================================================\n',
        'This section allows you to enter a target symbol (e.g., OGDC, HUBC, BOP) to view its historical baseline performance and how its returns, volume, and KSE-30 index weight act before, during, and after the June index recomposition each year.'
    ]
}

code_cell = {
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': [
        'TARGET_SYMBOL = "OGDC" # Change this to any KSE-30 ticker\n',
        '\n',
        'print(f"\\n=== Analysis for {TARGET_SYMBOL} ===")\n',
        'symbol_data = co_raw[co_raw["company"] == TARGET_SYMBOL].copy()\n',
        'if len(symbol_data) == 0:\n',
        '    print(f"Symbol {TARGET_SYMBOL} not found in the dataset.")\n',
        'else:\n',
        '    symbol_data = symbol_data.sort_values("date")\n',
        '    \n',
        '    # 1. Historical Summary\n',
        '    start_price = symbol_data["price"].iloc[0]\n',
        '    end_price = symbol_data["price"].iloc[-1]\n',
        '    cum_return = (end_price - start_price) / start_price * 100\n',
        '    avg_volume = symbol_data["volume"].mean()\n',
        '    \n',
        '    print(f"Data available from: {symbol_data[\'date\'].min().date()} to {symbol_data[\'date\'].max().date()}")\n',
        '    print(f"Start Price: {start_price:.2f} | End Price: {end_price:.2f}")\n',
        '    print(f"Cumulative Return: {cum_return:.2f}%")\n',
        '    print(f"Average Daily Volume: {avg_volume:,.0f}")\n',
        '    \n',
        '    # 2. June Recomposition Analysis\n',
        '    # We want to examine May, June, July for every year.\n',
        '    symbol_data["month_num"] = symbol_data["date"].dt.month\n',
        '    symbol_data["year"] = symbol_data["date"].dt.year\n',
        '    symbol_data["return_daily"] = symbol_data["price"].pct_change()\n',
        '    \n',
        '    recomp_data = []\n',
        '    \n',
        '    for year in symbol_data["year"].unique():\n',
        '        # Get data for May(5), June(6), July(7)\n',
        '        window = symbol_data[(symbol_data["year"] == year) & (symbol_data["month_num"].isin([5, 6, 7]))]\n',
        '        if not window.empty:\n',
        '            # Aggregate by month\n',
        '            monthly_stats = window.groupby("month_num").agg(\n',
        '                avg_wt = ("idx_wt", "mean"),\n',
        '                total_ret = ("return_daily", "sum"),\n',
        '                avg_vol = ("volume", "mean")\n',
        '            )\n',
        '            # Flatten to a single row for the year\n',
        '            row = {"year": year}\n',
        '            for m, m_name in zip([5,6,7], ["May", "June", "July"]):\n',
        '                if m in monthly_stats.index:\n',
        '                    row[f"{m_name}_wt"] = monthly_stats.loc[m, "avg_wt"]\n',
        '                    row[f"{m_name}_ret"] = monthly_stats.loc[m, "total_ret"] * 100\n',
        '                    row[f"{m_name}_vol"] = monthly_stats.loc[m, "avg_vol"]\n',
        '                else:\n',
        '                    row[f"{m_name}_wt"] = np.nan\n',
        '                    row[f"{m_name}_ret"] = np.nan\n',
        '                    row[f"{m_name}_vol"] = np.nan\n',
        '            recomp_data.append(row)\n',
        '            \n',
        '    recomp_df = pd.DataFrame(recomp_data)\n',
        '    print("\\n[Recomposition Performance (May -> June -> July)]")\n',
        '    display(recomp_df.round(2))\n',
        '    \n',
        '    # Plotting Recomposition Effect\n',
        '    recomp_df_clean = recomp_df.dropna()\n',
        '    if not recomp_df_clean.empty:\n',
        '        fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n',
        '        years = recomp_df_clean["year"].astype(int).astype(str)\n',
        '        \n',
        '        # 1. Weights\n',
        '        axes[0].plot(years, recomp_df_clean["May_wt"], marker="o", label="May (Pre)")\n',
        '        axes[0].plot(years, recomp_df_clean["June_wt"], marker="s", label="June (Recomp)")\n',
        '        axes[0].plot(years, recomp_df_clean["July_wt"], marker="^", label="July (Post)")\n',
        '        axes[0].set_title(f"{TARGET_SYMBOL}: Index Weight by Year")\n',
        '        axes[0].set_ylabel("Avg Index Weight (%)")\n',
        '        axes[0].legend()\n',
        '        \n',
        '        # 2. Returns\n',
        '        width = 0.25\n',
        '        x = np.arange(len(years))\n',
        '        axes[1].bar(x - width, recomp_df_clean["May_ret"], width=width, label="May")\n',
        '        axes[1].bar(x, recomp_df_clean["June_ret"], width=width, label="June")\n',
        '        axes[1].bar(x + width, recomp_df_clean["July_ret"], width=width, label="July")\n',
        '        axes[1].set_title(f"{TARGET_SYMBOL}: Total Monthly Return (%)")\n',
        '        axes[1].set_xticks(x)\n',
        '        axes[1].set_xticklabels(years)\n',
        '        axes[1].axhline(0, color="black", linewidth=0.8)\n',
        '        axes[1].legend()\n',
        '        \n',
        '        # 3. Volume\n',
        '        axes[2].plot(years, recomp_df_clean["May_vol"], marker="o", label="May")\n',
        '        axes[2].plot(years, recomp_df_clean["June_vol"], marker="s", label="June")\n',
        '        axes[2].plot(years, recomp_df_clean["July_vol"], marker="^", label="July")\n',
        '        axes[2].set_title(f"{TARGET_SYMBOL}: Average Volume")\n',
        '        axes[2].legend()\n',
        '        \n',
        '        plt.tight_layout()\n',
        '        if SAVE_PLOTS:\n',
        '            plt.savefig(os.path.join(OUT_DIR, f"10_{TARGET_SYMBOL}_recomposition.png"))\n',
        '            print(f"Saved {TARGET_SYMBOL} recomposition plot.")\n',
        '        plt.show()\n'
    ]
}

nb['cells'].append(markdown_cell)
nb['cells'].append(code_cell)

with open("3_final_model/enhanced-final.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f)

print('Successfully created enhanced-final.ipynb')
