🧺 Market Basket Analysis Web App

Overview

This web app performs Market Basket Analysis on transaction data to automatically discover product associations and generate actionable insights. Built with Streamlit, it allows you to upload a dataset, run Apriori analysis, and view interactive dashboards — just like a data scientist would.

Key outcomes:
	•	Discover frequent itemsets and association rules.
	•	Visualize item frequencies, co-occurrence heatmaps, and rule networks.
	•	Estimate cross-sell opportunities and potential Average Order Value (AOV) uplift (e.g., +12% in test scenarios).

⸻

✨ Features
	•	📂 CSV Upload – Supports:
	•	Row-per-item format (TransactionID, Product).
	•	Basket format (each row = basket).
	•	⚙️ Automated Preprocessing – Detects dataset format, cleans data, and one-hot encodes items.
	•	🔍 Apriori Algorithm – Generates frequent itemsets and association rules using mlxtend.
	•	📊 Dashboards & Visuals:
	•	Top item frequencies (bar chart).
	•	Item co-occurrence heatmap.
	•	Association rules table with support, confidence, lift.
	•	Interactive network graph of rules.
	•	💡 Business Insight – AOV uplift simulator to quantify potential impact of cross-sell recommendations.
	•	📥 Export – Download discovered rules as CSV for offline use.
🛠️ Tech Stack
	•	Python
	•	Streamlit – Web app framework
	•	pandas – Data processing
	•	mlxtend – Apriori & association rules
	•	plotly / seaborn / matplotlib / networkx – Visualization

⸻

📊 Example Output
	•	Frequent itemsets: e.g., {Milk, Bread} with 8% support.
	•	Association rules: Milk → Bread with 60% confidence and lift of 1.3.
	•	Visualizations:
	•	Top product frequencies
	•	Co-occurrence heatmap
	•	Rule network graph
	•	Business insight:
Targeted cross-selling recommendations based on discovered associations increased simulated AOV by 12%.

📈 Next Steps
	•	Add FP-Growth for faster large-scale analysis.
	•	Integrate time filtering for recent transactions.
	•	Segment rules by customer type/region.
	•	Deploy to Streamlit Cloud / Heroku / GCP / AWS for public use.

⸻

👤 Author

Eric Kumi
Data Analyst | Business Analytics | Machine Learning | Data Visualization
