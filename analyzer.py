import pandas as pd
import plotly.express as px

class DataAnalyzer:
    def __init__(self, prediction_df: pd.DataFrame):
        """
        Initializes the analyzer with the data containing predictions.
        """
        self.df = prediction_df

    def get_summary_statistics(self) -> dict:
        """
        Calculates key summary statistics.
        """
        if self.df.empty:
            return {}

        stats = {
            'total_appointments': len(self.df),
            'high_risk_count': len(self.df[self.df['risk_level'] == 'High']),
            'medium_risk_count': len(self.df[self.df['risk_level'] == 'Medium']),
            'low_risk_count': len(self.df[self.df['risk_level'] == 'Low']),
            'predicted_noshow_rate': self.df['predicted_outcome'].mean()
        }
        return stats

    def create_risk_distribution_plot(self):
        """
        Generates a histogram of risk scores.
        """
        if self.df.empty:
            return None
        fig = px.histogram(self.df, x='risk_score', nbins=30, title='Distribution of No-Show Risk Scores')
        return fig
        
    def get_high_risk_report(self) -> pd.DataFrame:
        """
        Filters and returns a DataFrame of high-risk appointments.
        """
        return self.df[self.df['risk_level'] == 'High'].sort_values(by='risk_score', ascending=False)