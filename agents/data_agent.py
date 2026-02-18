"""
Data Analysis Agent - Powerful data processing and analysis tools
"""

import asyncio
import pandas as pd
import numpy as np
import json
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

from utils.config import Config
from utils.logger import setup_logger

class DataAnalysisAgent:
    """Agent for data analysis, visualization, and insights"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logger("data_analysis_agent")
        
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data analysis task"""
        task_type = params.get("task_type", "analyze")
        
        if task_type == "analyze":
            return await self._analyze_data(params)
        elif task_type == "visualize":
            return await self._create_visualization(params)
        elif task_type == "clean":
            return await self._clean_data(params)
        elif task_type == "statistics":
            return await self._calculate_statistics(params)
        elif task_type == "correlation":
            return await self._find_correlations(params)
        else:
            return {"error": f"Unknown task type: {task_type}"}
    
    async def _analyze_data(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive data analysis"""
        data_input = params.get("data", "")
        data_format = params.get("format", "csv")
        
        if not data_input:
            return {"error": "No data provided"}
        
        try:
            # Load data
            if data_format == "csv":
                df = pd.read_csv(StringIO(data_input))
            elif data_format == "json":
                df = pd.read_json(StringIO(data_input))
            else:
                return {"error": f"Unsupported format: {data_format}"}
            
            # Basic analysis
            analysis = {
                "shape": df.shape,
                "columns": list(df.columns),
                "data_types": df.dtypes.to_dict(),
                "missing_values": df.isnull().sum().to_dict(),
                "basic_stats": df.describe().to_dict(),
                "memory_usage": df.memory_usage(deep=True).sum()
            }
            
            # Data quality assessment
            quality_score = self._assess_data_quality(df)
            analysis["quality_score"] = quality_score
            
            # Recommendations
            analysis["recommendations"] = self._generate_recommendations(df, quality_score)
            
            return {
                "analysis": analysis,
                "summary": f"Analyzed dataset with {df.shape[0]} rows and {df.shape[1]} columns",
                "status": "success"
            }
            
        except Exception as e:
            return {"error": f"Data analysis failed: {str(e)}"}
    
    async def _create_visualization(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create data visualizations"""
        data_input = params.get("data", "")
        chart_type = params.get("chart_type", "histogram")
        column = params.get("column", "")
        
        if not data_input:
            return {"error": "No data provided"}
        
        try:
            # Load data
            df = pd.read_csv(StringIO(data_input))
            
            # Create visualization
            plt.figure(figsize=(10, 6))
            
            if chart_type == "histogram" and column in df.columns:
                plt.hist(df[column].dropna(), bins=30)
                plt.title(f'Distribution of {column}')
                plt.xlabel(column)
                plt.ylabel('Frequency')
            elif chart_type == "scatter" and len(params.get("columns", [])) >= 2:
                cols = params["columns"][:2]
                plt.scatter(df[cols[0]], df[cols[1]], alpha=0.6)
                plt.title(f'{cols[0]} vs {cols[1]}')
                plt.xlabel(cols[0])
                plt.ylabel(cols[1])
            elif chart_type == "correlation":
                # Correlation heatmap
                numeric_df = df.select_dtypes(include=[np.number])
                corr_matrix = numeric_df.corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
                plt.title('Correlation Matrix')
            else:
                # Default: histogram of first numeric column
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    plt.hist(df[numeric_cols[0]].dropna(), bins=30)
                    plt.title(f'Distribution of {numeric_cols[0]}')
            
            # Save plot to string
            import io
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', bbox_inches='tight')
            img_buffer.seek(0)
            plt.close()
            
            # Convert to base64 for easy handling
            import base64
            img_str = base64.b64encode(img_buffer.getvalue()).decode()
            
            return {
                "chart_type": chart_type,
                "image_data": img_str,
                "status": "success"
            }
            
        except Exception as e:
            return {"error": f"Visualization failed: {str(e)}"}
    
    async def _clean_data(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and preprocess data"""
        data_input = params.get("data", "")
        operations = params.get("operations", ["remove_duplicates", "handle_missing"])
        
        if not data_input:
            return {"error": "No data provided"}
        
        try:
            # Load data
            df = pd.read_csv(StringIO(data_input))
            original_shape = df.shape
            
            changes = []
            
            # Remove duplicates
            if "remove_duplicates" in operations:
                before = len(df)
                df = df.drop_duplicates()
                if before > len(df):
                    changes.append(f"Removed {before - len(df)} duplicate rows")
            
            # Handle missing values
            if "handle_missing" in operations:
                for column in df.columns:
                    missing_count = df[column].isnull().sum()
                    if missing_count > 0:
                        if df[column].dtype in ['int64', 'float64']:
                            # Fill numeric with median
                            median_val = df[column].median()
                            df[column].fillna(median_val, inplace=True)
                            changes.append(f"Filled {missing_count} missing values in {column} with median ({median_val})")
                        else:
                            # Fill categorical with mode
                            mode_val = df[column].mode()[0] if not df[column].mode().empty else "Unknown"
                            df[column].fillna(mode_val, inplace=True)
                            changes.append(f"Filled {missing_count} missing values in {column} with mode ({mode_val})")
            
            # Remove outliers (simple method)
            if "remove_outliers" in operations:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                outliers_removed = 0
                for col in numeric_cols:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                    outliers_removed += len(outliers)
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                
                if outliers_removed > 0:
                    changes.append(f"Removed {outliers_removed} outlier rows")
            
            # Convert back to CSV
            cleaned_data = df.to_csv(index=False)
            
            return {
                "original_shape": original_shape,
                "cleaned_shape": df.shape,
                "changes_made": changes,
                "cleaned_data": cleaned_data,
                "status": "success"
            }
            
        except Exception as e:
            return {"error": f"Data cleaning failed: {str(e)}"}
    
    async def _calculate_statistics(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate detailed statistics"""
        data_input = params.get("data", "")
        columns = params.get("columns", [])
        
        if not data_input:
            return {"error": "No data provided"}
        
        try:
            # Load data
            df = pd.read_csv(StringIO(data_input))
            
            # Filter columns if specified
            if columns:
                df = df[columns]
            
            # Calculate statistics
            stats = {}
            
            # Numeric columns
            numeric_df = df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                stats["numeric"] = {
                    "count": numeric_df.count().to_dict(),
                    "mean": numeric_df.mean().to_dict(),
                    "std": numeric_df.std().to_dict(),
                    "min": numeric_df.min().to_dict(),
                    "max": numeric_df.max().to_dict(),
                    "median": numeric_df.median().to_dict(),
                    "skewness": numeric_df.skew().to_dict(),
                    "kurtosis": numeric_df.kurtosis().to_dict()
                }
            
            # Categorical columns
            categorical_df = df.select_dtypes(include=['object'])
            if not categorical_df.empty:
                stats["categorical"] = {
                    "unique_values": categorical_df.nunique().to_dict(),
                    "most_frequent": {col: categorical_df[col].mode()[0] if not categorical_df[col].mode().empty else None 
                                    for col in categorical_df.columns},
                    "value_counts": {col: categorical_df[col].value_counts().head(5).to_dict() 
                                   for col in categorical_df.columns}
                }
            
            return {
                "statistics": stats,
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "status": "success"
            }
            
        except Exception as e:
            return {"error": f"Statistics calculation failed: {str(e)}"}
    
    async def _find_correlations(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Find and analyze correlations"""
        data_input = params.get("data", "")
        method = params.get("method", "pearson")  # pearson, spearman, kendall
        threshold = params.get("threshold", 0.5)
        
        if not data_input:
            return {"error": "No data provided"}
        
        try:
            # Load data
            df = pd.read_csv(StringIO(data_input))
            
            # Get numeric columns only
            numeric_df = df.select_dtypes(include=[np.number])
            
            if numeric_df.empty:
                return {"error": "No numeric columns found for correlation analysis"}
            
            # Calculate correlation matrix
            corr_matrix = numeric_df.corr(method=method)
            
            # Find strong correlations
            strong_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) >= threshold:
                        strong_correlations.append({
                            "variables": [corr_matrix.columns[i], corr_matrix.columns[j]],
                            "correlation": float(corr_value),
                            "strength": "strong" if abs(corr_value) > 0.7 else "moderate"
                        })
            
            # Sort by absolute correlation value
            strong_correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
            
            return {
                "correlation_method": method,
                "threshold": threshold,
                "strong_correlations": strong_correlations,
                "correlation_matrix": corr_matrix.to_dict(),
                "status": "success"
            }
            
        except Exception as e:
            return {"error": f"Correlation analysis failed: {str(e)}"}
    
    def _assess_data_quality(self, df: pd.DataFrame) -> float:
        """Assess overall data quality (0.0 to 1.0)"""
        score = 1.0
        
        # Missing values penalty
        missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        score -= missing_ratio * 0.3
        
        # Duplicate rows penalty
        duplicate_ratio = df.duplicated().sum() / len(df)
        score -= duplicate_ratio * 0.2
        
        # Data type consistency (simple check)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Check for extreme values that might be errors
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[col] < Q1 - 3*IQR) | (df[col] > Q3 + 3*IQR)]
                outlier_ratio = len(outliers) / len(df)
                score -= outlier_ratio * 0.1
        
        return max(0.0, min(1.0, score))
    
    def _generate_recommendations(self, df: pd.DataFrame, quality_score: float) -> List[str]:
        """Generate data analysis recommendations"""
        recommendations = []
        
        # Quality-based recommendations
        if quality_score < 0.5:
            recommendations.append("Data quality needs significant improvement")
        
        # Missing values
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            recommendations.append(f"Handle missing values in columns: {', '.join(missing_cols[:3])}")
        
        # Duplicates
        if df.duplicated().sum() > 0:
            recommendations.append(f"Remove {df.duplicated().sum()} duplicate rows")
        
        # Data types
        object_cols = df.select_dtypes(include=['object']).columns
        if len(object_cols) > df.shape[1] * 0.5:
            recommendations.append("Consider converting some text columns to categorical types")
        
        # Add generic recommendations
        if not recommendations:
            recommendations.extend([
                "Data looks good for analysis",
                "Consider exploring relationships between variables",
                "Create visualizations to better understand patterns"
            ])
        
        return recommendations