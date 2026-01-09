"""
AI-Powered ML Algorithm Recommender - Main GUI Application
Advanced GUI for intelligent ML algorithm recommendations
"""

import customtkinter as ctk
import pandas as pd
import numpy as np
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from typing import Optional, List
import threading

from dataset_analyzer import DatasetAnalyzer
from algorithm_recommender import AlgorithmRecommender, AlgorithmRecommendation

# Configure appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class MLRecommenderApp(ctk.CTk):
    """Main application window for ML Algorithm Recommender"""
    
    def __init__(self):
        super().__init__()
        
        self.title("AI-Powered ML Algorithm Recommender")
        self.geometry("1400x900")
        self.minsize(1200, 800)
        
        self.dataset: Optional[pd.DataFrame] = None
        self.target_column: Optional[str] = None
        self.analysis_results: Optional[dict] = None
        self.recommendations: List[AlgorithmRecommendation] = []
        
        self.analyzer = DatasetAnalyzer()
        self.recommender = AlgorithmRecommender()
        
        self._create_widgets()
        self._setup_layout()
    
    def _create_widgets(self):
        """Create all GUI widgets with tabbed navigation"""
        # Main container
        self.main_container = ctk.CTkFrame(self)
        self.main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create tabview for main navigation
        self.tabview = ctk.CTkTabview(self.main_container, height=850)
        self.tabview.pack(fill="both", expand=True)
        
        # Create tabs
        self.tab_home = self.tabview.add("Home")
        self.tab_dataset = self.tabview.add("Dataset")
        self.tab_configure = self.tabview.add("Configure")
        self.tab_recommendations = self.tabview.add("Recommendations")
        self.tab_visualization = self.tabview.add("Visualization")
        self.tab_about = self.tabview.add("About")
        
        # Create content for each tab
        self._create_home_tab()
        self._create_dataset_tab()
        self._create_configure_tab()
        self._create_recommendations_tab()
        self._create_visualization_tab()
        self._create_about_tab()
        
        # Create content for each tab
        self._create_home_tab()
        self._create_dataset_tab()
        self._create_configure_tab()
        self._create_recommendations_tab()
        self._create_visualization_tab()
        self._create_about_tab()
    
    def _create_home_tab(self):
        """Create home/welcome tab"""
        # Welcome title
        welcome_title = ctk.CTkLabel(
            self.tab_home,
            text="≡ƒÜÇ AI-Powered ML Algorithm Recommender",
            font=ctk.CTkFont(size=32, weight="bold")
        )
        welcome_title.pack(pady=(50, 20))
        
        # Subtitle
        subtitle = ctk.CTkLabel(
            self.tab_home,
            text="Intelligent Algorithm Selection for Machine Learning",
            font=ctk.CTkFont(size=18),
            text_color="gray"
        )
        subtitle.pack(pady=(0, 40))
        
        # Features frame
        features_frame = ctk.CTkFrame(self.tab_home)
        features_frame.pack(fill="both", expand=True, padx=100, pady=20)
        
        features_text = """
        KEY FEATURES
        
        Comprehensive Analysis
        ΓÇó Analyzes 268+ dataset characteristics
        ΓÇó Data quality assessment
        ΓÇó Feature correlation and distribution analysis
        ΓÇó Automatic task type detection
        
        Intelligent Recommendations
        ΓÇó 60+ Machine Learning algorithms
        ΓÇó Multi-factor scoring system
        ΓÇó Confidence metrics
        ΓÇó Detailed reasoning for each recommendation
        
        ≡ƒôÜ Algorithm Categories
        ΓÇó Regression: 18 algorithms
        ΓÇó Classification: 15 algorithms
        ΓÇó Clustering: 10 algorithms
        ΓÇó Dimensionality Reduction: 5 algorithms
        
        ≡ƒöì What We Analyze
        ΓÇó Dataset size and complexity
        ΓÇó Feature types and distributions
        ΓÇó Missing values and data quality
        ΓÇó Class imbalance and target characteristics
        ΓÇó Feature correlations
        
        
        ≡ƒÜª GET STARTED
        
        1∩╕ÅΓâú Go to "Dataset" tab and load your CSV/Excel file
        2∩╕ÅΓâú Configure task type and target column in "Configure" tab
        3∩╕ÅΓâú Click "Analyze & Recommend" to get intelligent suggestions
        4∩╕ÅΓâú View recommendations and visualizations
        """
        
        features_label = ctk.CTkLabel(
            features_frame,
            text=features_text,
            font=ctk.CTkFont(size=14),
            justify="left"
        )
        features_label.pack(pady=20, padx=40, anchor="w")
        
        # Quick start button
        start_btn = ctk.CTkButton(
            self.tab_home,
            text="Load Dataset to Begin",
            command=lambda: self.tabview.set("Dataset"),
            height=50,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="#2B7A0B",
            hover_color="#1F5808"
        )
        start_btn.pack(pady=30)
    
    def _create_dataset_tab(self):
        """Create dataset loading tab"""
        # Title
        title_label = ctk.CTkLabel(
            self.tab_dataset,
            text="Dataset Management",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title_label.pack(pady=(20, 30))
        
        # Upload section
        upload_frame = ctk.CTkFrame(self.tab_dataset)
        upload_frame.pack(fill="x", padx=100, pady=20)
        
        upload_title = ctk.CTkLabel(
            upload_frame,
            text="Load Your Dataset",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        upload_title.pack(pady=(20, 10))
        
        upload_desc = ctk.CTkLabel(
            upload_frame,
            text="Upload a CSV or Excel file containing your dataset",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        upload_desc.pack(pady=(0, 20))
        
        self.upload_btn = ctk.CTkButton(
            upload_frame,
            text="Browse and Load Dataset",
            command=self._load_dataset,
            height=50,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="#1F6AA5",
            hover_color="#144870"
        )
        self.upload_btn.pack(pady=20, padx=50, fill="x")
        
        self.file_label = ctk.CTkLabel(
            upload_frame,
            text="No dataset loaded",
            font=ctk.CTkFont(size=14),
            text_color="gray"
        )
        self.file_label.pack(pady=(10, 20))
        
        # Dataset info section
        info_frame = ctk.CTkFrame(self.tab_dataset)
        info_frame.pack(fill="both", expand=True, padx=100, pady=20)
        
        info_title = ctk.CTkLabel(
            info_frame,
            text="Dataset Information",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        info_title.pack(pady=(20, 10))
        
        self.info_text = ctk.CTkTextbox(
            info_frame,
            height=300,
            font=ctk.CTkFont(size=12, family="Consolas"),
            wrap="word"
        )
        self.info_text.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        
        # Next button
        next_btn = ctk.CTkButton(
            self.tab_dataset,
            text="Next: Configure Analysis Γ₧í∩╕Å",
            command=lambda: self.tabview.set("Configure"),
            height=40,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        next_btn.pack(pady=20)
    
    def _create_configure_tab(self):
        """Create configuration tab"""
        # Title
        title_label = ctk.CTkLabel(
            self.tab_configure,
            text="Analysis Configuration",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title_label.pack(pady=(20, 30))
        
        # Configuration frame
        config_frame = ctk.CTkFrame(self.tab_configure)
        config_frame.pack(fill="both", expand=True, padx=150, pady=20)
        
        # Target column selection
        target_section = ctk.CTkFrame(config_frame)
        target_section.pack(fill="x", padx=40, pady=20)
        
        target_label = ctk.CTkLabel(
            target_section,
            text="Target Column (Optional)",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        target_label.pack(pady=(20, 10))
        
        target_desc = ctk.CTkLabel(
            target_section,
            text="Select the column you want to predict (or 'None' for unsupervised learning)",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        target_desc.pack(pady=(0, 15))
        
        self.target_dropdown = ctk.CTkComboBox(
            target_section,
            values=["None - No target (Unsupervised)"],
            command=self._on_target_selected,
            state="readonly",
            height=35,
            font=ctk.CTkFont(size=14)
        )
        self.target_dropdown.pack(fill="x", padx=50, pady=(0, 20))
        self.target_dropdown.set("None - No target (Unsupervised)")
        
        # Task type selection
        task_section = ctk.CTkFrame(config_frame)
        task_section.pack(fill="x", padx=40, pady=20)
        
        task_label = ctk.CTkLabel(
            task_section,
            text="Task Type",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        task_label.pack(pady=(20, 10))
        
        task_desc = ctk.CTkLabel(
            task_section,
            text="Select the type of machine learning task",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        task_desc.pack(pady=(0, 15))
        
        self.task_type_var = ctk.StringVar(value="auto")
        
        # Create task type radio buttons in a grid
        radio_frame = ctk.CTkFrame(task_section)
        radio_frame.pack(pady=10, padx=50)
        
        tasks = [
            ("Auto-detect", "auto"),
            ("Classification", "classification"),
            ("Regression", "regression"),
            ("Clustering", "clustering"),
            ("Dimensionality Reduction", "dimensionality_reduction"),
            ("Unsupervised (Clustering + DimRed)", "unsupervised")
        ]
        
        for i, (text, value) in enumerate(tasks):
            task_radio = ctk.CTkRadioButton(
                radio_frame,
                text=text,
                variable=self.task_type_var,
                value=value,
                font=ctk.CTkFont(size=14)
            )
            task_radio.grid(row=i//2, column=i%2, sticky="w", padx=30, pady=10)
        
        # Analyze button
        analyze_frame = ctk.CTkFrame(config_frame)
        analyze_frame.pack(fill="x", padx=40, pady=30)
        
        self.analyze_btn = ctk.CTkButton(
            analyze_frame,
            text="≡ƒöì Analyze & Recommend",
            command=self._analyze_and_recommend,
            height=60,
            font=ctk.CTkFont(size=18, weight="bold"),
            state="disabled",
            fg_color="#2B7A0B",
            hover_color="#1F5808"
        )
        self.analyze_btn.pack(fill="x", padx=50, pady=20)
        
        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(analyze_frame)
        self.progress_bar.pack(fill="x", padx=50, pady=10)
        self.progress_bar.set(0)
        self.progress_bar.pack_forget()
    
    def _create_recommendations_tab(self):
        """Create recommendations display tab"""
        # Title
        title_label = ctk.CTkLabel(
            self.tab_recommendations,
            text="Algorithm Recommendations",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title_label.pack(pady=(20, 10))
        
        # Info label
        self.rec_info_label = ctk.CTkLabel(
            self.tab_recommendations,
            text="Run analysis to see recommendations",
            font=ctk.CTkFont(size=14),
            text_color="gray"
        )
        self.rec_info_label.pack(pady=10)
        
        # Scrollable frame for recommendations
        self.rec_scroll_frame = ctk.CTkScrollableFrame(
            self.tab_recommendations,
            label_text=""
        )
        self.rec_scroll_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        self._show_recommendations_placeholder()
    
    def _create_visualization_tab(self):
        """Create visualization tab"""
        # Title
        title_label = ctk.CTkLabel(
            self.tab_visualization,
            text="Analysis Visualizations",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title_label.pack(pady=(20, 10))
        
        # Info label
        self.viz_info_label = ctk.CTkLabel(
            self.tab_visualization,
            text="Run analysis to see visualizations",
            font=ctk.CTkFont(size=14),
            text_color="gray"
        )
        self.viz_info_label.pack(pady=10)
        
        # Visualization canvas frame
        self.viz_canvas_frame = ctk.CTkFrame(self.tab_visualization)
        self.viz_canvas_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        self._show_visualization_placeholder()
    
    def _create_about_tab(self):
        """Create about tab"""
        # Title
        title_label = ctk.CTkLabel(
            self.tab_about,
            text="About ML Algorithm Recommender",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title_label.pack(pady=(30, 20))
        
        # About frame
        about_frame = ctk.CTkFrame(self.tab_about)
        about_frame.pack(fill="both", expand=True, padx=100, pady=20)
        
        about_text = """
        AI-Powered ML Algorithm Recommender v2.0
        
        
        ≡ƒôÜ ALGORITHM DATABASE
        
        This system includes 60+ machine learning algorithms:
        
        ΓÇó Regression Algorithms: 18
          Linear, Polynomial, Ridge, Lasso, Elastic Net, Bayesian, Decision Tree,
          Random Forest, Extra Trees, Gradient Boosting, AdaBoost, XGBoost,
          LightGBM, CatBoost, KNN, SVR, Neural Network
        
        ΓÇó Classification Algorithms: 15
          Logistic Regression, Ridge Classifier, Decision Tree, Random Forest,
          Extra Trees, Gradient Boosting, AdaBoost, XGBoost, LightGBM,
          CatBoost, KNN, SVM, Naive Bayes, Neural Network
        
        ΓÇó Clustering Algorithms: 10
          K-Means, Mini-Batch K-Means, DBSCAN, HDBSCAN, Agglomerative,
          Gaussian Mixture Model, Spectral Clustering, OPTICS
        
        ΓÇó Dimensionality Reduction: 5
          PCA, LDA, t-SNE, UMAP, Autoencoders
        
        
HOW IT WORKS
        
        1. Dataset Analysis
           Analyzes 268+ characteristics including size, features, data quality,
           correlations, distributions, and complexity metrics
        
        2. Intelligent Scoring
           Each algorithm is scored based on dataset characteristics using
           advanced heuristics and multi-factor analysis
        
        3. Ranked Recommendations
           Algorithms are ranked by relevance score with detailed reasoning,
           pros/cons, and hyperparameter suggestions
        
        
        ≡ƒôû DOCUMENTATION
        
        For detailed information about each algorithm, see:
        ΓÇó ALGORITHMS_REFERENCE.md - Complete algorithm guide
        ΓÇó CHANGELOG.md - Version history and updates
        ΓÇó README.md - Project overview and installation
        
        
        ≡ƒÆ╗ TECHNICAL DETAILS
        
        Built with:
        ΓÇó Python 3.8+
        ΓÇó CustomTkinter for modern GUI
        ΓÇó Pandas & NumPy for data analysis
        ΓÇó Scikit-learn for ML fundamentals
        ΓÇó Matplotlib & Seaborn for visualizations
        
        
        ≡ƒÄô USE CASES
        
        ΓÇó Academic research and project evaluation
        ΓÇó ML pipeline optimization
        ΓÇó Algorithm selection for production systems
        ΓÇó Educational purposes and ML learning
        ΓÇó Data science workflow improvement
        
        
        ≡ƒôº SUPPORT
        
        For questions, issues, or feature requests:
        ΓÇó Check the documentation files
        ΓÇó Review the algorithm reference guide
        ΓÇó Explore sample datasets
        """
        
        about_label = ctk.CTkLabel(
            about_frame,
            text=about_text,
            font=ctk.CTkFont(size=13),
            justify="left"
        )
        about_label.pack(pady=20, padx=40, anchor="w")
    
    def _setup_layout(self):
        """Setup initial layout"""
        # Set default tab to home
        self.tabview.set("Home")
    
    def _show_recommendations_placeholder(self):
        """Show placeholder in recommendations tab"""
        placeholder_text = """
        No Recommendations Yet
        
        To get algorithm recommendations:
        
        1. Load a dataset in the "Dataset" tab
        2. Configure your analysis in the "Configure" tab
        3. Click "Analyze & Recommend"
        
        The system will analyze your dataset and provide
        intelligent algorithm recommendations with detailed
        reasoning, pros/cons, and hyperparameter suggestions.
        """
        
        placeholder_label = ctk.CTkLabel(
            self.rec_scroll_frame,
            text=placeholder_text,
            font=ctk.CTkFont(size=16),
            justify="center"
        )
        placeholder_label.pack(pady=100, padx=50)
    
    def _show_visualization_placeholder(self):
        """Show placeholder in visualization tab"""
        placeholder_text = """
        No Visualizations Yet
        
        After running the analysis, this tab will display:
        
        ΓÇó Score comparison bar charts
        ΓÇó Confidence vs Score scatter plots
        ΓÇó Algorithm ranking visualizations
        
        Run the analysis to see your results visualized!
        """
        
        placeholder_label = ctk.CTkLabel(
            self.viz_canvas_frame,
            text=placeholder_text,
            font=ctk.CTkFont(size=16),
            justify="center"
        )
        placeholder_label.pack(expand=True)
    
    def _load_dataset(self):
        """Load dataset from file"""
        file_path = filedialog.askopenfilename(
            title="Select Dataset",
            filetypes=[
                ("CSV files", "*.csv"),
                ("Excel files", "*.xlsx *.xls"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            print("No file selected")
            return
        
        print(f"Attempting to load file: {file_path}")
        
        try:
            # Show loading
            self.file_label.configure(text="Loading...", text_color="yellow")
            self.update()
            
            # Load based on file extension
            print(f"File extension detected: {file_path.split('.')[-1]}")
            if file_path.endswith('.csv'):
                print("Loading as CSV...")
                self.dataset = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                print("Loading as Excel...")
                self.dataset = pd.read_excel(file_path)
            else:
                # Try CSV as default
                print("Loading as CSV (default)...")
                self.dataset = pd.read_csv(file_path)
            
            print(f"Dataset loaded: {len(self.dataset)} rows, {len(self.dataset.columns)} columns")
            print(f"Columns: {list(self.dataset.columns)[:5]}...")
            
            # Update UI on main thread
            self._update_ui_after_load()
            
        except Exception as e:
            import traceback
            error_msg = f"Failed to load dataset:\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            print(error_msg)
            messagebox.showerror("Error", error_msg)
            self.file_label.configure(text="Load failed", text_color="red")
    
    def _update_ui_after_load(self):
        """Update UI after dataset is loaded"""
        if self.dataset is None:
            return
        
        try:
            # Update file label to success
            self.file_label.configure(text="Dataset loaded successfully", text_color="green")
            self.file_label.update()
            print("File label updated")
            
            # Update target dropdown with dataset columns
            columns = ["None - No target (Unsupervised)"] + list(self.dataset.columns)
            self.target_dropdown.configure(values=columns, state="readonly")
            print(f"Target dropdown updated with {len(columns)} options")
            
            # Set to first actual column (not None)
            if len(columns) > 1:
                self.target_dropdown.set(columns[1])
                self.target_column = columns[1]
                print(f"Default target set to: {columns[1]}")
            else:
                self.target_dropdown.set(columns[0])
                self.target_column = None
                print("No columns found, target set to None")
            
            self.target_dropdown.update()
            
            # Update dataset info
            print("Updating dataset info...")
            info_text = f"""Rows: {len(self.dataset):,}
Columns: {len(self.dataset.columns)}
Memory: {self.dataset.memory_usage(deep=True).sum() / 1024**2:.2f} MB

Numeric: {len(self.dataset.select_dtypes(include=[np.number]).columns)}
Categorical: {len(self.dataset.select_dtypes(include=['object', 'category']).columns)}
Missing: {self.dataset.isnull().sum().sum():,} values
Duplicates: {self.dataset.duplicated().sum():,} rows"""
            
            # CTkTextbox doesn't need state changes, just update content
            self.info_text.delete("0.0", "end")
            self.info_text.insert("0.0", info_text)
            self.info_text.update()
            print("Dataset info updated")
            
            # Enable analyze button
            self.analyze_btn.configure(state="normal")
            self.analyze_btn.update()
            print("Analyze button enabled")
            
            # Force complete window update
            self.update()
            self.update_idletasks()
            
            # Switch to Dataset tab
            self.tabview.set("Dataset")
            self.tabview.update()
            
            # Another full update after tab switch
            self.update()
            self.update_idletasks()
            
            # Small delay to ensure rendering, then show message
            self.after(200, lambda: messagebox.showinfo(
                "Success",
                f"Dataset loaded successfully!\n\n{len(self.dataset):,} rows ├ù {len(self.dataset.columns)} columns"
            ))
            
            print("Dataset loading complete!")
            
        except Exception as e:
            import traceback
            error_msg = f"Failed to update UI:\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            print(error_msg)
            messagebox.showerror("Error", error_msg)
    
    def _update_dataset_info(self):
        """Update dataset information display"""
        if self.dataset is None:
            return
        
        info = f"""Rows: {len(self.dataset):,}
Columns: {len(self.dataset.columns)}
Memory: {self.dataset.memory_usage(deep=True).sum() / 1024**2:.2f} MB

Numeric: {len(self.dataset.select_dtypes(include=[np.number]).columns)}
Categorical: {len(self.dataset.select_dtypes(include=['object', 'category']).columns)}
Missing: {self.dataset.isnull().sum().sum():,} values
Duplicates: {self.dataset.duplicated().sum():,} rows"""
        
        self.info_text.delete("1.0", "end")
        self.info_text.insert("1.0", info)
    
    def _on_target_selected(self, choice):
        """Handle target column selection"""
        # Handle "None" selection for unsupervised learning
        if choice.startswith("None - "):
            self.target_column = None
            print("Target column set to None (Unsupervised learning)")
        else:
            self.target_column = choice
            print(f"Target column set to: {choice}")
    
    def _analyze_and_recommend(self):
        """Perform analysis and generate recommendations"""
        if self.dataset is None:
            messagebox.showwarning("Warning", "Please load a dataset first!")
            return
        
        # Show progress
        self.progress_bar.pack(fill="x", padx=10, pady=5)
        self.progress_bar.set(0.2)
        self.analyze_btn.configure(state="disabled", text="Analyzing...")
        self.update()
        
        # Run in thread to prevent UI freezing
        thread = threading.Thread(target=self._perform_analysis)
        thread.daemon = True
        thread.start()
    
    def _perform_analysis(self):
        """Perform analysis in background thread"""
        try:
            # Analyze dataset
            self.progress_bar.set(0.3)
            self.analysis_results = self.analyzer.analyze(self.dataset, self.target_column)
            
            # Get task type
            task_type = None if self.task_type_var.get() == "auto" else self.task_type_var.get()
            
            # Generate recommendations
            self.progress_bar.set(0.7)
            self.recommendations = self.recommender.recommend(self.analysis_results, task_type)
            
            # Update UI in main thread
            self.after(0, self._display_results)
            
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error", f"Analysis failed:\n{str(e)}"))
            self.after(0, self._reset_analyze_button)
    
    def _display_results(self):
        """Display analysis results and recommendations"""
        # Clear previous results
        for widget in self.rec_scroll_frame.winfo_children():
            widget.destroy()
        
        # Update info label
        self.rec_info_label.configure(
            text=f"Showing top {min(5, len(self.recommendations))} recommendations out of {len(self.recommendations)}",
            text_color="green"
        )
        
        # Display top recommendations
        top_n = min(5, len(self.recommendations))
        
        for i, rec in enumerate(self.recommendations[:top_n]):
            self._create_recommendation_card(rec, i + 1)
        
        # Create visualization
        self._create_visualization()
        
        # Reset button
        self._reset_analyze_button()
        
        # Switch to recommendations tab
        self.tabview.set("Recommendations")
        
        messagebox.showinfo("Success", f"Analysis complete! Found {len(self.recommendations)} algorithm recommendations.")
    
    def _create_recommendation_card(self, rec: AlgorithmRecommendation, rank: int):
        """Create a recommendation card widget"""
        # Main card frame
        card = ctk.CTkFrame(self.rec_scroll_frame)
        card.pack(fill="x", padx=10, pady=10)
        
        # Header with rank and name
        header = ctk.CTkFrame(card)
        header.pack(fill="x", padx=10, pady=10)
        
        rank_label = ctk.CTkLabel(
            header,
            text=f"#{rank}",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color="#4A90E2"
        )
        rank_label.pack(side="left", padx=10)
        
        name_label = ctk.CTkLabel(
            header,
            text=rec.name,
            font=ctk.CTkFont(size=20, weight="bold")
        )
        name_label.pack(side="left", padx=10)
        
        category_label = ctk.CTkLabel(
            header,
            text=f"({rec.category})",
            font=ctk.CTkFont(size=14),
            text_color="gray"
        )
        category_label.pack(side="left", padx=5)
        
        # Score and confidence
        score_frame = ctk.CTkFrame(header)
        score_frame.pack(side="right", padx=10)
        
        score_label = ctk.CTkLabel(
            score_frame,
            text=f"Score: {rec.score:.2f}",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#50C878"
        )
        score_label.pack(side="left", padx=5)
        
        conf_label = ctk.CTkLabel(
            score_frame,
            text=f"Confidence: {rec.confidence:.1%}",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        conf_label.pack(side="left", padx=5)
        
        # Content tabs
        tabview = ctk.CTkTabview(card)
        tabview.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Reasoning tab
        reasoning_tab = tabview.add("Reasoning")
        reasoning_text = "\n".join([f"ΓÇó {r}" for r in rec.reasoning])
        reasoning_label = ctk.CTkLabel(
            reasoning_tab,
            text=reasoning_text if reasoning_text else "No specific reasoning available",
            font=ctk.CTkFont(size=12),
            justify="left",
            wraplength=700
        )
        reasoning_label.pack(anchor="w", padx=10, pady=10)
        
        # Pros/Cons tab
        pros_cons_tab = tabview.add("Pros & Cons")
        
        pros_frame = ctk.CTkFrame(pros_cons_tab)
        pros_frame.pack(side="left", fill="both", expand=True, padx=5, pady=10)
        
        pros_title = ctk.CTkLabel(
            pros_frame,
            text="Pros",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#50C878"
        )
        pros_title.pack(anchor="w", padx=10, pady=5)
        
        pros_text = "\n".join([f"ΓÇó {p}" for p in rec.pros])
        pros_label = ctk.CTkLabel(
            pros_frame,
            text=pros_text,
            font=ctk.CTkFont(size=11),
            justify="left",
            wraplength=300
        )
        pros_label.pack(anchor="w", padx=10, pady=5)
        
        cons_frame = ctk.CTkFrame(pros_cons_tab)
        cons_frame.pack(side="right", fill="both", expand=True, padx=5, pady=10)
        
        cons_title = ctk.CTkLabel(
            cons_frame,
            text="Cons",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#FF6B6B"
        )
        cons_title.pack(anchor="w", padx=10, pady=5)
        
        cons_text = "\n".join([f"ΓÇó {c}" for c in rec.cons])
        cons_label = ctk.CTkLabel(
            cons_frame,
            text=cons_text,
            font=ctk.CTkFont(size=11),
            justify="left",
            wraplength=300
        )
        cons_label.pack(anchor="w", padx=10, pady=5)
        
        # Best For tab
        best_for_tab = tabview.add("Best For")
        best_for_text = "\n".join([f"ΓÇó {b}" for b in rec.best_for])
        best_for_label = ctk.CTkLabel(
            best_for_tab,
            text=best_for_text,
            font=ctk.CTkFont(size=12),
            justify="left",
            wraplength=700
        )
        best_for_label.pack(anchor="w", padx=10, pady=10)
        
        # Hyperparameters tab
        hyperparams_tab = tabview.add("Hyperparameters")
        hyperparams_text = "\n".join([f"{k}: {v}" for k, v in rec.hyperparameters_suggestions.items()])
        hyperparams_label = ctk.CTkLabel(
            hyperparams_tab,
            text=hyperparams_text if hyperparams_text else "Use default parameters",
            font=ctk.CTkFont(size=12, family="monospace"),
            justify="left",
            wraplength=700
        )
        hyperparams_label.pack(anchor="w", padx=10, pady=10)
    
    def _create_visualization(self):
        """Create visualization of recommendations"""
        # Clear previous visualization
        for widget in self.viz_canvas_frame.winfo_children():
            widget.destroy()
        
        # Update viz info label
        self.viz_info_label.configure(
            text=f"Visualizing top {min(5, len(self.recommendations))} recommendations",
            text_color="green"
        )
        
        if not self.recommendations:
            return
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.patch.set_facecolor('#212121')
        
        # Plot 1: Score comparison
        top_5 = self.recommendations[:5]
        names = [r.name for r in top_5]
        scores = [r.score for r in top_5]
        colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
        
        bars = ax1.barh(names, scores, color=colors)
        ax1.set_xlabel('Recommendation Score', color='white')
        ax1.set_title('Top 5 Algorithm Scores', color='white', fontweight='bold')
        ax1.set_facecolor('#2B2B2B')
        ax1.tick_params(colors='white')
        ax1.spines['bottom'].set_color('white')
        ax1.spines['top'].set_color('white')
        ax1.spines['right'].set_color('white')
        ax1.spines['left'].set_color('white')
        ax1.xaxis.label.set_color('white')
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax1.text(score + 0.01, i, f'{score:.2f}', va='center', color='white', fontweight='bold')
        
        # Plot 2: Confidence vs Score
        all_scores = [r.score for r in self.recommendations]
        all_confidences = [r.confidence for r in self.recommendations]
        all_names_short = [r.name[:15] + '...' if len(r.name) > 15 else r.name for r in self.recommendations]
        
        scatter = ax2.scatter(all_scores, all_confidences, s=100, alpha=0.6, c=range(len(all_scores)), cmap='viridis')
        ax2.set_xlabel('Score', color='white')
        ax2.set_ylabel('Confidence', color='white')
        ax2.set_title('Score vs Confidence', color='white', fontweight='bold')
        ax2.set_facecolor('#2B2B2B')
        ax2.tick_params(colors='white')
        ax2.spines['bottom'].set_color('white')
        ax2.spines['top'].set_color('white')
        ax2.spines['right'].set_color('white')
        ax2.spines['left'].set_color('white')
        ax2.xaxis.label.set_color('white')
        ax2.yaxis.label.set_color('white')
        
        # Annotate top 3
        for i, (name, score, conf) in enumerate(zip(all_names_short[:3], all_scores[:3], all_confidences[:3])):
            ax2.annotate(name, (score, conf), xytext=(5, 5), textcoords='offset points', 
                        color='white', fontsize=8, fontweight='bold')
        
        plt.tight_layout()
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, self.viz_canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def _reset_analyze_button(self):
        """Reset analyze button state"""
        self.analyze_btn.configure(state="normal", text="≡ƒöì Analyze & Recommend")
        self.progress_bar.set(0)
        self.progress_bar.pack_forget()


def main():
    """Main entry point"""
    app = MLRecommenderApp()
    app.mainloop()


if __name__ == "__main__":
    main()

