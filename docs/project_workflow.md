A. User Interaction Flow
1. Input Phase
   - User enters stock symbol
   - System validates symbol
   - Checks cache for recent predictions

2. Data Collection
   - Fetches real-time stock data
   - Processes historical data
   - Prepares data for models

3. Prediction Phase
   - If cached data exists:
     * Retrieves stored predictions
     * Updates if necessary
   - If new prediction needed:
     * Runs all three models
     * Stores results in cache
     * Generates visualizations

4. Result Display
   - Shows current stock data
   - Displays predictions from all models
   - Presents error metrics
   - Shows visualization graphs

B. Technical Workflow
1. Data Processing
   - Data cleaning
   - Feature scaling
   - Time series preparation
   - Train-test split (80-20)

2. Model Execution
   ARIMA Model:
   - Uses historical values
   - Implements (5,1,0) order
   - Handles prediction failures gracefully

   LSTM Model:
   - Preprocesses with MinMaxScaler
   - Creates 7-day sequences
   - Trains on scaled data
   - Makes future predictions

   Linear Regression:
   - Processes historical data
   - Calculates trends
   - Applies 1.04 adjustment factor
   - Generates forecast set

3. Visualization Generation
   - Creates separate plots for each model
   - Saves visualizations
   - Updates database with paths

4. Result Storage
   - Stores predictions in SQLite
   - Updates accuracy metrics
   - Maintains visualization paths
   - Handles caching logic