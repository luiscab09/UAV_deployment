# Generate predictions for the test data
    y_pred = model.predict(X_test)
    
    # Access the 'newpos' column in X_test
    newpos_values = X_test['newpos']

    # Compare the values of 'newpos'
    for i, value in enumerate(newpos_values):
            # Example: Apply a mathematical operation to adjust the predictions
        if value > 0:
                  y_pred[i] = y_pred[i] * 0.74
           
        else:
               y_pred[i] = y_pred[i] - 0.1
        
    