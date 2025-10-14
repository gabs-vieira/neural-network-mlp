from data_preprocessing import load_and_prepare_data
from mlp import SimpleMLP
from utils import compute_metrics, plot_loss, plot_roc

if __name__ == '__main__':
    X_train, X_test, y_train, y_test, mapping = load_and_prepare_data('nba_dados_2024.csv')
    n_features = X_train.shape[1]
    mlp = SimpleMLP([n_features, 12, 1], lr=0.05, momentum=0.9)
    history = mlp.fit(X_train, y_train, epochs=100, batch_size=32, verbose=True)
    y_pred = mlp.predict(X_test)
    y_scores = mlp.predict_proba(X_test)
    metrics = compute_metrics(y_test, y_pred)
    print(metrics)
    plot_loss(history['loss'])
    plot_roc(y_test, y_scores)
    print('Treinamento finalizado. Resultados salvos.')
