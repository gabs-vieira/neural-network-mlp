# ===============================
# Função para calcular a matriz de confusão
# ===============================
def confusion_matrix(y_true, y_pred):
    TP = FP = TN = FN = 0
    for yt, yp in zip(y_true.flatten(), y_pred.flatten()):
        if yt == 1 and yp == 1:
            TP += 1
        elif yt == 0 and yp == 1:
            FP += 1
        elif yt == 0 and yp == 0:
            TN += 1
        elif yt == 1 and yp == 0:
            FN += 1
    return TP, FP, TN, FN

# ===============================
# Função para calcular métricas
# ===============================
def calculate_metrics(y_true, y_pred):
    TP, FP, TN, FN = confusion_matrix(y_true, y_pred)

    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return accuracy, precision, recall, f1_score

# ===============================
# Avaliar modelo MLP
# ===============================
def evaluate_mlp(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy, precision, recall, f1 = calculate_metrics(y_test, y_pred)
    print("Matriz de Confusão (TP, FP, TN, FN):", confusion_matrix(y_test, y_pred))
    print(f"Acurácia: {accuracy:.4f}")
    print(f"Precisão: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    return accuracy, precision, recall, f1
