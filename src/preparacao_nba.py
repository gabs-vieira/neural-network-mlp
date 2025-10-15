import os, re, json
import numpy as np
import pandas as pd


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    def normalize(name: str) -> str:
        name = name.strip().replace('%', '_pct').lower()
        name = re.sub(r'[^0-9a-zA-Z]+', '_', name)
        name = re.sub(r'_+', '_', name).strip('_')
        return name

    df = df.copy()
    df.columns = [normalize(c) for c in df.columns]
    return df


def parse_time_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().astype(str).head(50)
            if len(sample) > 0 and sample.str.contains(':').mean() > 0.6:

                def mmss_to_min(x):
                    try:
                        s = str(x)
                        if ':' not in s:
                            return np.nan
                        parts = s.split(':')
                        if len(parts) == 2:
                            m, ss = parts
                            return float(m) + float(ss) / 60.0
                        elif len(parts) == 3:
                            h, m, ss = parts
                            return float(h) * 60.0 + float(m) + float(ss) / 60.0
                        return np.nan
                    except Exception:
                        return np.nan

                df[col] = df[col].apply(mmss_to_min)
    return df


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            s = df[col].astype(str)
            has_pct = s.str.contains('%', na=False).mean() > 0.6
            cleaned = s.str.replace(r'[\s,]', '', regex=True).str.replace('%', '', regex=False)
            num = pd.to_numeric(cleaned, errors='coerce')
            if has_pct:
                num = num / 100.0
            if num.notna().sum() > (df[col].notna().sum() * 0.5):
                df[col] = num
    return df


def drop_identifier_columns(df: pd.DataFrame):
    candidates = [
        'rk',
        'rank',
        'id',
        'player',
        'player_name',
        'name',
        'jogador',
        'nome_do_jogador',
        'url',
        'link',
        'birth_date',
        'college',
        'notes',
        'observacoes',
    ]
    present = [c for c in candidates if c in df.columns]
    return df.drop(columns=present, errors='ignore'), present


def one_hot_encode_small_categoricals(df: pd.DataFrame, max_unique: int = 40):
    df = df.copy()
    cats = []
    for col in df.columns:
        if df[col].dtype == object:
            n = df[col].nunique(dropna=True)
            if 2 <= n <= max_unique:
                cats.append(col)
    if cats:
        df = pd.get_dummies(df, columns=cats, dummy_na=True, drop_first=False)
    return df, cats


def impute_missing(df: pd.DataFrame):
    df = df.copy()
    imps = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            med = df[col].median()
            df[col] = df[col].fillna(med)
            imps[col] = None if pd.isna(med) else float(med)
        else:
            if df[col].isna().any():
                mode = df[col].mode(dropna=True)
                val = mode.iloc[0] if len(mode) > 0 else 'Unknown'
                df[col] = df[col].fillna(val)
                imps[col] = val
    return df, imps


def is_binary(series: pd.Series) -> bool:
    nn = series.dropna()
    if nn.empty:
        return False
    return nn.nunique() <= 2


def winsorize_clip(df: pd.DataFrame, numeric_cols, lq=0.01, uq=0.99):
    df = df.copy()
    for c in numeric_cols:
        col = df[c]
        if is_binary(col) or col.dropna().nunique() < 3:
            continue
        col = col.astype('float64')
        lo = col.quantile(lq)
        hi = col.quantile(uq)
        if pd.isna(lo) or pd.isna(hi) or lo >= hi:
            continue
        df[c] = col.clip(lower=lo, upper=hi)
    return df


def standardize(df: pd.DataFrame, numeric_cols):
    df = df.copy()
    params = {}
    for c in numeric_cols:
        col = df[c]
        if is_binary(col):
            continue
        m = col.mean()
        s = col.std(ddof=0)
        params[c] = {'mean': float(m), 'std': float(s)}
        df[c] = 0.0 if (s == 0 or pd.isna(s)) else (col - m) / s
    return df, params


def main():
    input_path = './data/nba_dados_2024.csv'
    if not os.path.exists(input_path):
        raise FileNotFoundError('Coloque o arquivo nba_dados_2024.csv no diretório ../data.')

    outdir = os.path.dirname(input_path)
    base = os.path.splitext(os.path.basename(input_path))[0]
    clean_csv = os.path.join(outdir, f'{base}_clean.csv')
    std_csv = os.path.join(outdir, f'{base}_clean_standardized.csv')
    scaler_js = os.path.join(outdir, f'{base}_scaler_params.json')
    report_js = os.path.join(outdir, f'{base}_prep_report.json')

    print(f'Lendo: {input_path}')
    raw = pd.read_csv(input_path)
    rows_before = len(raw)

    df = clean_column_names(raw)
    df = parse_time_like_columns(df)
    df = coerce_numeric(df)
    dup = df.duplicated().sum()
    df = df.drop_duplicates()
    df, dropped = drop_identifier_columns(df)
    df, imps1 = impute_missing(df)
    df, cats = one_hot_encode_small_categoricals(df, max_unique=40)

    # Separa listas para relatório
    all_numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    binary_cols = [c for c in all_numeric if is_binary(df[c])]
    # Numéricas não-binárias para winsorizar e padronizar
    num_cols_nb = [c for c in all_numeric if c not in binary_cols]

    # Winsorize só nas numéricas não-binárias
    df = winsorize_clip(df, num_cols_nb, 0.01, 0.99)
    df, imps2 = impute_missing(df)  # segurança

    df.to_csv(clean_csv, index=False)

    std = df.copy()
    std, params = standardize(std, num_cols_nb)
    std.to_csv(std_csv, index=False)
    with open(scaler_js, 'w') as f:
        json.dump(params, f, indent=2, ensure_ascii=False)

    report = {
        'input_path': input_path,
        'rows_before': rows_before,
        'rows_after': len(df),
        'duplicates_removed': int(dup),
        'columns_dropped': dropped,
        'one_hot_encoded_columns': cats,
        'numeric_columns_count': len(all_numeric),
        'numeric_binary_columns': binary_cols[:50],
        'numeric_nonbinary_columns': num_cols_nb[:50],
        'imputations_initial': imps1,
        'imputations_after_winsor': imps2,
        'outputs': {
            'clean_csv': clean_csv,
            'standardized_csv': std_csv,
            'scaler_params_json': scaler_js,
        },
    }
    with open(report_js, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print('\n=== PREP OK ===')
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print('\nArquivos:')
    print(clean_csv)
    print(std_csv)
    print(scaler_js)
    print(report_js)


if __name__ == '__main__':
    main()
