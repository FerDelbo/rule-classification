import pandas as pd
import numpy as np

# ============================================================
# CONFIGURAÇÃO — ajuste o caminho conforme seu ambiente
# ============================================================
INPUT_PATH  = './data/Single-Clauses-Data_Binary-Classification.csv'
OUTPUT_PATH = './data/accord_official_clean.csv'

# ============================================================
# 1. LEITURA
# ============================================================
print(">>> Lendo arquivo...")
df = pd.read_csv(INPUT_PATH, encoding='latin-1', header=None, low_memory=False)
print(f"    Shape bruto: {df.shape}")

# Renomeia colunas principais
df = df.rename(columns={0: 'ID', 1: 'label_main', 2: 'Text'})

# ============================================================
# 2. ENTENDIMENTO DA ESTRUTURA
# ============================================================
# Col 0 (ID)         → identificador da sentença no documento
# Col 1 (label_main) → rótulo principal: 'Others' ou 'self-contained'
# Col 2 (Text)       → texto da sentença
# Cols 3-72          → rótulos alternativos da MESMA sentença
#                      em outros documentos (estrutura matricial)
#                      apenas 6 colunas têm dados: 5, 6, 7, 20, 21, 72

# Identifica quais colunas extras têm valores não-nulos
extra_label_cols = []
for col in range(3, 73):
    if df[col].notna().sum() > 0:
        extra_label_cols.append(col)

print(f"    Colunas com rótulos alternativos encontradas: {extra_label_cols}")

# ============================================================
# 3. RESOLUÇÃO DO is_rule POR LINHA
#    Lógica: olha label_main + todas as colunas extras
#    Se 'self-contained' aparecer em QUALQUER coluna → is_rule = 1
#    Se apenas 'Others' → is_rule = 0
# ============================================================
print("\n>>> Resolvendo rótulos...")

def get_is_rule(row):
    labels = []
    
    # Rótulo principal
    v = str(row['label_main']).strip().lower() if pd.notna(row['label_main']) else ''
    if v and v != 'nan':
        labels.append(v)
    
    # Rótulos alternativos
    for col in extra_label_cols:
        v = str(row[col]).strip().lower() if pd.notna(row[col]) else ''
        if v and v != 'nan':
            labels.append(v)
    
    if 'self-contained' in labels:
        return 1
    elif 'others' in labels:
        return 0
    return None

df['is_rule'] = df.apply(get_is_rule, axis=1)

print(f"    Distribuição antes de deduplicar:")
print(f"    {df['is_rule'].value_counts(dropna=False).to_dict()}")

# ============================================================
# 4. DEDUPLICAÇÃO
#    Mesma sentença aparece múltiplas vezes (uma por documento)
#    groupby + max: prioriza is_rule=1 sobre is_rule=0 em conflitos
# ============================================================
print("\n>>> Deduplicando...")

# Verifica conflitos antes
conflicts = df.groupby('Text')['is_rule'].nunique()
n_conflicts = (conflicts > 1).sum()
print(f"    Textos com rótulos conflitantes: {n_conflicts} → resolvidos com prioridade self-contained")

df['Text'] = df['Text'].str.strip()

df_clean = df.groupby('Text', as_index=False).agg(
    is_rule=('is_rule', 'max'),   # self-contained (1) > others (0)
    ID=('ID', lambda x: ' | '.join(
        sorted(set(str(i) for i in x if pd.notna(i)))[:5]
    ))
).reset_index(drop=True)

print(f"    Após deduplicação: {len(df_clean)} sentenças únicas")
print(f"    NaN em is_rule: {df_clean['is_rule'].isna().sum()}")


# ============================================================
# 5. FINALIZAÇÃO
# ============================================================
df_clean['is_rule'] = df_clean['is_rule'].astype(int)
df_clean['source']  = 'accord_official'

print("\n" + "="*50)
print("RESULTADO FINAL")
print("="*50)
print(f"Total de sentenças: {len(df_clean)}")
print(f"\nDistribuição:")
vc = df_clean['is_rule'].value_counts()
for k, v in vc.items():
    label = 'self-contained (regra)' if k == 1 else 'others (não-regra)'
    print(f"  is_rule={k} [{label}]: {v} ({v/len(df_clean)*100:.1f}%)")
ratio = vc[0] / vc[1]
print(f"\nRazão de desbalanceamento: 1:{ratio:.1f}")
print(f"NaN restantes: {df_clean['is_rule'].isna().sum()}")

df_clean[['Text', 'is_rule', 'ID', 'source']].to_csv(OUTPUT_PATH, index=False)
print(f"\nSalvo em: {OUTPUT_PATH}")