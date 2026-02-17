import pandas as pd

def finalize_item_dataset():
    print("--- 🧹 Finalizing Item Dataset (ID-to-Name Version) ---")
    
    # 1. Load the files
    try:
        df_items = pd.read_excel('Updated_Item.xlsx')
        df_types = pd.read_excel('Type.xlsx')
        print("✅ Original datasets loaded.")
    except Exception as e:
        print(f"❌ Error loading files: {e}")
        return

    # 2. STRIP SPACES FROM COLUMN NAMES
    df_items.columns = df_items.columns.str.strip()
    df_types.columns = df_types.columns.str.strip()
    
    # 3. STANDARDIZE IDs FOR MERGING
    # Convert IDs to string so they match perfectly (prevents 1 vs '1' errors)
    df_items['AttractionTypeId'] = df_items['AttractionTypeId'].astype(str).str.strip()
    df_types['AttractionTypeId'] = df_types['AttractionTypeId'].astype(str).str.strip()

    # 4. PERFORM THE JOIN
    # We bring the TEXT name (AttractionType) from Type.xlsx into our Items list
    # using the ID as the bridge.
    print(f"Merging Items on 'AttractionTypeId'...")
    
    df_clean = pd.merge(
        df_items, 
        df_types[['AttractionTypeId', 'AttractionType']], 
        on='AttractionTypeId', 
        how='left'
    )

    # 5. ASSIGN UNIQUE ATTRACTION IDs
    # This fixes the "Kuta Beach" duplication/overwriting issue
    print("Assigning unique AttractionIds...")
    df_clean['AttractionId'] = range(1, len(df_clean) + 1)

    # 6. FILL MISSING DATA
    # If some IDs didn't match, we don't want NaNs crashing Streamlit
    df_clean['AttractionType'] = df_clean['AttractionType'].fillna('Other')

    # 7. SAVE RESULTS
    df_clean.to_excel('Updated_Item_Perfected.xlsx', index=False)
    df_clean.to_csv('Updated_Item_Perfected.csv', index=int(False))
    
    print("\n" + "="*40)
    print("✅ ITEM DATASET PERFECTED")
    print(f"Total Unique Attractions: {len(df_clean)}")
    print(f"Sample Types mapped: {df_clean['AttractionType'].unique().tolist()}")
    print("="*40)

if __name__ == "__main__":
    finalize_item_dataset()