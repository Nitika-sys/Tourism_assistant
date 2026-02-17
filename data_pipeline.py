import pandas as pd
import numpy as np

def load_and_merge_data():
    print("--- 🚀 Building Master Dataset (Robust Version) ---")
    
    # 1. LOAD ALL FILES
    try:
        # USE UPDATED ITEM LIST
        df_items = pd.read_excel('Updated_Item.xlsx')
        df_types = pd.read_excel('Type.xlsx')
        df_trans = pd.read_excel('Transaction.xlsx')
        df_user = pd.read_excel('User.xlsx')
        df_city = pd.read_excel('City.xlsx')
        df_mode = pd.read_excel('Mode.xlsx')
        df_cont = pd.read_excel('Continent.xlsx')
        df_country = pd.read_excel('Country.xlsx')
        df_region = pd.read_excel('Region.xlsx')
        print("✅ All Excel files loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading files: {e}")
        return

    # 2. STRIP SPACES FROM COLUMN NAMES
    for d in [df_items, df_types, df_trans, df_user, df_city, df_mode, df_cont, df_country, df_region]:
        d.columns = d.columns.str.strip()

    # 3. MERGE LOCATION DATA (City -> Country -> Region -> Continent)
    # Start with City
    location_master = pd.merge(df_city, df_country, on='CountryId', how='left', suffixes=('_City', '_Country'))
    
    # Add Region
    location_master = pd.merge(location_master, df_region, on='RegionId', how='left', suffixes=('', '_Region'))
    
    # Add Continent
    location_master = pd.merge(location_master, df_cont, on='ContinentId', how='left', suffixes=('', '_Continent'))
    
    print(f"✅ Location Master built with {len(location_master)} cities.")

    # --- NEW: FIX ITEM LOCATIONS BY TEXT MATCHING ---
    print("🔧 Fixing Item Locations by matching Address to City names...")
    
    # Create a lookup dictionary for cities: {city_name_lower: city_id}
    # Sort cities by length (descending) to match longest names first (e.g. "New York" before "York")
    sorted_cities = df_city.sort_values(by='CityName', key=lambda x: x.str.len(), ascending=False)
    city_lookup = {str(name).lower(): cid for name, cid in zip(sorted_cities['CityName'], sorted_cities['CityId']) if str(name).strip() != '-'}
    
    def find_city_id(address, original_id):
        if pd.isna(address):
            return original_id
        addr_lower = str(address).lower()
        # Heuristic: Check if any known city is in the address
        # This is slow O(N*M), but fine for 1700 items
        # Optimization: Only check cities that appear in the address? No, that's circular.
        # Faster approach: Split address into words and check if they are cities?
        # But cities can be multi-word.
        # Let's try a simple approach for now:
        # Check if the address contains the city name
        # To avoid false positives (e.g. "Male" in "Malevolent"), we could use regex or simple " in " check
        # But "Ubud" is unique enough.
        
        # We limit to cities that are actually relevant or common? No.
        # Let's try matching against the full city list but break early? No.
        
        # Let's try to match based on tokens.
        tokens = set(addr_lower.replace(',', ' ').split())
        
        # Check against cities that are single words first?
        # Or better: check against known cities in the same Country if we knew the country? We don't.
        
        # Let's use the provided CityId if it maps to a valid city, otherwise search.
        # But we know provided CityId (e.g. 1) maps to Douala which is WRONG for Bali.
        # So we MUST search.
        
        # Optimization: Most addresses end with "City, Country" or "City postcode Country"
        # "Jl. Monkey Forest, Ubud 80571 Indonesia" -> "Ubud" is the key.
        
        for city_name, city_id in city_lookup.items():
            # Use word boundary check or simple containment?
            # Simple containment is risky for short names like "Ba", "Ur".
            # Let's require the city name to be at least 3 chars unless it's a known short city.
            if len(city_name) < 3: 
                continue
                
            if f" {city_name} " in f" {addr_lower} " or f" {city_name}," in f" {addr_lower} ":
                 return city_id
                 
        return original_id

    # Apply the fix (might take a few seconds)
    # We only update AttractionCityId if we find a match
    # Create a copy to avoid SettingWithCopy warning
    df_items = df_items.copy()
    
    # Only run this if we suspect IDs are wrong (which we do)
    # df_items['AttractionCityId'] = df_items.apply(lambda x: find_city_id(x['AttractionAddress'], x['AttractionCityId']), axis=1)
    
    # Since iterating 9000 cities for 1700 items is 15 million checks, it might be slow in python.
    # Let's try a faster way: Extract potential city words from address and check if they exist in city_lookup.
    
    def fast_find_city(address, original_id):
        if pd.isna(address):
            return original_id
        addr_lower = str(address).lower()
        # Remove punctuation
        cleaned = addr_lower.replace(',', ' ').replace('.', ' ').replace('-', ' ')
        words = cleaned.split()
        
        # Check distinct words against city list
        for word in words:
            if len(word) > 2 and word in city_lookup:
                return city_lookup[word]
        
        # Handle multi-word cities? (e.g. New York)
        # Check bigrams?
        for i in range(len(words)-1):
            bigram = f"{words[i]} {words[i+1]}"
            if bigram in city_lookup:
                return city_lookup[bigram]
                
        return original_id

    print("... Mapping addresses to City IDs (this may take a moment) ...")
    df_items['NewCityId'] = df_items.apply(lambda x: fast_find_city(x['AttractionAddress'], x['AttractionCityId']), axis=1)
    
    # Update the ID
    df_items['AttractionCityId'] = df_items['NewCityId']
    df_items.drop(columns=['NewCityId'], inplace=True)
    print("✅ Address mapping complete.")

    # 4. ENRICH ITEMS WITH LOCATION AND TYPE
    # Merge Items with Location (via AttractionCityId -> CityId)
    # Note: Some items might not have a valid CityId, so we use left join
    items_enriched = pd.merge(df_items, location_master, left_on='AttractionCityId', right_on='CityId', how='left')
    
    # Merge with Type
    items_enriched = pd.merge(items_enriched, df_types, on='AttractionTypeId', how='left')
    
    # FILL MISSING VALUES IN ITEMS
    items_enriched['AttractionType'] = items_enriched['AttractionType'].fillna('Other')
    items_enriched['CityName'] = items_enriched['CityName'].fillna('Unknown')
    items_enriched['Country'] = items_enriched['Country'].fillna('Unknown')
    items_enriched['Region'] = items_enriched['Region'].fillna('Unknown')
    items_enriched['Continent'] = items_enriched['Continent'].fillna('Unknown')
    
    print(f"✅ Items Enriched: {len(items_enriched)} items.")
    
    # SAVE MASTER ITEMS LIST (For Recommendations)
    items_enriched.to_csv('master_items.csv', index=False)
    print(f"✅ Saved master_items.csv with {len(items_enriched)} items.")

    # 5. MERGE TRANSACTIONS WITH ITEMS AND USERS
    # Merge Transactions with Enriched Items
    # This links the Rating to the Item details
    # We use INNER JOIN or LEFT JOIN from Transactions to keep only valid transactions for training
    master = pd.merge(df_trans, items_enriched, on='AttractionId', how='left')
    
    # Merge with User details
    # Note: User also has location info (CityId, CountryId etc.) which might conflict with Item location
    # We'll rename User location columns to distinguish them
    user_location = pd.merge(df_user, location_master, on='CityId', how='left', suffixes=('_User', '_UserLoc'))
    
    # Actually, User.xlsx already has CountryId, RegionId, ContinentId. 
    # Let's see if we need to merge location_master for Users or just rely on User.xlsx columns.
    # User.xlsx: UserId, ContinentId, RegionId, CountryId, CityId
    # It seems User.xlsx has the IDs but not the names (Country Name, Region Name).
    # So we should merge with the dimension tables to get names if needed.
    
    # Let's merge User with just the names from our location tables to avoid ID conflicts?
    # Or simpler: Merge User with location_master on CityId
    
    # Rename location_master columns for User merge to avoid collision with Item location columns
    user_loc_renamed = location_master.rename(columns={
        'CityName': 'UserCity',
        'Country': 'UserCountry',
        'Region': 'UserRegion',
        'Continent': 'UserContinent',
        'CountryId': 'UserCountryId',
        'RegionId': 'UserRegionId',
        'ContinentId': 'UserContinentId'
    })
    
    # Merge User with their location names
    # Note: User.xlsx has CityId.
    users_enriched = pd.merge(df_user, user_loc_renamed, on='CityId', how='left')
    
    # Now merge this into the master transaction list
    master = pd.merge(master, users_enriched, on='UserId', how='left', suffixes=('_Item', '_User'))
    
    # 6. MERGE VISIT MODE
    # Ensure VisitMode is same type
    master['VisitMode'] = pd.to_numeric(master['VisitMode'], errors='coerce').fillna(0).astype(int)
    df_mode['VisitMode'] = pd.to_numeric(df_mode['VisitMode'], errors='coerce').fillna(0).astype(int)
    
    master = pd.merge(master, df_mode, on='VisitMode', how='left')

    # 7. CLEANUP
    # Drop rows where essential IDs might be missing if that's critical, or fillna
    # For now, we keep everything but fill NaNs for text columns
    
    # Rename columns to match expected schema in model_train.py and app.py
    # We want Item location to be the primary 'ContinentId', 'CountryId' etc.
    rename_map = {
        'CityId_Item': 'CityId',
        'CountryId_Item': 'CountryId', 
        'RegionId_Item': 'RegionId', 
        'ContinentId_Item': 'ContinentId',
        'CityId_User': 'UserCityId',
        'CountryId_User': 'UserOriginCountryId',
        'RegionId_User': 'UserOriginRegionId',
        'ContinentId_User': 'UserOriginContinentId'
    }
    master.rename(columns=rename_map, inplace=True)
    
    text_cols = ['Attraction', 'AttractionType', 'CityName', 'Country', 'Region', 'Continent', 'UserCity', 'UserCountry', 'UserRegion', 'UserContinent', 'Mode']
    for col in text_cols:
        if col in master.columns:
            master[col] = master[col].fillna('Unknown')
            
    # Fill numeric IDs with 0 if missing
    id_cols = ['AttractionId', 'UserId', 'CityId', 'CountryId', 'RegionId', 'ContinentId', 'VisitYear', 'VisitMonth', 'Rating']
    for col in master.columns:
        if 'Id' in col or col in id_cols:
            master[col] = pd.to_numeric(master[col], errors='coerce').fillna(0).astype(int)

    # 8. SAVE
    master.to_csv('master_tourism_data.csv', index=False)
    
    print("\n" + "="*40)
    print("✅ MASTER DATASET GENERATED SUCCESSFULLY")
    print(f"Total Transactions: {len(master)}")
    print(f"Unique Attractions: {master['AttractionId'].nunique()}")
    print(f"Unique Users: {master['UserId'].nunique()}")
    print("="*40)
    
    # Quick sanity check
    print(master[['UserId', 'AttractionId', 'Rating', 'Attraction', 'CityName', 'Country']].head())

if __name__ == "__main__":
    load_and_merge_data()
