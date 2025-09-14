# Load the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ast
import re
import json
#pip install osmnx
import osmnx as ox
import geopandas as gpd
from shapely.geometry import MultiPoint
from matplotlib.ticker import FuncFormatter
from scipy.spatial import cKDTree          # ← add this import
import pickle, gzip, os, networkx as nx


data = pd.read_csv('Data/orlando_house_data.csv', low_memory=False)


# Columns to keep
columns_to_keep = [
    'city', 'state', 'bathrooms', 'bedrooms', 'yearBuilt',
    'streetAddress', 'zipcode', 'longitude', 'latitude', 
    'homeType', 'livingAreaValue', 'lastSoldPrice', 'schools', 'dateSold', 
    'county', 'hoa_details', 'property', 'community_details']

house_data = data[[col for col in columns_to_keep if col in data.columns]]


# List of addresses to remove (zipcodes and coordiates dont correspond)
addresses_to_remove = [
    "13351 Camborne Pl",
    "10033 Pearson Ave",
    "6831 Oakway Rd",
    "13219 Winterton Ln",
    "2150 Whitney Marsh Aly",
    "7202 Summer Ivy Aly",
    "13566 McMillan Dr"
]

house_data = house_data[~house_data['streetAddress'].isin(addresses_to_remove)]



"""
Label-encode `homeType` and remove rows whose types are not in the mapping.
"""

def home_type(df, col='homeType'):

    mapping = { "CONDO": 1, "TOWNHOUSE": 2, "SINGLE_FAMILY": 3}
    df = df.copy()
    df[col] = (df[col].astype(str).str.upper().map(mapping).astype("Int64"))

    # Drop rows with NA and and convert to int64  
    df = df[df[col].notna()]             
    df[col] = df[col].astype("int64")   

    return df


"""
#Flatten schools column, extract ratings and distances to primary, middle and high schools 
"""

def schools(df, max_schools_per_level: int = 3):

    df = df.copy()                     
    original_cols = [c for c in df.columns if c != 'schools']

    # Helper function to flatten  `schools` column
    def _flatten_schools(entry):
        try:
            schools = ast.literal_eval(entry) if isinstance(entry, str) else entry
        except Exception:
            schools = []

        levels = {lvl: [] for lvl in ['Elementary', 'Primary', 'Middle', 'High']}
        if isinstance(schools, list):
            for sc in schools:
                lvl = sc.get('level')
                if lvl in levels:
                    levels[lvl].append(
                        {'rating': sc.get('rating'),
                         'distance': sc.get('distance'),
                         'type': sc.get('type')})
        out = {}
        for lvl, lst in levels.items():
            for i, sc in enumerate(lst[:max_schools_per_level], 1):
                suf = f"{lvl.lower()}{i}"
                out[f"{suf}_rating"]   = sc.get('rating')
                out[f"{suf}_distance"] = sc.get('distance')
                out[f"{suf}_type"]     = sc.get('type')
        return pd.Series(out)

    if 'schools' in df.columns:
        df = pd.concat([df.drop(columns=['schools']), df['schools'].apply(_flatten_schools)], axis=1)

    # if primary school is missing, fill the missing values with elementary school 
    have_all = {'primary1_rating','primary1_distance','primary1_type',
                'elementary1_rating','elementary1_distance','elementary1_type',
                'primary2_rating','primary2_distance','primary2_type'}.issubset(df.columns)

    if have_all:
        miss = (df['primary1_rating'].isna() & df['primary1_distance'].isna() & df['primary1_type'].isna())
        for suf in ['rating','distance','type']:
            p1, e1 = f'primary1_{suf}', f'elementary1_{suf}'
            df.loc[miss, p1] = df.loc[miss, e1]
            df.loc[miss, e1] = pd.NA

        # Use elementary1 if it has higher rating and is closer
        better_e1 = (
            df['elementary1_rating'].notna() & df['primary1_rating'].notna() &
            (df['elementary1_rating'] > df['primary1_rating']) &
            (df['elementary1_distance'] < df['primary1_distance'])
        )
        for suf in ['rating','distance','type']:
            p1, e1 = f'primary1_{suf}', f'elementary1_{suf}'
            df.loc[better_e1, p1] = df.loc[better_e1, e1]
            df.loc[better_e1, e1] = pd.NA

        # Use primary2 if it has higher rating and is closer
        better_p2 = (
            df['primary2_rating'].notna() & df['primary1_rating'].notna() &
            (df['primary2_rating'] > df['primary1_rating']) &
            (df['primary2_distance'] < df['primary1_distance'])
        )
        for suf in ['rating','distance','type']:
            p1, p2 = f'primary1_{suf}', f'primary2_{suf}'
            df.loc[better_p2, p1] = df.loc[better_p2, p2]
            df.loc[better_p2, p2] = pd.NA

    # Keep only required columns 
    new_cols = ['primary1_rating','primary1_distance', 'middle1_rating','middle1_distance',
        'high1_rating','high1_distance']
   
    for col in new_cols:
        if col not in df.columns: df[col] = pd.NA

    final_cols = original_cols + new_cols
    final_cols = list(dict.fromkeys(final_cols))

    return df[final_cols]


"""
Parse `hoa_details` column. Create has_hoa  → 1/0. One-hot encode amenities_included / services_included.
Add `recreational_facilities`  (integer count) and `park_trail` (0/1)
"""

def expand_hoa(df, hoa_col: str = 'hoa_details'):

    original_cols = [c for c in df.columns if c != hoa_col]
    df = df.copy()

    # Expand JSON  to dict
    def _parse(txt):
        if not isinstance(txt, str) or not txt.lstrip().startswith('{'):
            return {}
        try:
            return json.loads(txt)
        except Exception:
            return {}

    # Etract scalar and list fields
    def _extract(d):
        out = {
            'has_hoa': 0,
            'amenities': [],
            'services': []
        }
        if not isinstance(d, dict):
            return pd.Series(out)

        val = d.get('has_hoa')
        if isinstance(val, bool):
            out['has_hoa'] = int(val)
        elif isinstance(val, str):
            out['has_hoa'] = 1 if val.strip().lower().startswith('y') else 0

        if isinstance(d.get('amenities_included'), list):
            out['amenities'] = [a.strip() for a in d['amenities_included']
                                if isinstance(a, str)]
        if isinstance(d.get('services_included'), list):
            out['services']  = [s.strip() for s in d['services_included']
                                if isinstance(s, str)]
        return pd.Series(out)

    parsed   = df[hoa_col].apply(_parse)
    expanded = parsed.apply(_extract)
    df = pd.concat([df.drop(columns=[hoa_col]), expanded], axis=1)

    # Get dummies for amenities and services 
    def _san(txt): 
        return re.sub(r'[\s\-/()]+', '_', txt.strip().lower())

    for amen in {a for lst in df['amenities'] for a in lst}:
        df[f'amenity_{_san(amen)}'] = df['amenities'].apply(lambda lst, a=amen: int(a in lst))

    for srv in {s for lst in df['services'] for s in lst}:
        df[f'service_{_san(srv)}'] = df['services'].apply(lambda lst, s=srv: int(s in lst))

    # Create new columns: recreational_facilities and park_trail
    rec_cols = ['amenity_basketball_court', 'amenity_golf_course','amenity_pickleball_courts', 
            'amenity_racquetball','amenity_shuffleboard_court', 'amenity_tennis_courts', 'amenity_clubhouse',
            'amenity_pool','amenity_sauna','amenity_spa_hot_tub', 'amenity_fitness_center']
    present_rec = [c for c in rec_cols if c in df.columns]
    df['recreational_facilities'] = (df[present_rec].fillna(0).astype(int).sum(axis=1))

    pt_cols = [c for c in ['amenity_park', 'amenity_trails'] if c in df.columns]
    df['park_trail'] = (df[pt_cols].fillna(0).astype(int).max(axis=1) if pt_cols else 0)

    # Define the columns to keep 
    wanted_new = ['has_hoa', 'amenity_gated', 'recreational_facilities', 'park_trail']
    df_final = df[original_cols + wanted_new]

    return df_final



"""
From `community_details` column, pull only the values under the "Features" block
"""

def extract_community_features(df, col='community_details', new_col='comm_features'):

    def get_features(raw):
        # Parse JSON-like text to Python list
        try:
            blocks = ast.literal_eval(raw) if isinstance(raw, str) else raw
        except Exception:
            return None

        if not isinstance(blocks, list):
            return None

        for block in blocks:
            if isinstance(block, dict) and block.get('title') == 'Features':
                vals = block.get('values')
                if isinstance(vals, list):
                    return vals
        return None 

    df = df.copy()
    df[new_col] = df[col].apply(get_features)
    
    return df


"""
Add dummy columns: comm_playground, comm_park, comm_gated_yes
"""

def encode_community_features(df, col='comm_features', prefix='comm_'):

    df = df.copy()

    raw_to_safe = {
        'Gated':                      f'{prefix}gated',
        'Gated Community - Guard':    f'{prefix}gated_community_guard',
        'Gated Community - No Guard': f'{prefix}gated_community_no_guard',
        'Playground':                 f'{prefix}playground',
        'Park':                       f'{prefix}park'}

    # Initialise all dummy columns to 0
    for safe in raw_to_safe.values():
        df[safe] = 0
    # Populate dummies row-by-row
    for idx, lst in df[col].items():
        if not isinstance(lst, list):
            continue
        for raw, safe in raw_to_safe.items():
            if raw in lst:
                df.at[idx, safe] = 1
    # Combine gated variants into a single flag
    gated_cols = [
        raw_to_safe['Gated'],
        raw_to_safe['Gated Community - Guard'],
        raw_to_safe['Gated Community - No Guard']]
    df[f'{prefix}gated_yes'] = df[gated_cols].max(axis=1)
    # Upgrade comm_park if park_trail == 1
    if 'park_trail' in df.columns:
        df[f'{prefix}park'] = (
            df[f'{prefix}park'] | df['park_trail'].fillna(0).astype(int)
        )
    # Drop source gated dummies & original list column
    df.drop(columns=gated_cols + [col], inplace=True, errors='ignore')

    return df



"""
Split the nested property list into separate columns. From prop_lot, build one-hot 
binary columns for every distinct lot feature. Add variables: lot_feature_above_flood_plain,
lot_feature_city_lot, lot_feature_historic_district, greenbelt.
"""

def lot_features(df, prop_col: str = 'property', prefix: str = 'prop_'):
    df = df.copy()
    original_cols = [c for c in df.columns if c != prop_col]
    def _parse(cell):
        if isinstance(cell, str):
            try:
                return json.loads(cell)
            except Exception:
                return []
        return cell if isinstance(cell, list) else []

    def _san_title(t: str) -> str:
        return re.sub(r'\W+', '_', t.strip().lower())
    parsed  = df[prop_col].apply(_parse)
    titles  = {blk.get('title') for row in parsed for blk in row if blk.get('title')}
    col_map = {t: f"{prefix}{_san_title(t)}" for t in titles}
    new_dat = {col: [] for col in col_map.values()}

    for row in parsed:
        row_map = {col_map[blk['title']]: blk.get('values')
                   for blk in row if blk.get('title') in col_map}
        for col in new_dat:
            new_dat[col].append(row_map.get(col))

    df = pd.concat([df.drop(columns=[prop_col]),
                    pd.DataFrame(new_dat, index=df.index)], axis=1)
    # Drop unwanted block columns
    df.drop(columns=['prop_accessibility', 'prop_details'], errors='ignore', inplace=True)
    # Add lot feature binaries
    def _san_feat(s: str) -> str:
        return re.sub(r'\W+', '_', s.strip().lower())
    def _extract_feats(lst):
        if not isinstance(lst, list):
            return []
        for item in lst:
            if isinstance(item, str) and item.lower().startswith('features:'):
                return [p.strip() for p in item.split(':', 1)[1].split(',') if p.strip()]
        return []
    df['_lot_feats'] = df['prop_lot'].apply(_extract_feats)
    all_feats = {f for sub in df['_lot_feats'] for f in sub}

    for feat in sorted(all_feats):
        df[f'lot_feature_{_san_feat(feat)}'] = df['_lot_feats'].apply(
            lambda lst, f=feat: int(f in lst))
    df.drop(columns=['_lot_feats'], inplace=True)
    # Create greenbelt column
    if {'lot_feature_greenbelt', 'lot_feature_conservation_area'}.issubset(df.columns):
        df['greenbelt'] = (df[['lot_feature_greenbelt', 'lot_feature_conservation_area']]
              .fillna(0).astype(int).max(axis=1))
    else:
        df['greenbelt'] = 0

    # Define columns to keep 
    keep_new = ['lot_feature_above_flood_plain','lot_feature_city_lot',
                'lot_feature_historic_district','greenbelt']
    for col in keep_new:
        if col not in df.columns:
            df[col] = 0
    extra_keep = [c for c in ['prop_parking', 'prop_features'] if c in df.columns]
    final_cols = original_cols + extra_keep + keep_new
    final_cols = list(dict.fromkeys(final_cols))   

    return df[final_cols]



"""
From the list-of-strings in `prop_col`, extract parking_spaces and has_garage   
"""

def extract_parking_info(df, prop_col: str = 'prop_parking'):

    def _parse(lst):
        parking_spaces = 0
        has_garage     = 0
        if isinstance(lst, list):
            for s in lst:
                if not isinstance(s, str):
                    continue
                low = s.lower()
                if low.startswith('total spaces:'):
                    m = re.search(r'total spaces:\s*(\d+)', s, re.I)
                    if m:
                        parking_spaces = int(m.group(1))
                elif low.startswith('has garage:'):
                    m = re.search(r'has garage:\s*(yes|no)', s, re.I)
                    if m:
                        has_garage = 1 if m.group(1).lower() == 'yes' else 0
        return pd.Series({
            'parking_spaces': parking_spaces,
            'has_garage': has_garage})

    out = df[prop_col].apply(_parse)
    return pd.concat([df, out], axis=1)


"""
From the list-of-strings in `prop_col` extract 'levels' and add binary columns 
'has_view_yes', 'has_water_view_yes' and 'private_pool' 
"""

def levels_and_features(df, prop_col='prop_features'):
    # Define helper functions 
    def sanitize_key(s: str)  -> str:
        return re.sub(r'\W+', '_', s.strip().lower()).strip('_')
    def sanitize_feat(s: str) -> str:
        return re.sub(r'\W+', '_', s.strip().lower()).strip('_')
    def parse_level(text):
        if isinstance(text, (int, float)):
            return int(text) if int(text) in [1, 2, 3] else None
        if isinstance(text, str):
            t = text.strip().lower()
            if '1' in t or 'one' in t:
                return 1
            if '2' in t or 'two' in t:
                return 2
            if '3' in t or 'three' in t:
                return 3 
        return None
    levels   = []
    mappings = []
    # Parse each row into {key:[features]} mapping
    for lst in df[prop_col]:
        level_txt = None
        mapping   = {}
        if isinstance(lst, list):
            for item in lst:
                if isinstance(item, str) and ':' in item:
                    key, val = item.split(':', 1)
                    key, val = key.strip(), val.strip()
                    if key.lower() == 'levels':
                        level_txt = val
                    else:
                        parts = [v.strip() for v in val.split(',') if v.strip()]
                        mapping[key] = parts
        levels.append(parse_level(level_txt))
        mappings.append(mapping)
    # Collect unique featues across keys 
    unique_feats = {}
    for m in mappings:
        for k, feats in m.items():
            unique_feats.setdefault(k, set()).update(feats)
    # Build selected dummy columns 
    onehot = {}
    for key, feats in unique_feats.items():
        k_s = sanitize_key(key)
        for feat in feats:
            col = f"{k_s}_{sanitize_feat(feat)}"
            if col not in {'has_view_yes', 'has_water_view_yes'}:
                continue
            onehot[col] = [int(feat in m.get(key, [])) for m in mappings]
    # Private_pool flag 
    private_pool = [
        int(any(key.lower() == 'private pool' and
                any(p.lower().startswith('yes') for p in parts)
                for key, parts in m.items()))
        for m in mappings]
    # Assemble output 
    out = df.copy()
    out['levels'] = levels
    if onehot:
        out = pd.concat([out, pd.DataFrame(onehot, index=df.index)], axis=1)
    out['private_pool'] = private_pool    
    # Drop unwanted columns
    out = out.drop(columns=['prop_parking', 'prop_features', 'amenity_gated', 'park_trail'], errors='ignore')

    return out


"""
Add an 'age' column calculated from the 'year_built' column and optionally drops 
the original.
"""
def add_age_column(df, year_col='yearBuilt', date_col='dateSold', new_col='age', drop_original=True):

    df = df.copy()
    df[new_col] = pd.to_datetime(df[date_col]).dt.year - df[year_col]
    if drop_original:
        df = df.drop(columns=[year_col])
    return df


"""
Apply the functions to clean the data 
"""

house_data = home_type(house_data)
house_data = schools(house_data)
house_data = expand_hoa(house_data)
house_data = extract_community_features(house_data, col='community_details') 
house_data = encode_community_features(house_data, col='comm_features')
house_data = lot_features(house_data)
house_data = extract_parking_info(house_data)
house_data = levels_and_features(house_data)
house_data = add_age_column(house_data)



print("Data Cleaning is Completed")






"""
Rename and reorder the columns
"""

rename_dict = {
    'homeType': 'home_type',
    'streetAddress': 'address',
    'lastSoldPrice': 'sale_price',
    'dateSold': 'date_sold',
    'livingAreaValue': 'living_area',
    'bedrooms': 'bedrooms',
    'bathrooms': 'bathrooms',
    'has_garage': 'has_garage',
    'private_pool': 'private_pool',
    'comm_park' : 'park',
    'comm_playground': 'playground',
    'has_hoa': 'hoa',
    'comm_gated_yes': 'gated',
    'lot_feature_above_flood_plain': 'above_flood_plain',
    'lot_feature_city_lot': 'city_lot',
    'lot_feature_historic_district': 'historic_district',
    'has_view_yes': 'view',
    'has_water_view_yes': 'water_view',
    'primary1_rating': 'primary_school_rating',
    'primary1_distance': 'primary_school_distance',
    'middle1_rating': 'middle_school_rating',
    'middle1_distance': 'middle_school_distance',
    'high1_rating': 'high_school_rating',
    'high1_distance': 'high_school_distance'
}


house_data = house_data.rename(columns=rename_dict)

# Define new column order
new_col_order = [
    'city', 'state', 'county', 'address', 'zipcode', 'longitude', 'latitude',
    'sale_price', 'date_sold', 'home_type', 'age', 'living_area', 'bedrooms', 
    'bathrooms', 'levels', 'parking_spaces', 'has_garage', 'private_pool', 'hoa', 'gated',
    'recreational_facilities', 'park', 'playground', 'greenbelt', 'above_flood_plain',
    'city_lot', 'historic_district', 'view', 'water_view', 'primary_school_rating',
    'primary_school_distance', 'middle_school_rating', 'middle_school_distance',
    'high_school_rating', 'high_school_distance'
]

# Reorder the columns
house_data = house_data[new_col_order]


"""
Additional data cleaning: standardize city name, remove rare zipcodes, convert zipcodes 
to strings. Remove NAs. Convert date_sold to datetime format. Clean, bathrooms, bedrooms, 
living_area and parking_spaces columns.
"""

house_data['city'] = (house_data['city'].str.title())

bad_zips = ['32792', '32757', '34786', '32830', '34787', '34734', '32821', '32833']
house_data = house_data[~house_data['zipcode'].astype(str).isin(bad_zips)].reset_index(drop=True)
house_data['zipcode'] = house_data['zipcode'].astype(str)

#print(house_data.isna().sum())
house_data = house_data.dropna().reset_index(drop=True)

house_data['date_sold'] = pd.to_datetime(house_data['date_sold'], errors='coerce')

house_data['age'] = house_data['age'].apply(lambda x: max(x, 1) if pd.notnull(x) else x)


house_data = house_data[house_data['bathrooms'] > 1].reset_index(drop=True)
house_data = house_data[house_data['parking_spaces'] < 6].reset_index(drop=True)
house_data = house_data[house_data['bedrooms'] < 10].reset_index(drop=True)
house_data = house_data[house_data['bathrooms'] < 9].reset_index(drop=True)
house_data = house_data[house_data['living_area'] > 500].reset_index(drop=True)



"""
Build / load the road graph
"""

counties = [
    "Orange County, Florida, USA",
    "Seminole County, Florida, USA",
    "Osceola County, Florida, USA",
    "Lake County, Florida, USA",
]


def load_road_graph():
    """
    Load the lightweight .pkl if present, otherwise fall back to the
    full GraphML. Handles plain pickle, NetworkX read_gpickle, and gzip.
    """
    slim = os.path.join("Data", "orlando_road_network_light.pkl")
    full = os.path.join("Data", "orlando_road_network.graphml")

    if os.path.exists(slim):
        try:
            # Try regular NetworkX uncompressed read
            return nx.read_gpickle(slim)
        except Exception as e1:
            try:
                from networkx.readwrite import gpickle as gp
                return gp.read_gpickle(slim)
            except Exception as e2:
                try:
                    # Try gzip fallback
                    with gzip.open(slim, "rb") as fh:
                        return pickle.load(fh)
                except Exception as e3:
                    print("⚠ Failed all methods to read the slim graph:")
                    print("  Method 1:", repr(e1))
                    print("  Method 2:", repr(e2))
                    print("  Method 3:", repr(e3))
                    raise

    print("Slim graph not found – loading full GraphML (slow, high RAM)…")
    return ox.load_graphml(full)
print("Loading road network (slim if available)…")
G = load_road_graph()

#  Create full edges & exits GeoDFs
edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
# keep only motorway / trunk classes 
edges = edges[edges["highway"].isin(
    ["motorway", "motorway_link", "trunk", "trunk_link"]
)]

nodes = ox.graph_to_gdfs(G, nodes=True, edges=False)
exit_nodes = nodes[nodes["highway"] == "motorway_junction"].copy()
if "name" not in exit_nodes.columns:
    exit_nodes["name"] = ""        




"""
Calculate distances to the highways 
"""
M_TO_MI = 1 / 1609.34          
def _nearest_manhattan(home_xy, exit_xy):
    """
    home_xy  : (N, 2)  array of projected x,y coords for homes
    exit_xy  : (M, 2)  array of projected x,y coords for exits
    returns  : (N,)    minimum L¹ (Manhattan) distance for each home
    """
    tree = cKDTree(exit_xy)
    # p=1  -> Manhattan;  k=1 -> nearest neighbour only
    dist, _ = tree.query(home_xy, k=1, p=1)
    return dist

def highway_exit_distances(
        homes_df, *, edges, exits,
        crs_proj="EPSG:26917",
        min_exit_count=1):
    """
    As before, but Manhattan distance (miles) instead of Euclidean.
    """
    highway_keywords = {
        "I4":      ["i-4", "i 4", "i-4 express", "i 4 express",
                    "i-4 expressway", "i 4 toll"],
        "SR408":   ["408", "east-west expressway", "east-west exp",
                    "spessard l. holland"],
        "SR417":   ["417", "central florida greeneway",
                    "seminole expressway", "greeneway"],
        "SR528":   ["528", "beachline expressway", "beachline"],
        "Turnpike":["turnpike", "fl 91", "florida turnpike", "turnpike toll"],
    }

    # ---------- preparation (unchanged) ----------
    edges_proj = edges.to_crs(crs_proj)
    exits_proj = exits.to_crs(crs_proj)

    homes = homes_df.copy()
    if 'longtitude' in homes.columns and 'longitude' not in homes.columns:
        homes = homes.rename(columns={'longtitude': 'longitude'})

    homes["latitude"]  = pd.to_numeric(homes["latitude"],  errors="coerce")
    homes["longitude"] = pd.to_numeric(homes["longitude"], errors="coerce")
    homes = homes.dropna(subset=["latitude", "longitude"])

    g_homes = gpd.GeoDataFrame(
        homes,
        geometry=gpd.points_from_xy(homes.longitude, homes.latitude),
        crs="EPSG:4326").to_crs(crs_proj)

    # cache home coordinates (projected metres) as plain NumPy
    home_xy = np.column_stack([g_homes.geometry.x, g_homes.geometry.y])

    to_str = lambda v: ", ".join(v) if isinstance(v, list) else str(v)

    # ---------- loop over highways ----------
    for label, kw in highway_keywords.items():
        kw_lower = [k.lower() for k in kw]

        kw_mask = (
            edges_proj["name"].apply(to_str).str.lower().apply(
                lambda s: any(k in s for k in kw_lower))
            |
            edges_proj["ref"].apply(to_str).str.lower().apply(
                lambda s: any(k in s for k in kw_lower))
        )
        edges_this = edges_proj[kw_mask]
        if edges_this.empty:
            g_homes[f"dist_to_{label.lower()}"] = pd.NA
            continue

        exits_this = gpd.sjoin(
            exits_proj, edges_this[["geometry"]],
            how="inner", op="intersects")

        if len(exits_this) < min_exit_count:
            g_homes[f"dist_to_{label.lower()}"] = pd.NA
            continue

        # ----- Manhattan distance via cKDTree -----
        exit_xy = np.column_stack([exits_this.geometry.x, exits_this.geometry.y])
        manh_m  = _nearest_manhattan(home_xy, exit_xy)     # metres
        g_homes[f"dist_to_{label.lower()}"] = manh_m * M_TO_MI

    return g_homes.drop(columns="geometry")

# Apply the funcion 
house_data = highway_exit_distances(house_data,edges=edges,exits=exit_nodes)


print("Distance to the Highways is Calculated")


"""
Remove rows where the specified column falls outside the given quantile range.
"""

def remove_outliers(df, column, lower_quantile, upper_quantile):

    lower_bound = df[column].quantile(lower_quantile)
    upper_bound = df[column].quantile(upper_quantile)
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)].reset_index(drop=True)

house_data = remove_outliers(house_data, 'sale_price', 0.01, 0.99) 




# Plot distribution of sale price
formatter = FuncFormatter(lambda x, _: f'{x:,.0f}') # Formatter to display numbers 

plt.figure(figsize=(10, 5))
house_data['sale_price'].hist(bins=100, edgecolor='black', color='royalblue')
plt.xlabel('Sale Price')
plt.ylabel('Number of Properties')


plt.gca().xaxis.set_major_formatter(formatter)
plt.grid(False)
plt.tight_layout()
plt.savefig("Figures/sale_price_distribution.png", dpi=300) 



print('Figure /"sale_price_distribution.png/" is Created')


# Plot number of properties sld by month
monthly_sales = house_data.resample('M', on='date_sold').size() # Group by month and count sales

plt.figure(figsize=(12, 6))
monthly_sales.plot(kind='line', marker='o', linewidth=5.5)
plt.title('Number of Properties Sold Per Month')
plt.xlabel('Date Sold')
plt.ylabel('Number of Properties')
plt.ylim(100, 800)
plt.grid(True)
plt.tight_layout()
plt.savefig("Figures/monthly_sales.png", dpi=300)




# Correlations between columns
columns_to_check = [
    'home_type', 'age', 'living_area', 'bedrooms', 'bathrooms', 'levels',
    'parking_spaces', 'has_garage', 'private_pool', 'hoa', 'gated',
    'recreational_facilities', 'park', 'playground', 'greenbelt',
    'above_flood_plain', 'city_lot', 'historic_district', 'view',
    'water_view', 'primary_school_rating', 'primary_school_distance',
    'middle_school_rating', 'middle_school_distance',
    'high_school_rating', 'high_school_distance',
    'dist_to_i4', 'dist_to_sr408', 'dist_to_sr417',
    'dist_to_sr528', 'dist_to_turnpike'
]

# Compute correlation matrix
corr_matrix = house_data[columns_to_check].corr()

# Create a mask to extract only upper triangle (no repeats)
mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
filtered = corr_matrix.where(mask)

# Flatten and filter for abs(corr) > 0.5
high_corr = (
    filtered.stack()
    .reset_index()
    .rename(columns={"level_0": "Variable 1", "level_1": "Variable 2", 0: "Correlation"})
)

# Filter by absolute correlation value
high_corr = high_corr[high_corr["Correlation"].abs() > 0.5]

# Sort by absolute correlation
high_corr = high_corr.reindex(high_corr["Correlation"].abs().sort_values(ascending=False).index)

# Round for better formatting (optional)
high_corr = high_corr.copy()
high_corr["Correlation"] = high_corr["Correlation"].round(2)

latex_table = high_corr.to_latex(
    escape=True,
    index=False,
    float_format="%.2f",
    caption="Pairs of Variables with High Correlation",
    label="tab:high_corr"
)

# Save to .tex file
with open("Tables/high_corr_table.tex", "w") as f:
    f.write(latex_table)


print('Table /"high_corr_table.tex/" is Created')




"""
Add a column with the row-wise average of specified columns.
"""
def find_average(df, columns, new_col):
    df = df.copy()
    existing_cols = [col for col in columns if col in df.columns]
    df[new_col] = df[existing_cols].mean(axis=1, skipna=True)
    df = df.drop(columns=existing_cols)
    return df


"""
Compute the row-wise minimum of specified columns
"""

def find_min(df, columns, new_col):
 
    df = df.copy()
    existing_cols = [col for col in columns if col in df.columns]
    df[new_col] = df[existing_cols].min(axis=1, skipna=True)
    df = df.drop(columns=existing_cols)
    return df



# List of columns
school_dist_cols = ['primary_school_distance', 'middle_school_distance', 'high_school_distance']
school_rate_cols = ['primary_school_rating', 'middle_school_rating', 'high_school_rating']
highway_dist_cols = ['dist_to_i4', 'dist_to_sr408', 'dist_to_sr417', 'dist_to_sr528', 'dist_to_turnpike']


house_data = find_average(house_data, school_dist_cols, 'avg_distance_to_schools')
house_data = find_average(house_data, school_rate_cols, 'avg_schools_rating')
house_data = find_min(house_data, highway_dist_cols, 'min_distance_highway')

house_data['sqft_per_bedroom'] = house_data['living_area'] / house_data['bedrooms']






# --- Load CPI table ----------------------------------------------------------
cpi = pd.read_csv("Data/SeriesReport.csv")          # path as needed

# --- Reshape from wide to long ----------------------------------------------
month_lookup = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct':10, 'Nov':11, 'Dec':12
}

cpi_long = (
    cpi.melt(id_vars='Year', value_name='cpi', var_name='month')
       .query("month in @month_lookup.keys()")          # drop HALF1/HALF2
       .assign(month_num=lambda d: d['month'].map(month_lookup),
               date=lambda d: pd.to_datetime(
                     dict(year=d['Year'], month=d['month_num'], day=1)))
       .loc[:, ['date', 'cpi']]
       .dropna()
)

# --- Pick base period (May 2025) and build deflator ------------------------
base_date = pd.Timestamp("2025-05-01")
cpi_base  = cpi_long.loc[cpi_long['date'] == base_date, 'cpi'].iloc[0]

cpi_long['deflator'] = cpi_long['cpi'] / cpi_base       # ratio, not percent

# --- Load your sales data ----------------------------------------------------
house_data['month_date'] = house_data['date_sold'].values.astype('datetime64[M]')

# --- Merge and create real price --------------------------------------------
house_data = (
    house_data.merge(cpi_long[['date', 'deflator']],
                left_on='month_date', right_on='date', how='left')
          .assign(sale_price=lambda d: d['sale_price'] / d['deflator'])
          .drop(columns=['date', 'deflator', 'month_date'])
)



# Columns to summarize
columns = [
    'sale_price', 'home_type', 'age', 'living_area', 'bedrooms', 'bathrooms',
    'avg_distance_to_schools', 'avg_schools_rating', 'min_distance_highway', 
   'sqft_per_bedroom'
]


# Generate summary stats and filter desired rows
summary = house_data[columns].describe().T.loc[:, ['mean', 'std', 'min', '50%', 'max']]
summary = summary.round(2)

# Convert to LaTeX with escaping enabled
latex = summary.to_latex(
    escape=True,
    float_format="%.2f",
    caption="Summary Statistics",
    label="tab:summary_stats"
)

# Save to .tex file
with open("Tables/summary_statistics_table.tex", "w") as f:
    f.write(latex)
    
print('Table /"summary_statistics_table.tex/" is Created')
    

# Split house_data by zipcode and store in separate variables
for zc in house_data['zipcode'].unique():
    var_name = f"orlando_{zc}"
    globals()[var_name] = house_data[house_data['zipcode'] == zc].copy()
    

    
 # Save each zipcode subset to a CSV file
for zc in house_data['zipcode'].unique():
    filename = f"Data/orlando_{zc}.csv"
    df_zip = house_data[house_data['zipcode'] == zc]
    df_zip.to_csv(filename, index=False)   

house_data.to_csv("Data/house_data_full.csv",
                  index=False)          # index=False keeps the row numbers out

