import pandas as pd
import sys

# Force UTF-8 output to avoid Windows cp1252 encoding errors
sys.stdout.reconfigure(encoding="utf-8")

# -- Load the dataset ---------------------------------------------------------
csv_path   = "traffic_integration_data.csv"
clean_path = "traffic_integration_data_cleaned.csv"

df = pd.read_csv(csv_path)

print("=" * 45)
print("  EMBER 1 - Data Validation Report")
print("=" * 45)

# -- 1. Missing values --------------------------------------------------------
print("\n[1] Missing Values")
print("-" * 30)
missing = df.isnull().sum()
if missing.sum() == 0:
    print("  No missing values found [OK]")
else:
    print(missing[missing > 0].to_string())

# -- 2. Validate Vehicle_Count (must be positive integers) --------------------
print("\n[2] Vehicle_Count - Validity Check")
print("-" * 30)

df["Vehicle_Count"] = pd.to_numeric(df["Vehicle_Count"], errors="coerce")
invalid_vc = df[df["Vehicle_Count"].isna() | (df["Vehicle_Count"] <= 0)]

if invalid_vc.empty:
    print("  All values are valid positive numbers [OK]")
else:
    print(f"  {len(invalid_vc)} invalid row(s) found:")
    print(invalid_vc[["Timestamp", "Vehicle_Count"]].to_string(index=False))

# -- 3. Validate Average_Speed_kmh (must be non-negative numbers) -------------
print("\n[3] Average_Speed_kmh - Validity Check")
print("-" * 30)

df["Average_Speed_kmh"] = pd.to_numeric(df["Average_Speed_kmh"], errors="coerce")
invalid_sp = df[df["Average_Speed_kmh"].isna() | (df["Average_Speed_kmh"] < 0)]

if invalid_sp.empty:
    print("  All values are valid [OK]")
else:
    print(f"  {len(invalid_sp)} invalid row(s) found:")
    print(invalid_sp[["Timestamp", "Average_Speed_kmh"]].to_string(index=False))

# -- 4. Summary statistics ----------------------------------------------------
print("\n[4] Summary Statistics")
print("-" * 30)

vc = df["Vehicle_Count"].dropna()
sp = df["Average_Speed_kmh"].dropna()

print("\nVehicle Count")
print(f"  Min     : {vc.min():.0f}")
print(f"  Max     : {vc.max():.0f}")
print(f"  Average : {vc.mean():.2f}")

print("\nAverage Speed (km/h)")
print(f"  Min     : {sp.min():.2f}")
print(f"  Max     : {sp.max():.2f}")
print(f"  Average : {sp.mean():.2f}")

# -- 5. Drop invalid rows & save cleaned CSV ----------------------------------
before = len(df)
df_clean = df.dropna(subset=["Vehicle_Count", "Average_Speed_kmh"])
df_clean = df_clean[
    (df_clean["Vehicle_Count"] > 0) &
    (df_clean["Average_Speed_kmh"] >= 0)
]
after = len(df_clean)

df_clean.to_csv(clean_path, index=False)

print("\n[5] Cleaned File")
print("-" * 30)
print(f"  Rows before cleaning : {before}")
print(f"  Rows after  cleaning : {after}")
print(f"  Rows removed         : {before - after}")
print(f"  Saved to             : {clean_path}")
print("\n" + "=" * 45)
print("  Validation complete!")
print("=" * 45)
