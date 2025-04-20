# Purpose: Query Gaia DR3 for potential members of selected star clusters,
#          apply basic cleaning filters, and save the results.
# EDITED: Added more clusters to increase dataset size.

# Import necessary libraries
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.gaia import Gaia
from astropy.table import Table, vstack
import os

# Disable warnings for cleaner output (optional)
import warnings
warnings.filterwarnings('ignore', module='astropy.io.votable.tree')
warnings.filterwarnings('ignore', category=UserWarning, module='astroquery.gaia')


# --- Define Cluster Parameters ---
# ADDED MORE CLUSTERS to increase dataset size and improve ML training viability.
# Parameters are approximate and may require fine-tuning.
cluster_params = {
    # Open Clusters
    'Pleiades': {
        'coord': SkyCoord(ra=56.75*u.deg, dec=24.1167*u.deg, frame='icrs'), 'radius': 1.5 * u.deg,
        'parallax_range': (6.5, 8.5), 'pmra_range': (18, 22), 'pmdec_range': (-48, -42), 'type': 'Open'
    },
    'Hyades': {
        'coord': SkyCoord(ra=67.0*u.deg, dec=16.0*u.deg, frame='icrs'), 'radius': 5.0 * u.deg,
        'parallax_range': (19, 24), 'pmra_range': (90, 130), 'pmdec_range': (-40, -10), 'type': 'Open'
    },
    'Praesepe': { # M44 / Beehive
        'coord': SkyCoord(ra=130.0167*u.deg, dec=19.6667*u.deg, frame='icrs'), 'radius': 1.5 * u.deg,
        'parallax_range': (5.0, 6.5), 'pmra_range': (-38, -32), 'pmdec_range': (-16, -10), 'type': 'Open'
    },
    'M67': { # Older Open Cluster
        'coord': SkyCoord(ra=132.825*u.deg, dec=11.8167*u.deg, frame='icrs'), 'radius': 0.5 * u.deg,
        'parallax_range': (1.0, 1.4), 'pmra_range': (-12, -10), 'pmdec_range': (-4, -2), 'type': 'Open'
     },
     'NGC 188': { # Very Old Open Cluster
        'coord': SkyCoord(ra=12.125*u.deg, dec=85.258*u.deg, frame='icrs'), 'radius': 0.3 * u.deg,
        'parallax_range': (0.4, 0.7), 'pmra_range': (-3, 0), 'pmdec_range': (-2, 1), 'type': 'Open'
     },
     'NGC 6791': { # Old, Metal-Rich Open Cluster
         'coord': SkyCoord(ra=290.217*u.deg, dec=37.772*u.deg, frame='icrs'), 'radius': 0.2 * u.deg,
         'parallax_range': (0.20, 0.30), 'pmra_range': (-1.5, 0.5), 'pmdec_range': (-3, -1), 'type': 'Open'
     },
     'NGC 752': {
        'coord': SkyCoord(ra=28.38*u.deg, dec=37.85*u.deg, frame='icrs'), 'radius': 0.7 * u.deg,
        'parallax_range': (2.0, 2.6), 'pmra_range': (8, 14), 'pmdec_range': (-14, -9), 'type': 'Open'
    },
    # Globular Clusters
    'M13': { # Hercules Globular Cluster
        'coord': SkyCoord(ra=250.4234*u.deg, dec=36.4613*u.deg, frame='icrs'), 'radius': 0.3 * u.deg,
        'parallax_range': (0.05, 0.25), 'pmra_range': (-5, 5), 'pmdec_range': (-5, 5), 'type': 'Globular'
     },
    'M3': {
        'coord': SkyCoord(ra=205.5490*u.deg, dec=28.3774*u.deg, frame='icrs'), 'radius': 0.3 * u.deg,
        'parallax_range': (0.05, 0.25), 'pmra_range': (-5, 5), 'pmdec_range': (-5, 5), 'type': 'Globular'
    },
    '47Tuc': { # 47 Tucanae
        'coord': SkyCoord(ra=6.0225*u.deg, dec=-72.0814*u.deg, frame='icrs'), 'radius': 0.7 * u.deg,
        'parallax_range': (0.15, 0.30), 'pmra_range': (3, 8), 'pmdec_range': (-4, -1), 'type': 'Globular'
    },
    'M5': {
        'coord': SkyCoord(ra=229.6375*u.deg, dec=2.0811*u.deg, frame='icrs'), 'radius': 0.3 * u.deg,
        'parallax_range': (0.08, 0.20), 'pmra_range': (1, 5), 'pmdec_range': (-8, -4), 'type': 'Globular'
    },
    'M15': {
        'coord': SkyCoord(ra=322.4933*u.deg, dec=12.1671*u.deg, frame='icrs'), 'radius': 0.2 * u.deg,
        'parallax_range': (0.05, 0.15), 'pmra_range': (-1, 2), 'pmdec_range': (-6, -2), 'type': 'Globular'
    },
    'M92': {
        'coord': SkyCoord(ra=259.28*u.deg, dec=43.136*u.deg, frame='icrs'), 'radius': 0.2 * u.deg,
        'parallax_range': (0.08, 0.18), 'pmra_range': (0, 4), 'pmdec_range': (-6, -3), 'type': 'Globular'
    },
    'OmegaCen': { # NGC 5139 - Very massive globular, maybe former dwarf galaxy core
        'coord': SkyCoord(ra=201.697*u.deg, dec=-47.479*u.deg, frame='icrs'), 'radius': 0.8 * u.deg,
        'parallax_range': (0.15, 0.25), 'pmra_range': (-4, -2), 'pmdec_range': (-7, -5), 'type': 'Globular'
    }
}

# --- Function to Query Gaia and Clean Data ---
def get_clean_cluster_data(cluster_name, params):
    """
    Queries Gaia DR3 for a cluster, retrieves data, and applies basic cleaning filters.
    """
    print(f"--- Processing {cluster_name} ---")
    coord = params['coord']
    radius = params['radius']
    plx_min, plx_max = params['parallax_range']
    pmra_min, pmra_max = params['pmra_range']
    pmdec_min, pmdec_max = params['pmdec_range']
    cluster_type = params['type'] # Get type from params

    # Construct ADQL Query
    query = f"""
    SELECT
        source_id, ra, dec, parallax, parallax_error,
        pmra, pmra_error, pmdec, pmdec_error,
        radial_velocity, radial_velocity_error,
        phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
        ruwe
    FROM
        gaiadr3.gaia_source
    WHERE
        1=CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {coord.ra.deg}, {coord.dec.deg}, {radius.to(u.deg).value}))
    AND -- Apply broad filters in the query to limit download size
        parallax BETWEEN {plx_min - 2.0} AND {plx_max + 2.0}
    AND
        pmra BETWEEN {pmra_min - 10.0} AND {pmra_max + 10.0}
    AND
        pmdec BETWEEN {pmdec_min - 10.0} AND {pmdec_max + 10.0}
    """

    print(f"Launching Gaia job for {cluster_name}...")
    try:
        Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
        Gaia.ROW_LIMIT = -1
        job = Gaia.launch_job_async(query, dump_to_file=False)
        results = job.get_results()
        print(f"Retrieved {len(results)} stars initially for {cluster_name}.")
    except Exception as e:
        print(f"Error querying Gaia for {cluster_name}: {e}")
        # Return an empty table on error to avoid breaking the vstack later
        # Define columns based on a successful query's potential output
        colnames = ['source_id', 'ra', 'dec', 'parallax', 'parallax_error', 'pmra', 'pmra_error',
                    'pmdec', 'pmdec_error', 'radial_velocity', 'radial_velocity_error',
                    'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 'ruwe']
        # Create empty table with placeholder dtypes (adjust if needed)
        empty_table = Table(names=colnames + ['cluster_name', 'cluster_type'],
                            dtype=['i8'] + ['f8']*10 + ['f4']*3 + ['f4'] + ['<U20', '<U10'])
        return empty_table


    # Apply Cleaning Filters to the results table
    if len(results) == 0:
        print(f"No stars retrieved for {cluster_name} initially, skipping filters.")
        # Need to return an empty table structured correctly
        empty_table = Table(names=results.colnames + ['cluster_name', 'cluster_type'],
                           dtype=[results[col].dtype for col in results.colnames] + ['<U20', '<U10'])
        return empty_table


    print(f"Applying cleaning filters...")
    try:
      # Filter 0: Valid parallax measurement
      valid_plx_mask = ~results['parallax'].mask & (results['parallax_error'] > 0)
      # Filter 1: Parallax range and significance (S/N > 5)
      plx_filter = valid_plx_mask & \
                   (results['parallax'] > plx_min) & \
                   (results['parallax'] < plx_max) & \
                   (results['parallax'] / results['parallax_error'] > 5)
      # Filter 2: Proper Motion range (ensure pmra/pmdec are not masked)
      valid_pm_mask = ~results['pmra'].mask & ~results['pmdec'].mask
      pm_filter = valid_pm_mask & \
                  (results['pmra'] > pmra_min) & \
                  (results['pmra'] < pmra_max) & \
                  (results['pmdec'] > pmdec_min) & \
                  (results['pmdec'] < pmdec_max)
      # Filter 3: RUWE (astrometric quality)
      results['ruwe'] = results['ruwe'].filled(np.inf) # Replace masked values with infinity
      ruwe_filter = (results['ruwe'] < 1.4)
      # Combine filters
      final_mask = plx_filter & pm_filter & ruwe_filter
      cleaned_results = results[final_mask]
    except Exception as filter_error:
        print(f"  Error during filtering for {cluster_name}: {filter_error}")
        # Return empty table if filtering fails
        empty_table = Table(names=results.colnames + ['cluster_name', 'cluster_type'],
                           dtype=[results[col].dtype for col in results.colnames] + ['<U20', '<U10'])
        return empty_table


    print(f"Retained {len(cleaned_results)} likely members for {cluster_name} after cleaning.")
    print(f"Parallax range applied: {plx_min:.2f}-{plx_max:.2f} mas (with S/N > 5)")
    print(f"PM RA range applied:  {pmra_min:.2f}-{pmra_max:.2f} mas/yr")
    print(f"PM Dec range applied: {pmdec_min:.2f}-{pmdec_max:.2f} mas/yr")
    print(f"RUWE filter applied: < 1.4")
    print("-"*(len(cluster_name) + 25))

    # Add cluster name and type for easier identification later
    if len(cleaned_results) > 0:
       cleaned_results['cluster_name'] = cluster_name
       cleaned_results['cluster_type'] = cluster_type # Use type from params
    else:
        # Return an empty table with the correct columns if no stars are retained
        empty_table = Table(names=results.colnames + ['cluster_name', 'cluster_type'],
                           dtype=[results[col].dtype for col in results.colnames] + ['<U20', '<U10'])
        return empty_table

    return cleaned_results

# --- Main Execution ---
output_filename = 'cleaned_cluster_members.fits'

# Check if output file already exists
file_exists = os.path.isfile(output_filename)
if file_exists:
    print(f"\nNOTE: The output file '{output_filename}' already exists and will be overwritten.")
    print("If you want to keep the existing file, press Ctrl+C now to cancel.")
    # Optional: You could add a user prompt here, or add a parameter to skip overwriting

all_cluster_members = [] # Store results in a list

print("Starting Data Acquisition Process...")
print(f"Processing {len(cluster_params)} clusters...")
for name, params in cluster_params.items():
    data = get_clean_cluster_data(name, params)
    # Important: Check if data is None (query error) or is an Astropy Table before appending
    if data is not None and isinstance(data, Table):
        all_cluster_members.append(data)
    else:
        print(f"Function returned invalid data type or None for {name}, skipping.")

# --- Combine into a single table ---
# Ensure the list is not empty and contains only non-empty tables if needed,
# although vstack can handle empty tables if columns match.
valid_tables = [tbl for tbl in all_cluster_members if tbl is not None and len(tbl.colnames)>0]

if valid_tables:
    # Ensure all tables have the same columns before stacking
    # This should be handled by returning structured empty tables on error/no results
    try:
        combined_data = vstack(valid_tables, metadata_conflicts='silent')
        print("\n--- Data Acquisition Summary ---")
        print(f"Attempted processing for {len(cluster_params)} clusters.")
        if len(combined_data) > 0:
           print(f"Total likely members combined: {len(combined_data)}")
           # Display counts per cluster
           print("Members per cluster:")
           # Use group_by on the combined table
           grouped = combined_data.group_by('cluster_name')
           print(grouped.groups.keys) # Print cluster names
           # Calculate and print counts per group robustly
           counts = np.diff(grouped.groups.indices)
           for key, count in zip(grouped.groups.keys['cluster_name'], counts):
               print(f"- {key}: {count}")

           # Save the combined table to a FITS file for the next script
           try:
               # File existence was already checked at the start
               combined_data.write(output_filename, format='fits', overwrite=True)
               if file_exists:
                   print(f"\nExisting file was overwritten. Cleaned data saved to '{output_filename}'")
               else:
                   print(f"\nCleaned data saved to '{output_filename}'")
           except Exception as e:
                print(f"\nError saving data to FITS file: {e}")
        else:
            print("No likely members found across all clusters after filtering.")

    except Exception as stack_error:
        print(f"\nError combining cluster data tables: {stack_error}")
        print("Please check the output for individual cluster processing errors.")

else:
    print("\nNo valid cluster data tables were generated to combine.")

print("\nData Acquisition Script Finished.")