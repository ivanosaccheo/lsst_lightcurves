import sqlite3
import pandas as pd


#database_path = "../DATA/master.db"
#database_path = "/beegfs/AGILE/agile/data/catalog/dr1_new_new/db/20250703/master.db"
database_path = "/staff1/saccheo/agile/master.db"

def query_agile(query, params = None):
    conn = sqlite3.connect(database_path)
    table = pd.read_sql(query, conn, params=params)
    conn.close()
    return table

def query_available_patches():
    query = """SELECT DISTINCT patch FROM Object"""
    patches = query_agile(query)["patch"].to_list()
    return patches

def query_available_tracts():
    query = """SELECT DISTINCT tract FROM Object"""
    tracts = query_agile(query)["tract"].to_list()
    return tracts

def get_table_names():
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    return [row[0] for row in cursor.fetchall()]


def query_coadd_photometry(keep_psf = True, patch = query_available_patches(),
                           keep_cmodel = True, keep_extendedness = True, max_rows = 1e10):
    
    column_to_query = ["objectId", "refExtendedness", "detect_isPrimary"]
    for band in "ugrizy":
        temp = []
        if keep_psf:
            temp.extend([f"{band}_psfFlux", f"{band}_psfFluxErr"])
        if keep_cmodel:
            temp.extend([f"{band}_cModelFlux", f"{band}_cModelFluxErr"])
        if keep_extendedness: 
            temp.append(f"{band}_extendedness")
        column_to_query.extend(temp)
    column_to_query = ", ".join(column_to_query)

    if isinstance(patch, (list, tuple)):
        placeholders = ",".join("?" for _ in patch)
        query = f"""SELECT {column_to_query}
                    FROM Object
                    WHERE patch IN ({placeholders})
                    AND detect_isPrimary = 1
                    LIMIT ?
                """
        params = list(patch) + [max_rows]
    else:
        query = f"""SELECT {column_to_query}
                    FROM Object
                    WHERE patch = ?
                    AND detect_isPrimary = 1
                    LIMIT ?
                """
        params = [patch, max_rows]

    table = query_agile(query, params=params)  
    return table  


def query_force_photometry_without_coadd(snr=5, patch=23, band=None, max_rows=None,
                                         difference_flux = False):
    if not difference_flux:
        query = """
        SELECT f.objectId, f.band, ccd.expMidptMJD, f.psfFlux, f.psfFluxErr
        FROM ForcedSource AS f
        JOIN CcdVisit AS ccd USING (ccdVisitId)
        WHERE (f.psfFlux >= ? * f.psfFluxErr) AND f.psfFlux > 0
        AND f.detect_isPrimary = 1 
        """
        params = [snr]
    else:
        query = """
        SELECT f.objectId, f.band, ccd.expMidptMJD, f.psfDiffFlux AS psfFlux, 
        f.psfDiffFluxErr AS psfFluxErr
        FROM ForcedSource AS f
        JOIN CcdVisit AS ccd USING (ccdVisitId)
        WHERE f.psfDiffFlux IS NOT NULL AND f.psfDiffFluxErr IS NOT NULL 
        AND f.detect_isPrimary = 1 
        """
        params = []

    if patch is not None:
        if isinstance(patch, (list, tuple)):
            placeholders = ",".join("?" for _ in patch)
            query += f" AND f.patch IN ({placeholders})"
            params.extend(patch)
        else:
            query += " AND f.patch = ?"
            params.append(patch)

    if band is not None:
        if isinstance(band, (list, tuple)):
            placeholders = ",".join("?" for _ in band)
            query += f" AND f.band IN ({placeholders})"
            params.extend(band)
        else:
            query += " AND f.band = ?"
            params.append(band)

    if max_rows is not None:
        query += " LIMIT ?"
        params.append(max(1, int(max_rows)))

    table = query_agile(query, params=params)  
    return table



def query_force_photometry_with_coadd(snr = 5, patch = 23, max_rows = None, band = None,
                                       difference_flux = False):
    if not difference_flux:
        query = """
            SELECT f.objectId, f.band, ccd.expMidptMJD, f.psfFlux, f.psfFluxErr,
            o.u_psfFlux AS u_coadd,
            o.g_psfFlux AS g_coadd,
            o.r_psfFlux AS r_coadd,
            o.i_psfFlux AS i_coadd,
            o.z_psfFlux AS z_coadd,
            o.y_psfFlux AS y_coadd
            FROM ForcedSource as f
            JOIN CcdVisit as ccd USING (ccdVisitId)
            JOIN Object as o USING (objectId)
            WHERE (f.psfFlux >= ? * f.psfFluxErr) AND f.psfFlux > 0
            AND f.detect_isPrimary = 1 
            """
        params = [snr]

    else: 
        query = """
            SELECT f.objectId, f.band, ccd.expMidptMJD, f.psfDiffFlux AS psfFlux, 
            psfDiffFluxErr AS psfFluxErr,
            o.u_psfFlux AS u_coadd,
            o.g_psfFlux AS g_coadd,
            o.r_psfFlux AS r_coadd,
            o.i_psfFlux AS i_coadd,
            o.z_psfFlux AS z_coadd,
            o.y_psfFlux AS y_coadd
            FROM ForcedSource as f
            JOIN CcdVisit as ccd USING (ccdVisitId)
            JOIN Object as o USING (objectId)
            WHERE f.psfDiffFlux IS NOT NULL AND f.psfDiffFluxErr IS NOT NULL 
            AND f.detect_isPrimary = 1 
            """
        params = []
    
    if patch is not None:
        if isinstance(patch, (list, tuple)):
            placeholders = ",".join("?" for _ in patch)
            query += f" AND f.patch IN ({placeholders})"
            params.extend(patch)
        else:
            query += " AND f.patch = ?"
            params.append(patch)

    if band is not None:
        if isinstance(band, (list, tuple)):
            placeholders = ",".join("?" for _ in band)
            query += f" AND f.band IN ({placeholders})"
            params.extend(band)
        else:
            query += " AND f.band = ?"
            params.append(band)

    if max_rows is not None:
        query += " LIMIT ?"
        params.append(max(1, int(max_rows)))

    table = query_agile(query, params=params)  
    return table


def query_force_photometry(snr = 5, patch = 23, max_rows = None, band = None,
                            coadd = False, difference_flux = False):
    
    if coadd:
        table = query_force_photometry_with_coadd(snr=snr, patch=patch, max_rows = max_rows,
                                                  band = band, difference_flux=difference_flux)
    else:
        table = query_force_photometry_without_coadd(snr=snr, patch=patch, max_rows = max_rows,
                                                  band = band, difference_flux=difference_flux)
        
    return table


def query_truth_table(max_rows = None, 
                      columns = ["ID", "Z", "M", "is_optical_type2", "is_agn",
                      "[lsst-u_total]", "[lsst-g_total]","[lsst-r_total]", "[lsst-i_total]",
                      "[lsst-z_total]", "[lsst-y_total]"]):
    
    if isinstance(columns, str):
        if (columns.casefold() == "all") or (columns == "*"):
            columns = "Truth.*"
        else:
            columns = f"Truth.{columns}"
    elif isinstance(columns, list):
        columns = [f"Truth.{col}" for col in columns]
        columns = ",".join(columns)
    else:
        print("Invalid columns passed")
        return
    query = f"""SELECT {columns}, m.ObjectId
           FROM Truth
           JOIN MatchesTruth AS m ON Truth.ID=m.match_id
           """
    params = None
    if max_rows is not None:
        query += " LIMIT ?"
        params = [max(1, int(max_rows))]
    table = query_agile(query, params = params)
    return table

def query_lightcurve(objectid, snr=5, band=None, max_rows=None,
                                    difference_flux = False):
    if not difference_flux:
        query = f"""
        SELECT f.objectId, f.band, ccd.expMidptMJD, f.psfFlux, f.psfFluxErr
        FROM ForcedSource AS f
        JOIN CcdVisit AS ccd USING (ccdVisitId)
        WHERE (f.psfFlux >= ? * f.psfFluxErr) AND f.psfFlux > 0
        AND f.detect_isPrimary = 1 
        AND f.objectId = ?
        """
        params = [snr, int(objectid)]
    else:
        query = """
        SELECT f.objectId, f.band, ccd.expMidptMJD, f.psfDiffFlux AS psfFlux, 
        f.psfDiffFluxErr AS psfFluxErr
        FROM ForcedSource AS f
        JOIN CcdVisit AS ccd USING (ccdVisitId)
        WHERE f.psfDiffFlux IS NOT NULL AND f.psfDiffFluxErr IS NOT NULL 
        AND f.detect_isPrimary = 1 
        AND f.objectId = ?
        """
        params = [int(objectid)]

    if band is not None:
        if isinstance(band, (list, tuple)):
            placeholders = ",".join("?" for _ in band)
            query += f" AND f.band IN ({placeholders})"
            params.extend(band)
        else:
            query += " AND f.band = ?"
            params.append(band)

    if max_rows is not None:
        query += " LIMIT ?"
        params.append(max(1, int(max_rows)))

    table = query_agile(query, params=params)  
    return table

