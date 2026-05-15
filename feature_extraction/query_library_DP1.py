from lsst.rsp import get_tap_service


def query_dp1(query, max_rows = None):
    service = get_tap_service("tap")
    job = service.submit_job(query, maxrec = max_rows)
    job.run()
    job.wait(phases=['COMPLETED', 'ERROR'])
    print('Job phase is', job.phase)
    if job.phase == 'ERROR':
        job.raise_if_error()
        return
    table = job.fetch_result().to_table()
    return table.to_pandas()


def query_available_tracts(ra = 53.2, dec = -28.1, radius = 5):
    #coordinates are for CDFS
    query = f"""SELECT DISTINCT tract FROM dp1.Object
           WHERE CONTAINS(POINT('ICRS', coord_ra, coord_dec),
           CIRCLE('ICRS', {ra}, {dec}, {radius})) = 1"""
    res = query_dp1(query)
    return [int(i) for i in list(res["tract"])]

def query_available_patches(tract):
    query = f"""SELECT DISTINCT patch FROM dp1.Object
            WHERE tract = {tract}"""
    res = query_dp1(query)
    return [int(i) for i in list(res["patch"])]

def query_force_photometry_with_coadd(tract, patch, snr = 5, max_rows = None,  difference_flux = False):
    if not difference_flux:
        query = f"""
            SELECT f.objectId, f.band, v.expMidptMJD, f.psfFlux, f.psfFluxErr,
            o.u_psfFlux AS u_coadd,
            o.g_psfFlux AS g_coadd,
            o.r_psfFlux AS r_coadd,
            o.i_psfFlux AS i_coadd,
            o.z_psfFlux AS z_coadd,
            o.y_psfFlux AS y_coadd
            FROM dp1.ForcedSource AS f
            JOIN dp1.Visit AS v ON v.visit = f.visit
            JOIN dp1.Object AS o ON f.objectId =o.objectId
            WHERE o.tract = {tract} AND o.patch = {patch}
            AND (f.psfFlux >= {snr} * f.psfFluxErr)
                         AND f.psfFlux > 0 """
    else: 
        query = f"""
            SELECT f.objectId, f.band, v.expMidptMJD, f.psfDiffFlux AS psfFlux, 
            psfDiffFluxErr AS psfFluxErr,
            o.u_psfFlux AS u_coadd,
            o.g_psfFlux AS g_coadd,
            o.r_psfFlux AS r_coadd,
            o.i_psfFlux AS i_coadd,
            o.z_psfFlux AS z_coadd,
            o.y_psfFlux AS y_coadd
            FROM dp1.ForcedSource AS f
            JOIN dp1.Visit AS v ON v.visit = f.visit
            JOIN dp1.Object AS o ON f.objectId = o.objectId
            WHERE o.tract = {tract} AND o.patch = {patch}
            AND f.psfDiffFlux IS NOT NULL AND f.psfDiffFluxErr IS NOT NULL 
            """
    table = query_dp1(query, max_rows = max_rows)
    return table

def query_force_photometry(tract, patch , snr = 5, max_rows = None, 
                            coadd = False, difference_flux = False):
    
    if coadd:
        table = query_force_photometry_with_coadd(tract, patch, snr=snr, max_rows = max_rows,
                                                  difference_flux=difference_flux)
    else:
        table = query_force_photometry_without_coadd(tract, patch, snr=snr, max_rows = max_rows,
                                                     difference_flux=difference_flux)     
    return table


def get_object_table(ra = 53.2, dec = -28.1, radius = 1, max_err = 0.1, max_rows = 1e8):
    service = get_tap_service("tap")
    ra, dec, radius = float(ra), float(dec), float(radius)
    max_err = float(max_err)
    max_rows = int(max_rows) #avoid passing to the query strange stuff
    assert service is not None
    column_to_keep = ["objectId", "coord_ra", "coord_dec", "refExtendedness"]
    for band in "ugrizy":
        temp = [f"{band}_psfMag", f"{band}_cModelMag", f"{band}_extendedness"]
        column_to_keep.extend(temp)
    columns_query = ", ".join(column_to_keep)
    query = f"""SELECT {columns_query}
                FROM dp1.Object
                WHERE CONTAINS(POINT('ICRS', coord_ra, coord_dec),
                CIRCLE('ICRS', {ra}, {dec}, {radius})) = 1
                AND r_cModelMagerr < {max_err}       
                ORDER BY objectId ASC
                LIMIT {max_rows}
              """
    job = service.submit_job(query, maxrec = max_rows)
    job.run()
    job.wait(phases=['COMPLETED', 'ERROR'])
    print('Job phase is', job.phase)
    if job.phase == 'ERROR':
        job.raise_if_error()
        return
    return job.fetch_result().to_table()



def get_forced_photometry(SNR_min = 5,  Nsources = 10000):
    SNR_min = float(SNR_min)
    table = get_object_table(max_rows = Nsources)
    ids = table["objectId"].tolist()
    ids_list = ",".join(str(i) for i in ids)
    query_forced = f""" SELECT f.objectId, f.band, v.expMidptMJD, f.psfFlux, f.psfFluxErr
                         FROM dp1.ForcedSource AS f
                         JOIN dp1.Visit AS v ON v.visit = f.visit
                         WHERE f.objectId IN ({ids_list})
                         AND (f.psfFlux >= {SNR_min} * f.psfFluxErr)
                         AND f.psfFlux > 0
                      """
    service = get_tap_service("tap")
    job = service.submit_job(query_forced)
    job.run()
    job.wait(phases=['COMPLETED', 'ERROR'])
    print('Job phase is', job.phase)
    if job.phase == 'ERROR':
        job.raise_if_error()
        return
    return job.fetch_result().to_table().to_pandas()

    
    