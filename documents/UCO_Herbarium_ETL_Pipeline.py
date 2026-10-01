###################################################################################################
####
####UCO Herbarium ETL Pipeline
####created by: Marc Wright
####created on: 06.01.2026
####
###################################################################################################

import numpy as np
import pandas as pd
import geopandas as gpd
import os

##set current working directory to location of Python script file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

##location of CSUoccurrences csv file
csuFile = "./data/CSUoccurrences.csv"
csuData = pd.read_csv(csuFile, encoding = "latin-1",
                      low_memory = False,
                      na_values = [""],
                      dtype = {"identificationQualifier": str,
                               "references": str})
csuData = csuData.reset_index(drop = True)

##location of okCounties csv file
countiesFile = "./data/okCounties.geojson"
countyData = gpd.read_file(countiesFile)

##location of okTrackingList csv file
trackFile = "./data/okStateTracking.csv"
trackData = pd.read_csv(trackFile)

##create logs directory if it doesn't exist
os.makedirs("./logs/", exist_ok = True)

##create outputFiles directory if it doesn't exist
os.makedirs("./outputFiles/", exist_ok = True)


###################################################################################################
####
####Define sensitive species list for redaction in final dataset;
####this list includes all species within the Orchidaceae family, as
####well as Echinocactus texensis, a federally listed endangered species
####
####Note: to add additional species to the sensitive list, simply add the exact species name to
####the list in the following format:
####
####     excludedSpecies = [
####          "Echinocactus texensis",
####          "Genus species"
####     ]
####
####
###################################################################################################

excludedSpecies = [
    "Echinocactus texensis"
]

excludedFamilies = [
    "Orchidaceae"
]

###################################################################################################
####
####Block #1: Centroid assignment for records with null coordinates
####
###################################################################################################

##initial print statement
print("Beginning UCO Herbarium ETL Pipeline Script")
print("\n")

##print statement informing user that centroid calculation is taking place on entries
##with null coordinates and that results will be printed to console
print("Calculating county centroids for records with null coordinate values...")

##normalize county names in both datasets to ensure they match
csuData["cleanName"] = csuData["county"].str.strip().str.title()
countyData["cleanName"] = countyData["NAME"].str.strip().str.title()

##project into EPSG: 5070 to calculate accurate centroids
countyData["county_lat"] = countyData.geometry.to_crs(epsg = 5070).centroid.to_crs(epsg = 4326).y
countyData["county_lon"] = countyData.geometry.to_crs(epsg = 5070).centroid.to_crs(epsg = 4326).x

##create new columns in df for calculated Lat / Long 
##and copy original coordindates
csuData["calcLat"] = csuData["decimalLatitude"]
csuData["calcLon"] = csuData["decimalLongitude"]

##create a mask for rows where calcLat / calcLon are null
latMask = csuData["calcLat"].isnull()
lonMask = csuData["calcLon"].isnull()

##for rows where calcLat / calcLon are null, 
##fill in the values from the gdf using the cleanName as the key
lookupLat = countyData.set_index("cleanName")["county_lat"]
lookupLon = countyData.set_index("cleanName")["county_lon"]

##create boolean field for records that had centroid assigned
csuData["centroidAssigned"] = latMask

##use .loc to fill in the missing coordinate values in df
csuData.loc[latMask, "calcLat"] = csuData.loc[latMask, "cleanName"].map(lookupLat)
csuData.loc[lonMask, "calcLon"] = csuData.loc[lonMask, "cleanName"].map(lookupLon)

##test print statement to verify that records with null coordinates have been filled
print("Number of null values in calcLat and calcLon after centroid calculation:")
print(csuData[["calcLat", "calcLon"]].isna().sum())
print("\n")

##test to see is any records still have null values in calcLat / calcLon after centroid assignment;
##if so, flag these records as unreferenceable in georef_notes and export log file for review.
##if no values are still null, print statement confirming that all records have coordinates or county names
##for georeferencing and proceed with ETL pipeline
stillNull = csuData["calcLat"].isna()
numStillNull = stillNull.sum()
csuData.loc[stillNull, "georef_notes"] = "UNREFERENCEABLE - no coordinates, no county"
unreferencedSpecimensFile = "./logs/unreferenceableSpecimens.csv"
print(f"Records flagged as unreferenceable: {numStillNull}")
if numStillNull > 0:
    print(f"NOTE: check county column in '{unreferencedSpecimensFile}' log file for null county name values...")
    print("\n")
    print(f"Exporting unreferenceable records for manual review to '{unreferencedSpecimensFile}'...")
    csuData[stillNull].to_csv(unreferencedSpecimensFile, index = False)
    print(f"Dropping {numStillNull} unreferenceable specimens from CSUoccurrences.csv...")
    csuData = csuData[~stillNull].reset_index(drop = True)
    print("\n")
else:
    print("No unreferenceable records found - all records have either coordinates or county name for georeferencing.")
    print("Continuing with ETL pipeline...")
    print("\n")


###################################################################################################
####
####Block #2: Geographic precision column creation 
####
###################################################################################################

##block 2 initiation print statement
print("Creating geographicPrecision field and populating based on three values:")
print("    Precise, Assigned Centroid, Precision Unverified")
print("\n")

##create geographicPrecision field and populate based on conditions
conditions = [
    (csuData["georeferenceVerificationStatus"] == True),
    (csuData["centroidAssigned"] == True),
    ((csuData["centroidAssigned"] == False) & 
     (csuData["georeferenceVerificationStatus"].isna()) |
     (csuData["georeferenceVerificationStatus"] == False))
]

choices = ["PRECISE", "ASSIGNED CENTROID", "PRECISION UNVERIFIED"]
csuData["geographicPrecision"] = np.select(conditions,
                                           choices,
                                           default = "Unknown")

preciseRecords = len(csuData[(csuData["geographicPrecision"]) == "PRECISE"])
assignedCentroidRecords = len(csuData[(csuData["geographicPrecision"]) == "ASSIGNED CENTROID"])
precisionUnverifiedRecords = len(csuData[(csuData["geographicPrecision"]) == "PRECISION UNVERIFIED"])
unknownRecords = len(csuData[(csuData["geographicPrecision"]) == "Unknown"])
totalRecords = len(csuData)

##print statements to sum number of records in each category of geographicPrecision
print("""NOTE: 'ASSIGNED CENTROID' indicates records where county centroid was calculated
      and assigned due to no value given in the georeferenceVerificationStatus field and
      no geographic coordinates given; 'PRECISION UNVERIFIED' indicates records where geographic
      coordinates were provided, but geographicVerificationStatus had null or 'FALSE' value;
      coordinate accuracy could not be verified for these records.""")
print("\n")
print(f"Total number of specimen records: {totalRecords}")
print(f"Total number of geographically precise records: {preciseRecords}")
print(f"Total number of records with calculated county centroid: {assignedCentroidRecords}")
print(f"Total number of records with unverified precision: {precisionUnverifiedRecords}")
print("\n")
print("Percentages of each category:")
print(f"     Geographically precise records: {round(preciseRecords/totalRecords * 100,2)}%")
print(f"     Records with calculated county centroid: {round(assignedCentroidRecords/totalRecords * 100,2)}%")
print(f"     Records with unverified precision: {round(precisionUnverifiedRecords/totalRecords * 100,2)}%")
print(f"     Records outside of these categories: {round(unknownRecords/totalRecords * 100,2)}%")
print("\n")

##unknown condition: flags a warning if any records exist outside of the three conditions
if unknownRecords > 0:
    print(f"Warning: {unknownRecords} records fell into unknown category - review geographicPrecision values")
    print("\n")


###################################################################################################
####
####Block #3: Spatial join to check for geographic accuracy between county and coordinates
####
###################################################################################################

##assign csuOccurrences to geodataframe
csuGeoData = gpd.GeoDataFrame(
    csuData,
    geometry = gpd.points_from_xy(csuData.calcLon,
                                  csuData.calcLat),
    crs = "EPSG:4326"
)

##create a subset of countyData with only necessary columns for spatial join
reducedCountyRecords = countyData[["NAME", "GEOID", "geometry"]]

##perform spatial join between csuOccurrences and OKcounties to
##ensure geographic precision
joinedPointsWithCounty = gpd.sjoin(csuGeoData, reducedCountyRecords,
                                   how = "left",
                                   predicate = "within")

##create match column that formats county name between two datasets
joinedPointsWithCounty["countyMatch"] = (
    joinedPointsWithCounty["county"].str.strip().str.lower() ==
    joinedPointsWithCounty["NAME"].str.strip().str.lower()
)

notInOK = joinedPointsWithCounty[joinedPointsWithCounty["NAME"].isna()]
misMatched = joinedPointsWithCounty[(~joinedPointsWithCounty["countyMatch"]) &
                                    (joinedPointsWithCounty["NAME"].notna())
]

##print statements to summarize the results of the spatial join and accuracy check
print("Summary of spatial join and geographic accuracy check:")
print(f"Total points analyzed: {len(joinedPointsWithCounty)}")
print(f"Points listed in wrong county: {len(misMatched)}")
print(f"Points listed outside of Oklahoma: {len(notInOK)}")
print("Removing records listed outside of Oklahoma from dataset...")
print("\n")

##export mismatched entries w/ print statement showing location of export file
exportMismatches = misMatched[["id", "county", "stateProvince", "NAME",
                               "decimalLongitude", "decimalLatitude"]]
mismatchFile = "./logs/countyMismatches.csv"
exportMismatches.to_csv(mismatchFile, index = False)
print(f"Exporting mismatched specimen entries to: '{mismatchFile}...'")

##export entries not in Oklahoma w/ print statement showing location of export file
exportNotInOK = notInOK[["id", "county", "stateProvince", "NAME",
                               "decimalLongitude", "decimalLatitude"]]
notInOkFile = "./logs/entriesNotInOK.csv"
exportNotInOK.to_csv(notInOkFile, index = False)
print(f"Exporting specimen entries outside of Oklahoma to: '{notInOkFile}...'")
print("\n")

##new CSUoccurrences dataset with county FIPS information needed for geospatial analysis
csuData = joinedPointsWithCounty[~joinedPointsWithCounty["id"].isin(notInOK["id"])]
csuData = csuData.drop(columns = ["index_right"])
csuData = csuData.rename(columns = {"NAME": "countyNameFromJoin", "GEOID": "countyFIPS"})

###################################################################################################
####
####Block #4: Table join to add tracking list information 
####
###################################################################################################

csuData["genusSpecies"] = (
    csuData["genus"].str.strip().str.capitalize() + " " +
    csuData["specificEpithet"].str.strip().str.lower() +
    np.where(
        csuData["verbatimTaxonRank"].notna(),
        " " + csuData["verbatimTaxonRank"].str.replace("subsp.", "ssp.") + " " +
        csuData["infraspecificEpithet"].str.strip(),
        ""
    )
)

##merge tracking list with csuData to add tracking status information
csuData = csuData.merge(trackData[["Scientific Name", "State Rank", "Global Rank", "Federal Status"]], 
                        left_on = "genusSpecies", right_on = "Scientific Name", how = "left")
csuData["onTrackingList"] = csuData["State Rank"].notna()

matchRecords = csuData[csuData["genusSpecies"].isin(trackData["Scientific Name"])]
matchCount = len(matchRecords)
trackingMatchFile = "./logs/trackingListMatches.csv"

##print statements to confirm merge and # of successful matches 
##with tracking list and export log of successful matches
print("Merging OK tracking list with CSUoccurrences dataset to add ranking information...")
print(f"Number of records in CSUoccurrences after merge: {len(csuData)}")
print(f"Number of successful joins with tracking list: {matchCount}")
print("\n")
matchRecords.to_csv(trackingMatchFile, index = False)
print(f"Exporting successful joins with tracking list to: '{trackingMatchFile}...'")
print("\n")


###################################################################################################
####
####Block #5: Redaction of sensitive species records and export of cleaned dataset
####
###################################################################################################

print("Redacting records of sensitive species and exporting cleaned dataset...")

##filter out entries that fall within the excludedSpecies and excludedFamilies lists
finalFilteredData = csuData[
    (~csuData["family"].isin(excludedFamilies)) &
    (~csuData["scientificName"].isin(excludedSpecies))
]

##variables for number of sensitive species records and for final cleaned file
removedCount = len(csuData) - len(finalFilteredData)
cleanedFile = "./outputFiles/cleanedCSUoccurrences.csv"

##create separate dataframe of all excluded records and export to csv
excludedSpecimens = csuData[
    (csuData["family"].isin(excludedFamilies)) |
    (csuData["scientificName"].isin(excludedSpecies))
]
excludedSpecimensFile = "./logs/excludedSensitiveSpecimens.csv"
excludedSpecimens.to_csv(excludedSpecimensFile, index = False)

#print statement summarizing number of records and export information#
print(f"Number of specimens removed that belong to the excluded families / species: {removedCount}")
print(f"Exporting records of excluded sensitive species to: '{excludedSpecimensFile}...'")
print("\n")

##create final specimen dataset and export to csv
finalFilteredData.to_csv(cleanedFile, index = False)
print(f"Exporting cleaned dataset to: '{cleanedFile}...'")
print("\n")
print("CSU occurrences dataset has been cleaned and exported for use in the UCO Herbarium Specimen geospatial analysis.")
print("\n")

###################################################################################################
####
####Block #6: Aggregation counts of # of specimens per county and # of state rankings per county
####
###################################################################################################

print("Aggregating specimen and tracking list counts to county level...")
print("\n")

##reset dataframe index and set countyFIPS as string to match okCounty dataset
finalFilteredData = finalFilteredData.reset_index(drop = True)
finalFilteredData["countyFIPS"] = finalFilteredData["countyFIPS"].astype(str)

##create temporary dataset to store countyFIPS with specimen count per county
specCnts = finalFilteredData["countyFIPS"].value_counts().reset_index()
specCnts.columns = ["countyFIPS", "specCnt"]

##merge countyData with specCnts
countyData = countyData.merge(specCnts, left_on = "GEOID", right_on = "countyFIPS", how = "left")
countyData["specCnt"] = countyData["specCnt"].fillna(0)

##create temporary dataset to store countyFIPS with tracking list join count per county
trackCnts = finalFilteredData[finalFilteredData["onTrackingList"]]["countyFIPS"].value_counts().reset_index()
trackCnts.columns = ["countyFIPS", "trackCnt"]

##merge countyData with trackCnts
countyData = countyData.merge(trackCnts, left_on = "GEOID", right_on = "countyFIPS", how = "left")
countyData["trackCnt"] = countyData["trackCnt"].fillna(0)

##create temp dataset that creates pivot columns of state ranking per county
rankingPivot = finalFilteredData.groupby(["countyFIPS", "State Rank"]).size().unstack(fill_value = 0)
rankingPivot.columns = [f"rank{col}" for col in rankingPivot.columns]
rankingPivot = rankingPivot.reset_index()

##merge countyData with trackCnts
countyData = countyData.merge(rankingPivot, left_on = "GEOID", right_on = "countyFIPS", how = "left")

##filter out which columns to permanently join to countyExport shp file
rankCols = ["rankS1", "rankS2", "rankS3", "rankSH"]
countyData[rankCols] = countyData[rankCols].fillna(0)

##project into EPSG: 5070 to calculate accurate county area in square miles
countyData["areaSqMi"] = countyData.geometry.to_crs(epsg = 5070).area / 2.59e6  # Convert to square miles

##calculate density of specimens per square mile and add as new column
countyData["densSqMi"] = countyData["specCnt"] / countyData["areaSqMi"]

##calculate density of tracked species per square mile and add as new column
countyData["trkDnsSqMi"] = countyData["trackCnt"] / countyData["areaSqMi"]

##create countySHP folder as needed and export finalized county shp files
os.makedirs("./outputFiles/countySHP", exist_ok = True)
keepFields = ["GEOID", "NAME", "specCnt", "trackCnt", "rankS1", "rankS2", "rankS3", "rankSH", "areaSqMi", "densSqMi", "trkDnsSqMi", "geometry"]
countyExport = countyData[keepFields]
countyExportSHP = "./outputFiles/countySHP/countyDataOutput.shp"

##print statements informing user of file export
print(f"Exporting aggregated county information in shapefile format to: {countyExportSHP}")
print("\n")
countyExport.to_file(countyExportSHP)

###################################################################################################
####
####Block #7: This block creates two additional CSV log files:
####              1: a count of number of plant families within each county in long format, and
####              2: a percentage  each plant family equates to within the total number of 
####                    plant specimens
####
###################################################################################################

print("Creating csv file of number of plant family instances within each county")

##set variable of log file location
exportFamilyPivot = "./outputFiles/familyCntPerCounty.csv"
exportFamilyPercent = "./outputFiles/familyPercentage.csv"

##set up filter to export entries with no family value
noFamilyRecords = finalFilteredData[finalFilteredData["family"].isnull()]
noFamilyExport = "./logs/noFamilyValue.csv"
print(f"{len(noFamilyRecords)} records with null family values are excluded from {exportFamilyPivot}")
print(f"Exporting these entries to '{noFamilyExport}' for manual review.")
noFamilyRecords.to_csv(noFamilyExport, index = False)

##create temp dataset that filters data out by county and then create family count
familyPivot = finalFilteredData.groupby(["countyFIPS", "family"]).size().reset_index()
familyPivot.columns = ["countyFIPS", "family", "specCount"]

##export to csv log file
familyPivot.to_csv(exportFamilyPivot, index = False)
print(f"CSV log file created and saved to: {exportFamilyPivot}!")
print("\n")

print("Creating csv file of plant families and their respective percentage within dataset")

familyPercentage = finalFilteredData["family"].value_counts().reset_index()
familyPercentage.columns = ["family", "count"]
familyPercentage["percentage"] = round((familyPercentage["count"] / len(finalFilteredData) * 100), 2)

##export to csv log file
familyPercentage.to_csv(exportFamilyPercent, index = False)
print(f"CSV log file created and saved to: {exportFamilyPercent}!")
print("\n")

print("ETL pipeline complete!")


###################################################################################################
####
####End of UCO Herbarium ETL Pipeline
####
###################################################################################################

