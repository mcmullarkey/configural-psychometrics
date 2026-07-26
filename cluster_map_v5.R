##############################################################################
# cluster_map_v5.R  --  canonical item -> cluster map (SINGLE SOURCE OF TRUTH)
#
# The v5 cluster grain, after the folds decided 2026-07-12:
#   * depression   = depressed mood + anhedonia + suicidal ideation  (20 items)
#                    (was three fine labels: depmood / lossinterest / suicidality)
#   * somatoform   = all seven prior somatic labels merged into one   (14 items)
#                    (single-item "clusters" are not clusters; also a spectrum
#                     that is one cluster -- the only such case, with depression)
#   * physicalpanic= physical panic + fear-of-dying (MQRF_352)        (12 items)
#                    (fear of dying is a cardinal panic symptom)
#   * psychomot_ag_anx, agora_soc, fearavoid, sepanx : unchanged
#
# 7 clusters, 86 items. Cross-cluster is the unit of "cross-ness": a set is
# cross-cluster iff its items span >= 2 of these clusters. Every run script and
# the aggregate read THIS file so the grain is identical everywhere.
##############################################################################

CLUSTER_ITEMS <- list(

  # ---- Distress spectrum ----
  depression = c(
    # depressed mood (12)
    "MQRF_109","MQRF_111","MQRF_295","MQRF_318","MQRF_319","MQRF_324",
    "MQRF_373","MQRF_376","MQRF_504","MQRF_505","MQRF_63","MQRF_667",
    # anhedonia / loss of interest (4)
    "MQRF_110","MQRF_296","MQRF_595","MQRF_665",
    # suicidal ideation (4)
    "MQRF_303","MQRF_503","MQRF_551","MQRF_71"
  ),
  psychomot_ag_anx = c(
    "MQRF_276","MQRF_294","MQRF_304","MQRF_305","MQRF_321","MQRF_322",
    "MQRF_336","MQRF_360","MQRF_378","MQRF_379","MQRF_380","MQRF_612","MQRF_633"
  ),

  # ---- Fear spectrum ----
  agora_soc = c(
    "MQRF_332","MQRF_354","MQRF_355","MQRF_356","MQRF_357","MQRF_358",
    "MQRF_359","MQRF_564","MQRF_76","MQRF_88","MQRF_89"
  ),
  fearavoid = c(
    "MQRF_82","MQRF_83","MQRF_84","MQRF_85","MQRF_86","MQRF_87","MQRF_90"
  ),
  physicalpanic = c(
    "MQRF_337","MQRF_340","MQRF_342","MQRF_343","MQRF_344","MQRF_347",
    "MQRF_348","MQRF_349","MQRF_350","MQRF_634","MQRF_664",
    "MQRF_352"                                   # fear-of-dying, folded in
  ),
  sepanx = c(
    "MQRF_330","MQRF_331","MQRF_333","MQRF_498","MQRF_77","MQRF_78",
    "MQRF_79","MQRF_80","MQRF_81"
  ),

  # ---- Somatoform spectrum (one cluster) ----
  somatoform = c(
    "MQRF_115","MQRF_116","MQRF_384","MQRF_385","MQRF_386",   # illness anxiety
    "MQRF_345","MQRF_346",                                     # abdominal distress
    "MQRF_328",                                                # bloating
    "MQRF_460",                                                # headaches
    "MQRF_327","MQRF_362",                                     # muscle tension
    "MQRF_631","MQRF_662",                                     # fatigue
    "MQRF_341"                                                 # trembling
  )
)

# Cluster -> spectrum, for reference / secondary reporting only. Cross-CLUSTER
# (not cross-spectrum) is the operative measure.
CLUSTER_SPECTRUM <- c(
  depression = "Distress", psychomot_ag_anx = "Distress",
  agora_soc = "Fear", fearavoid = "Fear", physicalpanic = "Fear", sepanx = "Fear",
  somatoform = "Somatoform"
)

# Flat item -> cluster named vector (built from the list above).
ITEM_TO_CLUSTER <- unlist(lapply(names(CLUSTER_ITEMS),
  function(cl) setNames(rep(cl, length(CLUSTER_ITEMS[[cl]])), CLUSTER_ITEMS[[cl]])))

# When sourced at top level, (re)write the item->cluster CSV the aggregate reads.
if (sys.nframe() == 0L) {
  df <- data.frame(item = names(ITEM_TO_CLUSTER),
                   cluster = unname(ITEM_TO_CLUSTER),
                   stringsAsFactors = FALSE)
  df <- df[order(df$cluster, df$item), , drop = FALSE]
  write.csv(df, "item_cluster_map.csv", row.names = FALSE)
  cat(sprintf("Wrote item_cluster_map.csv: %d items across %d clusters.\n",
              nrow(df), length(CLUSTER_ITEMS)))
  print(table(df$cluster))
}
