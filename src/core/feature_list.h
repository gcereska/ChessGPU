#pragma once

// The search feature sets (docs/ARCHITECTURE.md, Search), listed once: the
// feature-set structs (core/features.h), FeatureKind, names, the variant
// switches on the CPU and the GPU, the UCI option and CMake's default for
// CHESSGPU_FEATURES (with a kernel unit per set and table layout) all come
// from this list. Adding a set is one line: its techniques are flags
// (core/features.h, namespace ft). Keep names in the order
// tt[-noqs|-order], ext, killers, see, qchecks, pvs, nmp, lmr, fut.
//
//   none        plain alpha-beta with quiescence
//   tt          transposition table: probe and store at every node, quiescence
//               included; stored results cut off, stored moves are tried first
//   tt-noqs     the same, without the table in quiescence search
//   tt-order    the table only orders moves (no cutoffs, no quiescence): never
//               changes a search's score, so tests can demand exact results
//   ext         check extensions          killers  killer moves
//   see         SEE and delta pruning in quiescence
//   qchecks     quiet checks at quiescence's first ply, besides the captures
//   pvs         principal variation search (zero windows after a node's first move)
//   nmp         null-move pruning         lmr      late move reductions
//   fut         futility pruning
//
// X(Kind, Type, "name", technique flags)
#define CB_FEATURE_SETS(X) \
    X(None, FeaturesNone, "none", 0)                                                                                              \
    X(TT, FeaturesTT, "tt", ft::kTTAll)                                                                                           \
    X(TTNoQS, FeaturesTTNoQS, "tt-noqs", ft::kTT | ft::kTTCutoffs)                                                                \
    X(TTOrder, FeaturesTTOrder, "tt-order", ft::kTT)                                                                              \
    X(TTNmp, FeaturesTTNmp, "tt-nmp", ft::kTTAll | ft::kNullMove)                                                                 \
    X(TTLmr, FeaturesTTLmr, "tt-lmr", ft::kTTAll | ft::kLmr)                                                                      \
    X(TTNmpLmr, FeaturesTTNmpLmr, "tt-nmp-lmr", ft::kTTAll | ft::kNullMove | ft::kLmr)                                            \
    X(TTFut, FeaturesTTFut, "tt-fut", ft::kTTAll | ft::kFutility)                                                                 \
    X(TTLmrFut, FeaturesTTLmrFut, "tt-lmr-fut", ft::kTTAll | ft::kLmr | ft::kFutility)                                            \
    X(TTExt, FeaturesTTExt, "tt-ext", ft::kTTAll | ft::kCheckExt)                                                                 \
    X(TTKillers, FeaturesTTKillers, "tt-killers", ft::kTTAll | ft::kKillers)                                                      \
    X(TTExtKillers, FeaturesTTExtKillers, "tt-ext-killers", ft::kTTAll | ft::kCheckExt | ft::kKillers)                            \
    X(TTKillersSee, FeaturesTTKillersSee, "tt-killers-see", ft::kTTAll | ft::kKillers | ft::kQPrune)                              \
    X(TTExtKillersSee, FeaturesTTExtKillersSee, "tt-ext-killers-see", ft::kTTAll | ft::kCheckExt | ft::kKillers | ft::kQPrune)    \
    X(TTKillersNmp, FeaturesTTKillersNmp, "tt-killers-nmp", ft::kTTAll | ft::kNullMove | ft::kKillers)                            \
    X(TTKillersLmr, FeaturesTTKillersLmr, "tt-killers-lmr", ft::kTTAll | ft::kLmr | ft::kKillers)                                 \
    X(TTExtKillersNmp, FeaturesTTExtKillersNmp, "tt-ext-killers-nmp", ft::kTTAll | ft::kNullMove | ft::kCheckExt | ft::kKillers)  \
    X(TTExtKillersLmr, FeaturesTTExtKillersLmr, "tt-ext-killers-lmr", ft::kTTAll | ft::kLmr | ft::kCheckExt | ft::kKillers)  \
    X(TTKillersSeePvs, FeaturesTTKillersSeePvs, "tt-killers-see-pvs", ft::kTTAll | ft::kKillers | ft::kQPrune | ft::kPvs)  \
    X(TTKillersSeeQchecks, FeaturesTTKillersSeeQchecks, "tt-killers-see-qchecks", ft::kTTAll | ft::kKillers | ft::kQPrune | ft::kQChecks) \
    X(TTKillersSeeQchecksNmp, FeaturesTTKillersSeeQchecksNmp, "tt-killers-see-qchecks-nmp", ft::kTTAll | ft::kKillers | ft::kQPrune | ft::kQChecks | ft::kNullMove)
